#!/usr/bin/env python3
r"""
CIFAR-10 / ResNet9 (p = 672, 890)

THE GAP THIS FILLS
------------------
Their Figure 1 covers only models with p <= ~18k. For CIFAR-10 / ResNet9 their Figure 2 
reports LOG-TRACE ONLY. This script supplies the exact KL metric, a direct test of 
whether log-trace really tracks KL at p = 672, 890, and the CIFAR-10 predictive table.

UPDATES
-------
- Includes ELLA (Accelerated Linearized Laplace Approximation via Nystrom).
- Includes exact KFAC Full Rank and Diagonal Full Rank baselines computed matrix-free.
- Uses ResNet9NoBN to bypass laplace-torch BatchNorm rejections.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call, jvp, vjp
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2, 3, 4)
LAMBDA_THEIRS = 68.1          
LAMBDA_OVERRIDE = None        

N_XPRIME = 20                 
N_XTEST = 50                  
S_LIST = (20, 50, 100, 200, 500)
N_FIT = 200                   
N_TOTAL = 50000               
N_EVAL = 500                  
CG_ITERS = 600                
CG_TOL = 1e-9
EPOCHS, LR, BATCH = 30, 0.004, 256   
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# --------------------------- MODELS (no BN) -----------------------------

def conv_relu(ci, co, pool=False):
    layers = [nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CifarCNN(nn.Module):
    def __init__(self, C=10, w=(32, 72, 112)):   
        super().__init__()
        a, b, c = w
        self.f = nn.Sequential(
            conv_relu(3, a), conv_relu(a, a, pool=True),      
            conv_relu(a, b), conv_relu(b, b, pool=True),      
            conv_relu(b, c), conv_relu(c, c, pool=True),      
            nn.Flatten(), nn.Linear(c * 16, 128), nn.ReLU(inplace=True),
            nn.Linear(128, C))

    def forward(self, x):
        return self.f(x)


class ResNet9NoBN(nn.Module):
    def __init__(self, num_classes=10, w=(32, 64, 136, 136)):
        super().__init__()
        w1, w2, w3, w4 = w
        self.prep = conv_relu(3, w1)
        self.layer1 = conv_relu(w1, w2, pool=True)
        self.res1 = nn.Sequential(conv_relu(w2, w2), conv_relu(w2, w2))
        self.layer2 = conv_relu(w2, w3, pool=True)
        self.layer3 = conv_relu(w3, w4, pool=True)
        self.res2 = nn.Sequential(conv_relu(w4, w4), conv_relu(w4, w4))
        self.pool = nn.MaxPool2d(4)
        self.classifier = nn.Linear(w4, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = x + self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.res2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_data(seed):
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tf_te = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])
    tr = datasets.CIFAR10("./data", train=True, download=True, transform=tf_tr)
    tr_clean = datasets.CIFAR10("./data", train=True, download=True, transform=tf_te)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=tf_te)
    g = torch.Generator().manual_seed(seed)
    pr = torch.randperm(len(tr), generator=g)
    pt = torch.randperm(len(te), generator=g)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4),
            DataLoader(Subset(tr_clean, pr[:N_FIT].tolist()), batch_size=128),
            DataLoader(Subset(tr_clean, pr[N_FIT:N_FIT + N_XPRIME].tolist()), batch_size=64),
            DataLoader(Subset(te, pt[:N_XTEST].tolist()), batch_size=64),
            DataLoader(Subset(te, pt[N_XTEST:N_XTEST + N_EVAL].tolist()), batch_size=128),
            DataLoader(Subset(te, pt[-1000:].tolist()), batch_size=128))


def train(model, loader):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    tot = EPOCHS * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR * 10, total_steps=tot, pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step(); sched.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}/{EPOCHS}", flush=True)
    model.eval()
    return model


# ------------------ matrix-free GGN and Jacobians --------------------------
class GGN:
    def __init__(self, model, loader, lam, scale):
        self.model, self.lam, self.scale = model, float(lam), float(scale)
        self.params = {n: p.detach() for n, p in model.named_parameters()}
        self.buffers = {n: b.detach() for n, b in model.named_buffers()}
        self.names = list(self.params)
        self.shapes = [self.params[n].shape for n in self.names]
        self.numels = [self.params[n].numel() for n in self.names]
        self.offs = np.cumsum([0] + self.numels).tolist()
        self.p = int(sum(self.numels))
        self.nmv = 0
        self.batches = [(x.to(DEVICE), y.to(DEVICE)) for x, y in loader]

    def _call(self, pr, x):
        return functional_call(self.model, (pr, self.buffers), (x,))

    def unflat(self, v):
        return {n: v[i:i + k].view(s) for n, i, k, s in
                zip(self.names, self.offs, self.numels, self.shapes)}

    def flat(self, dd):
        return torch.cat([dd[n].reshape(-1) for n in self.names])

    def __call__(self, v):
        self.nmv += 1
        vd = self.unflat(v.to(DEVICE, torch.float32))
        acc = torch.zeros(self.p, device=DEVICE, dtype=torch.float32)
        for x, _ in self.batches:
            def f(pr):
                return self._call(pr, x)
            out, Jv = jvp(f, (self.params,), (vd,))
            pr = torch.softmax(out, 1)
            HJv = pr * Jv - pr * (pr * Jv).sum(1, keepdim=True)
            _, vjp_fn = vjp(f, self.params)
            acc = acc + self.flat(vjp_fn(HJv)[0])
        return self.scale * acc + self.lam * v

    def jac(self, x):
        def f(pr):
            return self._call(pr, x.unsqueeze(0))
        out, vjp_fn = vjp(f, self.params)
        rows = []
        for c in range(out.shape[1]):
            e = torch.zeros_like(out); e[:, c] = 1.0
            rows.append(self.flat(vjp_fn(e)[0]))
        return torch.stack(rows), out[0]

    def jac_stack(self, loader, on_cpu=False):
        Js, lg = [], []
        for xb, _ in loader:
            for x in xb:
                J, o = self.jac(x.to(DEVICE))
                Js.append(J.cpu() if on_cpu else J)
                lg.append(o.cpu() if on_cpu else o)
        J = torch.cat(Js); L = torch.stack(lg)
        return J, L


def build_V(op):
    cols = []
    for x, _ in op.batches:
        for i in range(x.shape[0]):
            J, logits = op.jac(x[i])
            pr = torch.softmax(logits, 0)
            H = torch.diag(pr) - torch.outer(pr, pr)
            w, U = torch.linalg.eigh(H.double())
            Hh = (U * w.clamp_min(0).sqrt()) @ U.T
            cols.append((Hh.to(torch.float32) @ J).T)     
    V = torch.cat(cols, 1) * math.sqrt(op.scale)
    return V


def cg_solve(op, B, iters=CG_ITERS, tol=CG_TOL, tag=""):
    X = torch.zeros_like(B)
    worst, t0 = 0.0, time.time()
    for j in range(B.shape[1]):
        b = B[:, j]
        x = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = float(torch.dot(r, r)); bn = max(math.sqrt(rs), 1e-30)
        for _ in range(iters):
            Ap = op(p)
            den = float(torch.dot(p, Ap))
            if den <= 0: break
            a = rs / den
            x = x + a * p; r = r - a * Ap
            rs2 = float(torch.dot(r, r))
            if math.sqrt(rs2) < tol * bn:
                rs = rs2; break
            p = r + (rs2 / rs) * p
            rs = rs2
        X[:, j] = x
        worst = max(worst, math.sqrt(rs) / bn)
    return X, worst


# ---------------------------- Krylov / Sigma_m -----------------------------
def lanczos(Av, v, m):
    n = v.numel()
    Q = torch.zeros(n, m, device=DEVICE, dtype=torch.float32)
    al = torch.zeros(m, dtype=torch.float64)
    be = torch.zeros(m, dtype=torch.float64)
    Q[:, 0] = v / v.norm(); b = 0.0
    qp = torch.zeros(n, device=DEVICE, dtype=torch.float32)
    for j in range(m):
        w = Av(Q[:, j]) - b * qp
        a = torch.dot(Q[:, j], w)
        w = w - a * Q[:, j]
        w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        al[j] = a.item(); b = w.norm().item(); be[j] = b
        qp = Q[:, j].clone()
        if b < 1e-10:
            m = j + 1; Q, al, be = Q[:, :m], al[:m], be[:m]; break
        if j + 1 < m:
            Q[:, j + 1] = w / b
    T = (torch.diag(al) + torch.diag(be[:-1], 1) + torch.diag(be[:-1], -1)).double()
    return Q, T, float(be[-1]), (w / b if b > 1e-10 else torch.zeros_like(w))

def reproj_tridiag(Av, Pr, w, l):
    nw2 = float(torch.dot(w, w))
    if nw2 < 1e-24: return torch.zeros(0, 0, dtype=torch.float64), 0.0
    n = w.numel()
    V = torch.zeros(n, l, device=DEVICE, dtype=torch.float32)
    al = torch.zeros(l, dtype=torch.float64)
    be = torch.zeros(max(l - 1, 1), dtype=torch.float64)
    V[:, 0] = w / math.sqrt(nw2); b, st = 0.0, 0
    qp = torch.zeros(n, device=DEVICE, dtype=torch.float32)
    for j in range(l):
        z = Pr(Av(V[:, j])) - b * qp
        a = torch.dot(V[:, j], z)
        z = Pr(z - a * V[:, j])
        z = Pr(z - V[:, :j + 1] @ (V[:, :j + 1].T @ z))
        al[j] = a.item(); st = j + 1
        nb = z.norm().item(); qp = V[:, j].clone()
        if j + 1 < l:
            if nb < 1e-10: break
            be[j] = nb; b = nb; V[:, j + 1] = z / nb
    return (torch.diag(al[:st]) + torch.diag(be[:st - 1], 1) + torch.diag(be[:st - 1], -1)).double(), nw2

def quad_inv(T, nw2, power=1):
    if T.numel() == 0: return 0.0
    e1 = torch.zeros(T.shape[0], dtype=torch.float64); e1[0] = 1.0
    v = e1.clone()
    for _ in range(power): v = torch.linalg.solve(T, v)
    return float(nw2 * torch.dot(e1, v))

def cg_proj(Av, b, iters, Pr, tol=1e-10):
    b = Pr(b)
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = float(torch.dot(r, r)); bn = max(math.sqrt(rs), 1e-30)
    for _ in range(iters):
        Ap = Pr(Av(p))
        den = float(torch.dot(p, Ap))
        if den <= 0: break
        a = rs / den
        x = Pr(x + a * p); r = Pr(r - a * Ap)
        rs2 = float(torch.dot(r, r))
        if math.sqrt(rs2) < tol * bn: break
        p = Pr(r + (rs2 / rs) * p)
        rs = rs2
    return x

class SigmaM:
    def __init__(self, op, m, gen, lam, n_pslq=20, l_pslq=20, l_bnd=25, l_cg=100, uncorrected=False):
        Av = op
        p = op.p
        v0 = torch.randn(p, device=DEVICE, dtype=torch.float32, generator=gen)
        self.Q, T, bm, qn = lanczos(Av, v0, m)
        self.m = m = self.Q.shape[1]
        self.p, self.lam = p, lam
        self.Ti = torch.linalg.inv(T).to(torch.float32).to(DEVICE)
        self.unc = uncorrected
        if uncorrected:
            self.omega = 0.0; self.lam_omega = 0.0
            return

        Pr = lambda z: z - self.Q @ (self.Q.T @ z)
        Th, nw2 = reproj_tridiag(Av, Pr, qn, l_bnd)
        a1 = quad_inv(Th, nw2, 1); a2 = quad_inv(Th, nw2, 2)
        gamma = bm ** 2 * float(torch.linalg.inv(T)[m - 1, m - 1])
        den = 1.0 - gamma * a1
        e1 = torch.zeros(Th.shape[0], dtype=torch.float64); e1[0] = 1.0
        Thg = Th - gamma * torch.outer(e1, e1)
        p2 = quad_inv(Thg, nw2, 2); p3 = quad_inv(Thg, nw2, 3); p4 = quad_inv(Thg, nw2, 4)
        self.s11 = a1 / den
        self.sig_u = p3 / p2 if p2 > 0 else 0.0
        self.c_norm = (math.sqrt(max(p4 / p2 - (p3 / p2) ** 2, 0.0)) if p2 > 0 else 0.0)

        acc = 0.0
        for _ in range(n_pslq):
            xi = torch.randn(p, device=DEVICE, dtype=torch.float32, generator=gen)
            Thp, nw2p = reproj_tridiag(Av, Pr, Pr(xi), l_pslq)
            acc += quad_inv(Thp, nw2p, 1)
        tr_S = acc / n_pslq + gamma * a2 / den

        em = torch.zeros(m, dtype=torch.float64); em[m - 1] = 1.0
        self.q_par = bm * (self.Q @ (torch.linalg.inv(T) @ em).to(torch.float32).to(DEVICE))
        x = cg_proj(Av, bm * qn, l_cg, Pr)
        self.q_perp = Pr((x / bm) / den) if bm > 1e-10 else torch.zeros_like(x)
        nq = float(self.q_perp.norm())
        self.u = (self.q_perp / nq if nq > 1e-10 else torch.zeros_like(self.q_perp))
        self.omega = max((tr_S - self.sig_u) / max(p - m - 1, 1), 0.0)
        self.lam_omega = self.omega * lam
        self.tr_S = tr_S

    def apply(self, V):
        out = self.Q @ (self.Ti @ (self.Q.T @ V))
        if self.unc: return out
        a = self.q_par @ V; b = self.q_perp @ V; c = self.u @ V
        out = (out + self.s11 * torch.outer(self.q_par, a) - torch.outer(self.q_par, b) 
               - torch.outer(self.q_perp, a) + self.sig_u * torch.outer(self.u, c))
        return out + self.omega * (V - self.Q @ (self.Q.T @ V) - torch.outer(self.u, c))


# ---------------------------- Subspace & Full Rank Models ------------------------------
def sigma_from_P(P, op, V, JX, lam):
    if P.dtype == torch.long:
        VtP = V[P, :].T                      
        JP = JX[:, P]
        PtP = torch.eye(P.numel(), device=DEVICE, dtype=torch.float32)
    else:
        VtP = V.T @ P
        JP = JX @ P
        PtP = P.T @ P
    M = (VtP.T @ VtP).double() + lam * PtP.double()
    M = 0.5 * (M + M.T)
    M += 1e-8 * float(torch.diagonal(M).mean()) * torch.eye(M.shape[0], device=DEVICE, dtype=torch.float64)
    return (JP.double() @ torch.linalg.solve(M, JP.double().T))

def lowrank_P(psi_apply, Jxp, s):
    W = psi_apply(Jxp.T)
    Mk = (Jxp @ W).double()
    Mk = 0.5 * (Mk + Mk.T)
    ev, U = torch.linalg.eigh(Mk)
    idx = torch.argsort(ev, descending=True)[:s]
    return W @ U[:, idx].to(torch.float32)

def subset_P(scores, s):
    return torch.argsort(scores, descending=True)[:s]

def shift_sigma(P, op, V, JX, lam):
    U = torch.linalg.qr(P)[0]
    VtU = V.T @ U                            
    G = (VtU.T @ VtU).double()
    ev, W = torch.linalg.eigh(0.5 * (G + G.T))   
    Us = U @ W.to(torch.float32)
    inner = (1.0 / (ev.clamp_min(0.0) + lam) - 1.0 / lam)
    JU = (JX @ Us).double()
    return (JU * inner) @ JU.T + (JX.double() @ JX.double().T) / lam

# Exact Projections (Matrix-Free) for Full-Rank & ELLA
def get_S_diag(JX, diagA):
    return (JX.double() * (1.0 / diagA.double())) @ JX.double().T

def get_S_kfac(JX, psi_kfac):
    return JX.double() @ psi_kfac(JX.T).double()

def get_S_ella(JX, Jxp, lam):
    K = (Jxp @ Jxp.T).double()
    I = torch.eye(K.shape[0], dtype=torch.float64, device=DEVICE)
    invM = torch.linalg.inv(K / lam + I)
    J_jx = JX.double() @ Jxp.double().T
    term1 = (JX.double() @ JX.double().T) / lam
    term2 = (J_jx @ invM @ J_jx.T) / (lam * lam)
    return term1 - term2

def build_ella_var_fn(Jxp, lam):
    K = (Jxp @ Jxp.T).double()
    I = torch.eye(K.shape[0], dtype=torch.float64, device=DEVICE)
    invM = torch.linalg.inv(K / lam + I)
    def f(Jc, C):
        term1 = (Jc.double() * Jc.double()).sum(1) / lam
        Jc_jx = Jc.double() @ Jxp.double().T
        term2 = (Jc_jx @ invM * Jc_jx).sum(1) / (lam * lam)
        return (term1 - term2).view(-1, C).float()
    return f

# -------------------------------- metrics ----------------------------------
def blocks(S, C):
    n = S.shape[0] // C
    return torch.stack([S[i * C:(i + 1) * C, i * C:(i + 1) * C] for i in range(n)])

def kl_blocks(Shat, Sref, C, jitter=1e-10):
    Bh, Br = blocks(Shat, C), blocks(Sref, C)
    I = torch.eye(C, device=DEVICE, dtype=torch.float64)
    out, ndeg = [], 0
    for A_, B_ in zip(Bh, Br):
        sc = float(torch.trace(B_)) / C          
        A_ = A_ + jitter * sc * I; B_ = B_ + jitter * sc * I
        ld, ldB = float(torch.logdet(A_)), float(torch.logdet(B_))
        if (not math.isfinite(ld)) or ld < ldB - C * math.log(1e6):
            ndeg += 1; continue
        try:
            sol = torch.linalg.solve(A_, B_)
            v = float(0.5 * (torch.trace(sol) - C + ld - torch.logdet(B_)))
            out.append(v) if math.isfinite(v) else None
        except Exception:
            ndeg += 1
    return (float(np.mean(out)) if out else float("inf"), ndeg / Bh.shape[0])

def disp(Shat, Sref, C):
    r = (blocks(Shat, C).diagonal(dim1=1, dim2=2).sum(1) / blocks(Sref, C).diagonal(dim1=1, dim2=2).sum(1)).sqrt()
    return float(r.median())

def subspace_var_fn(P, V, lam, p):
    if P.dtype == torch.long:
        VtP = V[P, :].T
        PtP = torch.eye(P.numel(), device=DEVICE, dtype=torch.float32)
        sel = P
    else:
        VtP = V.T @ P
        PtP = P.T @ P
        sel = None
    M = (VtP.T @ VtP).double() + lam * PtP.double()
    M = 0.5 * (M + M.T)
    M += 1e-8 * float(torch.diagonal(M).mean()) * torch.eye(M.shape[0], device=DEVICE, dtype=torch.float64)
    L = torch.linalg.cholesky(M)

    def f(Jc, C):
        PtJ = (Jc[:, sel].T if sel is not None else P.T @ Jc.T).double()
        W = torch.linalg.solve_triangular(L, PtJ, upper=False)
        return (W * W).sum(0).view(-1, C).float()
    return f

def sigma_m_var_fn(sm):
    def f(Jc, C):
        return ((Jc @ sm.apply(Jc.T)).diagonal()).view(-1, C)
    return f

def shift_var_fn(P, V, lam):
    U = torch.linalg.qr(P)[0]
    VtU = V.T @ U
    G = (VtU.T @ VtU).double()
    ev, W = torch.linalg.eigh(0.5 * (G + G.T))
    Us = U @ W.to(torch.float32)
    inner = (1.0 / (ev.clamp_min(0.0) + lam) - 1.0 / lam).float()

    def f(Jc, C):
        JU = Jc @ Us
        d = (JU * JU * inner).sum(1) + (Jc * Jc).sum(1) / lam
        return d.view(-1, C)
    return f

def fit_temperature(op, loader):
    lg, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            lg.append(op.model(xb.to(DEVICE)).cpu()); ys.append(yb)
    lg, y = torch.cat(lg), torch.cat(ys)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())

def predictive_table(op, var_fns, loader, C, temp, chunk=25):
    keys = list(var_fns) + ["MAP", "temp. scaling"]
    acc = {k: [] for k in keys}
    labs, xs, ys = [], [], []
    for xb, yb in loader:
        xs.append(xb); ys.append(yb)
    X, Y = torch.cat(xs), torch.cat(ys)
    for i in range(0, X.shape[0], chunk):
        Js, lg = [], []
        for x in X[i:i + chunk]:
            J, o = op.jac(x.to(DEVICE))
            Js.append(J); lg.append(o)
        Jc = torch.cat(Js); L = torch.stack(lg)
        labs.append(Y[i:i + chunk])
        acc["MAP"].append(torch.softmax(L, 1).cpu())
        acc["temp. scaling"].append(torch.softmax(L / temp, 1).cpu())
        for nm, f in var_fns.items():
            acc[nm].append(probit(L, f(Jc, C).clamp_min(0)).cpu())
    y = torch.cat(labs)
    return {nm: pred_metrics(torch.cat(v), y) for nm, v in acc.items()}

def probit(logits, var):
    return torch.softmax(logits / torch.sqrt(1.0 + math.pi / 8.0 * var), 1)

def pred_metrics(probs, labels, n_bins=15):
    p = probs.clamp_min(1e-12).double()
    y = labels.to(p.device)
    nll = float(-torch.log(p[torch.arange(len(y)), y]).mean())
    conf, pred = p.max(1)
    corr = (pred == y).double()
    bins = torch.linspace(0, 1, n_bins + 1, device=p.device, dtype=p.dtype)
    ece = 0.0
    for i in range(n_bins):
        sel = (conf > bins[i]) & (conf <= bins[i + 1])
        if sel.sum() > 0: ece += float(sel.double().mean() * (corr[sel].mean() - conf[sel].mean()).abs())
    oh = torch.zeros_like(p); oh[torch.arange(len(y)), y] = 1.0
    return nll, ece, float(((oh - p) ** 2).sum(1).mean())


# --------------------------------- run -------------------------------------
def fit_lambda(model, loader, scale, fallback=LAMBDA_THEIRS):
    if LAMBDA_OVERRIDE is not None:
        return float(LAMBDA_OVERRIDE)
    try:
        from laplace import Laplace
        la = Laplace(model, "classification", subset_of_weights="all", hessian_structure="kron")
        la.fit(loader); la.optimize_prior_precision(method="marglik")
        return float(np.atleast_1d(la.prior_precision.detach().cpu().numpy()).mean()) * scale
    except Exception:
        return fallback

def run_once(seed, which="resnet9"):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_l, fit_l, xp_l, xt_l, ev_l, val_l = get_data(seed)
    
    if which == "resnet9":
        model = ResNet9NoBN().to(DEVICE); train(model, tr_l)
    else:
        model = CifarCNN().to(DEVICE); train(model, tr_l)
        
    model.eval()          
    with torch.no_grad():
        acc = np.mean([(model(x.to(DEVICE)).argmax(1).cpu() == y).float().mean().item() for x, y in ev_l])
    p = sum(q.numel() for q in model.parameters())

    scale = N_TOTAL / N_FIT
    lam = fit_lambda(model, fit_l, scale)
    op = GGN(model, fit_l, lam, scale)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    C = 10

    JX, _ = op.jac_stack(xt_l); Jxp, _ = op.jac_stack(xp_l)                
    Xsol, res = cg_solve(op, JX.T, tag="Sigma_X")
    Sref = (JX @ Xsol).double()
    logtr_ref = math.log(float(torch.trace(Sref)))
    
    V = build_V(op)
    diagA = (V * V).sum(1) + lam
    psi_diag = lambda Z: Z / diagA[:, None]

    psi_kfac = None
    try:
        from laplace import Laplace
        la = Laplace(model, "classification", subset_of_weights="all", hessian_structure="kron")
        la.fit(fit_l)
        la.prior_precision = torch.tensor(float(lam), device=DEVICE)
        def _kfac(Z): return la.posterior_precision.bmm(Z.T.unsqueeze(0).float(), exponent=-1).squeeze(0).T.to(torch.float32)
        _e = torch.zeros(op.p, 1, device=DEVICE, dtype=torch.float32); _e[0, 0] = 1.0
        if 1e-4 / lam < float(_kfac(_e)[0, 0].abs()) < 1e4 / lam: psi_kfac = _kfac
    except Exception:
        pass

    T_scale = fit_temperature(op, val_l)
    with torch.no_grad():
        Lref = torch.cat([op.model(x.to(DEVICE)) for x, _ in xt_l])
        Yref = torch.cat([y for _, y in xt_l])
    vref = Sref.diagonal().view(-1, C).float().clamp_min(0)
    gate_la = pred_metrics(probit(Lref, vref), Yref)[0]
    gate_map = pred_metrics(torch.softmax(Lref, 1), Yref)[0]

    rows, pred = [], []
    theta = torch.cat([q.detach().reshape(-1) for q in model.parameters()])

    # --- 1. Evaluate Full-Rank and ELLA approximations (matrix-free) ---
    full_cands = [
        ("Diagonal full rank", get_S_diag(JX, diagA), lambda Jc, C, dA=diagA: (Jc * Jc / dA).sum(1).view(-1, C).float()),
        ("ELLA (Nystrom)", get_S_ella(JX, Jxp, lam), build_ella_var_fn(Jxp, lam))
    ]
    if psi_kfac is not None:
        full_cands.append(("KFAC full rank", get_S_kfac(JX, psi_kfac), lambda Jc, C, pk=psi_kfac: (Jc * pk(Jc.T).T).sum(1).view(-1, C).float()))

    full_vfs = {}
    for nm, S_proj, vf in full_cands:
        t_ = time.time()
        kl, dg = kl_blocks(S_proj, Sref, C)
        rows.append(dict(s=p, method=nm, kl=kl, deg=dg, logtr=math.log(max(float(torch.trace(S_proj)), 1e-300)), disp=disp(S_proj, Sref, C)))
        full_vfs[nm] = vf
        print(f"    s=Full {nm}: {time.time() - t_:.1f}s", flush=True)

    pm_full = predictive_table(op, full_vfs, ev_l, C, T_scale)
    for nm, (nll, ece, br) in pm_full.items():
        if nm not in ["MAP", "temp. scaling"]:
            pred.append(dict(s=p, method=nm, nll=nll, ece=ece, brier=br))
    
    # --- 2. Evaluate Subspace models ---
    for s in S_LIST:
        cands = {
            "subset-Diagonal": subset_P(1.0 / diagA, s),
            "subset-Magnitude": subset_P(theta.abs(), s),
            "lowrank-Diagonal": lowrank_P(psi_diag, Jxp, s),
        }
        if psi_kfac is not None: cands["lowrank-KFAC"] = lowrank_P(psi_kfac, Jxp, s)

        vfs = {}
        for nm, P in list(cands.items()):
            S = sigma_from_P(P, op, V, JX, lam)
            vfs[nm] = subspace_var_fn(P, V, lam, op.p)
            kl, dg = kl_blocks(S, Sref, C)
            rows.append(dict(s=s, method=nm, kl=kl, deg=dg, logtr=math.log(max(float(torch.trace(S)), 1e-300)), disp=disp(S, Sref, C)))
                
        base = cands.get("lowrank-KFAC", cands["lowrank-Diagonal"])
        S = shift_sigma(base, op, V, JX, lam)
        vfs["lowrankopt + 1/lam shift"] = shift_var_fn(base, V, lam)
        kl, dg = kl_blocks(S, Sref, C)
        rows.append(dict(s=s, method="lowrankopt + 1/lam shift", kl=kl, deg=dg, logtr=math.log(max(float(torch.trace(S)), 1e-300)), disp=disp(S, Sref, C)))

        keep = []
        for nm, unc in (("Sigma_m (ours)", False), ("Krylov (uncorr.) [abl]", True)):
            sm = SigmaM(op, s, gen, lam, uncorrected=unc)
            SX = (JX @ sm.apply(JX.T)).double()
            kl, dg = kl_blocks(SX, Sref, C)
            rows.append(dict(s=s, method=nm, kl=kl, deg=dg, logtr=math.log(max(float(torch.trace(SX)), 1e-300)), disp=disp(SX, Sref, C), omega=sm.omega, lam_omega=sm.lam_omega))
            vfs[nm] = sigma_m_var_fn(sm)
            keep.append(sm)
            
        pm = predictive_table(op, vfs, ev_l, C, T_scale)
        for nm, (nll, ece, br) in pm.items(): pred.append(dict(s=s, method=nm, nll=nll, ece=ece, brier=br))
        print(f"  s = {s} done", flush=True)

    return rows, pred, logtr_ref, p, acc, (gate_la, gate_map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--model", choices=["cnn", "resnet9"], default="resnet9")
    a, _ = ap.parse_known_args()
              
    allr, allp, ceil, gates = [], [], [], []
    for sd in a.seeds:
        print(f"\n########## SEED {sd} ##########", flush=True)
        r, pr_, c, p, acc, g = run_once(sd, a.model)
        for x in r + pr_: x["seed"] = sd
        allr += r; allp += pr_; ceil.append(c); gates.append(g)

    meths = []
    for x in allr:
        if x["method"] not in meths: meths.append(x["method"])

    def agg(v):
        v = np.array([q for q in v if math.isfinite(q)], float)
        if v.size == 0: return float("inf"), 0.0
        return v.mean(), (v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0)

    print("\n" + "=" * 96)
    print(f"CIFAR-10 / ResNet9  |  {len(a.seeds)} seeds  |  ceiling log Tr Sigma_X = {np.mean(ceil):.4f}")
    print("=" * 96)
    hdr = f"{'s':>6s} {'method':26s} {'KL':>20s} {'degen':>7s} {'log Tr':>18s} {'disp':>8s}"
    print(hdr)
    
    S_ALL = sorted(list(set(x["s"] for x in allr)))
    for sv in S_ALL:
        print("-" * len(hdr))
        for nm in meths:
            sub = [x for x in allr if x["s"] == sv and x["method"] == nm]
            if not sub: continue
            km, ke = agg([x["kl"] for x in sub])
            lm, le = agg([x["logtr"] for x in sub])
            dm, _ = agg([x["disp"] for x in sub])
            dg = float(np.mean([x["deg"] for x in sub]))
            ks = "                 inf" if not math.isfinite(km) else f"{km:12.4f}±{ke:<6.4f}"
            print(f"{sv:6d} {nm:26s} {ks:>20s} {dg:7.2f} {lm:10.4f}±{le:<6.4f} {dm:8.4f}")
        tw = [x for x in allr if x["s"] == sv and "lam_omega" in x and x["method"].startswith("Sigma_m")]
        if tw:
            m_, e_ = agg([x["lam_omega"] for x in tw])
            o_, _ = agg([x["omega"] for x in tw])
            print(f"{'':6s} {'  [ours] omega':26s} = {o_:.4e}   lambda*omega = {m_:.4f}±{e_:.4f}")
    print("-" * len(hdr))
    print(f"{'':6s} {'(exact / ceiling)':26s} {0.0:20.4f} {0.0:7.2f} {np.mean(ceil):18.4f} {1.0:8.4f}")

    if allp:
        print("\n" + "=" * 96)
        print("PREDICTIVE METRICS (their Table 1 format)")
        print("=" * 96)
        pm = []
        for x in allp:
            if x["method"] not in pm: pm.append(x["method"])
        
        for met in ("nll", "ece", "brier"):
            print(f"\n{met.upper()}")
            print(f"{'s':>6s} " + "".join(f"{m[:19]:>23s}" for m in pm))
            for sv in S_ALL:
                line = f"{sv:6d} "
                for m in pm:
                    v = np.array([x[met] for x in allp if x["s"] == sv and x["method"] == m], dtype=float)
                    if v.size == 0: line += f"{'--':>23s}"
                    else: line += f"{v.mean():14.4f}±{(v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0):<8.4f}"
                print(line)


if __name__ == "__main__":
    main()
