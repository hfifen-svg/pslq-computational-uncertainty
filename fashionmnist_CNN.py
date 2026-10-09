#!/usr/bin/env python3
r"""
Faller & Martin (AISTATS 2026) FashionMNIST experiment, with Sigma_m added.

WHY THIS EXPERIMENT AND WHY FIRST
---------------------------------
p ~ 12k, so the FULL GGN Laplace is formable as a dense matrix and every
quantity is exact.  That makes this the decision point: if Sigma_m does not
separate from lowrank-KFAC here, where nothing is approximated, scaling to
ResNet will not change it.  One day of compute answers the question.

THE COMMON FRAME
----------------
Faller & Martin, eq (15):   Sigma_{P,X} = J_X P (P^T Psi^{-1} P)^{-1} P^T J_X^T
This script, for ours:      Sigma_hat_X = J_X Sigma_m J_X^T
Same Jacobians, same target Sigma_X = J_X Psi J_X^T; the methods differ only
in the middle matrix.  So every method in both papers is directly comparable.

TWO STRUCTURAL FACTS WORTH KEEPING IN VIEW
------------------------------------------
1. Their Lemma 2: Sigma_{P,X} <= Sigma_X in Loewner order for EVERY P, hence
   Tr Sigma_{P,X} <= Tr Sigma_X.  Every subspace method with s < p provably
   under-covers.  Sigma_m is not a subspace method -- it is full rank with an
   isotropic bulk -- so it is not bound by this and can, in principle, sit at
   the ceiling.  Whether it does is the experiment.

2. Their KL (eq 19) is computed per test point on C x C blocks, not on the
   full nC x nC matrix.  It has to be: Sigma_{P,X} has rank <= s, so for
   s < nC the full-matrix KL is infinite.  This script follows them.

METRICS
    KL          per-point Gaussian KL of the predictive covariance against
                the exact one, averaged.  The metric that sees DIRECTION.
    log Tr      their eq (20) proxy, WITH the ceiling log Tr Sigma_X drawn --
                which they can only do where the full LA is feasible.
    disp        sqrt(tr Sigma_hat / tr Sigma_X): the trace ratio, blind to
                direction.  Included to show what it does and does not catch.
    NLL/ECE     with temperature scaling as a baseline, which neither Faller
                & Martin nor Daxberger et al. report.

DEVIATIONS FROM THEIR SETUP, STATED
    - Their exact CNN is only in their released code; p here will differ from
      their 12,458.  The script prints p.
    - NO BatchNorm in the CNN, so that KFAC runs.  Their appendix lists BN;
      with BN the laplace-torch KFAC backend rejects the model, which is the
      same wall the ResNet-56 experiment hits.  Stated rather than hidden.
    - KFAC comes from laplace-torch if importable; otherwise the lowrank-KFAC
      row is skipped and the script says so.  Do not report a table without
      it -- lowrank-KFAC is their best method and the one to beat.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call, vjp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
SEEDS = (0, 1, 2, 3, 4)   # their five seeds
LAMBDA_THEIRS = 32.7     # their Table 9 value, for THEIR 12,458-param CNN
FIT_LAMBDA = True        # fit lambda on OUR model instead (see fit_lambda())
LAMBDA_FALLBACK = 32.7   # used only if the marglik fit is unavailable
N_FIT = 2000             # data actually used to estimate the GGN
N_TOTAL = 60000          # FashionMNIST training-set size the GGN must match
N_XPRIME = 100           # subset for subspace construction (their n, nC=1000)
N_XTEST = 100            # reference points for KL / trace  (their n)
N_EVAL = 2000            # points for NLL / ECE
S_LIST = (120, 230, 450, 670, 1000)   # their subspace dimensions
# --- FULL-RANK baselines.  Every method in Faller & Martin is a rank-s
# projection, so "a full-rank method beats rank-s methods" is close to
# tautological given their Lemma 2.  These are the comparisons a referee
# actually asks for.  Their KL does NOT depend on s (except ELLA, which has
# its own rank K), so they are computed once per seed and repeated across the
# s rows for readability.
ELLA_M = 2000                  # Nystrom sample count (Deng et al. use 2000)
ELLA_K = (20, 100, 500)        # Nystrom ranks
RUN_SWAG = True
SWAG_EPOCHS, SWAG_LR, SWAG_RANK = 10, 0.01, 20
EPOCHS, LR, BATCH = 40, 0.002, 256
ELLA_M, ELLA_K = 500, (20, 100, 500)   # Nystrom sample count and ranks
SWAG_EPOCHS, SWAG_LR, SWAG_RANK = 10, 0.01, 20


# ------------------------------- model -------------------------------------
class SmallCNN(nn.Module):
    """No BatchNorm, so KFAC runs.  See the deviations note above."""

    def __init__(self, C=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.c3 = nn.Conv2d(16, 32, 3, padding=1)
        self.f1 = nn.Linear(32 * 3 * 3, 20)
        self.f2 = nn.Linear(20, C)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    tr = datasets.FashionMNIST("./data", train=True, download=True,
                               transform=tf)
    te = datasets.FashionMNIST("./data", train=False, download=True,
                               transform=tf)
    g = torch.Generator().manual_seed(SEED)
    pt = torch.randperm(len(te), generator=g)
    pr = torch.randperm(len(tr), generator=g)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True),
            DataLoader(Subset(tr, pr[:N_FIT].tolist()), batch_size=256),
            DataLoader(Subset(tr, pr[N_FIT:N_FIT + N_XPRIME].tolist()),
                       batch_size=256),
            DataLoader(Subset(te, pt[:N_XTEST].tolist()), batch_size=256),
            DataLoader(Subset(te, pt[N_XTEST:N_XTEST + N_EVAL].tolist()),
                       batch_size=256),
            DataLoader(Subset(te, pt[-1000:].tolist()), batch_size=256))


def train(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * len(loader), pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step(); sched.step()
        print(f"  epoch {ep + 1}/{EPOCHS}", flush=True)
    model.eval()
    return model


# --------------------------- Jacobians and GGN -----------------------------
class Model:
    def __init__(self, model):
        self.model = model
        self.params = {n: p.detach() for n, p in model.named_parameters()}
        self.buffers = {n: b.detach() for n, b in model.named_buffers()}
        self.names = list(self.params)
        self.numels = [self.params[n].numel() for n in self.names]
        self.p = int(sum(self.numels))

    def flat(self, dd):
        return torch.cat([dd[n].reshape(-1) for n in self.names])

    def jacobian(self, x):
        """J(x) in R^{C x p} for a single input."""
        def f(pr):
            return functional_call(self.model, (pr, self.buffers),
                                   (x.unsqueeze(0),))
        out, vjp_fn = vjp(f, self.params)
        C = out.shape[1]
        rows = []
        for c in range(C):
            e = torch.zeros_like(out); e[:, c] = 1.0
            rows.append(self.flat(vjp_fn(e)[0]))
        return torch.stack(rows), out[0]

    def jacobian_stack(self, loader):
        """J_X in R^{nC x p}, and the softmax Hessians per point."""
        Js, Hs = [], []
        for xb, _ in loader:
            for x in xb:
                J, logits = self.jacobian(x.to(DEVICE))
                pr = torch.softmax(logits, 0)
                Hs.append(torch.diag(pr) - torch.outer(pr, pr))
                Js.append(J)
        return torch.cat(Js).double(), torch.stack(Hs).double()


def fit_lambda(model, fit_loader, n_fit=N_FIT, n_total=N_TOTAL):
    """Fit the prior precision by marginal likelihood, as Faller & Martin do
    (their Appendix A.4 uses the laplace library's marglik optimisation).

    WHY THIS IS NOT OPTIONAL.  Their lambda = 32.7 was fitted on THEIR CNN:
    12,458 parameters, with BatchNorm. Ours has 11,878 and no BN. Borrowing
    the constant produced an exact Laplace posterior WORSE than the MAP
    (NLL 0.338 vs 0.260), which is impossible when lambda and the GGN are
    consistent -- the posterior was simply far too wide.

    laplace-torch fits on the data it is given, so a lambda fitted against a
    2,000-point GGN is inconsistent with the n_total/n_fit rescale applied in
    full_ggn(). We therefore scale the fitted precision by the same factor,
    so that lambda and the data term refer to the same effective sample size.

    The check that this worked is the 'FULL LA (exact)' row of the predictive
    table: it must beat the MAP. If it does not, do not interpret the
    predictive numbers -- and say so rather than reporting them.
    """
    try:
        from laplace import Laplace
        la = Laplace(model, "classification", subset_of_weights="all",
                     hessian_structure="kron")
        la.fit(fit_loader)
        la.optimize_prior_precision(method="marglik")
        lam = float(np.atleast_1d(
            la.prior_precision.detach().cpu().numpy()).mean())
        lam *= (n_total / max(n_fit, 1))
        print(f"[lambda] marglik fit on OUR model: {lam:.4f} "
              f"(their Table 9 value for their CNN was {LAMBDA_THEIRS})")
        return lam
    except Exception as e:
        print(f"[lambda] marglik fit unavailable ({type(e).__name__}: {e}); "
              f"falling back to {LAMBDA_FALLBACK}. The FULL LA (exact) row "
              "may then lose to the MAP, in which case the predictive table "
              "is uninterpretable.")
        return LAMBDA_FALLBACK


def full_ggn(M, loader, lam, n_fit=N_FIT, n_total=N_TOTAL):
    """Dense precision Psi^{-1} = sum_{i=1}^{N} J_i^T H_i J_i + lambda I.

    SCALE MATTERS AND IS EASY TO GET WRONG.  The sum in Faller & Martin runs
    over the WHOLE training set (N = 60,000 here), and their lambda = 32.7
    was fitted by marginal likelihood against that full-data GGN.  Summing
    over a subset of n_fit points and adding the same lambda gives a
    precision ~n_total/n_fit times too small, i.e. a posterior ~30x too wide.
    The symptom is unmistakable and was observed before this fix: predictive
    NLL got monotonically WORSE the better a method approximated Psi
    (lowrank-Diagonal 0.253 < KFAC 0.274 < their optimum 0.308 < Sigma_m
    0.551), and the MAP beat all of them -- because every method was being
    rewarded for failing to reproduce an over-dispersed target.

    We therefore rescale the subset sum to an unbiased estimate of the
    full-data GGN before adding the prior.  If you change n_fit, the estimate
    gets noisier but its SCALE stays right.
    """
    J, H = M.jacobian_stack(loader)
    C = H.shape[1]
    P = torch.zeros(M.p, M.p, dtype=torch.float64, device=DEVICE)
    for i in range(H.shape[0]):
        Ji = J[i * C:(i + 1) * C]
        P += Ji.T @ H[i] @ Ji
    P *= (n_total / max(n_fit, 1))
    P += lam * torch.eye(M.p, dtype=torch.float64, device=DEVICE)
    return P


# ------------------------- Krylov machinery (ours) -------------------------
def lanczos(Av, v, m):
    n = v.numel()
    Q = torch.zeros(n, m, dtype=torch.float64, device=DEVICE)
    al = torch.zeros(m, dtype=torch.float64)
    be = torch.zeros(m, dtype=torch.float64)
    Q[:, 0] = v / v.norm(); b = 0.0
    qp = torch.zeros(n, dtype=torch.float64, device=DEVICE)
    for j in range(m):
        w = Av(Q[:, j]) - b * qp
        a = torch.dot(Q[:, j], w)
        w = w - a * Q[:, j]
        w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        al[j] = a.item(); b = w.norm().item(); be[j] = b
        qp = Q[:, j].clone()
        if b < 1e-13:
            m = j + 1; Q, al, be = Q[:, :m], al[:m], be[:m]; break
        if j + 1 < m:
            Q[:, j + 1] = w / b
    T = (torch.diag(al) + torch.diag(be[:-1], 1)
         + torch.diag(be[:-1], -1)).to(DEVICE)
    return Q, T, float(be[-1]), (w / b if b > 1e-13 else torch.zeros_like(w))


def reproj_tridiag(Av, Pr, w, l):
    nw2 = float(torch.dot(w, w))
    if nw2 < 1e-24:
        return torch.zeros(0, 0, dtype=torch.float64, device=DEVICE), 0.0
    n = w.numel()
    V = torch.zeros(n, l, dtype=torch.float64, device=DEVICE)
    al = torch.zeros(l, dtype=torch.float64)
    be = torch.zeros(max(l - 1, 1), dtype=torch.float64)
    V[:, 0] = w / math.sqrt(nw2); b, st = 0.0, 0
    qp = torch.zeros(n, dtype=torch.float64, device=DEVICE)
    for j in range(l):
        z = Pr(Av(V[:, j])) - b * qp
        a = torch.dot(V[:, j], z)
        z = Pr(z - a * V[:, j])
        z = Pr(z - V[:, :j + 1] @ (V[:, :j + 1].T @ z))
        al[j] = a.item(); st = j + 1
        nb = z.norm().item(); qp = V[:, j].clone()
        if j + 1 < l:
            if nb < 1e-13:
                break
            be[j] = nb; b = nb; V[:, j + 1] = z / nb
    return (torch.diag(al[:st]) + torch.diag(be[:st - 1], 1)
            + torch.diag(be[:st - 1], -1)).to(DEVICE), nw2


def quad_inv(T, nw2, power=1):
    if T.numel() == 0:
        return 0.0
    e1 = torch.zeros(T.shape[0], dtype=torch.float64, device=DEVICE)
    e1[0] = 1.0
    v = e1.clone()
    for _ in range(power):
        v = torch.linalg.solve(T, v)
    return float(nw2 * torch.dot(e1, v))


def cg(Av, b, iters, Pr=None, tol=1e-12):
    if Pr is not None:
        b = Pr(b)
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = float(torch.dot(r, r)); bn = max(math.sqrt(rs), 1e-30)
    for _ in range(iters):
        Ap = Av(p)
        if Pr is not None:
            Ap = Pr(Ap)
        den = float(torch.dot(p, Ap))
        if den <= 0:
            break
        a = rs / den
        x = x + a * p; r = r - a * Ap
        if Pr is not None:
            x, r = Pr(x), Pr(r)
        rs2 = float(torch.dot(r, r))
        if math.sqrt(rs2) < tol * bn:
            break
        p = r + (rs2 / rs) * p
        if Pr is not None:
            p = Pr(p)
        rs = rs2
    return x


def sigma_m(A, m, gen, lam, n_pslq=20, l_pslq=20, l_bnd=25, l_cg=200,
            uncorrected=False):
    """Dense Sigma_m of eq (25), built matrix-free from A (p ~ 12k)."""
    n = A.shape[0]
    Av = lambda v: A @ v
    v0 = torch.randn(n, dtype=torch.float64, device=DEVICE, generator=gen)
    Q, T, bm, qn = lanczos(Av, v0, m)
    m = Q.shape[1]
    Ti = torch.linalg.inv(T)
    S_res = Q @ Ti @ Q.T
    if uncorrected:
        return S_res, dict(m=m, omega=0.0, tau_omega=0.0)

    Pr = lambda z: z - Q @ (Q.T @ z)
    Th, nw2 = reproj_tridiag(Av, Pr, qn, l_bnd)
    a1 = quad_inv(Th, nw2, 1); a2 = quad_inv(Th, nw2, 2)
    gamma = bm ** 2 * float(Ti[m - 1, m - 1])
    den = 1.0 - gamma * a1
    e1 = torch.zeros(Th.shape[0], dtype=torch.float64, device=DEVICE)
    e1[0] = 1.0
    Thg = Th - gamma * torch.outer(e1, e1)
    p2 = quad_inv(Thg, nw2, 2); p3 = quad_inv(Thg, nw2, 3)
    s11 = a1 / den
    sig_u = p3 / p2 if p2 > 0 else 0.0

    acc = 0.0
    for _ in range(n_pslq):
        xi = torch.randn(n, dtype=torch.float64, device=DEVICE, generator=gen)
        Thp, nw2p = reproj_tridiag(Av, Pr, Pr(xi), l_pslq)
        acc += quad_inv(Thp, nw2p, 1)
    tr_S = acc / n_pslq + gamma * a2 / den

    q_par = bm * (Q @ Ti[:, m - 1])
    x = cg(lambda z: Pr(Av(Pr(z))), bm * qn, l_cg, Pr=Pr)
    q_perp = Pr((x / bm) / den) if bm > 1e-13 else torch.zeros_like(x)
    nq = float(q_perp.norm())
    u = q_perp / nq if nq > 1e-13 else torch.zeros_like(q_perp)
    omega = max((tr_S - sig_u) / max(n - m - 1, 1), 0.0)
    Prest = (torch.eye(n, dtype=torch.float64, device=DEVICE)
             - Q @ Q.T - torch.outer(u, u))
    Sig = (S_res + s11 * torch.outer(q_par, q_par)
           - torch.outer(q_par, q_perp) - torch.outer(q_perp, q_par)
           + sig_u * torch.outer(u, u) + omega * Prest)
    return Sig, dict(m=m, omega=omega, tau_omega=omega * lam, tr_S=tr_S)


# ----------------------------- subspace models -----------------------------
def sigma_from_P(P, Prec, JX):
    """Faller & Martin eq (15).  P is p x s (dense) or a LongTensor of
    indices for the subset methods."""
    if P.dtype == torch.long:
        M = Prec[P][:, P]
        JP = JX[:, P]
    else:
        M = P.T @ Prec @ P
        JP = JX @ P
    return JP @ torch.linalg.solve(M, JP.T)


def lowrank_P(Psi_approx_apply, Jxp, s):
    """Their Algorithm 1: M = J' Psi_approx J'^T, top-s eigvecs U_s,
    P = Psi_approx J'^T U_s."""
    W = Psi_approx_apply(Jxp.T)                # p x nC
    Mk = Jxp @ W                               # nC x nC
    Mk = 0.5 * (Mk + Mk.T)
    ev, U = torch.linalg.eigh(Mk)
    idx = torch.argsort(ev, descending=True)[:s]
    return W @ U[:, idx]


def subset_P(scores, s):
    return torch.argsort(scores, descending=True)[:s]


def _shift_dense(P, A, lam):
    """Dense Sigma for low-rank-plus-shift, for the predictive metrics."""
    U = torch.linalg.qr(P.double())[0]
    G = U.T @ A @ U - lam * (U.T @ U)
    ev, V = torch.linalg.eigh(0.5 * (G + G.T))
    Us = U @ V
    inner = 1.0 / (ev.clamp_min(0.0) + lam) - 1.0 / lam
    p = A.shape[0]
    return (Us * inner) @ Us.T + torch.eye(
        p, dtype=torch.float64, device=DEVICE) / lam


def sigma_lowrank_shift(P, A, JX, lam):
    """Low-rank-plus-shift on THEIR subspace -- the baseline Faller & Martin
    do not have.  Every method in their paper is a pure projection with zero
    variance off-subspace; Sigma_m fills the complement, and at
    lambda*omega ~ 1 that filling IS the 1/lambda prior shift.  So part of
    the margin against their baselines is 'we put something there and they
    put nothing'.  This row removes that part.

    Sigma = U (Lambda_s + lambda)^-1 U^T + (1/lambda)(I - U U^T).
    """
    if P.dtype == torch.long:
        p = A.shape[0]
        U = torch.zeros(p, P.numel(), dtype=torch.float64, device=A.device)
        U[P, torch.arange(P.numel(), device=A.device)] = 1.0
    else:
        U = torch.linalg.qr(P.double())[0]
    G = U.T @ A @ U - lam * (U.T @ U)
    ev, V = torch.linalg.eigh(0.5 * (G + G.T))
    Us = U @ V
    inner = 1.0 / (ev.clamp_min(0.0) + lam) - 1.0 / lam
    JU = JX @ Us
    return (JU * inner) @ JU.T + (JX @ JX.T) / lam


# ------------------- FULL-RANK baselines (added for submission) ------------
# Every method above is a rank-s PROJECTION.  Their Lemma 2 proves each one
# satisfies Tr Sigma_{P,X} <= Tr Sigma_X, so a full-rank method beating them
# is close to tautological and a referee will say so.  These four are the
# standard full-rank Laplace baselines a reviewer will actually ask for.
# They do NOT depend on s (except ELLA's K and SWAG's rank), so they are
# computed once per seed and repeated across the s columns for readability.


def diag_laplace(A):
    """Sigma = diag(A)^{-1} over ALL p parameters (LeCun et al. 1989;
    Daxberger et al. 2021).  Distinct from subset-/lowrank-Diagonal, which
    only USE the diagonal to pick a rank-s subspace."""
    return torch.diag(1.0 / torch.diagonal(A))


def kfac_laplace_full(psi_kfac, p):
    """Sigma = Psi_KFAC over ALL p parameters (Ritter et al. 2018).

    NOT lowrank-KFAC: that is a rank-s projection BUILT FROM KFAC.  This is
    the Kronecker-factored posterior itself, the standard scalable Laplace.
    """
    if psi_kfac is None:
        return None
    return psi_kfac(torch.eye(p, dtype=torch.float64, device=DEVICE))


def ella_sigma(M, A, lam, K, n_sample, gen, xs):
    """ELLA (Deng, Zhou & Zhu, NeurIPS 2022).

    Draw M pairs (x_m, i_m), stack the SCALAR gradients as X in R^{M x p},
    eigendecompose X X^T, and set v_k = X^T u_k / sqrt(lam_k) -- orthonormal,
    so V is a basis of the top-K Nystrom subspace.  Then

        Sigma_ELLA = V (V^T G V + lambda I)^{-1} V^T,   G = A - lambda I.

    The prior enters ONLY inside the K-dimensional subspace, so like the
    plain Krylov projection ELLA assigns exactly zero variance to the other
    p - K directions.  Expect it to under-cover.
    """
    p = M.p
    idx = torch.randint(0, xs.shape[0], (n_sample,), generator=gen,
                        device=DEVICE)
    cls = torch.randint(0, 10, (n_sample,), generator=gen, device=DEVICE)
    X = torch.zeros(n_sample, p, dtype=torch.float64, device=DEVICE)
    for i in range(n_sample):
        J, _ = M.jacobian(xs[idx[i]])
        X[i] = J[cls[i]].double()
    G = X @ X.T
    G = 0.5 * (G + G.T)
    ev, U = torch.linalg.eigh(G)
    sel = torch.argsort(ev, descending=True)[:K]
    V = X.T @ (U[:, sel] / ev[sel].clamp_min(1e-12).sqrt())
    V = torch.linalg.qr(V)[0]
    del X
    GK = V.T @ (A - lam * torch.eye(p, dtype=torch.float64,
                                    device=DEVICE)) @ V
    GK = 0.5 * (GK + GK.T) + lam * torch.eye(K, dtype=torch.float64,
                                             device=DEVICE)
    return V @ torch.linalg.inv(GK) @ V.T


def swag_sigma(model, loader, epochs, lr, rank, p, names_numels):
    """SWAG (Maddox et al. 2019): a Gaussian fitted to a short SGD
    trajectory, NOT to the GGN.  It therefore targets a DIFFERENT posterior
    and its KL to the Laplace predictive is not a fidelity number in the same
    sense -- it is included because reviewers expect it, and flagged.

    Restores BOTH parameters and buffers: restoring only the parameters
    leaves the BatchNorm running statistics from the SGD run in place and
    silently changes the MAP for every method evaluated afterwards.
    """
    theta0 = torch.cat([q.detach().reshape(-1).clone()
                        for q in model.parameters()])
    buf0 = {n: b.detach().clone() for n, b in model.named_buffers()}
    shapes = [q.shape for q in model.parameters()]
    numels = [q.numel() for q in model.parameters()]
    mean = torch.zeros_like(theta0); sq = torch.zeros_like(theta0)
    devs, n = [], 0
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
        with torch.no_grad():
            th = torch.cat([q.detach().reshape(-1) for q in
                            model.parameters()])
            n += 1
            mean += (th - mean) / n
            sq += (th ** 2 - sq) / n
            devs.append((th - mean).clone())
            if len(devs) > rank:
                devs.pop(0)
    model.eval()
    with torch.no_grad():
        i = 0
        for q, k, sh in zip(model.parameters(), numels, shapes):
            q.copy_(theta0[i:i + k].view(sh)); i += k
        for nm, b in model.named_buffers():
            b.copy_(buf0[nm])
    var = (sq - mean ** 2).clamp_min(1e-30).double()
    D = torch.stack(devs, dim=1).double()
    K = D.shape[1]
    return torch.diag(var) / 2.0 + (D @ D.T) / (2 * max(K - 1, 1))


# -------------------------------- metrics ----------------------------------
def blocks(S, C):
    n = S.shape[0] // C
    return torch.stack([S[i * C:(i + 1) * C, i * C:(i + 1) * C]
                        for i in range(n)])


def kl_blocks(Shat, Sref, C, jitter=1e-10):
    """Mean KL( N(0,Sref) || N(0,Shat) ) over test points, on C x C blocks,
    plus the fraction of points where Shat's block is singular.

    Their eq (19) is per point because Sigma_{P,X} has rank <= s: for s < nC
    the full nC x nC KL is infinite.

    A singular block is NOT a numerical failure -- it is the correct answer.
    It means the method assigns zero predictive variance across a test
    point's entire output space, which is infinitely far from the exact
    posterior there.  Faller & Martin hit the same thing and report
    log-trace = -inf for subset-Diagonal (their Table 13); Table 2 omits it
    entirely as "indistinguishable from the MAP due to negligible
    uncertainty".  Their Table 15 explains why: 34-92% of parameters in
    these small nets are dead (gradient ~ 0), so a diagonal-variance subset
    can select a block on which J_i P = 0 exactly.

    NOTE the jitter is scaled by the REFERENCE block, not by Shat: scaling
    it by Shat adds nothing to a zero block and the solve still fails.

    Returns (mean KL over finite points, fraction degenerate).
    """
    Bh, Br = blocks(Shat, C), blocks(Sref, C)
    I = torch.eye(C, dtype=torch.float64, device=DEVICE)
    out, ndeg = [], 0
    for A_, B_ in zip(Bh, Br):
        scale = float(torch.trace(B_)) / C          # reference scale, > 0
        A_ = A_ + jitter * scale * I
        B_ = B_ + jitter * scale * I
        # rank tests with a relative tolerance do NOT catch a block that is
        # uniformly ~1e-30: it is numerically rank C but its inverse is ~1e30,
        # which is what produced the spurious KL ~ 5e10 for subset-Diagonal.
        # Test the log-determinant against the reference scale instead.
        ld = float(torch.logdet(A_))
        if (not math.isfinite(ld)) or ld < math.log(1e-12) * C \
                or int(torch.linalg.matrix_rank(A_, rtol=1e-6)) < C:
            ndeg += 1
            continue
        try:
            sol = torch.linalg.solve(A_, B_)
            val = 0.5 * (torch.trace(sol) - C
                         + torch.logdet(A_) - torch.logdet(B_))
            if math.isfinite(float(val)):
                out.append(float(val))
            else:
                ndeg += 1
        except Exception:
            ndeg += 1
    n = Bh.shape[0]
    return (float(np.mean(out)) if out else float("inf"), ndeg / n)


def disp(Shat, Sref, C):
    r = (blocks(Shat, C).diagonal(dim1=1, dim2=2).sum(1)
         / blocks(Sref, C).diagonal(dim1=1, dim2=2).sum(1)).sqrt()
    return float(r.median())


# ------------------- predictive metrics: their Table 1 ---------------------
def eval_jacobians(M, loader, chunk=200):
    """Yield (J_chunk in R^{bC x p}, logits in R^{b x C}, labels)."""
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb); ys.append(yb)
    X = torch.cat(xs); Y = torch.cat(ys)
    for i in range(0, X.shape[0], chunk):
        Js, lg = [], []
        for x in X[i:i + chunk]:
            J, o = M.jacobian(x.to(DEVICE))
            Js.append(J); lg.append(o)
        yield (torch.cat(Js).double(), torch.stack(lg).double(),
               Y[i:i + chunk])


def probit_probs(logits, var, C):
    """Their eq (8): softmax( f / sqrt(1 + pi/8 * diag Sigma_x) )."""
    return torch.softmax(logits / torch.sqrt(1.0 + math.pi / 8.0 * var), 1)


def metrics(probs, labels, n_bins=15):
    """NLL, ECE, Brier.  Brier with their convention: average over points,
    SUM over classes."""
    p = probs.clamp_min(1e-12)
    y = labels.to(p.device)
    nll = float(-torch.log(p[torch.arange(len(y)), y]).mean())
    conf, pred = p.max(1)
    corr = (pred == y).double()
    bins = torch.linspace(0, 1, n_bins + 1, device=p.device, dtype=p.dtype)
    ece = 0.0
    for i in range(n_bins):
        sel = (conf > bins[i]) & (conf <= bins[i + 1])
        if sel.sum() > 0:
            ece += float(sel.double().mean()
                         * (corr[sel].mean() - conf[sel].mean()).abs())
    oh = torch.zeros_like(p)
    oh[torch.arange(len(y)), y] = 1.0
    brier = float(((oh - p) ** 2).sum(1).mean())
    return nll, ece, brier


def subspace_var_fn(P, A):
    """Returns f(J_chunk) -> per-point diag of Sigma_{P,x}, via a Cholesky of
    P^T A P.  Never forms a p x p matrix."""
    if P.dtype == torch.long:
        Ms = A[P][:, P]
        sel = P
    else:
        Ms = P.T @ A @ P
        sel = None
    Ms = 0.5 * (Ms + Ms.T)
    Ms = Ms + 1e-10 * float(torch.diagonal(Ms).mean()) * torch.eye(
        Ms.shape[0], dtype=torch.float64, device=DEVICE)
    L = torch.linalg.cholesky(Ms)

    def f(Jc, C):
        PtJ = Jc[:, sel].T if sel is not None else P.T @ Jc.T   # s x bC
        W = torch.linalg.solve_triangular(L, PtJ, upper=False)
        d = (W * W).sum(0)                                      # bC
        return d.view(-1, C)
    return f


def dense_var_fn(Sig):
    def f(Jc, C):
        d = ((Jc @ Sig) * Jc).sum(1)
        return d.view(-1, C)
    return f


def fit_temperature(M, loader):
    """One scalar on the validation split -- the baseline neither Faller &
    Martin nor Daxberger et al. report, and the one every Laplace variant
    here has to beat before a calibration claim is worth making."""
    lg, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            lg.append(M.model(xb.to(DEVICE)).double().cpu()); ys.append(yb)
    lg = torch.cat(lg); y = torch.cat(ys)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def predictive_table(M, var_fns, loader, C, temp=None):
    """MAP + every method under their probit predictive."""
    keys = list(var_fns) + ["MAP"] + (["temp. scaling"] if temp else [])
    acc = {k: [] for k in keys}
    labs = []
    for Jc, lg, yb in eval_jacobians(M, loader):
        labs.append(yb)
        acc["MAP"].append(torch.softmax(lg, 1))
        if temp:
            acc["temp. scaling"].append(torch.softmax(lg / temp, 1))
        for nm, f in var_fns.items():
            acc[nm].append(probit_probs(lg, f(Jc, C), C))
        del Jc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    y = torch.cat(labs)
    return {nm: metrics(torch.cat(v), y) for nm, v in acc.items()}


# --------------------------------- main ------------------------------------
def run_once(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    tr_l, fit_l, xp_l, xt_l, ev_l, val_l = get_data()

    model = SmallCNN().to(DEVICE)
    print("[train] FashionMNIST CNN")
    train(model, tr_l)
    M = Model(model)
    print(f"[model] p = {M.p:,}  (their CNN reports 12,458; exact "
          "architecture is only in their released code)")

    LAM = fit_lambda(model, fit_l) if FIT_LAMBDA else LAMBDA_FALLBACK
    print("[GGN] forming the dense precision ...", flush=True)
    Prec = full_ggn(M, fit_l, LAM)
    Psi = torch.linalg.inv(Prec)
    A = Prec                                    # A = Psi^{-1}
    diagPsi = torch.diag(Psi)
    # Psi is kept for: the reference Sigma_X, their P* construction, and the
    # exact-full-LA row of the predictive table (the consistency check).

    JX, _ = M.jacobian_stack(xt_l)              # nC x p, test points
    Jxp, _ = M.jacobian_stack(xp_l)             # nC x p, construction subset
    C = 10
    Sref = JX @ Psi @ JX.T
    logtr_ref = math.log(float(torch.trace(Sref)))
    print(f"[ref] log Tr Sigma_X = {logtr_ref:.4f}  (the ceiling of their "
          "Lemma 2; every subspace method must sit below it)")

    # Psi_approx for their low-rank construction
    dg = 1.0 / torch.diag(Prec)
    psi_diag = lambda Z: dg[:, None] * Z

    psi_kfac = None
    try:
        from laplace import Laplace
        la = Laplace(model, "classification", subset_of_weights="all",
                     hessian_structure="kron")
        la.fit(fit_l)
        la.prior_precision = torch.tensor(float(LAM), device=DEVICE)
        Kd = la.posterior_precision.to_matrix().double()
        Kinv = torch.linalg.inv(Kd)
        psi_kfac = lambda Z: Kinv @ Z
        print("[KFAC] laplace-torch Kron available")
    except Exception as e:
        print(f"[KFAC] unavailable ({type(e).__name__}: {e}). "
              "lowrank-KFAC is their BEST method -- do not report a table "
              "without it.")

    T_scale = fit_temperature(M, val_l)
    print(f"[temp] temperature scaling T = {T_scale:.4f}", flush=True)

    # ---- FULL-RANK baselines: computed ONCE, they do not depend on s
    full = {}
    full["diagonal LA (full rank)"] = diag_laplace(A)
    Kf = kfac_laplace_full(psi_kfac, M.p)
    if Kf is not None:
        full["KFAC LA (full rank)"] = Kf
    xs_fit = torch.cat([x for x, _ in fit_l]).to(DEVICE)
    for K in ELLA_K:
        try:
            full[f"ELLA (K={K})"] = ella_sigma(M, A, LAM, K, ELLA_M, gen,
                                               xs_fit)
        except Exception as e:
            print(f"[ELLA K={K}] failed: {type(e).__name__}: {e}")
    try:
        full["SWAG [different target]"] = swag_sigma(
            model, tr_l, SWAG_EPOCHS, SWAG_LR, SWAG_RANK, M.p, None)
        M2 = Model(model)          # buffers restored; refresh the cached MAP
        M.params, M.buffers = M2.params, M2.buffers
    except Exception as e:
        print(f"[SWAG] failed: {type(e).__name__}: {e}")
    print(f"[full-rank] {list(full)}", flush=True)

    rows, pred_rows = [], []
    for s in S_LIST:
        cands = {}
        cands["subset-Diagonal"] = subset_P(diagPsi, s)
        cands["subset-Magnitude"] = subset_P(
            torch.cat([p.detach().reshape(-1).abs()
                       for p in model.parameters()]).double(), s)
        cands["lowrank-Diagonal"] = lowrank_P(psi_diag, Jxp, s)
        if psi_kfac is not None:
            cands["lowrank-KFAC"] = lowrank_P(psi_kfac, Jxp, s)
        cands["lowrankopt-GGN"] = lowrank_P(lambda Z: Psi @ Z, Jxp, s)

        for nm, P in cands.items():
            S = sigma_from_P(P, Prec, JX)
            kl, deg = kl_blocks(S, Sref, C)
            rows.append(dict(s=s, method=nm, kl=kl, deg=deg,
                             logtr=math.log(max(float(torch.trace(S)), 1e-300)),
                             disp=disp(S, Sref, C)))

        S = sigma_lowrank_shift(cands["lowrankopt-GGN"], A, JX, LAM)
        kl, deg = kl_blocks(S, Sref, C)
        rows.append(dict(s=s, method="lowrankopt + 1/lam shift", kl=kl,
                         deg=deg,
                         logtr=math.log(max(float(torch.trace(S)), 1e-300)),
                         disp=disp(S, Sref, C)))
        del S

        # full-rank methods: no P, the middle matrix is used directly
        Sm, info = sigma_m(A, s, gen, lam=LAM)
        SmX = JX @ Sm @ JX.T
        kl, deg = kl_blocks(SmX, Sref, C)
        rows.append(dict(s=s, method="Sigma_m (ours)", kl=kl, deg=deg,
                         logtr=math.log(max(float(torch.trace(SmX)), 1e-300)),
                         disp=disp(SmX, Sref, C),
                         **{k: info[k] for k in ("omega", "tau_omega")}))
        Su, _ = sigma_m(A, s, gen, lam=LAM, uncorrected=True)
        SuX = JX @ Su @ JX.T
        kl, deg = kl_blocks(SuX, Sref, C)
        rows.append(dict(s=s, method="Krylov (uncorr.) [abl]", kl=kl, deg=deg,
                         logtr=math.log(max(float(torch.trace(SuX)), 1e-300)),
                         disp=disp(SuX, Sref, C)))
        for nm, Sig in full.items():
            SX = JX @ Sig @ JX.T
            kl, deg = kl_blocks(SX, Sref, C)
            rows.append(dict(s=s, method=nm, kl=kl, deg=deg,
                             logtr=math.log(max(float(torch.trace(SX)),
                                                1e-300)),
                             disp=disp(SX, Sref, C)))
            del SX

        # --- their Table 1: NLL / ECE / Brier under the probit predictive
        vfs = {nm: subspace_var_fn(P, A) for nm, P in cands.items()}
        for nm, Sig in full.items():
            vfs[nm] = dense_var_fn(Sig)
        vfs["lowrankopt + 1/lam shift"] = dense_var_fn(
            _shift_dense(cands["lowrankopt-GGN"], A, LAM))
        vfs["FULL LA (exact)"] = dense_var_fn(Psi)
        vfs["Sigma_m (ours)"] = dense_var_fn(Sm)
        vfs["Krylov (uncorr.) [abl]"] = dense_var_fn(Su)
        pm = predictive_table(M, vfs, ev_l, C, temp=T_scale)
        for nm, (nll, ece, br) in pm.items():
            pred_rows.append(dict(s=s, method=nm, nll=nll, ece=ece, brier=br))
        del vfs

        del Sm, Su, SmX, SuX
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  s = {s} done", flush=True)

    print(f"[check] lambda used = {LAM:.4f}")
    return rows, pred_rows, logtr_ref


def main():
    all_rows, all_pred, ceils = [], [], []
    for seed in SEEDS:
        print(f"\n########## SEED {seed} ##########", flush=True)
        r, pr, c = run_once(seed)
        for x in r + pr:
            x["seed"] = seed
        all_rows += r
        all_pred += pr
        ceils.append(c)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    methods = []
    for x in all_rows:
        if x["method"] not in methods:
            methods.append(x["method"])

    def agg(vals):
        v = np.array([x for x in vals if math.isfinite(x)], dtype=float)
        if v.size == 0:
            return float("inf"), 0.0
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size)
                                      if v.size > 1 else 0.0)

    print("\n" + "=" * 96)
    print(f"FashionMNIST, p reported above, {len(SEEDS)} seeds, "
          f"mean +/- s.e.m.   ceiling log Tr Sigma_X = "
          f"{np.mean(ceils):.4f} +/- {np.std(ceils, ddof=1) / math.sqrt(len(ceils)) if len(ceils) > 1 else 0:.4f}")
    print("=" * 96)
    hdr = (f"{'s':>6s} {'method':24s} {'KL':>20s} {'degen':>7s} "
           f"{'log Tr':>18s} {'disp':>16s}")
    print(hdr)
    for sv in S_LIST:
        print("-" * len(hdr))
        for nm in methods:
            sub = [x for x in all_rows if x["s"] == sv and x["method"] == nm]
            if not sub:
                continue
            dg = float(np.mean([x["deg"] for x in sub]))
            klm, kls = agg([x["kl"] for x in sub])
            ltm, lts = agg([x["logtr"] for x in sub])
            dpm, dps = agg([x["disp"] for x in sub])
            kss = "                 inf" if not math.isfinite(klm) else \
                f"{klm:12.4f}±{kls:<7.4f}"
            lss = "                -inf" if ltm < -300 else \
                f"{ltm:10.4f}±{lts:<7.4f}"
            print(f"{sv:6d} {nm:24s} {kss:>20s} {dg:7.2f} {lss:>18s} "
                  f"{dpm:8.4f}±{dps:<7.4f}")
        tw = [x for x in all_rows if x["s"] == sv and "tau_omega" in x]
        if tw:
            m_, e_ = agg([x["tau_omega"] for x in tw])
            om, _ = agg([x["omega"] for x in tw])
            flag = ("   *** ~1: the bulk HAS degenerated into the "
                    "1/lambda shift; say so ***" if m_ > 0.99 else "")
            print(f"{'':6s} {'  [ours] omega':24s} = {om:.6e}   "
                  f"lambda*omega = {m_:.4f}±{e_:.4f}{flag}")
    print("-" * len(hdr))
    print(f"{'':6s} {'(exact / ceiling)':24s} {0.0:20.4f} {0.0:7.2f} "
          f"{np.mean(ceils):18.4f} {1.0:16.4f}")
    # ---- their Table 1 format
    print("\n" + "=" * 96)
    print("PREDICTIVE METRICS (their Table 1 format), probit predictive of "
          "their eq (8)")
    print("=" * 96)
    pm = []
    for x in all_pred:
        if x["method"] not in pm:
            pm.append(x["method"])
    nseed = len({x.get("seed") for x in all_pred})
    print(f"mean +/- s.e.m. over {nseed} seed(s).  Report BOTH: on CIFAR-10 "
          "the ECE gaps between\nSigma_m and lowrank-KFAC vanished once the "
          "error bars were added.  Do not claim a\ncalibration margin whose "
          "size is under two s.e.m.")
    for met in ("nll", "ece", "brier"):
        print(f"\n{met.upper()}")
        print(f"{'s':>6s} " + "".join(f"{m[:19]:>23s}" for m in pm))
        for sv in S_LIST:
            line = f"{sv:6d} "
            for m in pm:
                v = np.array([x[met] for x in all_pred
                              if x["s"] == sv and x["method"] == m],
                             dtype=float)
                if v.size == 0:
                    line += f"{'--':>23s}"
                else:
                    se = (v.std(ddof=1) / math.sqrt(v.size)
                          if v.size > 1 else 0.0)
                    line += f"{v.mean():14.4f}±{se:<8.4f}"
            print(line)
        for sv in S_LIST:
            vals = {}
            for m in pm:
                v = [x[met] for x in all_pred
                     if x["s"] == sv and x["method"] == m]
                if v:
                    vals[m] = float(np.mean(v))
            if vals:
                b = min(vals, key=vals.get)
                print(f"       s={sv}: best = {b} ({vals[b]:.4f})")

    # ---- gate: the exact LA must beat the MAP, or the table means nothing
    fl = [x["nll"] for x in all_pred if x["method"] == "FULL LA (exact)"]
    mp = [x["nll"] for x in all_pred if x["method"] == "MAP"]
    if fl and mp:
        if np.mean(fl) > np.mean(mp):
            print(f"\n*** SANITY CHECK FAILED: FULL LA (exact) NLL "
                  f"{np.mean(fl):.4f} > MAP {np.mean(mp):.4f}. The exact "
                  "Laplace posterior is worse than no posterior, so lambda "
                  "and the GGN are still inconsistent. DO NOT report the "
                  "predictive table; the KL table is unaffected. ***")
        else:
            print(f"\n[sanity] FULL LA (exact) NLL {np.mean(fl):.4f} <= MAP "
                  f"{np.mean(mp):.4f}: lambda and the GGN are consistent, "
                  "the predictive table is interpretable.")

    # ---- persist EVERY per-seed record.  Nothing here is saved otherwise,
    # and regenerating a figure would mean rerunning the whole experiment.
    import csv
    for fn, rec in (("fashionmnist_kl.csv", all_rows),
                    ("fashionmnist_predictive.csv", all_pred)):
        if not rec:
            continue
        keys = sorted({k for r in rec for k in r})
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rec:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"wrote {fn} ({len(rec)} rows)")


if __name__ == "__main__":
    main()
