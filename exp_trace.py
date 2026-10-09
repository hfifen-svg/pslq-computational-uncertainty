#!/usr/bin/env python3
"""Trace estimation at matched matrix-vector budgets (Section 6.1).

Target: tr(A^{-1}).  Every method receives the same budget B of products with
A and the same Krylov depth L per probe or solve; the free parameter of each
method (number of probes, sketch size, Krylov dimension) is derived from B.

    Hutchinson + CG   N = B / L Rademacher probes, each solved by L CG steps
    SLQ               N = B / L Rademacher probes, L-step Gauss quadrature
                      (Ubaru, Chen and Saad, 2017)
    Hutch++           k = g = B / (3L) solves for the sketch, its trace and the
                      deflated Hutchinson part (Meyer et al., 2021)
    ULISSE            CG with Russian-roulette truncation, unbiased
                      (Filippone and Engler, 2015): L guaranteed steps, then
                      continuation with probability P_CONT, increments
                      reweighted by the inverse survival probability
    P-SLQ (ours)      tr(A^{-1}) = tr(T_m^{-1}) + (S_perp^{-1})_11 beta_m^2
                      (T_m^{-2})_mm + tr(S_perp^{-1}): one Lanczos run of
                      dimension m, a boundary probe of depth L, and N probes
                      of depth L.  With B' = B - L:  m = B'/4,  N = (B' - m)/L.

All methods use Rademacher probes (projected onto the complement for P-SLQ;
Hutchinson's identity needs only E[z z^T] = I).

Two conditioning regimes, one row of the figure each:
    kappa ~ 1e3   the matrices of the other experiments (the kernel matrix
                  has kappa ~ 3e3 with its 0.01 nugget)
    kappa = 1e5   the same families at kappa = 1e5 (see make_matrix: the
                  kernel matrix uses a squared-exponential kernel here)

Reported: relative RMSE over REPS independent repetitions, against the exact
trace.  Output: exp_trace.pdf, exp_trace.csv.

Usage:  python exp_trace.py                  both regimes, then plot (~40 min)
        python exp_trace.py --kappa 1e5      one regime only; merged into an
                                             existing exp_trace.csv
        python exp_trace.py --plot           replot from exp_trace.csv
"""
import csv
import sys

import numpy as np
from pslq_core import test_matrix, lanczos, random_orthogonal

D = 1000
L = 30                                # Krylov depth for every method
BUDGETS = [200, 400, 800, 1600, 3200]
REPS = 30
P_CONT = 0.5                          # ULISSE continuation probability
REGIMES = [("1e3", r"$\kappa\approx10^3$"), ("5000", r"$\kappa=5000$")]
KINDS = [("kernel", "kernel"), ("power-law", "power-law"),
         ("bulk+outliers", "bulk + outliers"), ("exponential", "exponential")]
METHODS = ["Hutchinson + CG", "SLQ", "Hutch++", "ULISSE", "P-SLQ (ours)"]


# ============================================================ helpers =======

def batched_lanczos(A, U, l, Q=None):
    """l Lanczos steps for every column of U at once, full reorthogonalisation.
    With Q given, every vector is projected onto range(Q)^perp (P-SLQ).
    Returns the tridiagonal entries (alpha l x B, beta l-1 x B) and ||u||^2."""
    P = (lambda Z: Z) if Q is None else (lambda Z: Z - Q @ (Q.T @ Z))
    d, B = U.shape
    nu2 = (U * U).sum(0)
    V = np.zeros((l, d, B))
    V[0] = U / np.sqrt(nu2)
    al, be = np.zeros((l, B)), np.zeros((l - 1, B))
    prev, bprev = np.zeros((d, B)), np.zeros(B)
    for j in range(l):
        W = P(A @ V[j]) - bprev * prev
        al[j] = (V[j] * W).sum(0)
        W -= al[j] * V[j]
        for _ in range(2):
            W -= np.einsum("jdb,jb->db", V[:j + 1], np.einsum("jdb,db->jb", V[:j + 1], W))
        W = P(W)
        b = np.linalg.norm(W, axis=0)
        if j + 1 < l:
            be[j] = b
            V[j + 1] = W / np.maximum(b, 1e-300)
        prev, bprev = V[j], b
    return al, be, nu2


def gauss(al, be, nu2, k=1):
    """nu2 * e1^T T^{-k} e1 for every column (k = 1 or 2)."""
    out = np.empty(al.shape[1])
    l = al.shape[0]
    for i in range(al.shape[1]):
        T = np.diag(al[:, i]) + np.diag(be[:, i], 1) + np.diag(be[:, i], -1)
        x = np.zeros(l); x[0] = 1.0
        for _ in range(k):
            x = np.linalg.solve(T, x)
        out[i] = nu2[i] * x[0]
    return out


def cg(A, Bm, iters):
    """iters CG steps for every column of Bm, from zero; returns the iterates."""
    X = np.zeros_like(Bm); R = Bm.copy(); P = R.copy()
    rr = (R * R).sum(0)
    for _ in range(iters):
        AP = A @ P
        alpha = rr / np.maximum((P * AP).sum(0), 1e-300)
        X += alpha * P
        R -= alpha * AP
        rr_new = (R * R).sum(0)
        P = R + (rr_new / np.maximum(rr, 1e-300)) * P
        rr = rr_new
    return X


def rademacher(rng, d, n):
    return rng.choice([-1.0, 1.0], size=(d, n))


def make_matrix(key, regime, rng):
    """Test matrix for a regime.  "1e3" returns the matrices used elsewhere in
    the paper, unchanged.  "1e5" gives the same families at kappa = 1e5:
      kernel         squared-exponential kernel on the same inputs, with the
                     nugget set to reach kappa (the exponential kernel's own
                     smallest eigenvalue is too large for this kappa)
      bulk+outliers  smallest eigenvalue pinned to 1/kappa
      others         the spectrum of test_matrix stretched to kappa"""
    if regime == "1e3":
        return test_matrix(key, D, rng)
    kappa = float(regime)
    if key == "kernel":
        X = rng.random((D, 5))
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        K = np.exp(-d2 / (2 * (0.3 * np.sqrt(5)) ** 2))
        K = 0.5 * (K + K.T)
        w = np.linalg.eigvalsh(K)
        nugget = max((w[-1] - kappa * w[0]) / (kappa - 1.0), 0.0)
        return K + nugget * np.eye(D)
    if key == "bulk+outliers":
        lam = np.concatenate([rng.uniform(0.5, 1.0, 10),
                              rng.uniform(1 / kappa, 0.05, D - 10)])
        lam[np.argmin(lam)] = 1 / kappa
        U = random_orthogonal(D, rng)
        return (U * lam) @ U.T
    return test_matrix(key, D, rng, kappa=kappa)


# ============================================================ methods =======

def hutchinson_cg(A, B, rng):
    Z = rademacher(rng, A.shape[0], max(B // L, 1))
    return float(np.mean((Z * cg(A, Z, L)).sum(0)))


def slq(A, B, rng):
    al, be, nu2 = batched_lanczos(A, rademacher(rng, A.shape[0], max(B // L, 1)), L)
    return float(np.mean(gauss(al, be, nu2)))


def hutchpp(A, B, rng):
    d = A.shape[0]
    k = max(B // (3 * L), 1)
    Q = np.linalg.qr(cg(A, rademacher(rng, d, k), L))[0]      # sketch of A^{-1}
    t1 = np.trace(Q.T @ cg(A, Q, L))
    G = rademacher(rng, d, k)
    G -= Q @ (Q.T @ G)
    t2 = float(np.mean((G * cg(A, G, L)).sum(0)))
    return t1 + t2


def ulisse(A, B, rng):
    """CG with Russian-roulette truncation; unbiased for the converged solve."""
    d = A.shape[0]
    cost = L + P_CONT / (1.0 - P_CONT)                        # expected steps
    n = max(int(B / cost), 1)
    Z = rademacher(rng, d, n)
    extra = rng.geometric(1.0 - P_CONT, size=n) - 1           # steps after L
    X = np.zeros_like(Z); R = Z.copy(); P = R.copy()
    rr = (R * R).sum(0)
    est = np.zeros(n)
    for k in range(L + extra.max()):
        AP = A @ P
        alpha = rr / np.maximum((P * AP).sum(0), 1e-300)
        dX = alpha * P
        w = np.where(k < L, 1.0, P_CONT ** -(k - L + 1))       # inverse survival
        alive = k < L + extra
        est += alive * w * (Z * dX).sum(0)
        X += dX
        R -= alpha * AP
        rr_new = (R * R).sum(0)
        P = R + (rr_new / np.maximum(rr, 1e-300)) * P
        rr = rr_new
    return float(est.mean())


def pslq(A, B, rng):
    d = A.shape[0]
    Bp = B - L                                                 # boundary probe
    m = min(max(Bp // 4, 5), d - L - 2)                       # room for probes
    N = max((Bp - m) // L, 1)
    Q, T, beta_m, q_next = lanczos(A, rng.standard_normal(d), m)
    Ti = np.linalg.inv(T)
    gamma = beta_m ** 2 * Ti[m - 1, m - 1]
    al, be, _ = batched_lanczos(A, q_next[:, None], L, Q)      # boundary probe
    a1, a2 = gauss(al, be, np.ones(1), 1)[0], gauss(al, be, np.ones(1), 2)[0]
    s11 = a1 / (1.0 - gamma * a1)
    U = rademacher(rng, d, N)                                  # same probes as
    U -= Q @ (Q.T @ U)                                         # the baselines
    al, be, nu2 = batched_lanczos(A, U, L, Q)                  # P-SLQ
    tr_S = float(np.mean(gauss(al, be, nu2))) + gamma * a2 / (1.0 - gamma * a1)
    return np.trace(Ti) + s11 * beta_m ** 2 * (Ti @ Ti)[m - 1, m - 1] + tr_S


ESTIMATORS = dict(zip(METHODS, [hutchinson_cg, slq, hutchpp, ulisse, pslq]))


# ============================================================== run =========

def run(regimes):
    """Run the given regimes; rows of other regimes already in exp_trace.csv
    are kept, so one regime can be added without recomputing the other."""
    keep = [r for r in load() if r["regime"] not in regimes] if _have_csv() else []
    rows = []
    for regime in regimes:
        for key, _ in KINDS:
            rng = np.random.default_rng(0)
            A = make_matrix(key, regime, rng)
            w = np.linalg.eigvalsh(A)
            exact = float(np.sum(1.0 / w))
            print(f"[{regime}] {key}: kappa = {w[-1] / w[0]:.2e}", flush=True)
            for B in BUDGETS:
                for name, est in ESTIMATORS.items():
                    vals = np.array([est(A, B, rng) for _ in range(REPS)])
                    rows.append(dict(
                        regime=regime, matrix=key, budget=B, method=name,
                        rel_rmse=float(np.sqrt(np.mean((vals / exact - 1.0) ** 2))),
                        rel_bias=float(np.mean(vals) / exact - 1.0)))
                line = "  ".join(f"{r['method'][:12]:>12s} {r['rel_rmse']:.1e}"
                                 for r in rows if r["regime"] == regime
                                 and r["matrix"] == key and r["budget"] == B)
                print(f"  B={B:5d}  {line}", flush=True)
    rows = keep + rows
    with open("exp_trace.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w_.writeheader(); w_.writerows(rows)
    return rows


def _have_csv():
    try:
        open("exp_trace.csv").close()
        return True
    except OSError:
        return False


def load():
    with open("exp_trace.csv") as fh:
        return [dict(r, regime=r.get("regime") or "1e3", budget=int(r["budget"]),
                     rel_rmse=float(r["rel_rmse"]), rel_bias=float(r["rel_bias"]))
                for r in csv.DictReader(fh)]


def _sci(v, _):
    """Tick label: 10^k for powers of ten, else c x 10^k."""
    e = int(np.floor(np.log10(v) + 1e-9))
    c = round(v / 10 ** e)
    return rf"$10^{{{e}}}$" if c == 1 else rf"${c}\times10^{{{e}}}$"


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False})
    style = {"Hutchinson + CG": dict(color="#7f7f7f", marker="s", ls="-"),
             "SLQ": dict(color="#2a9d8f", marker="^", ls="-"),
             "Hutch++": dict(color="#e9a03b", marker="D", ls="-"),
             "ULISSE": dict(color="#b2182b", marker="v", ls="-"),
             "P-SLQ (ours)": dict(color="#1f5fa8", marker="o", ls="-", lw=2.0)}
    regimes = [(k, lab) for k, lab in REGIMES if any(r["regime"] == k for r in rows)]
    fig, axes = plt.subplots(len(regimes), 4, figsize=(6.5, 1.75 * len(regimes) + 0.25),
                             sharex=True, squeeze=False)
    for i, (regime, rlab) in enumerate(regimes):
        for j, (key, title) in enumerate(KINDS):
            ax = axes[i, j]
            for name in METHODS:
                st = style[name]
                y = [next(r["rel_rmse"] for r in rows if r["regime"] == regime
                          and r["matrix"] == key and r["budget"] == B
                          and r["method"] == name) for B in BUDGETS]
                ax.plot(BUDGETS, y, color=st["color"], marker=st["marker"], ms=2.8,
                        lw=st.get("lw", 1.1), zorder=3 if "ours" in name else 2)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
            ax.tick_params(axis="x", labelsize=6.2)
            ax.minorticks_off()
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(_sci))
            if i == 0:
                ax.set_title(title, pad=4)
        axes[i, 0].set_ylabel(f"{rlab}\nrelative RMSE")
    handles = [plt.Line2D([], [], color=style[n]["color"], marker=style[n]["marker"],
                          ms=2.8, lw=style[n].get("lw", 1.1)) for n in METHODS]
    fig.legend(handles, METHODS, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=0.8, h_pad=1.0)
    pos = [a_.get_position() for a_ in axes[-1]]
    fig.text((pos[0].x0 + pos[-1].x1) / 2, pos[0].y0 - 0.075,
             "matrix-vector budget $B$", ha="center", va="top", fontsize=8)
    fig.savefig("exp_trace.pdf", bbox_inches="tight")
    fig.savefig("exp_trace.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        rows = load()
    elif "--kappa" in sys.argv:
        rows = run([sys.argv[sys.argv.index("--kappa") + 1]])
    else:
        rows = run([k for k, _ in REGIMES])
    plot(rows)
    # per regime: the budget from which P-SLQ has the lowest RMSE on a matrix
    for regime, _ in REGIMES:
        for key, _ in KINDS:
            sub = [r for r in rows if r["regime"] == regime and r["matrix"] == key]
            if not sub:
                continue
            wins = [B for B in BUDGETS if min((r for r in sub if r["budget"] == B),
                    key=lambda r: r["rel_rmse"])["method"] == "P-SLQ (ours)"]
            print(f"[{regime}] {key:14s} P-SLQ lowest at B = {wins or 'none'}")
