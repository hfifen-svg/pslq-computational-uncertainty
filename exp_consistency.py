#!/usr/bin/env python3
"""P-SLQ consistency (Theorem 17): bias against quadrature depth, variance
against the number of probes.

Target: t = tr(T_perp^{-1}), the complement trace estimated by P-SLQ.

Bias.  Each probe's l-point Gauss estimate is compared with the exact quadratic
form of the SAME probe, so the curve shows the quadrature bias alone, free of
Monte Carlo noise.  The depth-l tridiagonal is the leading l x l block of the
depth-L one, so one run per probe serves every l.

Variance.  The probe pool is split into disjoint groups of N probes; each group
gives an independent estimate, and their spread is the empirical variance.
Theorem 17 predicts Var = 2 ||T_perp^{-1}||_F^2 / N for Gaussian probes once the
quadrature is converged; that line is drawn for each matrix.

Probes are run in batches (independent Lanczos processes advanced together),
so the whole experiment takes a few minutes.

Reference lines.  Left: the rate rho_perp^{-2l} of Theorem 16, from each
matrix's own kappa_perp, anchored at l = 2; it is an upper bound, so the data
lie on or below it.  Right: the variance 2 ||T_perp^{-1}||_F^2 / N predicted by
Theorem 17.

Usage:  python exp_consistency.py           run and plot (about 5 min)
        python exp_consistency.py --plot    replot from exp_consistency.npz
"""
import numpy as np
from pslq_core import test_matrix, lanczos, complement_basis

D, M = 1000, 20
L_MAX = 30                       # quadrature depth of every probe run
LS = list(range(2, L_MAX + 1, 2))
POOL = 4096                      # probes per matrix
NS = [1, 2, 4, 8, 16, 32, 64, 128]
SEEDS = range(5)                 # independent matrices per spectrum
KINDS = [("kernel", "kernel"), ("power-law", "power-law"),
         ("bulk+outliers", "bulk + outliers"), ("exponential", "exponential")]
COLORS = {"kernel": "#1f5fa8", "power-law": "#2a9d8f",
          "bulk+outliers": "#e9a03b", "exponential": "#b2182b"}


def batched_reprojected_lanczos(A, Q, U, L):
    """L reprojected Lanczos steps for every column of U at once.
    Returns alphas (L x B), betas (L-1 x B) and the squared probe norms."""
    P = lambda Z: Z - Q @ (Q.T @ Z)
    d, B = U.shape
    nu2 = (U * U).sum(0)
    V = np.zeros((L, d, B))
    V[0] = U / np.sqrt(nu2)
    al = np.zeros((L, B)); be = np.zeros((L - 1, B))
    prev, bprev = np.zeros((d, B)), np.zeros(B)
    for j in range(L):
        W = P(A @ V[j]) - bprev * prev
        al[j] = (V[j] * W).sum(0)
        W -= al[j] * V[j]
        for _ in range(2):                              # full reorthogonalisation
            C = np.einsum("jdb,db->jb", V[:j + 1], W)
            W -= np.einsum("jdb,jb->db", V[:j + 1], C)
        W = P(W)
        b = np.linalg.norm(W, axis=0)
        if j + 1 < L:
            be[j] = b
            V[j + 1] = W / np.maximum(b, 1e-300)
        prev, bprev = V[j], b
    return al, be, nu2


def gauss_batch(al, be, nu2, l):
    """nu2 * e1^T T_l^{-1} e1 for every probe, from the leading l x l block."""
    out = np.empty(al.shape[1])
    for i in range(al.shape[1]):
        T = np.diag(al[:l, i]) + np.diag(be[:l - 1, i], 1) + np.diag(be[:l - 1, i], -1)
        x = np.zeros(l); x[0] = 1.0
        out[i] = nu2[i] * np.linalg.solve(T, x)[0]
    return out


def one_matrix(kind, seed):
    rng = np.random.default_rng(seed)
    A = test_matrix(kind, D, rng)
    Q, T, bm, qn = lanczos(A, rng.standard_normal(D), M)
    Qp = complement_basis(Q, qn)
    Tpi = np.linalg.inv(Qp.T @ A @ Qp)
    Aperp = Qp @ Tpi @ Qp.T                              # (P A P)^+
    t = np.trace(Tpi)
    var_theory = 2.0 * np.sum(Tpi ** 2) / t ** 2          # relative, times N
    U = rng.standard_normal((D, POOL))
    U -= Q @ (Q.T @ U)                                    # projected probes
    exact_qf = (U * (Aperp @ U)).sum(0)
    bias, est_L = [], None
    for start in range(0, POOL, 512):                    # batches of 512 probes
        al, be, nu2 = batched_reprojected_lanczos(A, Q, U[:, start:start + 512], L_MAX)
        qf = exact_qf[start:start + 512]
        bias.append([(gauss_batch(al, be, nu2, l) - qf) for l in LS])
        g = gauss_batch(al, be, nu2, L_MAX)
        est_L = g if est_L is None else np.concatenate([est_L, g])
    rel_bias = np.array([np.concatenate([b[k] for b in bias]).mean() / t
                         for k in range(len(LS))])
    rel_var = np.array([np.var(est_L[: (POOL // n) * n].reshape(-1, n).mean(1), ddof=1)
                        / t ** 2 for n in NS])
    kperp = np.linalg.cond(Qp.T @ A @ Qp)
    return rel_bias, rel_var, var_theory, kperp


def main(plot_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False})

    res = {}
    if plot_only:
        z = np.load("exp_consistency.npz")
        for kind, _ in KINDS:
            res[kind] = {f: z[f"{kind}|{f}"] for f in ("bias", "var", "theory", "kperp")}
    else:
        for kind, _ in KINDS:
            runs = [one_matrix(kind, s) for s in SEEDS]
            res[kind] = dict(bias=np.array([r[0] for r in runs]),
                             var=np.array([r[1] for r in runs]),
                             theory=np.array([r[2] for r in runs]),
                             kperp=np.array([r[3] for r in runs]))
        np.savez("exp_consistency.npz", **{f"{k}|{f}": v for k, d in res.items()
                                           for f, v in d.items()})
    for kind, _ in KINDS:
        b = res[kind]["bias"]
        print(f"{kind:14s} kappa_perp = {res[kind]['kperp'].mean():8.1f}   "
              f"all biases negative: {bool((b < 0).all())}   "
              f"rel. bias at l={LS[-1]}: {np.abs(b[:, -1]).mean():.1e}   "
              f"var/theory at N=1: "
              f"{(res[kind]['var'][:, 0] / res[kind]['theory']).mean():.3f}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.35))
    for kind, title in KINDS:
        c, r = COLORS[kind], res[kind]
        lab = rf"{title} ($\kappa^\perp={r['kperp'].mean():.0f}$)"
        # left: bias, with the rate of Theorem 16 for this matrix (dotted)
        lb = np.log10(np.abs(r["bias"]))
        mu, sd = lb.mean(0), lb.std(0)
        ax1.fill_between(LS, 10 ** (mu - sd), 10 ** (mu + sd), color=c, alpha=0.15, lw=0)
        ax1.plot(LS, 10 ** mu, color=c, marker="o", ms=2.6, label=lab, zorder=3)
        sk = np.sqrt(r["kperp"].mean())
        rho = (sk + 1) / (sk - 1)
        ls_ = np.array(LS, float)
        ax1.plot(ls_, 10 ** mu[0] * rho ** (-2 * (ls_ - LS[0])), color=c,
                 ls=(0, (1, 1.2)), lw=1.0, zorder=2)
        # right: variance
        lv = np.log10(r["var"])
        mu, sd = lv.mean(0), lv.std(0)
        # theory of Theorem 17, drawn wide and light under the measurements
        ax2.plot(NS, r["theory"].mean() / np.array(NS, float), color=c, lw=6.0,
                 alpha=0.18, solid_capstyle="round", zorder=1)
        ax2.plot(NS, 10 ** mu, color=c, marker="o", ms=2.6, zorder=3)

    ax1.set_yscale("log"); ax1.set_xlabel("quadrature depth $l$")
    ax1.set_ylabel("|relative bias|")
    ax1.set_xticks([2, 10, 20, 30])
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xlabel("number of probes $N$"); ax2.set_ylabel("relative variance")
    ax2.set_xticks(NS); ax2.set_xticklabels([str(n) for n in NS])
    ax2.minorticks_off()
    h, l = ax1.get_legend_handles_labels()
    h += [plt.Line2D([], [], color="0.35", ls=(0, (1, 1.2)), lw=1.0),
          plt.Line2D([], [], color="0.35", lw=6.0, alpha=0.25, solid_capstyle="round")]
    l += [r"rate $\rho_\perp^{-2l}$",
          r"$2\|T_{m,\perp}^{-1}\|_F^2/N$"]
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    fig.savefig("exp_consistency.pdf", bbox_inches="tight")
    fig.savefig("exp_consistency.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    main(plot_only="--plot" in sys.argv)
