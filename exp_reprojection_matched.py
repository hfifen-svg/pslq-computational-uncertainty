#!/usr/bin/env python3
r"""Figure 2, redone with matched targets (reviewer item 2).

The original figure compared naive SLQ against tr(T_perp^{-1}).  That was the
wrong reference.  For a projected probe u = P_perp xi, plain Lanczos on A
approximates u^T A^{-1} u, and

    E[u^T A^{-1} u] = tr(P_perp A^{-1} P_perp) = tr(S_perp^{-1})

by Lemma 7(ii).  So naive SLQ is an estimator of the COUPLING-CORRECTED trace,
not a corrupted estimator of tr(T_perp^{-1}).  The gap in the old figure was
tr(S_perp^{-1}) - tr(T_perp^{-1}) >= 0, not drift.

Here both methods estimate the same quantity, tr(S_perp^{-1}):
    naive SLQ           : Lanczos on A from u, no reprojection
    P-SLQ + SM          : reprojected Lanczos -> tr(T_perp^{-1}), then the
                          Sherman-Morrison correction from one boundary probe

QUADRATURE bias is isolated from Monte Carlo noise by comparing each probe's
quadrature value with the exact quadratic form of the SAME probe.  The same
probes are used for both methods.

Also checked: Theta_gamma (boundary tridiagonal with gamma removed from its
(1,1) entry) equals the Lanczos matrix of S_perp started at e_1 -- the
reviewer's observation that makes the separate CG solve unnecessary.
"""
import numpy as np
from pslq_core import (test_matrix, lanczos, lanczos_tridiag, quad,
                       exact_pieces)

D, M, N, LMAX = 1000, 20, 300, 24
LS = list(range(2, LMAX + 1, 2))
KINDS = ["power-law", "exponential", "bulk+outliers", "kernel"]


def run(kind, seed=0):
    rng = np.random.default_rng(seed)
    A = test_matrix(kind, D, rng)
    Ainv = np.linalg.inv(A)
    Q, T, bm, qn = lanczos(A, rng.standard_normal(D), M)
    ex = exact_pieces(A, Q, T, bm, qn)
    P = lambda z: z - Q @ (Q.T @ z)
    Aperp_pinv = ex["Qp"] @ ex["Tpi"] @ ex["Qp"].T
    trS, trT = ex["trS"], np.trace(ex["Tpi"])

    # Theta_gamma == Lanczos matrix of S_perp from e1 ?
    th, _, _ = lanczos_tridiag(A, qn, 12, P=P)
    thg = th.copy(); thg[0, 0] -= ex["gamma"]
    e1 = np.zeros(D - M); e1[0] = 1.0
    ts, _, _ = lanczos_tridiag(ex["S"], e1, 12)
    theta_check = np.abs(thg - ts).max() / np.abs(ts).max()

    # boundary probe at full depth, sliced per l
    thL, _, _ = lanczos_tridiag(A, qn, LMAX, P=P)
    g = ex["gamma"]
    sm_exact = g * (ex["Tpi"] @ ex["Tpi"])[0, 0] / (1 - g * ex["Tpi"][0, 0])

    nv_est = np.zeros((N, len(LS))); ps_est = np.zeros((N, len(LS)))
    nv_ref = np.zeros(N); ps_ref = np.zeros(N)
    for i in range(N):
        u = P(rng.standard_normal(D))
        tn, w2, _ = lanczos_tridiag(A, u, LMAX)          # naive
        tp, w2p, _ = lanczos_tridiag(A, u, LMAX, P=P)    # reprojected
        nv_ref[i] = u @ Ainv @ u
        ps_ref[i] = u @ Aperp_pinv @ u
        for k, l in enumerate(LS):
            nv_est[i, k] = quad(tn, w2, 1, l)
            ps_est[i, k] = quad(tp, w2p, 1, l)

    out = dict(kind=kind, trS=trS, trT=trT, theta_check=theta_check,
               kappa=np.linalg.cond(A),
               kappa_perp=np.linalg.cond(ex["Tp"]),
               kappaS_perp=np.linalg.cond(ex["S"]))
    nb, pb, nt, pt = [], [], [], []
    for k, l in enumerate(LS):
        a1, a2 = quad(thL, 1.0, 1, l), quad(thL, 1.0, 2, l)
        sm_hat = g * a2 / (1 - g * a1)
        nb.append((nv_est[:, k] - nv_ref).mean() / trS)
        pb.append(((ps_est[:, k] - ps_ref).mean() + sm_hat - sm_exact) / trS)
        nt.append(abs(nv_est[:, k].mean() - trS) / trS)
        pt.append(abs(ps_est[:, k].mean() + sm_hat - trS) / trS)
    out.update(naive_bias=np.array(nb), pslq_bias=np.array(pb),
               naive_total=np.array(nt), pslq_total=np.array(pt))
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False})

    res = [run(k) for k in KINDS]
    order = ["kernel", "power-law", "bulk+outliers", "exponential"]   # as Fig. calibration
    res = sorted(res, key=lambda r: order.index(r["kind"]))
    titles = {"kernel": "kernel", "power-law": "power-law",
              "bulk+outliers": "bulk + outliers", "exponential": "exponential"}
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.95), sharey=True)
    for ax, r in zip(axes, res):
        # P-SLQ underneath, slightly thicker; the plain run on top, so that where
        # the two coincide (exponential spectrum) both remain visible.
        ax.semilogy(LS, np.abs(r["pslq_bias"]), color="#1f5fa8", marker="o",
                    ms=3.6, lw=2.0, zorder=2)
        ax.semilogy(LS, np.abs(r["naive_bias"]), color="#b2182b", ls=(0, (4, 2)),
                    marker="x", ms=3.2, mew=1.1, lw=1.2, zorder=3)
        ax.set_title(titles[r["kind"]], pad=4)
        mant, ex = f"{r['kappa']:.0e}".split("e")
        kap = (rf"$\kappa=10^{{{int(ex)}}}$" if mant == "1"
               else rf"$\kappa={mant}\times10^{{{int(ex)}}}$")
        ax.text(0.04, 0.04, kap + "\n" + rf"$\kappa^\perp_S={r['kappaS_perp']:.0f}$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
        ax.set_xticks([4, 12, 20])
    axes[0].set_ylabel("relative quadrature bias")
    h = [plt.Line2D([], [], color="#b2182b", ls=(0, (4, 2)), marker="x", ms=3.2,
                    mew=1.1, lw=1.2),
         plt.Line2D([], [], color="#1f5fa8", marker="o", ms=3.6, lw=2.0)]
    fig.legend(h, ["plain Lanczos, no reprojection", "P-SLQ + Sherman-Morrison"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), w_pad=0.8)
    pos = [a_.get_position() for a_ in axes]
    fig.text((pos[0].x0 + pos[-1].x1) / 2, pos[0].y0 - 0.12,
             "quadrature depth $l$", ha="center", va="top", fontsize=8)
    fig.savefig("exp_reprojection_matched.pdf", bbox_inches="tight")
    fig.savefig("exp_reprojection_matched.png", dpi=300, bbox_inches="tight")

    print(f"d={D}, m={M}, N={N} probes (same probes for both methods)")
    for r in res:
        print(f"{r['kind']:14s} kappa={r['kappa']:.2e} kappa_S_perp={r['kappaS_perp']:.2e}"
              f"  bias at l={LS[-1]}: plain {r['naive_bias'][-1]:.2e}"
              f"  P-SLQ {r['pslq_bias'][-1]:.2e}")
