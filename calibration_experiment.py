#!/usr/bin/env python3
"""Calibration of the computational-uncertainty belief (Section 6.3).

From the estimator to the paper's figure and table, in one file:

    1. test matrices        four spectra, plus the matrix of Cockayne et al.
    2. Lanczos              the m-step Krylov basis Q_m and T_m
    3. P-SLQ                pslq_belief(): Algorithm 1, the estimated belief
    4. reference beliefs    exact, and without the boundary coupling
    5. statistics           Z of Cockayne et al. (2019)
    6. outputs              fig_calibration.pdf, tab_calibration.tex

Setting (Section 4.4).  The basis Q_m is built once from a fixed v_ref and
reused for every problem.  Each problem draws x* ~ N(0, A^{-1}), so the exact
belief N(Q_m T_m^{-1} Q_m^T v, Sigma_m^perp) is exactly calibrated and serves as
the reference.  The test matrices are stored dense only to compute that
reference; P-SLQ accesses A through matrix-vector products alone.

Usage
    python calibration_experiment.py            # run and plot (about 30 min)
    python calibration_experiment.py --plot     # replot from results.pkl

Requires numpy, scipy and matplotlib.
"""
import argparse
import pickle

import numpy as np
from scipy.linalg import null_space
from scipy.stats import chi2

D = 1000                      # dimension of the synthetic matrices
M_SWEEP = [10, 25, 50, 100, 200, 400, 700]
M_SHOW = 100                  # m shown in the top row of the figure (10% of D)
N_PROBLEMS = 500              # test problems x* per matrix, as in Cockayne et al.
N_PROBES, DEPTH = 30, 60      # P-SLQ probes and quadrature depth


# ========================================================= 1. matrices ======

def random_orthogonal(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def test_matrix(kind, d, rng, kappa=1e3):
    """Symmetric positive definite test matrix with a prescribed spectrum."""
    if kind == "kernel":                       # exponential kernel, inputs in R^5
        X = rng.random((d, 5))
        dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        A = np.exp(-dist / (0.3 * np.sqrt(5))) + 0.01 * np.eye(d)
        return 0.5 * (A + A.T)
    if kind == "cockayne":                     # Cockayne et al. (2019), Sec. 6.1
        lam = rng.exponential(scale=10.0, size=d)
    elif kind == "power-law":
        lam = np.arange(1, d + 1, dtype=float) ** -1.0
        lam = 1 / kappa + (1 - 1 / kappa) * (lam - lam.min()) / (lam.max() - lam.min())
    elif kind == "exponential":
        lam = np.exp(-np.linspace(0.0, np.log(kappa), d))
    elif kind == "bulk+outliers":
        lam = np.concatenate([rng.uniform(0.5, 1.0, 10),
                              rng.uniform(1 / kappa, 0.05, d - 10)])
    else:
        raise ValueError(f"unknown matrix: {kind}")
    U = random_orthogonal(d, rng)
    return (U * lam) @ U.T


# ========================================================== 2. Lanczos ======

def lanczos(matvec, v, m):
    """m Lanczos steps with full reorthogonalisation.
    Returns Q_m, T_m, beta_m and the next Lanczos vector q_{m+1}."""
    d = v.size
    Q = np.zeros((d, m)); a = np.zeros(m); b = np.zeros(m)
    Q[:, 0] = v / np.linalg.norm(v)
    beta, q_prev = 0.0, np.zeros(d)
    for j in range(m):
        w = matvec(Q[:, j]) - beta * q_prev
        a[j] = Q[:, j] @ w
        w -= a[j] * Q[:, j]
        for _ in range(2):
            w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        beta = b[j] = np.linalg.norm(w)
        q_prev = Q[:, j].copy()
        if j + 1 < m:
            Q[:, j + 1] = w / beta
    T = np.diag(a) + np.diag(b[:-1], 1) + np.diag(b[:-1], -1)
    return Q, T, b[-1], w / b[-1]


def reprojected_lanczos(matvec, P, u, l):
    """l Lanczos steps kept in range(P_perp) by reprojecting every vector.
    Returns the tridiagonal matrix, ||u||^2 and the Lanczos basis."""
    d = u.size
    nu2 = float(u @ u)
    V = np.zeros((d, l)); a = np.zeros(l); b = np.zeros(l)
    V[:, 0] = u / np.sqrt(nu2)
    beta, q_prev, n = 0.0, np.zeros(d), l
    for j in range(l):
        z = P(matvec(V[:, j])) - beta * q_prev
        a[j] = V[:, j] @ z
        z -= a[j] * V[:, j]
        for _ in range(2):
            z -= V[:, :j + 1] @ (V[:, :j + 1].T @ z)
        z = P(z)
        beta = b[j] = np.linalg.norm(z)
        q_prev = V[:, j].copy()
        if j + 1 < l:
            if beta < 1e-12:                   # invariant subspace found
                n = j + 1
                break
            V[:, j + 1] = z / beta
    Th = np.diag(a[:n]) + np.diag(b[:n - 1], 1) + np.diag(b[:n - 1], -1)
    return Th, nu2, V[:, :n]


def gauss(Th, nu2, k=1):
    """Gauss quadrature  nu2 * e1^T Th^{-k} e1."""
    x = np.zeros(Th.shape[0]); x[0] = 1.0
    for _ in range(k):
        x = np.linalg.solve(Th, x)
    return nu2 * x[0]


# ============================================================ 3. P-SLQ ======

def pslq_belief(matvec, Q, T, beta_m, q_next, N, l, rng):
    """Algorithm 1: the computable belief Sigma_m^perp, matrix-free.

    (a) boundary probe    reprojected Lanczos from q_{m+1}: a_1, a_2, and the
                          moments p_k of Theta_gamma = Theta - gamma e1 e1^T,
                          the Lanczos matrix of S_perp (Lemma theta_gamma)
    (b) P-SLQ             N probes projected onto the complement, l reprojected
                          Lanczos steps each, Gauss quadrature -> tr(T_perp^-1)
    (c) Sherman-Morrison  tr(S_perp^-1) = tr(T_perp^-1) + gamma a_2/(1 - gamma a_1)
    (d) assembly          s_11, sigma_u = p_3/p_2, omega(m), and
                          q_perp = V Theta_gamma^{-1} e1 from the boundary probe

    Returns Sigma_m^perp as a dense matrix (formed for evaluation only).
    """
    d, m = Q.shape
    P = lambda z: z - Q @ (Q.T @ z)
    Ti = np.linalg.inv(T)
    gamma = beta_m ** 2 * Ti[m - 1, m - 1]

    # (a) boundary probe
    Th, _, V = reprojected_lanczos(matvec, P, q_next, l)
    a1, a2 = gauss(Th, 1.0, 1), gauss(Th, 1.0, 2)
    Thg = Th.copy(); Thg[0, 0] -= gamma
    p2, p3 = gauss(Thg, 1.0, 2), gauss(Thg, 1.0, 3)

    # (b) P-SLQ
    t_perp = 0.0
    for _ in range(N):
        u = P(rng.standard_normal(d))
        Thu, nu2, _ = reprojected_lanczos(matvec, P, u, l)
        t_perp += gauss(Thu, nu2, 1) / N

    # (c) Sherman-Morrison
    tr_S = t_perp + gamma * a2 / (1.0 - gamma * a1)

    # (d) assembly
    s11 = a1 / (1.0 - gamma * a1)
    sigma_u = p3 / p2
    omega = (tr_S - sigma_u) / (d - m - 1)
    q_par = beta_m * (Q @ Ti[:, m - 1])
    e1 = np.zeros(Thg.shape[0]); e1[0] = 1.0
    q_perp = V @ np.linalg.solve(Thg, e1)
    u = q_perp / np.linalg.norm(q_perp)
    P_rest = np.eye(d) - Q @ Q.T - np.outer(u, u)
    return (s11 * np.outer(q_par, q_par)
            - np.outer(q_par, q_perp) - np.outer(q_perp, q_par)
            + sigma_u * np.outer(u, u) + omega * P_rest)


# =============================================== 4. reference beliefs ======

def exact_belief(Ainv, Q, T):
    """Sigma_m^perp = A^{-1} - Q T^{-1} Q^T (Proposition 16)."""
    return Ainv - Q @ np.linalg.solve(T, Q.T)


def decoupled_belief(A, Q, T, beta_m, q_next):
    """The belief without boundary coupling (beta_m = 0, Corollary 8), given
    its exact bulk magnitude -- an advantage over the estimated belief."""
    d, m = Q.shape
    Qp = np.column_stack([q_next, null_space(np.column_stack([Q, q_next]).T)])
    omega_dec = np.trace(np.linalg.inv(Qp.T @ A @ Qp)) / (d - m)
    return omega_dec * (np.eye(d) - Q @ Q.T)


def kappa_S_perp(kind, d, m, seed=0):
    """Complement condition number kappa_S^perp(m), same matrix and basis as run()."""
    rng = np.random.default_rng(seed)
    A = test_matrix(kind, d, rng)
    Q, T, beta_m, q_next = lanczos(lambda v: A @ v, rng.standard_normal(d), m)
    Qp = np.column_stack([q_next, null_space(np.column_stack([Q, q_next]).T)])
    S = Qp.T @ A @ Qp
    S[0, 0] -= beta_m ** 2 * np.linalg.inv(T)[m - 1, m - 1]
    w = np.linalg.eigvalsh(S)
    return w[-1] / w[0]


# ======================================================= 5. statistics ======

def z_statistic(E, Sigma, tol=1e-10):
    """Z = e^T Sigma^+ e for each column of E, and the rank r of Sigma.
    Under calibration Z ~ chi^2_r (Cockayne et al., 2019)."""
    w, U = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    keep = w > tol * w.max()
    C = U[:, keep].T @ E
    return ((C ** 2) / w[keep][:, None]).sum(0), int(keep.sum())


def run(kind, d, ms, n_problems, N, l, seed=0):
    """Z and traces of each belief, for every m, on one matrix."""
    rng = np.random.default_rng(seed)
    A = test_matrix(kind, d, rng)
    Ainv = np.linalg.inv(A)
    matvec = lambda v: A @ v                   # all that P-SLQ sees of A
    v_ref = rng.standard_normal(d)
    X = np.linalg.solve(np.linalg.cholesky(A).T,
                        rng.standard_normal((d, n_problems)))   # x* ~ N(0, A^-1)
    V = A @ X
    res = {}
    for m in ms:
        Q, T, beta_m, q_next = lanczos(matvec, v_ref, m)
        E = X - Q @ np.linalg.solve(T, Q.T @ V)                  # true errors
        beliefs = {
            "exact": exact_belief(Ainv, Q, T),
            "ours": pslq_belief(matvec, Q, T, beta_m, q_next, N,
                                min(l, d - m - 1), rng),
            "decoupled": decoupled_belief(A, Q, T, beta_m, q_next),
        }
        out = {"mse": float(np.mean((E ** 2).sum(0)))}
        for name, S in beliefs.items():
            Z, r = z_statistic(E, S)
            out[name] = dict(Z=Z, r=r, zr=float(Z.mean() / r),
                             trace=float(np.trace(S)))
        out["lanczos"] = dict(zr=np.inf, trace=0.0)              # zero covariance
        res[m] = out
        print(f"  {kind:14s} m = {m:4d}   Z/r: exact {out['exact']['zr']:.3f}  "
              f"ours {out['ours']['zr']:.3f}  decoupled {out['decoupled']['zr']:.3f}",
              flush=True)
    return res


# ========================================================== 6. outputs ======

TEXTWIDTH = 6.5                                # inches; set to the journal's
# Ordered by kappa_S^perp at m = 100, best- to worst-conditioned complement.
SPECTRA = [("kernel", "kernel"), ("power-law", "power-law"),
           ("bulk+outliers", "bulk + outliers"), ("exponential", "exponential")]
BLUE, GREY = "#1f5fa8", "0.86"


def setup_matplotlib():
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
    return plt


def standardised_z(b):
    """(Z - r)/sqrt(2r): approximately N(0, 1) under calibration."""
    return (b["Z"] - b["r"]) / np.sqrt(2 * b["r"])


def kde(t, xs):
    """Gaussian kernel density estimate, Silverman's bandwidth."""
    h = 0.9 * t.std() * t.size ** -0.2
    return np.exp(-0.5 * ((xs[:, None] - t) / h) ** 2).sum(1) / (t.size * h * np.sqrt(2 * np.pi))


def fig_calibration(R, fn="fig_calibration"):
    """Top: distribution of the standardised Z at m = M_SHOW against the
    chi^2_r reference.  Bottom: its width over the whole sweep of m."""
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, len(SPECTRA), figsize=(TEXTWIDTH, 3.15),
                             gridspec_kw=dict(height_ratios=[1.0, 0.9]))
    xs = np.linspace(-4.5, 4.5, 400)
    for j, (key, title) in enumerate(SPECTRA):
        # top: density at M_SHOW
        ax = axes[0, j]
        r = R[key][M_SHOW]["exact"]["r"]
        ax.fill_between(xs, np.sqrt(2 * r) * chi2.pdf(r + xs * np.sqrt(2 * r), r),
                        color=GREY, lw=0, zorder=0)
        ax.plot(xs, kde(standardised_z(R[key][M_SHOW]["exact"]), xs),
                color="k", ls=(0, (4, 2)), lw=1.0)
        ax.plot(xs, kde(standardised_z(R[key][M_SHOW]["ours"]), xs), color=BLUE)
        k = kappa_S_perp(key, D, M_SHOW)
        ax.text(0.97, 0.97, rf"$\kappa^\perp_S={k:.1f}$" if k < 10 else
                rf"$\kappa^\perp_S={k:.0f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
        ax.set_title(title, pad=4)
        ax.set_xlim(-4.5, 4.5); ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_ylim(0, None); ax.set_yticks([])
        ax.spines["left"].set_visible(False)

        # bottom: width against m
        ax = axes[1, j]
        ms = sorted(R[key])
        ax.axhline(1.0, color=GREY, lw=1.0, zorder=0)
        ax.plot(ms, [np.std(standardised_z(R[key][m]["exact"])) for m in ms],
                color="k", ls=(0, (4, 2)), lw=1.0)
        ax.plot(ms, [np.std(standardised_z(R[key][m]["ours"])) for m in ms],
                color=BLUE, marker="o", ms=2.8)
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1000"])
        ax.set_xlim(8, 1000)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylim(0.9, 2.0); ax.set_yticks([1.0, 1.5, 2.0])
        if j:
            ax.set_yticklabels([])

    axes[0, 0].set_ylabel(f"density, $m={M_SHOW}$")
    axes[1, 0].set_ylabel("width of $Z$")
    handles = [plt.Line2D([], [], color="k", ls=(0, (4, 2)), lw=1.0),
               plt.Line2D([], [], color=BLUE, marker="o", ms=2.8),
               plt.matplotlib.patches.Patch(color=GREY)]
    fig.legend(handles, ["exact belief", "ours (P-SLQ)", r"calibrated ($\chi^2_r$)"],
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04), handlelength=2.2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=0.8, h_pad=2.6)
    # one axis label per row, centred under the row's tick labels
    for row, label in ((0, r"standardised statistic $(Z-r)/\sqrt{2r}$"),
                       (1, "Krylov dimension $m$")):
        pos = [a.get_position() for a in axes[row]]
        fig.text((pos[0].x0 + pos[-1].x1) / 2, pos[0].y0 - 0.085, label,
                 ha="center", va="top", fontsize=8)
    fig.savefig(fn + ".pdf", bbox_inches="tight")
    fig.savefig(fn + ".png", dpi=300, bbox_inches="tight")


def tab_calibration(R, rows, fn="tab_calibration.tex"):
    """Z/r and predicted-over-realised variance for each belief."""
    f = lambda v: r"$\infty$" if not np.isfinite(v) else f"{v:.2f}"
    out = [r"\begin{table}[t]", r"\centering\small",
           r"\caption{Calibration of the computational-uncertainty belief. $Z/r$ is "
           r"the statistic of \citet{cockayne2019bayesiancg} divided by its degrees "
           r"of freedom; \emph{var.\ ratio} is the predicted variance divided by the "
           r"realised squared error. Both equal $1$ for a calibrated belief. Standard "
           r"Lanczos assigns zero variance to the truncation error. The "
           r"$\beta_m{=}0$ variant is given its exact bulk magnitude, whereas ours "
           r"is estimated by P-SLQ. Mean over 500 problems with "
           r"$x^\star\sim\mathcal N(0,A^{-1})$.}",
           r"\label{tab:calibration}", r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{4}{c}{$Z/r$} & \multicolumn{2}{c}{var.\ ratio} \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
           r"Matrix & $m$ & exact & ours & ours, $\beta_m{=}0$ & Lanczos & ours "
           r"& ours, $\beta_m{=}0$ \\", r"\midrule"]
    for i, (key, title, ms) in enumerate(rows):
        for k, m in enumerate(ms):
            o = R[key][m]
            zr = [o[b]["zr"] for b in ("exact", "ours", "decoupled", "lanczos")]
            vr = [o[b]["trace"] / o["mse"] for b in ("ours", "decoupled")]
            out.append(f"{title if k == 0 else ''} & {m} & "
                       + " & ".join(map(f, zr)) + " & " + " & ".join(map(f, vr)) + r" \\")
        if i < len(rows) - 1:
            out.append(r"\addlinespace[2pt]")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(fn, "w") as fh:
        fh.write("\n".join(out) + "\n")


# ============================================================== main ========

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plot", action="store_true", help="replot from results.pkl")
    # Safeguard against Jupyter/Colab kernel background flags like -f
    args, unknown = ap.parse_known_args()

    if not args.plot:
        R = {key: run(key, D, M_SWEEP, N_PROBLEMS, N_PROBES, DEPTH)
             for key, _ in SPECTRA}
        # Two eigenvalues near 0.01 make A^{-1} heavy-tailed, the worst case for
        # Hutchinson probing: 200 probes instead of 30.
        R["cockayne"] = run("cockayne", 100, [10, 50], N_PROBLEMS, 200, 40)
        with open("results.pkl", "wb") as fh:
            pickle.dump(R, fh)
    with open("results.pkl", "rb") as fh:
        R = pickle.load(fh)

    fig_calibration(R)
    tab_calibration(R, [(k, t, (10, M_SHOW)) for k, t in SPECTRA]
                    + [("cockayne", "Cockayne et al.", (10, 50))])
    print("wrote fig_calibration.pdf and tab_calibration.tex")
