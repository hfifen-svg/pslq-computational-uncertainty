"""Shared NumPy machinery for the TMLR revision experiments.

Dense, small-d (d ~ 1000) so that every quantity has an exact reference.
"""
import numpy as np
from scipy.linalg import null_space


# ------------------------------- matrices ----------------------------------
def random_orthogonal(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def test_matrix(kind, d, rng, kappa=1e3):
    """The three synthetic regimes of Section 6.1 plus the kernel matrix."""
    if kind == "kernel":
        X = rng.random((d, 5))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        A = np.exp(-D / (0.3 * np.sqrt(5))) + 0.01 * np.eye(d)
        return 0.5 * (A + A.T)
    if kind == "cockayne":
        # Cockayne et al. (2019, Sec. 6.1): d = 100, eigenvalues drawn i.i.d.
        # from an exponential distribution.  They then embed the spectrum in
        # a 20%-sparse matrix with MATLAB's sprandsym; our method only sees
        # the spectrum through matrix-vector products, so a dense random
        # orthogonal basis is used here.  Z is scale-invariant, so the
        # exponential's parameter only fixes the units.
        lam = rng.exponential(scale=10.0, size=d)
        U = random_orthogonal(d, rng)
        return (U * lam) @ U.T
    if kind == "power-law":
        lam = np.arange(1, d + 1, dtype=float) ** -1.0
        lam = 1 / kappa + (1 - 1 / kappa) * (lam - lam.min()) / (lam.max() - lam.min())
    elif kind == "exponential":
        lam = np.exp(-np.linspace(0.0, np.log(kappa), d))
    elif kind == "bulk+outliers":
        lam = np.concatenate([rng.uniform(0.5, 1.0, 10),
                              rng.uniform(1 / kappa, 0.05, d - 10)])
    else:
        raise ValueError(kind)
    U = random_orthogonal(d, rng)
    return (U * lam) @ U.T


# ------------------------------- Lanczos -----------------------------------
def lanczos(A, v, m):
    """m Lanczos steps, full reorthogonalisation. Returns Q, T, beta_m, q_{m+1}."""
    d = v.size
    Q = np.zeros((d, m)); al = np.zeros(m); be = np.zeros(m)
    Q[:, 0] = v / np.linalg.norm(v); b = 0.0; qp = np.zeros(d)
    for j in range(m):
        w = A @ Q[:, j] - b * qp
        a = Q[:, j] @ w
        w -= a * Q[:, j]
        for _ in range(2):
            w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        al[j] = a; b = np.linalg.norm(w); be[j] = b; qp = Q[:, j].copy()
        if j + 1 < m:
            Q[:, j + 1] = w / b
    T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)
    return Q, T, be[-1], w / be[-1]


def lanczos_tridiag(A, u, L, P=None):
    """L Lanczos steps from u/||u||.

    P=None : plain Lanczos on A, full reorthogonalisation within its own
             basis ("naive SLQ"). Its Krylov space leaves range(P_perp).
    P=proj : reproject onto range(P_perp) every step ("P-SLQ").

    The depth-l tridiagonal is the leading l x l block of the depth-L one,
    so one run serves every l <= L.
    """
    nw2 = float(u @ u)
    d = u.size
    V = np.zeros((d, L)); al = np.zeros(L); be = np.zeros(L)
    V[:, 0] = u / np.sqrt(nw2); b = 0.0; qp = np.zeros(d); n = L
    for j in range(L):
        z = A @ V[:, j]
        if P is not None:
            z = P(z)
        z -= b * qp
        a = V[:, j] @ z
        z -= a * V[:, j]
        for _ in range(2):
            z -= V[:, :j + 1] @ (V[:, :j + 1].T @ z)
        if P is not None:
            z = P(z)
        al[j] = a; b = np.linalg.norm(z); be[j] = b; qp = V[:, j].copy()
        if j + 1 < L:
            if b < 1e-12:
                n = j + 1
                break
            V[:, j + 1] = z / b
    T = np.diag(al[:n]) + np.diag(be[:n - 1], 1) + np.diag(be[:n - 1], -1)
    return T, nw2, V[:, :n]


def quad(T, nw2, k=1, l=None):
    """nw2 * e1^T T_l^{-k} e1 with T_l the leading l x l block."""
    if l is not None:
        T = T[:l, :l]
    e1 = np.zeros(T.shape[0]); e1[0] = 1.0
    x = e1.copy()
    for _ in range(k):
        x = np.linalg.solve(T, x)
    return nw2 * (e1 @ x)


# --------------------------- exact references ------------------------------
def complement_basis(Q, qn):
    """Orthonormal basis of range(P_perp) whose first column is q_{m+1}."""
    rest = null_space(np.column_stack([Q, qn]).T)
    return np.column_stack([qn, rest])


def exact_pieces(A, Q, T, bm, qn):
    """Exact S_perp, coupling vectors and bulk magnitude (dense)."""
    d, m = Q.shape
    Qp = complement_basis(Q, qn)
    Tp = Qp.T @ A @ Qp
    Ti = np.linalg.inv(T)
    gamma = bm ** 2 * Ti[m - 1, m - 1]
    S = Tp.copy(); S[0, 0] -= gamma
    Si = np.linalg.inv(S)
    q_par = bm * (Q @ Ti[:, m - 1])
    q_perp = Qp @ Si[:, 0]
    u = q_perp / np.linalg.norm(q_perp)
    sig_u = float(u @ (Qp @ Si @ Qp.T) @ u)
    omega = (np.trace(Si) - sig_u) / (d - m - 1)
    return dict(Qp=Qp, Tp=Tp, Tpi=np.linalg.inv(Tp), gamma=gamma, S=S, Si=Si,
                s11=Si[0, 0], q_par=q_par, q_perp=q_perp, u=u, sig_u=sig_u,
                omega=omega, trS=np.trace(Si), Ti=Ti)


# ------------------------- estimated (matrix-free) --------------------------
def estimated_pieces(A, Q, T, bm, qn, N, l, rng):
    """Everything Algorithm 1 computes, from matrix-vector products only.

    q_perp is taken from the boundary probe itself, q_perp ~ V Theta_gamma^{-1} e1
    (the reviewer's suggestion): no separate CG solve.
    """
    d, m = Q.shape
    P = lambda z: z - Q @ (Q.T @ z)
    Ti = np.linalg.inv(T)
    gamma = bm ** 2 * Ti[m - 1, m - 1]

    th, _, V = lanczos_tridiag(A, qn, l, P=P)         # boundary probe
    a1, a2 = quad(th, 1.0, 1), quad(th, 1.0, 2)
    thg = th.copy(); thg[0, 0] -= gamma
    p2, p3, p4 = (quad(thg, 1.0, k) for k in (2, 3, 4))

    acc = 0.0
    for _ in range(N):                                # P-SLQ
        uu = P(rng.standard_normal(d))
        t, nw2, _ = lanczos_tridiag(A, uu, l, P=P)
        acc += quad(t, nw2, 1)
    t_hat = acc / N

    trS = t_hat + gamma * a2 / (1 - gamma * a1)       # Sherman-Morrison
    s11 = a1 / (1 - gamma * a1)
    sig_u = p3 / p2
    omega = (trS - sig_u) / (d - m - 1)
    q_par = bm * (Q @ Ti[:, m - 1])
    e1 = np.zeros(thg.shape[0]); e1[0] = 1.0
    q_perp = V @ np.linalg.solve(thg, e1)
    u = q_perp / np.linalg.norm(q_perp)
    c_norm = np.sqrt(max(p4 / p2 - (p3 / p2) ** 2, 0.0))
    return dict(gamma=gamma, s11=s11, q_par=q_par, q_perp=q_perp, u=u,
                sig_u=sig_u, omega=omega, trS=trS, c_norm=c_norm, Ti=Ti)


def structured_remainder(Q, pc):
    """The remainder Sigma_m^perp built from pieces (coupling + sigma_u + bulk)."""
    d = Q.shape[0]
    qp, qq, u = pc["q_par"], pc["q_perp"], pc["u"]
    Prest = np.eye(d) - Q @ Q.T - np.outer(u, u)
    return (pc["s11"] * np.outer(qp, qp) - np.outer(qp, qq) - np.outer(qq, qp)
            + pc["sig_u"] * np.outer(u, u) + pc["omega"] * Prest)
