"""Hypotheses of the pointing-constraint LCvx result (Theorem 2) as Malyuta et al. 2022 state them,
p.7-8 and p.46-47, attributed there to Carson, Açıkmeşe & Blackmore 2011 [87] and Açıkmeşe, Carson &
Blackmore 2013 [96]. Secondary source: the 2013 paper itself is not in refs/.

Following the tutorial (p.46), z is a fictitious state and the state is x = (r, rdot) in R^6.
"""
import numpy as np
from scipy.linalg import expm, null_space

from dynamics.model import Vehicle, continuous_matrices


def rv_matrices(omega: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    # eq. (105): A = [[0, I], [-w x w x, -2 w x]], B = [0; I].
    Ac, Bc = continuous_matrices(1.0, omega)
    return Ac[:6, :6], Bc[:6, :3]


def kalman_singular_values(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return np.linalg.svd(np.hstack(blocks), compute_uv=False)


def gramian_eigs(A: np.ndarray, B: np.ndarray, T: float) -> tuple[float, float]:
    """Smallest and largest eigenvalues of the controllability Gramian W(T) = int_0^T e^{As} B B^T e^{A^T s} ds
    (Van Loan). Mixed units (position and velocity rows), so it is only compared as a ratio."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n], M[:n, n:], M[n:, n:] = -A, B @ B.T, A.T
    F = expm(M * T)
    W = F[n:, n:].T @ F[:n, n:]
    ev = np.linalg.eigvalsh(0.5 * (W + W.T))
    return float(ev.min()), float(ev.max())


def controllability(A: np.ndarray, B: np.ndarray, T: float) -> dict:
    # For LTI, total controllability (p.7) is the Kalman rank test.
    sv = kalman_singular_values(A, B)
    rank = int(np.sum(sv > 1e-12 * sv.max()))
    lo, hi = gramian_eigs(A, B, T)
    return {"rank": rank, "holds": rank == A.shape[0], "min_singular_value": float(sv[min(5, len(sv) - 1)]),
            "gramian_min_eig": lo, "gramian_max_eig": hi}


def condition1(omega: np.ndarray | None, T: float) -> dict:
    # Condition 1, p.7: {A, B} totally controllable.
    return controllability(*rv_matrices(omega), T)


def condition3(omega: np.ndarray | None, n_hat: np.ndarray, T: float) -> dict:
    # Condition 3, p.8: N spans null(n_hat^T), the thrust components orthogonal to the cone axis;
    # {A, B N} totally controllable. p.46: with n_hat vertical it holds iff w x n_hat != 0.
    A, B = rv_matrices(omega)
    N = null_space(np.asarray(n_hat, float)[None, :])
    return controllability(A, B @ N, T)


def condition2(sigma_tf: float) -> dict:
    """Condition 2, eq. (8): m_LCvx = [grad_x m; grad_t m + zeta l(sigma(tf))] and the columns of
    B_LCvx = [grad_x b^T; grad_t b^T] linearly independent. Here m = 0, zeta = 1, l(sigma) = sigma (cost
    (47)) and b = x(tf) = 0 (eq. (5)), so m_LCvx = [0; sigma(tf)] and B_LCvx = [I_6; 0]: independent iff
    sigma(tf) != 0. The tutorial reaches the same test on p.47, eq. (107)."""
    m = np.zeros(7)
    m[6] = sigma_tf
    Bm = np.vstack([np.eye(6), np.zeros((1, 6))])
    return {"sigma_tf": float(sigma_tf), "holds": int(np.linalg.matrix_rank(np.column_stack([m, Bm]))) == 7}


def glideslope_force_balance(veh: Vehicle, glideslope_deg: float, half_angle_deg: float, n: int = 2001) -> dict:
    """Sufficient test (106a-b), p.47, that the glide slope is only instantaneously active: for all
    theta in [pi/2 - gamma_p - gamma_gs, pi/2 + gamma_p - gamma_gs],
    rho_min cos(theta) < m_dry ||g|| sin(gamma_gs) and rho_max cos(theta) > m_wet ||g|| sin(gamma_gs).
    The tutorial's own argument (not attributed to [96]); it relies on bang-bang thrust [45]."""
    gp, gs = np.deg2rad(half_angle_deg), np.deg2rad(glideslope_deg)
    th = np.linspace(np.pi / 2 - gp - gs, np.pi / 2 + gp - gs, n)
    g = np.linalg.norm(veh.g)
    a_margin = veh.m_dry * g * np.sin(gs) - veh.rho1 * np.cos(th)      # N, > 0 required
    b_margin = veh.rho2 * np.cos(th) - veh.m_wet * g * np.sin(gs)      # N, > 0 required
    return {"theta_range_deg": [float(np.degrees(th[0])), float(np.degrees(th[-1]))],
            "a_holds": bool(a_margin.min() > 0), "a_min_margin_N": float(a_margin.min()),
            "b_holds": bool(b_margin.min() > 0), "b_min_margin_N": float(b_margin.min())}
