import numpy as np

from problem.pdg import PDGProblem, Solution


def heuristic_gamma(u: np.ndarray, v: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Section VI heuristic (p.1363). r-hat_k = u_k/||u_k|| from the solve without pointing estimates
    R2(t_k)/||R2(t_k)||. kappa_1k = v^T r-hat_k, kappa_2k = ||v - (v^T r-hat_k) r-hat_k||.
    Case 1, kappa_1k >= 0 or kappa_2k >= gamma: keep (61) exactly, so the per-node gamma stays gamma.
    Case 2, otherwise: relax to (62), v^T u_k >= kappa_2k sigma_k, so the per-node gamma becomes kappa_2k.
    """
    rhat = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    k1 = rhat @ v
    k2 = np.linalg.norm(v[None, :] - k1[:, None] * rhat, axis=1)
    exact = (k1 >= 0) | (k2 >= gamma)
    return np.where(exact, gamma, k2), k1, k2


def solve_heuristic(P: PDGProblem, r0, v0, m0: float, tf: float, gamma: float) -> Solution:
    # P must be built with Options(pointing=True); gamma = None turns (61) off (vacuous -1).
    base = P.solve(r0, v0, m0, tf, gamma=None)
    if not base.ok:
        return base
    g, k1, k2 = heuristic_gamma(base.U, np.asarray(P.opts.pointing_axis, float), gamma)
    s = P.solve(r0, v0, m0, tf, gamma=g)
    s.extra["relaxed_nodes"] = np.flatnonzero(g != gamma).tolist()
    s.extra["kappa1"], s.extra["kappa2"] = k1.tolist(), k2.tolist()
    s.wall_s += base.wall_s
    return s


def pointing_violation_deg(sol: Solution, v: np.ndarray, gamma: float, min_accel: float = 1e-3) -> np.ndarray:
    """Angle by which each node's thrust lies outside the cone (60), in degrees; 0 inside. Nodes with
    ||u|| below min_accel (m/s^2) have no direction and are reported as nan."""
    n = np.linalg.norm(sol.U, axis=1)
    cosang = (sol.U @ v) / np.maximum(n, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    out = np.maximum(0.0, ang - np.degrees(np.arccos(gamma)))
    out[n < min_accel] = np.nan
    return out
