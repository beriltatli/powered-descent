import numpy as np

from problem.pdg import PDGProblem


def sweep_tf(problem: PDGProblem, r0, v0, m0: float, tfs: np.ndarray) -> dict:
    betas, statuses = [], []
    for tf in tfs:
        s = problem.solve(r0, v0, m0, float(tf))
        betas.append(s.cost if s.ok else np.inf)
        statuses.append(s.status)
    betas = np.array(betas)
    fuel = m0 * (1 - np.exp(-problem.veh.alpha * betas))
    return {"tf": np.asarray(tfs).tolist(), "beta": betas.tolist(), "fuel": fuel.tolist(),
            "status": statuses, **unimodality(np.asarray(tfs), betas)}


def unimodality(tfs: np.ndarray, betas: np.ndarray, rel_tol: float = 1e-6) -> dict:
    """Remark 10 claims beta(tf) is unimodal. Check on the sampled curve: the feasible tf form one
    interval, and on it beta is non-increasing then non-decreasing. rel_tol absorbs interior-point noise
    (solver tolerance ~1e-8 relative), not real bumps."""
    fin = np.isfinite(betas)
    if not fin.any():
        return {"feasible_interval": False, "unimodal": False, "violations": []}
    idx = np.flatnonzero(fin)
    contiguous = bool(np.all(np.diff(idx) == 1))
    b = betas[idx]
    k = int(np.argmin(b))
    tol = rel_tol * np.abs(b).max()
    viol = [float(tfs[idx[j + 1]]) for j in range(k) if b[j + 1] > b[j] + tol]
    viol += [float(tfs[idx[j + 1]]) for j in range(k, len(b) - 1) if b[j + 1] < b[j] - tol]
    return {"feasible_interval": contiguous, "unimodal": contiguous and not viol, "violations": viol,
            "tf_min_feasible": float(tfs[idx[0]]), "tf_argmin": float(tfs[idx[k]])}
