import numpy as np

from dynamics.model import Vehicle, interval_states
from problem.pdg import Solution


def continuous_trajectory(sol: Solution, veh: Vehicle, per_interval: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Continuous states under the computed piecewise-constant (u, sigma), sampled inside every interval.
    Uses the closed-form solution of (26), (31); tests/test_verify.py checks it against solve_ivp."""
    taus = np.linspace(0.0, sol.dt, per_interval + 1)[:-1]
    ts, ys = [], []
    for k in range(len(sol.S)):
        w = np.concatenate([sol.U[k], [sol.S[k]]])
        ys.append(interval_states(sol.Y[k], w, veh, taus))
        ts.append(k * sol.dt + taus)
    ts.append([sol.tf])
    ys.append(sol.Y[-1:])
    return np.concatenate(ts), np.vstack(ys)


def internode_violation(sol: Solution, veh: Vehicle, glideslope_deg: float | None,
                        vmax: float | None, per_interval: int = 40) -> dict:
    """Largest constraint violation between nodes, in physical units.

    Glide slope: ||(r2, r3)|| - tan(theta) r1 in metres of horizontal distance outside the cone (positive is
    a violation). Speed: ||rdot|| - vmax in m/s. Altitude: -min r1 in metres (subsurface depth).
    The discrete problem cannot see any of these; they are what the vehicle would fly through.
    """
    t, Y = continuous_trajectory(sol, veh, per_interval)
    out = {"subsurface_depth_m": float(max(0.0, -Y[:, 0].min()))}
    if glideslope_deg is not None:
        g = np.linalg.norm(Y[:, 1:3], axis=1) - np.tan(np.deg2rad(glideslope_deg)) * Y[:, 0]
        j = int(np.argmax(g))
        out["glideslope_violation_m"] = float(max(0.0, g[j]))
        out["glideslope_violation_t"] = float(t[j])
    if vmax is not None:
        s = np.linalg.norm(Y[:, 3:6], axis=1) - vmax
        out["speed_violation_mps"] = float(max(0.0, s.max()))
    return out
