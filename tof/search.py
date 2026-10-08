from dataclasses import dataclass, field

import numpy as np

from problem.pdg import PDGProblem, Solution

INVPHI = (np.sqrt(5.0) - 1.0) / 2.0


def tf_bounds(veh, v0, m0: float, m_fuel: float) -> tuple[float, float]:
    # (55): t_min = (m_wet - m_fuel)||rdot0|| / rho2, t_max = m_fuel / (alpha rho1).
    # On a closed-loop re-solve m0 is the current mass and m_fuel what is left of it.
    tmin = (m0 - m_fuel) * np.linalg.norm(v0) / veh.rho2
    tmax = m_fuel / (veh.alpha * veh.rho1)
    return max(tmin, 1e-3), tmax


@dataclass
class TofResult:
    tf: float
    sol: Solution | None
    beta: float
    classification: str     # valid | insufficient_fuel | insufficient_thrust | solver_failure | initial_state_infeasible
    fuel: float
    probes: list = field(default_factory=list)     # (tf, beta or inf, status)


def initial_state_violations(problem: PDGProblem, r0, v0) -> list[str]:
    o, out = problem.opts, []
    if o.glideslope_deg is not None and np.linalg.norm(r0[1:3]) > np.tan(np.deg2rad(o.glideslope_deg)) * r0[0]:
        out.append("glideslope")
    if o.vmax is not None and np.linalg.norm(v0) > o.vmax:
        out.append("vmax")
    if r0[0] <= 0:
        out.append("assumption (6) r0_1 > 0")
    return out


def algorithm1(problem: PDGProblem, r0, v0, m0: float, m_fuel: float, coarse: int = 14,
               tol: float = 0.05, bounds: tuple[float, float] | None = None, **solve_kw) -> TofResult:
    """Algorithm 1 (p.1361) as a line search over tf with N fixed and Delta-t = tf/N.

    The paper fixes Delta-t and searches the integer N; both are a 1-D search over tf for the minimum of
    beta(tf), with beta = inf where Problem 4 is infeasible (58). A coarse grid brackets the minimum,
    then golden-section refines it; golden-section is only valid if beta is unimodal, which is
    Remark 10's observation and is checked separately by tof/sweep.py.
    """
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    bad = initial_state_violations(problem, r0, v0)
    if bad:
        return TofResult(np.nan, None, np.inf, "initial_state_infeasible", np.inf,
                         [("violates", ",".join(bad))])
    tmin, tmax = bounds or tf_bounds(problem.veh, v0, m0, m_fuel)
    probes, cache = [], {}

    def beta(tf):
        if tf in cache:
            return cache[tf][0]
        s = problem.solve(r0, v0, m0, tf, **solve_kw)
        b = s.cost if s.ok else np.inf
        cache[tf] = (b, s)
        probes.append((float(tf), float(b), s.status))
        return b

    grid = np.linspace(tmin, tmax, coarse)
    vals = np.array([beta(t) for t in grid])
    if not np.isfinite(vals).any():
        # Densify once near t_min before declaring infeasibility: a narrow feasible window can fall
        # between coarse points.
        fine = np.linspace(tmin, tmin + (tmax - tmin) / 3, 3 * coarse)
        vals_f = np.array([beta(t) for t in fine])
        if np.isfinite(vals_f).any():
            grid, vals = np.concatenate([grid, fine]), np.concatenate([vals, vals_f])
            order = np.argsort(grid)
            grid, vals = grid[order], vals[order]
    if not np.isfinite(vals).any():
        statuses = {s for _, _, s in probes}
        cls = "insufficient_thrust" if statuses <= {"infeasible"} else "solver_failure"
        return TofResult(np.nan, None, np.inf, cls, np.inf, probes)

    i = int(np.argmin(vals))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, len(grid) - 1)]
    c, d = b - INVPHI * (b - a), a + INVPHI * (b - a)
    fc, fd = beta(c), beta(d)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - INVPHI * (b - a)
            fc = beta(c)
        else:
            a, c, fc = c, d, fd
            d = a + INVPHI * (b - a)
            fd = beta(d)
    tf_best = min(cache, key=lambda t: cache[t][0])
    b_best, sol = cache[tf_best]
    fuel = m0 * (1.0 - np.exp(-problem.veh.alpha * b_best))
    # Algorithm 1 cases 1 and 2a: fuel is checked after the solve (Remark 8), not constrained inside it.
    cls = "valid" if fuel <= m_fuel else "insufficient_fuel"
    return TofResult(tf_best, sol, b_best, cls, fuel, probes)
