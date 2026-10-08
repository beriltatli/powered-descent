from dataclasses import dataclass, field

import numpy as np

from dynamics.model import Vehicle
from guidance.plant import PlantState, step_rk4, thrust_command
from problem.pdg import PDGProblem, Solution
from tof.search import algorithm1


@dataclass
class FlightResult:
    mode: str
    eps: float
    status: str                 # landed_at_tf | ground_contact | no_initial_plan
    t_end: float = np.nan
    r_end: np.ndarray | None = None
    v_end: np.ndarray | None = None
    m_end: float = np.nan
    fuel_used: float = np.nan
    fuel_exhausted: bool = False
    tf_plan0: float = np.nan
    n_resolves: int = 0
    n_feasible: int = 0         # nominal fixed-tf re-solves that returned optimal
    n_recovery: int = 0         # nominal infeasible; soft-state recovery problem used at the same tf
    n_retimed: int = 0          # recovery also infeasible; recovered by re-running the tf search on it
    n_held: int = 0             # nothing feasible: kept flying the previous plan
    saturated_time: float = 0.0
    solve_ms: list = field(default_factory=list)
    glideslope_violation_m: float = 0.0
    trace: list = field(default_factory=list)

    @property
    def landing_error(self) -> float:
        return float(np.linalg.norm(self.r_end)) if self.r_end is not None else np.inf

    @property
    def touchdown_speed(self) -> float:
        return float(np.linalg.norm(self.v_end)) if self.v_end is not None else np.inf

    @property
    def recursively_feasible(self) -> bool:
        return self.status != "no_initial_plan" and self.n_feasible == self.n_resolves


def fly(P: PDGProblem, veh: Vehicle, r0, v0, m0: float, eps: float, mode: str = "closed",
        R: PDGProblem | None = None,
        dt_resolve: float = 1.0, t_stop: float = 1.0, sim_dt: float = 0.02, rho2_margin: float = 1.0,
        initial: tuple[float, Solution] | None = None, keep_trace: bool = False) -> FlightResult:
    """Fly the true plant (1)-(2) under the convex guidance.

    Mass error: guidance believes m-hat = (1 + eps) m at all times (the brief's model: a persistent
    multiplicative bias on the mass estimate). It plans in u = Tc/m-hat and commands Tc = u m-hat, so the
    plant's acceleration is (1 + eps) u. Position and velocity are perfectly known.

    mode "closed": re-solve Problem 4 every dt_resolve from the current state with tf fixed at the
    original landing time (shrinking horizon, N fixed so Delta-t shrinks). If that is infeasible, solve
    the recovery problem R (same N; glide slope and speed bound soft, full rho2) at the same tf; then
    re-run the tf search on R (re-time); if all fail, keep flying the old plan. Only the first,
    nominal solve counts toward recursive feasibility.
    Re-solving stops once time-to-go is below t_stop.
    mode "open": plan once at t = 0 and fly it blind.
    rho2_margin < 1 plans with rho2 * margin, leaving thrust headroom (an explicit design choice, off by
    default).
    """
    mhat = lambda m: (1.0 + eps) * m  # noqa: E731
    rho2p = veh.rho2 * rho2_margin
    out = FlightResult(mode=mode, eps=eps, status="no_initial_plan")
    if initial is None:
        res = algorithm1(P, r0, v0, mhat(m0), veh.m_fuel, rho2=rho2p)
        if res.sol is None:
            return out
        tf_plan, plan = res.tf, res.sol
    else:
        tf_plan, plan = initial
    out.tf_plan0 = tf_plan
    plan_t0 = 0.0
    s = PlantState(0.0, np.asarray(r0, float), np.asarray(v0, float), m0)
    next_resolve = dt_resolve
    tan_gs = np.tan(np.deg2rad(P.opts.glideslope_deg)) if P.opts.glideslope_deg is not None else None

    while s.t < tf_plan - 1e-9:
        if mode == "closed" and s.t >= next_resolve - 1e-9 and tf_plan - s.t >= t_stop:
            next_resolve += dt_resolve
            out.n_resolves += 1
            sol = P.solve(s.r, s.v, mhat(s.m), tf_plan - s.t, rho2=rho2p)
            out.solve_ms.append(sol.wall_s * 1e3)
            if sol.ok:
                out.n_feasible += 1
                plan, plan_t0 = sol, s.t
            elif R is not None and (rs := R.solve(s.r, s.v, mhat(s.m), tf_plan - s.t)).ok:
                out.n_recovery += 1
                out.solve_ms.append(rs.wall_s * 1e3)
                plan, plan_t0 = rs, s.t
            else:
                t_rem = tf_plan - s.t
                rt = algorithm1(R or P, s.r, s.v, mhat(s.m), veh.m_fuel, coarse=8,
                                bounds=(max(t_rem * 0.8, 0.5), t_rem + 40.0),
                                **({} if R is not None else {"rho2": rho2p}))
                if rt.sol is not None:
                    out.n_retimed += 1
                    plan, plan_t0, tf_plan = rt.sol, s.t, s.t + rt.tf
                else:
                    out.n_held += 1
        k = min(int((s.t - plan_t0) / plan.dt + 1e-9), P.N - 1)
        t_next = min(s.t + sim_dt, plan_t0 + (k + 1) * plan.dt, tf_plan)
        if mode == "closed" and next_resolve > s.t + 1e-9:
            t_next = min(t_next, next_resolve)
        T, sat = thrust_command(plan.U[k], mhat(s.m), veh)
        dt = t_next - s.t
        if sat:
            out.saturated_time += dt
        new = step_rk4(s, T, dt, veh)
        if keep_trace:
            out.trace.append((s.t, *s.r, *s.v, s.m, *T))
        if tan_gs is not None and new.t < tf_plan - 1.0:
            out.glideslope_violation_m = max(out.glideslope_violation_m,
                                             np.linalg.norm(new.r[1:3]) - tan_gs * new.r[0])
        if new.r[0] < 0.0:
            # Ground contact before tf: interpolate the crossing inside this step.
            a = s.r[0] / (s.r[0] - new.r[0])
            out.status = "ground_contact"
            out.t_end = s.t + a * dt
            out.r_end = s.r + a * (new.r - s.r)
            out.v_end = s.v + a * (new.v - s.v)
            out.m_end = s.m + a * (new.m - s.m)
            break
        s = new
    else:
        out.status = "landed_at_tf"
        out.t_end, out.r_end, out.v_end, out.m_end = s.t, s.r, s.v, s.m
    out.fuel_used = m0 - out.m_end
    out.fuel_exhausted = out.m_end <= veh.m_dry + 1e-6
    out.glideslope_violation_m = float(max(0.0, out.glideslope_violation_m))
    return out
