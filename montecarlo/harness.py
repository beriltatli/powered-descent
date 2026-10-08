"""One initial state, one mass-error sample, every guidance method. Each worker process compiles its own
problems once (module-level cache), so runs inside a worker reuse the DPP compilation."""
import numpy as np

from baselines.constant_tw import fly_constant_tw
from baselines.open_loop import fly_open_loop
from baselines.polynomial import fly_polynomial
from dynamics.model import load_config, vehicle_from_config
from guidance.config import guidance_kwargs, guidance_problems
from guidance.mpc import fly

_CACHE: dict = {}


def _setup():
    if not _CACHE:
        cfg = load_config()
        P, R = guidance_problems(cfg)
        _CACHE.update(cfg=cfg, veh=vehicle_from_config(cfg), P=P, R=R, kw=guidance_kwargs(cfg))
    return _CACHE


def flight_metrics(f) -> dict:
    return {"status": f.status, "landing_error": f.landing_error, "touchdown_speed": f.touchdown_speed,
            "fuel_used": f.fuel_used, "fuel_exhausted": bool(getattr(f, "fuel_exhausted", False)),
            "glideslope_violation_m": getattr(f, "glideslope_violation_m", np.nan)}


def run_closed(r0, v0, eps: float, plan_mass_margin: float | None = None) -> dict:
    c = _setup()
    kw = dict(c["kw"])
    if plan_mass_margin is not None:
        kw["plan_mass_margin"] = plan_mass_margin
    f = fly(c["P"], c["veh"], r0, v0, c["veh"].m_wet, eps, mode="closed", R=c["R"], **kw)
    out = flight_metrics(f)
    out.update(n_resolves=f.n_resolves, n_feasible=f.n_feasible, n_physical=f.n_physical,
               n_retimed_hard=f.n_retimed_hard, n_recovery=f.n_recovery,
               n_retimed=f.n_retimed, n_held=f.n_held, recursively_feasible=f.recursively_feasible,
               strictly_feasible=f.strictly_feasible, t_end=f.t_end,
               tf_plan0=f.tf_plan0, fuel_remaining=f.m_end - c["veh"].m_dry if np.isfinite(f.m_end) else np.nan,
               max_solve_ms=max(f.solve_ms) if f.solve_ms else np.nan)
    return out


def run_all(args) -> dict:
    r0, v0, eps, tw, tgo_poly = args
    c = _setup()
    veh, kw = c["veh"], c["kw"]
    out = {"eps": eps, "closed": run_closed(r0, v0, eps)}
    ol = fly_open_loop(c["P"], veh, r0, v0, veh.m_wet, eps, **kw)
    out["open_loop"] = flight_metrics(ol)
    p = fly_polynomial(veh, r0, v0, veh.m_wet, eps, tgo_poly, t_hold=kw["t_stop"])
    out["polynomial"] = {**flight_metrics(p), "time_outside_annulus_frac": (p.time_above_rho2 + p.time_below_rho1) / tgo_poly}
    pi = fly_polynomial(veh, r0, v0, veh.m_wet, eps, tgo_poly, t_hold=kw["t_stop"], ideal_actuator=True)
    out["polynomial_ideal"] = {**flight_metrics(pi), "time_outside_annulus_frac": (pi.time_above_rho2 + pi.time_below_rho1) / tgo_poly,
                               "subsurface_depth_m": pi.subsurface_depth_m}
    t = fly_constant_tw(veh, r0, v0, veh.m_wet, eps, tw)
    out["constant_tw"] = {**flight_metrics(t), "horizontal_miss": t.horizontal_miss}
    return out


def run_closed_args(args) -> dict:
    r0, v0, eps, margin = args
    return {"eps": eps, **run_closed(r0, v0, eps, margin)}


def run_envelope_cell(args) -> dict:
    """Feasibility at the true mass (Algorithm 1, no margin), the closed loop with perfect mass
    knowledge (no margin needed), and the closed loop at both mass-error corners with the margin."""
    from tof.search import algorithm1
    r0, v0, corners = args
    c = _setup()
    veh = c["veh"]
    res = algorithm1(c["P"], r0, v0, veh.m_wet, veh.m_fuel)
    out = {"r0": r0, "v0": v0, "class": res.classification, "tf": res.tf, "fuel": res.fuel}
    if res.sol is None:
        return out
    out["no_error"] = run_closed(r0, v0, 0.0, 0.0)
    out["corners"] = {f"{e:+.2f}": run_closed(r0, v0, e) for e in corners}
    return out
