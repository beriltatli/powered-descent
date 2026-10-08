"""The guidance design choices, each switched off, on the first 300 Monte Carlo mass-error samples:
no planning mass margin, no rho2 tightening, and re-solving stopped 1 s (not 0.25 s) before touchdown.
Evidence for why each exists."""
import os
from multiprocessing import Pool

import numpy as np

from common import save

from dynamics.model import load_config, vehicle_from_config
from guidance.config import guidance_kwargs, guidance_problems
from guidance.mpc import fly
from metrics.dispersion import summarize

WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))
_C: dict = {}


def job(args):
    variant, eps = args
    if not _C:
        cfg = load_config()
        P, R = guidance_problems(cfg)
        _C.update(cfg=cfg, veh=vehicle_from_config(cfg), P=P, R=R)
    cfg, veh = _C["cfg"], _C["veh"]
    kw = guidance_kwargs(cfg)
    if variant == "no_mass_margin":
        kw["plan_mass_margin"] = 0.0
    elif variant == "no_rho2_tightening":
        kw["rho2_margin"] = 1.0
    elif variant == "t_stop_1s":
        kw["t_stop"] = 1.0
    f = fly(_C["P"], veh, cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"], veh.m_wet, eps, R=_C["R"], **kw)
    return {"variant": variant, "eps": eps, "status": f.status, "landing_error": f.landing_error,
            "touchdown_speed": f.touchdown_speed, "fuel_used": f.fuel_used, "fuel_exhausted": f.fuel_exhausted,
            "n_resolves": f.n_resolves, "n_feasible": f.n_feasible, "n_physical": f.n_physical,
            "n_retimed_hard": f.n_retimed_hard, "n_recovery": f.n_recovery, "n_retimed": f.n_retimed,
            "n_held": f.n_held, "recursively_feasible": f.recursively_feasible,
            "strictly_feasible": f.strictly_feasible, "t_end": f.t_end,
            "fuel_remaining": f.m_end - veh.m_dry}


def main(n: int = 300, n_full: int | None = None):
    """n samples per variant; n_full (all Monte Carlo samples) for no_rho2_tightening, whose effect on
    rare fuel-exhaustion failures needs the full sample to be distinguishable."""
    cfg = load_config()
    mc = cfg["montecarlo"]
    eps_all = np.random.default_rng(mc["seed"]).uniform(-mc["mass_error"], mc["mass_error"], mc["runs"])
    n_full = n_full or mc["runs"]
    variants = {"no_mass_margin": n, "no_rho2_tightening": n_full, "t_stop_1s": n}
    with Pool(WORKERS) as pool:
        rows = pool.map(job, [(v, float(e)) for v, k in variants.items() for e in eps_all[:k]], chunksize=4)
    sc = cfg["success"]
    out = {"runs_per_variant": variants, "summary": {}, "runs": rows}
    for v in variants:
        rv = [r for r in rows if r["variant"] == v]
        s = summarize(rv, sc["landing_radius"], sc["touchdown_speed"])
        neg = [r for r in rv if r["eps"] < 0]
        s["success_eps_negative"] = float(np.mean([r["landing_error"] < sc["landing_radius"]
                                                   and r["touchdown_speed"] < sc["touchdown_speed"] for r in neg]))
        s["runs_eps_negative"] = len(neg)
        out["summary"][v] = s
        print(v, {k: s[k] for k in ("success", "below_speed", "within_radius", "fuel_exhausted",
                                    "success_eps_negative", "recursive_feasibility_resolves")})
    save("ablation", out)


if __name__ == "__main__":
    main()
