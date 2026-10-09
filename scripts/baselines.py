"""Baselines vs the convex closed loop at eps in {-5 %, 0, +5 %}, on the Monte Carlo state and the
Section V state."""
import numpy as np

from common import save

from baselines.constant_tw import calibrate_tw
from dynamics.model import load_config, vehicle_from_config
from montecarlo.harness import run_all, run_closed
from problem.setup import make_problem
from tof.search import algorithm1


def poly_tgo(cfg, veh, r0, v0) -> float:
    # The polynomial law is given the convex optimum's time of flight (at eps = 0): the most favourable
    # choice it could be handed without an optimizer of its own.
    P = make_problem(cfg)
    return float(algorithm1(P, r0, v0, veh.m_wet, veh.m_fuel).tf)


def main():
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    states = {"montecarlo": (cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"]),
              "section_v": (cfg["scenario"]["r0"], cfg["scenario"]["v0"])}
    out = {}
    for name, (r0, v0) in states.items():
        tw = calibrate_tw(veh, r0, v0, veh.m_wet)
        tgo = poly_tgo(cfg, veh, r0, v0)
        rows = {f"{e:+.2f}": run_all((r0, v0, e, tw, tgo)) for e in (-0.05, 0.0, 0.05)}
        # The same closed loop without the planning mass margin: why the margin exists.
        no_margin = {f"{e:+.2f}": run_closed(r0, v0, e, 0.0) for e in (-0.05, 0.0, 0.05)}
        out[name] = {"r0": r0, "v0": v0, "tw": tw, "poly_tgo": tgo, "by_eps": rows, "closed_no_margin": no_margin}
        for e, r in rows.items():
            for m, v in r.items():
                if m == "eps":
                    continue
                print(name, e, m, v["status"], "err %.2f spd %.2f fuel %.1f" % (v["landing_error"], v["touchdown_speed"], v["fuel_used"]))
    save("baselines", out)


if __name__ == "__main__":
    main()
