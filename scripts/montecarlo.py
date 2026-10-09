"""1000 closed-loop runs with eps ~ U[-5 %, +5 %], every method on the same eps samples, plus the
closed loop on the Section V state."""
import os
from multiprocessing import Pool

import numpy as np

from common import FIGURES, machine, save
from figstyle import INK2, MUTED, SERIES, plt, savefig

from baselines.constant_tw import calibrate_tw
from dynamics.model import load_config, vehicle_from_config
from metrics.dispersion import summarize
from montecarlo.harness import run_all, run_closed_args
from problem.setup import make_problem
from tof.search import algorithm1

WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))


def main():
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    mc, sc = cfg["montecarlo"], cfg["success"]
    rng = np.random.default_rng(mc["seed"])
    eps = rng.uniform(-mc["mass_error"], mc["mass_error"], mc["runs"])
    r0, v0 = mc["r0"], mc["v0"]
    tw = calibrate_tw(veh, r0, v0, veh.m_wet)
    tgo = float(algorithm1(make_problem(cfg), r0, v0, veh.m_wet, veh.m_fuel).tf)
    with Pool(WORKERS) as pool:
        rows = pool.map(run_all, [(r0, v0, float(e), tw, tgo) for e in eps], chunksize=8)
        paper = pool.map(run_closed_args, [(cfg["scenario"]["r0"], cfg["scenario"]["v0"], float(e), None) for e in eps],
                         chunksize=8)
    methods = ("closed", "open_loop", "polynomial", "polynomial_ideal", "constant_tw")
    summary = {m: summarize([r[m] for r in rows], sc["landing_radius"], sc["touchdown_speed"]) for m in methods}
    summary["section_v_closed"] = summarize(paper, sc["landing_radius"], sc["touchdown_speed"])
    out = {"machine": machine(), "workers": WORKERS, "N": cfg["discretization"]["N"], "r0": r0, "v0": v0,
           "mass_error": mc["mass_error"], "plan_mass_margin": cfg["guidance"]["plan_mass_margin"],
           "poly_tgo": tgo, "tw": tw, "summary": summary, "runs": rows, "section_v_runs": paper}
    save("montecarlo", out)

    cl = [r["closed"] for r in rows]
    e = np.array([r["eps"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(100 * e, [c["landing_error"] for c in cl], s=8, color=SERIES[0], label="landing error (m)")
    ax[0].scatter(100 * e, [c["touchdown_speed"] for c in cl], s=8, color=SERIES[1], label="touchdown speed (m/s)")
    ax[0].axhline(sc["touchdown_speed"], color=MUTED, ls="--", lw=1)
    ax[0].set(xlabel="mass error ε (%)", ylabel="m  or  m/s", yscale="log",
              title=f"Closed loop, {len(cl)} runs, N = {out['N']}")
    ax[0].legend(loc="upper left")
    ax[1].scatter(100 * e, [c["fuel_used"] for c in cl], s=8, color=SERIES[0], label="closed loop")
    ax[1].scatter(100 * e, [r["open_loop"]["fuel_used"] for r in rows], s=8, color=SERIES[2], label="open loop")
    ax[1].axhline(veh.m_fuel, color=MUTED, ls="--", lw=1)
    ax[1].text(5, veh.m_fuel - 8, "fuel on board", color=INK2, fontsize=8, ha="right")
    ax[1].set(xlabel="mass error ε (%)", ylabel="fuel used (kg)", title="Fuel vs mass error")
    ax[1].legend(loc="upper left")
    savefig(fig, FIGURES / "montecarlo.png")
    for m, s in summary.items():
        print(m, {k: v for k, v in s.items() if k != "statuses"}, s["statuses"])


if __name__ == "__main__":
    main()
