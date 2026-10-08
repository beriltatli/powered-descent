"""Remark 10: is beta(tf) unimodal? Dense sweep on the nominal case plus random instances."""
import numpy as np

from common import FIGURES, save
from figstyle import INK2, MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from montecarlo.sampling import draw_instance
from problem.setup import make_problem
from tof.search import algorithm1, tf_bounds
from tof.sweep import sweep_tf


def main(n_random: int = 40, seed: int = 5):
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    sc = cfg["scenario"]
    P = make_problem(cfg)
    lo, hi = tf_bounds(veh, np.array(sc["v0"]), veh.m_wet, veh.m_fuel)
    tfs = np.linspace(lo, hi, cfg["tof"]["sweep_points"])
    nominal = sweep_tf(P, sc["r0"], sc["v0"], veh.m_wet, tfs)
    alg = algorithm1(P, sc["r0"], sc["v0"], veh.m_wet, veh.m_fuel)
    nominal["algorithm1_tf"], nominal["algorithm1_fuel"] = alg.tf, alg.fuel
    nominal["algorithm1_probes"] = len(alg.probes)
    nominal["sweep_best_tf"] = float(tfs[int(np.argmin(nominal["beta"]))])

    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        r0, v0 = draw_instance(rng)
        a, b = tf_bounds(veh, v0, veh.m_wet, veh.m_fuel)
        sw = sweep_tf(P, r0, v0, veh.m_wet, np.linspace(a, b, 80))
        alg = algorithm1(P, r0, v0, veh.m_wet, veh.m_fuel)
        best = min((x for x in sw["beta"] if np.isfinite(x)), default=np.inf)
        rand.append({"r0": r0, "v0": v0, "unimodal": sw["unimodal"], "feasible_interval": sw["feasible_interval"],
                     "violations": sw["violations"], "any_feasible": bool(np.isfinite(best)),
                     "alg1_beta": alg.beta, "sweep_best_beta": best,
                     "alg1_not_worse_than_grid": bool(alg.beta <= best * (1 + 1e-6)) if np.isfinite(best) else None})
    feas = [r for r in rand if r["any_feasible"]]
    summary = {"N": P.N, "nominal_unimodal": nominal["unimodal"], "nominal_violations": nominal["violations"],
               "nominal_tf_min_feasible": nominal.get("tf_min_feasible"), "nominal_tf_argmin": nominal.get("tf_argmin"),
               "algorithm1_tf": nominal["algorithm1_tf"], "algorithm1_fuel": nominal["algorithm1_fuel"],
               "random_instances": len(rand), "random_feasible": len(feas),
               "random_unimodal": sum(r["unimodal"] for r in feas),
               "random_alg1_not_worse_than_grid": sum(bool(r["alg1_not_worse_than_grid"]) for r in feas)}
    save("tof_sweep", {"summary": summary, "nominal": nominal, "random": rand})

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    f = np.array(nominal["fuel"], float)
    f[~np.isfinite(np.array(nominal["beta"], float))] = np.nan
    ax.plot(tfs, f, color=SERIES[0], marker="o", ms=2.5)
    ax.axhline(veh.m_fuel, color=MUTED, ls="--", lw=1)
    ax.text(tfs[-1], veh.m_fuel + 3, "m_fuel = 400 kg", ha="right", color=INK2, fontsize=8)
    ax.axvline(alg_tf := nominal["algorithm1_tf"], color=SERIES[1], lw=1)
    ax.text(alg_tf + 1, np.nanmax(f) - 10, f"Algorithm 1: {alg_tf:.1f} s", color=INK2, fontsize=8)
    ax.set_xlim(tfs[0], tfs[-1])
    ax.set(xlabel="time of flight tf (s)", ylabel="fuel (kg)",
           title=f"Fuel vs tf, nominal case, N = {P.N} (gaps = infeasible)")
    savefig(fig, FIGURES / "tof_sweep.png")
    print(summary)


if __name__ == "__main__":
    main()
