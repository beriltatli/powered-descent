"""Fuel vs node count N (Algorithm 1 at each N), with a short timing sample per N to choose N."""
import numpy as np

from common import FIGURES, machine, save
from figstyle import INK2, MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from metrics.timing import percentiles
from problem.setup import make_problem
from tof.search import algorithm1


def main(timing_samples: int = 60):
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    cases = {"section_v": (cfg["scenario"]["r0"], cfg["scenario"]["v0"]),
             "montecarlo": (cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"])}
    rows = {k: [] for k in cases}
    rng = np.random.default_rng(0)
    for N in cfg["discretization"]["N_sweep"]:
        P = make_problem(cfg, N=N)
        for name, (r0, v0) in cases.items():
            res = algorithm1(P, r0, v0, veh.m_wet, veh.m_fuel, tol=0.01)
            row = {"N": N, "tf": res.tf, "fuel": res.fuel, "class": res.classification}
            if name == "montecarlo":
                ms = []
                for _ in range(timing_samples):
                    tf = res.tf * rng.uniform(0.97, 1.1)
                    ms.append(P.solve(r0, v0, veh.m_wet, tf).wall_s * 1e3)
                row["solve_ms"] = percentiles(ms)
            rows[name].append(row)
            print(name, row["N"], round(row["tf"], 2), round(row["fuel"], 3), row["class"],
                  round(row.get("solve_ms", {}).get("p99", np.nan), 1))
    chosen = cfg["discretization"]["N"]
    out = {"machine": machine(), "chosen_N": chosen, "rows": rows}
    for name in cases:
        ref = rows[name][-1]
        at = next(r for r in rows[name] if r["N"] == chosen)
        out[f"{name}_reference_N"] = ref["N"]
        out[f"{name}_fuel_gap_kg_at_chosen_N"] = at["fuel"] - ref["fuel"]
        out[f"{name}_fuel_gap_pct_at_chosen_N"] = 100 * (at["fuel"] - ref["fuel"]) / ref["fuel"]
    save("convergence", out)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, name in enumerate(cases):
        Ns = [r["N"] for r in rows[name]]
        ax[0].plot(Ns, [r["fuel"] for r in rows[name]], "o-", color=SERIES[i], label=name.replace("_", " "))
    ax[0].axhline(veh.m_fuel, color=MUTED, ls="--", lw=1)
    ax[0].text(320, veh.m_fuel + 2, "fuel on board", color=INK2, fontsize=8, ha="right")
    ax[0].axvline(chosen, color=MUTED, lw=1)
    ax[0].set(xscale="log", xlabel="node count N", ylabel="optimal fuel (kg)", title="Fuel vs N (Algorithm 1 at each N)")
    ax[0].legend()
    mc = rows["montecarlo"]
    ax[1].plot([r["N"] for r in mc], [r["solve_ms"]["p99"] for r in mc], "o-", color=SERIES[0], label="p99")
    ax[1].plot([r["N"] for r in mc], [r["solve_ms"]["p50"] for r in mc], "o-", color=SERIES[2], label="p50")
    ax[1].axhline(cfg["success"]["p99_solve_ms"], color=MUTED, ls="--", lw=1)
    ax[1].axvline(chosen, color=MUTED, lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel="node count N", ylabel="wall time per solve (ms)",
              title=f"Solve time vs N ({timing_samples} solves each, Clarabel)")
    ax[1].legend()
    savefig(fig, FIGURES / "convergence.png")


if __name__ == "__main__":
    main()
