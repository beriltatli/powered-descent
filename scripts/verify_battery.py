"""Lemma 1 / Lemma 2 battery on random instances without the pointing constraint.

Draws initial states until `target` instances have a finite beta (Problem 4 feasible for some tf). Every
draw is recorded with its Algorithm 1 class; none is dropped. Insufficient-fuel instances are Problem 4
feasible (fuel is checked after the solve, Remark 8) and are included in the slack statistics.
"""
import numpy as np

from common import FIGURES, save
from figstyle import INK2, MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from problem.setup import make_problem
from tof.search import algorithm1
from verify.bangbang import classify_thrust
from verify.exact32 import exact32_margins
from verify.internode import internode_violation
from verify.slack import mass_tightness, slack_gaps
from montecarlo.sampling import draw_instance



def characterize_loose(P, veh, draw: dict, tol: float) -> dict:
    """Re-solve a loose instance at Algorithm 1's tf with Clarabel tolerances tightened from 1e-8 to
    1e-10: a gap that collapses is an interior-point residual, one that survives is a real violation."""
    res = algorithm1(P, draw["r0"], draw["v0"], veh.m_wet, veh.m_fuel)
    P.set_parameters(draw["r0"], draw["v0"], veh.m_wet, res.tf)
    P.prob.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    gap = P.S.value - np.linalg.norm(P.U.value, axis=1)
    return {"tf": res.tf, "max_gap_default_tol": float(slack_gaps(res.sol).max()),
            "max_gap_tight_tol": float(gap.max()), "status_tight": P.prob.status,
            "tight_after_resolve": bool(gap.max() <= tol)}


def main(target: int = 500, seed: int = 1):
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    P = make_problem(cfg)
    rng = np.random.default_rng(seed)
    tol = cfg["solver"]["slack_tol"]
    draws, gaps, gam, rows = [], [], [], []
    while len(rows) < target:
        r0, v0 = draw_instance(rng)
        res = algorithm1(P, r0, v0, veh.m_wet, veh.m_fuel,
                         coarse=cfg["tof"]["coarse_points"], tol=cfg["tof"]["golden_tol"])
        draws.append({"r0": r0, "v0": v0, "class": res.classification, "tf": res.tf, "fuel": res.fuel})
        if res.sol is None:
            continue
        s = res.sol
        g = slack_gaps(s)
        bb = classify_thrust(s, veh, P.opts, veh.m_wet)
        e32 = exact32_margins(s, veh)
        rows.append({"i": len(draws) - 1, "class": res.classification, "max_gap": float(g.max()),
                     "loose_nodes": np.flatnonzero(g > tol).tolist(),
                     "exact32_node_ok": e32["node_ok"], **{f"e32_{k}": v for k, v in e32.items() if k != "node_ok"},
                     "mass13_diff_kg": mass_tightness(s, veh.alpha)["difference_kg"],
                     "bb_counts": bb["counts"], "bb_unexplained": [k for k, r in bb["intermediate"].items() if r == "unexplained"],
                     **internode_violation(s, veh, P.opts.glideslope_deg, P.opts.vmax)})
        gaps.append(g)
        gam.extend(bb["gamma_over_rho2"])
    gaps = np.concatenate(gaps)
    classes = {c: sum(d["class"] == c for d in draws) for c in sorted({d["class"] for d in draws})}
    loose = [r for r in rows if r["loose_nodes"]]
    summary = {
        "N": P.N, "draws": len(draws), "feasible_instances": len(rows), "classes": classes,
        "slack_tol": tol,
        "slack_gap_quantiles": {q: float(np.quantile(gaps, q)) for q in (0, 0.5, 0.9, 0.99, 0.999, 1.0)},
        "instances_with_loose_slack": len(loose),
        "loose_detail": [{"i": r["i"], "nodes": r["loose_nodes"], "max_gap": r["max_gap"]} for r in loose],
        "exact32_all_nodes_ok": all(r["exact32_node_ok"] for r in rows),
        "exact32_worst_lower_node_N": min(r["e32_min_lower_margin_node_N"] for r in rows),
        "exact32_worst_upper_node_N": min(r["e32_min_upper_margin_node_N"] for r in rows),
        "exact32_worst_lower_interval_end_N": min(r["e32_min_lower_margin_interval_end_N"] for r in rows),
        "mass13_max_abs_diff_kg": max(abs(r["mass13_diff_kg"]) for r in rows),
        "bangbang_counts": {k: sum(r["bb_counts"][k] for r in rows) for k in rows[0]["bb_counts"]},
        "bangbang_unexplained_instances": [r["i"] for r in rows if r["bb_unexplained"]],
        "internode_glideslope_max_m": max(r.get("glideslope_violation_m", 0) for r in rows),
        "internode_glideslope_p95_m": float(np.quantile([r.get("glideslope_violation_m", 0) for r in rows], 0.95)),
        "internode_speed_max_mps": max(r.get("speed_violation_mps", 0) for r in rows),
        "internode_subsurface_max_m": max(r["subsurface_depth_m"] for r in rows),
        "gamma_over_rho2_hist": np.histogram(gam, bins=40, range=(0.3, 1.0))[0].tolist(),
    }
    for d in summary["loose_detail"]:
        d["resolve_tight_tolerance"] = characterize_loose(P, veh, draws[d["i"]], tol)
    save("lemma1_battery", {"summary": summary, "instances": rows, "draws": draws})

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ag = np.maximum(np.abs(gaps), 1e-14)
    ax[0].hist(np.log10(ag), bins=60, color=SERIES[0])
    ax[0].axvline(np.log10(tol), color=MUTED, ls="--", lw=1)
    ax[0].text(np.log10(tol) + 0.1, ax[0].get_ylim()[1] * 0.9, "loose threshold", color=INK2, fontsize=8)
    ax[0].set(xlabel="log₁₀ |σ_k − ‖u_k‖|  (m/s²)", ylabel="nodes",
              title=f"Slack gap, {len(rows)} instances × {P.N} nodes")
    ax[1].hist(gam, bins=70, range=(0.3, 1.02), color=SERIES[0])
    ax[1].axvline(veh.rho1 / veh.rho2, color=MUTED, ls="--", lw=1)
    ax[1].axvline(1.0, color=MUTED, ls="--", lw=1)
    ax[1].set(xlabel="Γ_k / ρ2", ylabel="nodes", yscale="log", title="Thrust level Γ = σm (bang-bang)")
    savefig(fig, FIGURES / "lemma1_battery.png")
    print(summary)


if __name__ == "__main__":
    main()
