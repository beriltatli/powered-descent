"""Render REPORT.md from scripts/report_template.md and results/*.json.

Every number in the report is a {placeholder} filled here from a results file; tests/test_report.py
re-renders and fails if the committed REPORT.md differs, so the report cannot drift from results/.
"""
import json
import string

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text())


def f(x, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.{nd}f}"


def pct(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f} %"


def _load_config() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def mc_narrative(mc: dict, crit: dict) -> str:
    """The failed closed-loop runs, listed individually with their mass error and cause."""
    bad = [r for r in mc["runs"] if not (r["closed"]["landing_error"] < crit["landing_radius"]
                                         and r["closed"]["touchdown_speed"] < crit["touchdown_speed"])]
    if not bad:
        return "No closed-loop run failed the landing criteria."
    parts = []
    for r in sorted(bad, key=lambda r: r["eps"]):
        c = r["closed"]
        cause = "fuel exhausted" if c["fuel_exhausted"] else c["status"].replace("_", " ")
        parts.append(f"ε = {100 * r['eps']:+.2f} %: {f(c['landing_error'], 2)} m, {f(c['touchdown_speed'], 2)} m/s, "
                     f"{cause}, flight {f(c['t_end'])} s, {c['n_retimed_hard']} hard re-times")
    eps = [r["eps"] for r in bad]
    return (f"{len(bad)} of {len(mc['runs'])} runs failed a landing criterion, all with ε between "
            f"{100 * min(eps):+.2f} % and {100 * max(eps):+.2f} %: " + "; ".join(parts) + ".")


def tightening_sentence(mc: dict, ab: dict, crit: dict) -> str:
    """Matched-sample comparison of the closed loop with and without the 0.97 rho2 tightening."""
    nr = [r for r in ab["runs"] if r["variant"] == "no_rho2_tightening"]
    main = [r["closed"] for r in mc["runs"][:len(nr)]]

    def fails(rows):
        return sum(not (r["landing_error"] < crit["landing_radius"] and r["touchdown_speed"] < crit["touchdown_speed"])
                   for r in rows)
    f_with, f_without = fails(main), fails(nr)
    d_fuel = np.mean([r["fuel_used"] for r in main]) - np.mean([r["fuel_used"] for r in nr])
    verdict = ("so on this evidence the tightening buys nothing and costs fuel" if f_with >= f_without
               else f"so it removes {f_without - f_with} of the {f_without} failures, at that fuel cost")
    return (f"On the same {len(nr)} mass-error samples: {f_with} failures with the tightening, {f_without} without it; "
            f"the tightening costs {d_fuel:.1f} kg of fuel on average, {verdict}.")


def env_sentence(env: dict) -> str:
    n0, n1 = env["landable_no_error"], env["landable_with_error"]
    return (f"A ±5 % mass error shrinks the landable region from {n0} to {n1} grid cells, "
            f"a {100 * (1 - n1 / n0):.1f} % loss.")


def values() -> dict:
    v = {}
    so, bat, nc = _load("solve_once"), _load("lemma1_battery")["summary"], _load("nonconvex")["summary"]
    tof, conv, tim = _load("tof_sweep")["summary"], _load("convergence"), _load("timing")
    pt, bl = _load("pointing")["summary"], _load("baselines")
    mc, env = _load("montecarlo"), _load("envelope")["summary"]
    ms = mc["summary"]

    v.update(N=so["N"], nom_tf=f(so["tf"]), nom_fuel=f(so["fuel_kg"]), nom_class=so["classification"].replace("_", " "),
             nom_slack_max=f"{so['slack']['max_gap']:.1e}", nom_gs_internode=f(so["internode"]["glideslope_violation_m"], 2),
             nom_gs_t=f(so["internode"]["glideslope_violation_t"]),
             nom_bb_rho1=so["bangbang"]["n_rho1"], nom_bb_rho2=so["bangbang"]["n_rho2"],
             nom_e32_end=f(so["exact32"]["min_lower_margin_interval_end_N"]),
             rho1=f(so["rho1"]), rho2=f(so["rho2"]), alpha=f"{so['alpha']:.4e}")

    v.update(bat_n=bat["feasible_instances"], bat_draws=bat["draws"], bat_valid=bat["classes"].get("valid", 0),
             bat_fuel=bat["classes"].get("insufficient_fuel", 0), bat_thrust=bat["classes"].get("insufficient_thrust", 0),
             bat_loose=bat["instances_with_loose_slack"],
             bat_loose_gap=f"{bat['loose_detail'][0]['max_gap']:.1e}" if bat["loose_detail"] else "n/a",
             bat_loose_i=bat["loose_detail"][0]["i"] if bat["loose_detail"] else "n/a",
             bat_loose_tight=(f"{bat['loose_detail'][0]['resolve_tight_tolerance']['max_gap_tight_tol']:.1e}"
                              if bat["loose_detail"] else "n/a"),
             bat_q50=f"{bat['slack_gap_quantiles']['0.5']:.1e}", bat_q99=f"{bat['slack_gap_quantiles']['0.99']:.1e}",
             bat_q999=f"{bat['slack_gap_quantiles']['0.999']:.1e}", bat_qmax=f"{bat['slack_gap_quantiles']['1.0']:.1e}",
             bat_e32=("all nodes" if bat["exact32_all_nodes_ok"] else "NOT all nodes"),
             bat_e32_up=f(bat["exact32_worst_upper_node_N"], 4), bat_e32_end_abs=f(abs(bat["exact32_worst_lower_interval_end_N"])),
             bat_mass13=f"{bat['mass13_max_abs_diff_kg']:.1e}",
             bat_gs_max=f(bat["internode_glideslope_max_m"], 2), bat_gs_p95=f(bat["internode_glideslope_p95_m"], 2),
             bat_speed=f(bat["internode_speed_max_mps"], 3), bat_sub=f"{bat['internode_subsurface_max_m']:.1e}",
             **{f"bb_{k}": c for k, c in bat["bangbang_counts"].items()})

    t, e = nc["taylor"], nc["exact"]
    v.update(nc_n=nc["instances"], nc_N=nc["N"], nc_starts=nc["starts"], nc_t_ok=t["instances_convex_at_least_as_good"],
             nc_t_feas=t["starts_feasible"], nc_t_total=t["starts_total"], nc_t_gap=f"{t['median_rel_gap_best_start']:.1e}",
             nc_t_viol=f"{t['max_convex_point_violation']:.1e}",
             nc_e_ok=e["instances_convex_at_least_as_good"], nc_e_none=e["instances_no_feasible_start"],
             nc_e_beaten=nc["instances"] - e["instances_convex_at_least_as_good"],
             nc_e_gap=f(100 * -e["min_rel_gap_best_start"], 3), nc_e_warm=f(100 * e["from_convex_rel_improvement_max"], 3))

    v.update(tof_uni=("unimodal" if tof["nominal_unimodal"] else "NOT unimodal"), tof_rand=tof["random_feasible"],
             tof_rand_uni=tof["random_unimodal"], tof_alg_ok=tof["random_alg1_not_worse_than_grid"],
             tof_min=f(tof["nominal_tf_min_feasible"]), tof_arg=f(tof["nominal_tf_argmin"]))

    rows = {r["N"]: r for r in conv["rows"]["section_v"]}
    mrows = {r["N"]: r for r in conv["rows"]["montecarlo"]}
    v.update(cv_ref=conv["section_v_reference_N"], cv_sv_gap=f(conv["section_v_fuel_gap_kg_at_chosen_N"], 2),
             cv_mc_gap=f(conv["montecarlo_fuel_gap_kg_at_chosen_N"], 2), cv_mc_gap_pct=f(conv["montecarlo_fuel_gap_pct_at_chosen_N"], 2),
             cv_first_valid=min(n for n, r in rows.items() if r["class"] == "valid"),
             cv_first_valid_p99=f(mrows[min(n for n, r in rows.items() if r["class"] == "valid")]["solve_ms"]["p99"]),
             cv_sv_fuel_ref=f(rows[conv["section_v_reference_N"]]["fuel"], 2), cv_sv_fuel_N=f(rows[so["N"]]["fuel"], 2),
             cv_mc_fuel_N=f(mrows[so["N"]]["fuel"], 2), cv_mc_fuel_ref=f(mrows[conv["montecarlo_reference_N"]]["fuel"], 2),
             cv_p99_80=f(mrows[80]["solve_ms"]["p99"]), cv_p99_120=f(mrows[120]["solve_ms"]["p99"]),
             cv_p99_160=f(mrows[160]["solve_ms"]["p99"]))
    v["cv_table"] = "\n".join(
        f"| {n} | {f(rows[n]['fuel'], 2)} | {rows[n]['class'].replace('_', ' ')} | {f(mrows[n]['fuel'], 2)} | {f(mrows[n]['solve_ms']['p50'])} | {f(mrows[n]['solve_ms']['p99'])} |"
        for n in sorted(rows))

    w, so_ = tim["per_solve_wall_ms_at_tf_opt"], tim["per_solve_solver_ms_at_tf_opt"]
    hw = tim["machine"]
    v.update(hw=f"{hw['cpu']}, {hw['memory_gb']} GB, {hw['os'].split('-')[0]}", py=hw["python"], cvxpy=hw["cvxpy"],
             clarabel=hw["clarabel"], tm_n=w["n"], tm_p50=f(w["p50"]), tm_p95=f(w["p95"]), tm_p99=f(w["p99"]), tm_max=f(w["max"]),
             ts_p50=f(so_["p50"]), ts_p99=f(so_["p99"]), ts_max=f(so_["max"]),
             ti_p50=f(tim["per_solve_wall_ms_infeasible_probes"]["p50"]), ti_p99=f(tim["per_solve_wall_ms_infeasible_probes"]["p99"]),
             ti_n=tim["per_solve_wall_ms_infeasible_probes"]["n"],
             ta_p50=f(tim["algorithm1_total_ms"]["p50"], 0), ta_p99=f(tim["algorithm1_total_ms"]["p99"], 0),
             compile_ms=f(tim["compile_ms_median"]), build_ms=f(tim["build_ms_median"]),
             dpp_p50=f(tim["dpp_vs_recompile"]["dpp"]["p50"]), rc_p50=f(tim["dpp_vs_recompile"]["recompile_every_call"]["p50"]),
             dpp_p99=f(tim["dpp_vs_recompile"]["dpp"]["p99"]), rc_p99=f(tim["dpp_vs_recompile"]["recompile_every_call"]["p99"]))

    v["pt_n"] = pt["90deg"]["off"]["n"]
    ex = pt["example"]
    v.update(pt_off_tf=f(ex["off"]["tf"]), pt_off_fuel=f(ex["off"]["fuel"]), pt_ex_tf=f(ex["exact61"]["tf"]),
             pt_ex_fuel=f(ex["exact61"]["fuel"]), pt_ex_loose=len(ex["exact61"]["loose_nodes"]),
             pt_off_loose=len(ex["off"]["loose_nodes"]), pt_off_min=f(ex["off"]["min_thrust_over_rho1"], 2))
    for k in ("90deg", "45deg"):
        for m in ("off", "exact61", "heuristic62"):
            s = pt[k][m]
            v[f"pt_{k}_{m}_mat"] = s["material_loose"]
            v[f"pt_{k}_{m}_thr"] = s["thrust_below_0p99_rho1"]
            v[f"pt_{k}_{m}_raw"] = s["loose_instances"]
            v[f"pt_{k}_{m}_viol"] = s["pointing_violated_instances"]
            v[f"pt_{k}_{m}_n"] = s["n"]
        v[f"pt_{k}_pen"] = f(pt[k]["heuristic_fuel_penalty_kg_median"], 2)
        v[f"pt_{k}_maxviol"] = f(pt[k]["heuristic_max_violation_deg"])

    v.update(mc_runs=ms["closed"]["runs"], mc_r0=str(mc["r0"]), mc_v0=str(mc["v0"]),
             mc_margin=f(100 * mc["plan_mass_margin"], 2), mc_tgo=f(mc["poly_tgo"]), mc_tw=f(mc["tw"], 3))
    for m in ("closed", "open_loop", "polynomial", "polynomial_ideal", "constant_tw", "section_v_closed"):
        s = ms[m]
        for key in ("within_radius", "below_speed", "success"):
            v[f"{m}_{key}"] = pct(s[key])
        for key in ("landing_error_p50", "landing_error_p95", "landing_error_max", "touchdown_speed_p50",
                    "touchdown_speed_p95", "touchdown_speed_max", "fuel_mean", "fuel_p95"):
            v[f"{m}_{key}"] = f(s[key], 2)
        v[f"{m}_exhausted"] = s["fuel_exhausted"]
    c = ms["closed"]
    crit = _load_config()["success"]
    v.update(strict_res=pct(c["strict_feasibility_resolves"], 2), strict_runs=pct(c["strict_feasibility_runs"]),
             rf_retimed_hard=c["retimed_hard"], rf_physical=c["physical_bounds_used"],
             crit_radius="met" if c["within_radius"] >= 0.95 else "MISSED",
             crit_speed="met" if c["below_speed"] >= 0.95 else "MISSED",
             crit_rf="met" if c["recursive_feasibility_resolves"] > crit["recursive_feasibility"] else "MISSED")
    v["mc_narrative"] = mc_narrative(mc, crit)
    v.update(rf_runs=pct(c["recursive_feasibility_runs"]), rf_res=pct(c["recursive_feasibility_resolves"], 2),
             rf_total=c["resolves_total"], rf_recovery=c["recovery_used"], rf_retimed=c["retimed"], rf_held=c["held"],
             fuel_rem_p05=f(c["fuel_remaining_p05"]), resolves_per_flight=f(c["resolves_total"] / c["runs"], 0),
             sv_rf_runs=pct(ms["section_v_closed"]["recursive_feasibility_runs"]),
             poly_outside=pct(ms["polynomial"]["time_outside_annulus_frac_mean"]),
             poly_outside_runs=pct(ms["polynomial"]["runs_commanding_outside_annulus"]),
             polyi_outside=pct(ms["polynomial_ideal"]["time_outside_annulus_frac_mean"]))

    v.update(env_grid=f"{env['grid'][1]} × {env['grid'][0]}", env_n0=env["landable_no_error"], env_n1=env["landable_with_error"],
             env_shrink=pct(env["shrink_fraction"]), env_lost=len(env["lost_cells"]),
             env_lost_fuel=sum(1 for c in env["lost_cells"] if c["with_error"] == "insufficient_fuel"),
             env_lost_flight=sum(1 for c in env["lost_cells"] if c["with_error"] == "failed_in_flight"),
             env_thrust=env["counts_no_error"]["insufficient_thrust"], env_fuel0=env["counts_no_error"]["insufficient_fuel"],
             env_gs=env["counts_no_error"]["initial_state_infeasible"], env_fail0=env["counts_no_error"]["failed_in_flight"])

    nm = bl["montecarlo"]["closed_no_margin"]
    for e in ("-0.05", "+0.00", "+0.05"):
        tag = f"nm_{e.replace('-', 'm').replace('+', 'p').replace('.', '')}"
        v[f"{tag}_status"] = nm[e]["status"].replace("_", " ")
        v[f"{tag}_spd"], v[f"{tag}_err"] = f(nm[e]["touchdown_speed"], 2), f(nm[e]["landing_error"], 2)
    v["env_sentence"] = env_sentence(env)
    ab = _load("ablation")
    a_nm, a_nr = ab["summary"]["no_mass_margin"], ab["summary"]["no_rho2_tightening"]
    v["tightening_sentence"] = tightening_sentence(mc, ab, crit)
    v.update(ab_n=ab["runs_per_variant"]["no_mass_margin"], ab_nm_success=pct(a_nm["success"]), ab_nm_neg_success=pct(a_nm["success_eps_negative"]),
             ab_nm_neg=a_nm["runs_eps_negative"], ab_nr_success=pct(a_nr["success"]), ab_nr_exhausted=a_nr["fuel_exhausted"],
             ab_ts_p95=f(ab["summary"]["t_stop_1s"]["touchdown_speed_p95"], 2), ab_ts_below=pct(ab["summary"]["t_stop_1s"]["below_speed"]))
    b = bl["montecarlo"]["by_eps"]
    for e in ("-0.05", "+0.00", "+0.05"):
        for m in ("closed", "open_loop", "polynomial", "polynomial_ideal", "constant_tw"):
            r = b[e][m]
            tag = f"bl_{m}_{e.replace('-', 'm').replace('+', 'p').replace('.', '')}"
            v[f"{tag}_err"], v[f"{tag}_spd"], v[f"{tag}_fuel"] = f(r["landing_error"], 2), f(r["touchdown_speed"], 2), f(r["fuel_used"])
    return v


def render() -> str:
    tpl = (ROOT / "scripts" / "report_template.md").read_text()
    return string.Template(tpl).substitute(values())


def main():
    (ROOT / "REPORT.md").write_text(render())
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
