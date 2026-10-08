"""Section VI: switch on (61), measure how often the 2007 relaxation goes loose, then apply the heuristic."""
import numpy as np

from common import FIGURES, save
from figstyle import INK2, MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from montecarlo.sampling import draw_pointing_instance
from pointing.constraint import pointing_violation_deg, solve_heuristic
from problem.setup import make_problem
from tof.search import algorithm1
from verify.slack import classify_loose, slack_gaps


def run_case(P, veh, r0, v0, gamma, tol, coarse, gtol):
    v = np.asarray(P.opts.pointing_axis, float)
    m0 = veh.m_wet
    rows = {}
    modes = {"off": dict(gamma=None), "exact61": dict(gamma=gamma)}
    for name, kw in modes.items():
        res = algorithm1(P, r0, v0, m0, veh.m_fuel, coarse=coarse, tol=gtol, **kw)
        rows[name] = res
    rows["heuristic62"] = algorithm1(P, r0, v0, m0, veh.m_fuel, coarse=coarse, tol=gtol,
                                     solve_fn=lambda tf: solve_heuristic(P, r0, v0, m0, tf, gamma))
    out = {}
    for name, res in rows.items():
        if res.sol is None:
            out[name] = {"class": res.classification}
            continue
        s = res.sol
        g = slack_gaps(s)
        thrust = np.linalg.norm(s.U, axis=1) * np.exp(s.Y[:-1, 6])
        viol = pointing_violation_deg(s, v, gamma)
        out[name] = {"class": res.classification, "tf": res.tf, "fuel": res.fuel,
                     "loose_nodes": np.flatnonzero(g > tol).tolist(), "max_gap": float(g.max()),
                     "loose_reasons": sorted(set(classify_loose(s, tol).values())),
                     "min_thrust_over_rho1": float(thrust.min() / veh.rho1),
                     "max_pointing_violation_deg": float(np.nanmax(viol)) if np.isfinite(viol).any() else None,
                     "n_relaxed": len(s.extra.get("relaxed_nodes", [])),
                     "t": (s.dt * np.arange(P.N)).tolist(), "thrust": thrust.tolist(),
                     "angle_from_v_deg": np.degrees(np.arccos(np.clip(
                         (s.U @ v) / np.maximum(np.linalg.norm(s.U, axis=1), 1e-12), -1, 1))).tolist()}
    return out


def physical_counts(batch: list, mode: str) -> dict:
    """Loose slack that matters: gaps above the 1e-3 m/s^2 tolerance band, split by mechanism, and the
    instances where the commanded thrust ||u|| m actually drops below rho1 (not Problem 1 feasible)."""
    rows = [b[mode] for b in batch if "loose_nodes" in b[mode]]
    return {"material_loose": sum(1 for r in rows if set(r["loose_reasons"]) - {"tolerance"}),
            "material_loose_not_reversal": sum(1 for r in rows if set(r["loose_reasons"]) - {"tolerance", "reversal"}),
            "thrust_below_0p99_rho1": sum(1 for r in rows if r["min_thrust_over_rho1"] < 0.99),
            "n": len(rows)}


def plot_example(example: dict, veh) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = {"off": "no pointing", "exact61": "(61) exact", "heuristic62": "heuristic (62), identical to (61) at γ = 0"}
    styles = {"off": "-", "exact61": "-", "heuristic62": ":"}
    for i, (mode, lab) in enumerate(labels.items()):
        e = example[mode]
        if "t" not in e:
            continue
        thrust = np.array(e["thrust"])
        ang = np.array(e["angle_from_v_deg"], float)
        # ||u|| ~ 0 has no direction; leave those nodes out of the angle panel.
        ang[thrust < 1.0] = np.nan
        ax[0].step(e["t"], thrust / 1e3, where="post", color=SERIES[i], ls=styles[mode], label=lab, lw=2)
        ax[1].step(e["t"], ang, where="post", color=SERIES[i], ls=styles[mode], label=lab, lw=2)
    ax[0].axhline(veh.rho1 / 1e3, color=MUTED, ls="--", lw=1)
    ax[0].text(1, veh.rho1 / 1e3 + 0.3, "ρ1", color=INK2, fontsize=8)
    ax[0].set(xlabel="time (s)", ylabel="‖Tc‖ = ‖u‖m (kN)", title="Section VI example (5 km, at rest): thrust")
    ax[1].axhline(90, color=MUTED, ls="--", lw=1)
    ax[1].text(70, 93, "cone edge, γ = 0", color=INK2, fontsize=8, ha="right")
    ax[1].set(xlabel="time (s)", ylabel="angle from v (deg)", title="Thrust angle from +altitude (blank: u = 0)")
    ax[0].legend(loc="upper left")
    savefig(fig, FIGURES / "pointing_example.png")


def main():
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    pc = cfg["pointing"]
    tol = cfg["solver"]["slack_tol"]
    P = make_problem(cfg, pointing=True)
    co, gt = cfg["tof"]["coarse_points"], cfg["tof"]["golden_tol"]

    example = run_case(P, veh, np.array(pc["r0"]), np.array(pc["v0"]), pc["gamma"], tol, co, gt)
    rng = np.random.default_rng(pc["seed"])
    gammas = {"90deg": pc["gamma"], "45deg": float(np.cos(np.deg2rad(45)))}
    batches = {k: [] for k in gammas}
    for i in range(pc["random_instances"]):
        r0, v0 = draw_pointing_instance(rng)
        for k, gm in gammas.items():
            c = run_case(P, veh, r0, v0, gm, tol, co, gt)
            for m in c.values():
                for key in ("t", "thrust", "angle_from_v_deg"):
                    m.pop(key, None)
            batches[k].append({"r0": r0, "v0": v0, **c})

    def frac(batch, mode, pred):
        rows = [b[mode] for b in batch if "loose_nodes" in b[mode]]
        return sum(pred(r) for r in rows), len(rows)

    summary = {"N": P.N, "slack_tol": tol, "example": {k: {kk: vv for kk, vv in v.items()
                                                         if kk not in ("t", "thrust", "angle_from_v_deg")}
                                                     for k, v in example.items()},
               "paper_fig10": {"tf": 69, "fuel_kg": 293.6}, "paper_fig11": {"tf": 78, "fuel_kg": 303}}
    for k, batch in batches.items():
        s = {}
        for mode in ("off", "exact61", "heuristic62"):
            nl, n = frac(batch, mode, lambda r: bool(r["loose_nodes"]))
            reasons = {}
            for b in batch:
                for r in b[mode].get("loose_reasons", []):
                    reasons[r] = reasons.get(r, 0) + 1
            nv, _ = frac(batch, mode, lambda r: (r["max_pointing_violation_deg"] or 0) > 0.5)
            classes = {}
            for b in batch:
                classes[b[mode]["class"]] = classes.get(b[mode]["class"], 0) + 1
            s[mode] = {"solved": n, "loose_instances": nl, "loose_fraction": nl / n if n else None,
                       "instances_by_loose_reason": reasons,
                       "pointing_violated_instances": nv, "classes": classes,
                       "min_thrust_over_rho1": min((b[mode]["min_thrust_over_rho1"] for b in batch
                                                    if "min_thrust_over_rho1" in b[mode]), default=None)}
        fuel_pen = [b["heuristic62"]["fuel"] - b["off"]["fuel"] for b in batch
                    if "fuel" in b["heuristic62"] and "fuel" in b["off"]]
        for mode in ("off", "exact61", "heuristic62"):
            s[mode].update(physical_counts(batch, mode))
        s["heuristic_fuel_penalty_kg_median"] = float(np.median(fuel_pen)) if fuel_pen else None
        s["heuristic_max_violation_deg"] = max((b["heuristic62"].get("max_pointing_violation_deg") or 0)
                                               for b in batch)
        summary[k] = s
    save("pointing", {"summary": summary, "example": example, "instances": batches})

    plot_example(example, veh)
    print({k: v for k, v in summary.items()})


if __name__ == "__main__":
    main()
