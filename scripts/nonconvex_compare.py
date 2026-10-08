"""Problem 4 (convex) vs Problem 1 by SLSQP from random starts, on 20 small instances (same N, same tf)."""
import time

import numpy as np

from common import FIGURES, save
from figstyle import MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from montecarlo.sampling import draw_instance
from nonconvex.nlp import NonconvexProblem1, random_start
from problem.setup import make_problem
from tof.search import algorithm1
from verify.slack import slack_report


def main():
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    nc = cfg["nonconvex"]
    P = make_problem(cfg, N=nc["N"])
    rng = np.random.default_rng(nc["seed"])
    rows, skipped = [], []
    while len(rows) < nc["instances"]:
        r0, v0 = draw_instance(rng)
        res = algorithm1(P, r0, v0, veh.m_wet, veh.m_fuel)
        if res.classification != "valid":
            skipped.append({"r0": r0, "v0": v0, "class": res.classification})
            continue
        s = res.sol
        row = {"r0": r0, "v0": v0, "tf": res.tf, "beta_convex": s.cost,
               "convex_slack_max_gap": slack_report(s, cfg["solver"]["slack_tol"])["max_gap"]}
        for variant in ("taylor", "exact"):
            nlp = NonconvexProblem1(P.N, res.tf, veh, P.opts, r0, v0, veh.m_wet, variant)
            # The convex optimum, plugged into Problem 1 with sigma = ||u||: is it Problem 1 feasible?
            row[f"{variant}_convex_point_violation"] = nlp.violation(s.U.ravel())
            starts = []
            t0 = time.perf_counter()
            for _ in range(nc["starts"]):
                r = nlp.solve(random_start(rng, P.N, veh, veh.m_wet))
                starts.append({"beta": r.beta, "feasible": r.feasible, "violation": r.max_violation,
                               "success": r.success, "nit": r.nit})
            row[f"{variant}_starts"] = starts
            row[f"{variant}_seconds"] = time.perf_counter() - t0
            feas = [st["beta"] for st in starts if st["feasible"]]
            row[f"{variant}_best"] = min(feas) if feas else None
            row[f"{variant}_n_feasible"] = len(feas)
            warm = nlp.solve(s.U.ravel())
            row[f"{variant}_from_convex"] = {"beta": warm.beta, "feasible": warm.feasible}
        rows.append(row)
        print(len(rows), round(row["beta_convex"], 4), row["taylor_best"], row["exact_best"],
              row["taylor_n_feasible"], row["exact_n_feasible"])

    def gap(r, v):
        return None if r[f"{v}_best"] is None else (r[f"{v}_best"] - r["beta_convex"]) / r["beta_convex"]

    summary = {
        "N": P.N, "instances": len(rows), "starts": nc["starts"], "skipped_not_valid": len(skipped),
        "taylor": {
            "instances_convex_at_least_as_good": sum(1 for r in rows if r["taylor_best"] is None
                                                     or r["beta_convex"] <= r["taylor_best"] * (1 + 1e-7)),
            "instances_no_feasible_start": sum(1 for r in rows if r["taylor_best"] is None),
            "starts_feasible": sum(r["taylor_n_feasible"] for r in rows),
            "starts_total": len(rows) * nc["starts"],
            "median_rel_gap_best_start": float(np.median([g for r in rows if (g := gap(r, "taylor")) is not None])),
            "max_convex_point_violation": max(r["taylor_convex_point_violation"] for r in rows)},
        "exact": {
            "instances_convex_at_least_as_good": sum(1 for r in rows if r["exact_best"] is None
                                                     or r["beta_convex"] <= r["exact_best"] * (1 + 1e-7)),
            "instances_no_feasible_start": sum(1 for r in rows if r["exact_best"] is None),
            "starts_feasible": sum(r["exact_n_feasible"] for r in rows),
            "min_rel_gap_best_start": float(np.min([g for r in rows if (g := gap(r, "exact")) is not None])),
            "max_convex_point_violation": max(r["exact_convex_point_violation"] for r in rows),
            "from_convex_rel_improvement_max": float(max(
                (r["beta_convex"] - r["exact_from_convex"]["beta"]) / r["beta_convex"]
                for r in rows if r["exact_from_convex"]["feasible"]))},
    }
    save("nonconvex", {"summary": summary, "instances": rows, "skipped": skipped})

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, r in enumerate(rows):
        for st in r["taylor_starts"]:
            if st["feasible"]:
                ax.plot(i, 100 * (st["beta"] - r["beta_convex"]) / r["beta_convex"], "o",
                        color=SERIES[0], alpha=0.6, ms=5)
            else:
                ax.plot(i, -0.5, "x", color=MUTED, ms=5)
    ax.axhline(0, color=SERIES[1], lw=1.5)
    ax.set(xlabel="instance", ylabel="(β_NLP − β_convex)/β_convex  (%)",
           title=f"SLSQP on Problem 1, {nc['starts']} random starts (× = no feasible point), N = {P.N}")
    ax.set_yscale("symlog", linthresh=0.01)
    savefig(fig, FIGURES / "nonconvex_compare.png")
    print(summary)


if __name__ == "__main__":
    main()
