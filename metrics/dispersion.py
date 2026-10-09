import numpy as np


def success(row: dict, radius: float, speed: float) -> bool:
    return row["landing_error"] < radius and row["touchdown_speed"] < speed


def summarize(rows: list[dict], radius: float, speed: float) -> dict:
    err = np.array([r["landing_error"] for r in rows], float)
    spd = np.array([r["touchdown_speed"] for r in rows], float)
    fuel = np.array([r["fuel_used"] for r in rows], float)
    out = {"runs": len(rows),
           "within_radius": float(np.mean(err < radius)), "below_speed": float(np.mean(spd < speed)),
           "success": float(np.mean([success(r, radius, speed) for r in rows])),
           "landing_error_p50": float(np.median(err)), "landing_error_p95": float(np.percentile(err, 95)),
           "landing_error_max": float(err.max()),
           "touchdown_speed_p50": float(np.median(spd)), "touchdown_speed_p95": float(np.percentile(spd, 95)),
           "touchdown_speed_max": float(spd.max()),
           "fuel_mean": float(fuel.mean()), "fuel_p95": float(np.percentile(fuel, 95)),
           "fuel_exhausted": int(sum(bool(r.get("fuel_exhausted")) for r in rows)),
           "statuses": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}}
    if "recursively_feasible" in rows[0]:
        # physical: hard constraints, true thrust bounds, free tf had a solution at the re-solve.
        # strict: the nominal re-solve at the original landing time was feasible.
        out["recursive_feasibility_runs"] = float(np.mean([r["recursively_feasible"] for r in rows]))
        out["strict_feasibility_runs"] = float(np.mean([r["strictly_feasible"] for r in rows]))
        n = sum(r["n_resolves"] for r in rows)
        hard = sum(r["n_feasible"] + r["n_physical"] + r["n_retimed_hard"] for r in rows)
        out["recursive_feasibility_resolves"] = hard / n if n else None
        out["strict_feasibility_resolves"] = sum(r["n_feasible"] for r in rows) / n if n else None
        out["resolves_total"] = n
        out["retimed_hard"] = sum(r["n_retimed_hard"] for r in rows)
        out["physical_bounds_used"] = sum(r["n_physical"] for r in rows)
        out["recovery_used"] = sum(r["n_recovery"] for r in rows)
        out["retimed"] = sum(r["n_retimed"] for r in rows)
        out["held"] = sum(r["n_held"] for r in rows)
        out["fuel_remaining_p05"] = float(np.nanpercentile([r["fuel_remaining"] for r in rows], 5))
    if "time_outside_annulus_frac" in rows[0]:
        out["time_outside_annulus_frac_mean"] = float(np.mean([r["time_outside_annulus_frac"] for r in rows]))
        out["runs_commanding_outside_annulus"] = float(np.mean([r["time_outside_annulus_frac"] > 0 for r in rows]))
    return out
