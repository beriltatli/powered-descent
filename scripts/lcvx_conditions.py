"""Theorem 2 (pointing LCvx, Malyuta et al. 2022 p.8, attributed to [87, 96]): check Conditions 1-3 and
the glide-slope test (106) for this repo's model, then re-run the Section VI pointing battery with the
planet rotation that Condition 3 needs, to see whether satisfying it closes the loose-slack gap."""
import dataclasses
import os
from multiprocessing import Pool

import numpy as np

from common import save

from dynamics.model import load_config, vehicle_from_config
from montecarlo.sampling import draw_pointing_instance
from problem.pdg import PDGProblem
from problem.setup import options_from_config
from tof.search import algorithm1
from verify.lcvx_conditions import condition1, condition2, condition3, glideslope_force_balance
from verify.slack import classify_loose, slack_gaps

WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))
_C: dict = {}


def omegas(cfg: dict) -> dict[str, np.ndarray]:
    lc = cfg["lcvx_conditions"]
    w = np.deg2rad(np.asarray(lc["omega_tutorial_deg_s"], float))
    return {"none_2007": np.zeros(3), "mars_tutorial": w, "diagnostic_x100": lc["diagnostic_scale"] * w}


def problem_for(name: str) -> PDGProblem:
    if name not in _C:
        cfg = load_config()
        veh = dataclasses.replace(vehicle_from_config(cfg), omega=omegas(cfg)[name])
        _C[name] = PDGProblem(cfg["discretization"]["N"], veh, options_from_config(cfg, pointing=True))
        _C["cfg"] = cfg
    return _C[name]


def job(args):
    wname, half_angle, i, r0, v0 = args
    P = problem_for(wname)
    cfg = _C["cfg"]
    veh, tol = P.veh, cfg["solver"]["slack_tol"]
    gamma = None if half_angle is None else float(np.cos(np.deg2rad(half_angle)))
    res = algorithm1(P, np.asarray(r0), np.asarray(v0), veh.m_wet, veh.m_fuel, coarse=cfg["tof"]["coarse_points"],
                     tol=cfg["tof"]["golden_tol"], gamma=gamma)
    row = {"omega": wname, "half_angle_deg": half_angle, "i": i, "class": res.classification}
    if res.sol is None:
        return row
    s = res.sol
    reasons = set(classify_loose(s, tol).values())
    thrust = np.linalg.norm(s.U, axis=1) * np.exp(s.Y[:-1, 6])
    tan_t = np.tan(np.deg2rad(cfg["scenario"]["glideslope_deg"]))
    gs_margin = tan_t * s.Y[1:, 0] - np.linalg.norm(s.Y[1:, 1:3], axis=1)   # m, >= 0 by (11)
    row.update(tf=res.tf, fuel=res.fuel, max_gap=float(slack_gaps(s).max()), loose_reasons=sorted(reasons),
               material=bool(reasons - {"tolerance"}), material_not_reversal=bool(reasons - {"tolerance", "reversal"}),
               thrust_below_0p99_rho1=bool(thrust.min() < 0.99 * veh.rho1),
               glideslope_active_nodes=int(np.sum(gs_margin[:-1] < 1e-2)),
               condition2=condition2(float(s.S[-1]))["holds"])
    return row


def main():
    cfg = load_config()
    lc, pc = cfg["lcvx_conditions"], cfg["pointing"]
    veh = vehicle_from_config(cfg)
    T, n_hat = lc["gramian_horizon_s"], np.asarray(pc["v"], float)
    ws = omegas(cfg)
    conditions = {k: {"omega_rad_s": w, "condition1": condition1(w, T), "condition3": condition3(w, n_hat, T)}
                  for k, w in ws.items()}
    force_balance = {f"{a:g}deg": glideslope_force_balance(veh, cfg["scenario"]["glideslope_deg"], a)
                     for a in lc["pointing_half_angles_deg"]}

    rng = np.random.default_rng(pc["seed"])
    inst = [draw_pointing_instance(rng) for _ in range(pc["random_instances"])]
    angles = [None] + list(lc["pointing_half_angles_deg"])
    args = [(w, a, i, r0.tolist(), v0.tolist()) for w in ws for a in angles for i, (r0, v0) in enumerate(inst)]
    with Pool(WORKERS) as pool:
        rows = pool.map(job, args, chunksize=4)

    summary = {}
    for w in ws:
        for a in angles:
            rs = [r for r in rows if r["omega"] == w and r["half_angle_deg"] == a]
            ok = [r for r in rs if "tf" in r]
            key = f"{w}/{'off' if a is None else f'{a:g}deg'}"
            classes: dict = {}
            for r in rs:
                classes[r["class"]] = classes.get(r["class"], 0) + 1
            summary[key] = {"n": len(rs), "solved": len(ok), "classes": classes,
                            "material_loose": sum(r["material"] for r in ok),
                            "material_loose_not_reversal": sum(r["material_not_reversal"] for r in ok),
                            "thrust_below_0p99_rho1": sum(r["thrust_below_0p99_rho1"] for r in ok),
                            "glideslope_touching": sum(r["glideslope_active_nodes"] > 0 for r in ok),
                            "condition2_holds": sum(r["condition2"] for r in ok),
                            "median_fuel": float(np.median([r["fuel"] for r in ok])) if ok else None}
    save("lcvx_conditions", {"T_s": T, "n_hat": n_hat, "conditions": conditions, "force_balance": force_balance,
                             "summary": summary, "rows": rows})
    for k, v in conditions.items():
        print(k, v["condition3"])
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
