"""Per-solve latency distribution at the chosen N, compile time reported separately."""
import subprocess
import sys

import numpy as np

from common import FIGURES, ROOT, machine, save
from figstyle import INK2, MUTED, SERIES, plt, savefig

from dynamics.model import load_config, vehicle_from_config
from metrics.timing import percentiles, timed
from montecarlo.sampling import draw_instance
from problem.setup import make_problem
from tof.search import algorithm1

COLD = """
import sys, time; sys.path.insert(0, %r)
import numpy as np
from dynamics.model import load_config, vehicle_from_config
from problem.setup import make_problem
cfg = load_config(); veh = vehicle_from_config(cfg)
t0 = time.perf_counter(); P = make_problem(cfg); t1 = time.perf_counter()
s = P.solve(cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"], veh.m_wet, 60.0); t2 = time.perf_counter()
s2 = P.solve(cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"], veh.m_wet, 61.0); t3 = time.perf_counter()
print((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
"""


def dpp_vs_recompile(cfg, veh, n: int = 60) -> dict:
    """Same solves, once through the cached DPP mapping and once with ignore_dpp=True (full
    canonicalization each call). Separate problem objects: an ignore_dpp call evicts the DPP cache."""
    r0, v0 = cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"]
    out = {}
    for name, kw in (("dpp", {}), ("recompile_every_call", {"ignore_dpp": True})):
        P = make_problem(cfg)
        if not kw:
            P.solve(r0, v0, veh.m_wet, 58.0)
        ms = []
        for i in range(n):
            P.set_parameters(r0, v0, veh.m_wet, 58.0 + 0.05 * i)
            _, t = timed(P.prob.solve, solver="CLARABEL", **kw)
            ms.append(t)
        out[name] = percentiles(ms)
    return out


def main():
    cfg = load_config()
    veh = vehicle_from_config(cfg)
    tc = cfg["timing"]
    cold = []
    for _ in range(5):
        out = subprocess.run([sys.executable, "-c", COLD % str(ROOT)], capture_output=True, text=True, check=True)
        build, first, second = map(float, out.stdout.split()[-3:])
        cold.append({"build_ms": build, "first_solve_ms": first, "second_solve_ms": second,
                     "compile_ms": first - second})

    P = make_problem(cfg)
    P.solve(cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"], veh.m_wet, 60.0)   # compile, excluded
    rng = np.random.default_rng(tc["seed"])
    at_opt, solver_ms, probe_ms, alg_ms, statuses = [], [], [], [], {}
    while len(at_opt) < tc["solves"]:
        r0, v0 = draw_instance(rng)
        res, ms = timed(algorithm1, P, r0, v0, veh.m_wet, veh.m_fuel,
                        coarse=cfg["tof"]["coarse_points"], tol=cfg["tof"]["golden_tol"])
        if res.sol is None:
            continue
        alg_ms.append(ms)
        s = P.solve(r0, v0, veh.m_wet, res.tf)
        at_opt.append(s.wall_s * 1e3)
        solver_ms.append(s.solver_s * 1e3)
        statuses[s.status] = statuses.get(s.status, 0) + 1
    # every solve inside the searches, feasible or not, timed by the same wrapper
    P2 = make_problem(cfg)
    P2.solve(cfg["montecarlo"]["r0"], cfg["montecarlo"]["v0"], veh.m_wet, 60.0)
    rng = np.random.default_rng(tc["seed"] + 1)
    infeas_ms = []
    for _ in range(100):
        r0, v0 = draw_instance(rng)
        res = algorithm1(P2, r0, v0, veh.m_wet, veh.m_fuel)
        # re-time a sample of probes individually so feasible and infeasible solves are separated
        for tf, b, st in res.probes[:: max(1, len(res.probes) // 6)]:
            s = P2.solve(r0, v0, veh.m_wet, tf)
            (probe_ms if s.ok else infeas_ms).append(s.wall_s * 1e3)

    out = {"machine": machine(), "N": P.N, "backend": "CLARABEL",
           "per_solve_wall_ms_at_tf_opt": percentiles(at_opt),
           "per_solve_solver_ms_at_tf_opt": percentiles(solver_ms),
           "per_solve_wall_ms_feasible_probes": percentiles(probe_ms),
           "per_solve_wall_ms_infeasible_probes": percentiles(infeas_ms) if infeas_ms else None,
           "algorithm1_total_ms": percentiles(alg_ms),
           "statuses": statuses, "cold_start": cold,
           "compile_ms_median": float(np.median([c["compile_ms"] for c in cold])),
           "build_ms_median": float(np.median([c["build_ms"] for c in cold])),
           "dpp_vs_recompile": dpp_vs_recompile(cfg, veh),
           "samples_wall_ms": at_opt}
    save("timing", out)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    a = np.sort(at_opt)
    ax.plot(a, 1 - np.arange(len(a)) / len(a), color=SERIES[0], label="wall (cvxpy solve call)")
    b = np.sort(solver_ms)
    ax.plot(b, 1 - np.arange(len(b)) / len(b), color=SERIES[2], label="Clarabel-reported")
    ax.axvline(cfg["success"]["p99_solve_ms"], color=MUTED, ls="--", lw=1)
    ax.text(cfg["success"]["p99_solve_ms"], 0.5, " 50 ms target", color=INK2, fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="time per solve (ms)", ylabel="fraction of solves ≥ x",
           title=f"Solve-time tail, N = {P.N}, {len(at_opt)} solves, {out['machine']['cpu']}")
    ax.legend()
    savefig(fig, FIGURES / "solve_time_tail.png")
    print({k: v for k, v in out.items() if k not in ("samples_wall_ms", "cold_start", "machine")})


if __name__ == "__main__":
    main()
