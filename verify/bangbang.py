import numpy as np

from dynamics.model import Vehicle
from problem.pdg import Options, Solution
from problem.taylor import taylor_coefficients


def classify_thrust(sol: Solution, veh: Vehicle, opts: Options, m0: float,
                    rel_tol: float = 1e-3, active_tol_m: float = 1.0) -> dict:
    """Classify every node's thrust level for the Lemma 1 bang-bang check.

    A node is "at rho1" / "at rho2" when sigma_k sits on the lower / upper Taylor bound of (34), the
    constraints the discrete problem actually has. Gamma = sigma m sits slightly inside [rho1, rho2] there,
    because (34) is conservative (Lemma 2), so comparing Gamma with rho directly would mislabel those nodes.

    An intermediate node is explained, in this order, by:
      final_dir   - k = N-1, where (37) pins the direction and the magnitude is a free choice;
      switch      - its neighbours sit on different bounds (or k = 0 and node 1 sits on a bound), so the
                    continuous switch time falls inside this interval and ZOH averages it;
      short_arc   - both neighbours sit on the same bound: an arc of the other level shorter than one
                    interval, again averaged by ZOH;
      state_active - glide slope (within active_tol_m metres) or speed bound active at node k or k+1.
                    Lemma 1's proof has no state-constraint multipliers (THEORY.md section 2), so it makes
                    no claim on such arcs;
      unexplained - none of the above: a candidate counterexample, reported individually.
    """
    N = len(sol.S)
    c = taylor_coefficients(m0, veh.alpha, veh.rho1, veh.rho2, sol.dt * np.arange(N))
    z = sol.Y[:-1, 6]
    lo = c["lo_c0"] + c["lo_c1"] * z + 0.5 * (c["lo_s"] * z - c["lo_w"]) ** 2
    up = c["up_c0"] + c["up_c1"] * z
    at_lo = np.abs(sol.S - lo) <= rel_tol * up
    at_up = np.abs(sol.S - up) <= rel_tol * up
    level = np.where(at_up, 2, np.where(at_lo, 1, 0))

    gs_active = np.zeros(N + 1, bool)
    if opts.glideslope_deg is not None:
        d = np.tan(np.deg2rad(opts.glideslope_deg)) * sol.Y[:, 0] - np.linalg.norm(sol.Y[:, 1:3], axis=1)
        gs_active = d <= active_tol_m
        gs_active[-1] = False       # the target itself sits on the cone apex
    v_active = np.zeros(N + 1, bool)
    if opts.vmax is not None:
        v_active = opts.vmax - np.linalg.norm(sol.Y[:, 3:6], axis=1) <= 1e-2

    reasons = {}
    for k in np.flatnonzero(level == 0):
        if k == N - 1 and opts.final_thrust_dir is not None:
            r = "final_dir"
        elif (0 < k < N - 1 and {level[k - 1], level[k + 1]} == {1, 2}) or (k == 0 and N > 1 and level[1] > 0):
            r = "switch"
        elif 0 < k < N - 1 and level[k - 1] == level[k + 1] > 0:
            r = "short_arc"
        elif _in_short_run(level, k, max_len=2):
            # two partial intervals in a row: an arc between one and two intervals long, entered and
            # left inside the two intervals
            r = "short_arc"
        elif gs_active[k] or gs_active[k + 1] or v_active[k] or v_active[k + 1]:
            r = "state_active"
        else:
            r = "unexplained"
        reasons[int(k)] = r
    counts = {r: sum(1 for v in reasons.values() if v == r)
              for r in ("final_dir", "switch", "short_arc", "state_active", "unexplained")}
    return {"n_rho1": int((level == 1).sum()), "n_rho2": int((level == 2).sum()),
            "intermediate": reasons, "counts": counts,
            "gamma_over_rho2": (sol.gamma / veh.rho2).tolist()}


def _in_short_run(level: np.ndarray, k: int, max_len: int) -> bool:
    a = k
    while a > 0 and level[a - 1] == 0:
        a -= 1
    b = k
    while b < len(level) - 1 and level[b + 1] == 0:
        b += 1
    return a > 0 and b < len(level) - 1 and b - a + 1 <= max_len and level[a - 1] > 0 and level[b + 1] > 0
