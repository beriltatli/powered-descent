import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from dynamics.model import Vehicle, discretize
from problem.taylor import taylor_coefficients


@dataclass(frozen=True)
class Options:
    glideslope_deg: float | None = 86.0
    subsurface: bool = False          # (7); redundant when the glide slope is on
    vmax: float | None = None         # (10)
    final_thrust_dir: tuple | None = (1.0, 0.0, 0.0)   # second equality of (37)
    pointing: bool = False            # (61)/(62) with a per-node gamma Parameter
    pointing_axis: tuple = (1.0, 0.0, 0.0)
    # Recovery variant for closed loop only: glide slope and speed bound become soft, with slack
    # penalized by soft_weight per metre (per m/s) in the cost. Never used for the open-loop results.
    soft_state: bool = False
    soft_weight: float = 100.0


@dataclass
class Solution:
    status: str                # optimal | infeasible | inaccurate | solver_error
    tf: float
    dt: float
    Y: np.ndarray | None = None        # (N+1, 7): r, rdot, z
    U: np.ndarray | None = None        # (N, 3)
    S: np.ndarray | None = None        # (N,) sigma
    cost: float = np.inf               # sum sigma dt  (beta in (58))
    wall_s: float = np.nan             # whole cvxpy solve() call, parameter mapping included
    solver_s: float = np.nan           # backend-reported time
    backend: str = ""
    raw_status: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "optimal"

    def fuel(self, m0: float) -> float:
        # m(tf) = m0 exp(-alpha int sigma) from (27); reported via z so no alpha is needed here.
        return m0 - np.exp(self.Y[-1, 6]) if self.Y is not None else np.inf

    @property
    def gamma(self) -> np.ndarray:
        # Gamma = sigma m, the slack in thrust units (25); m taken at the interval start.
        return self.S * np.exp(self.Y[:-1, 6])

    @property
    def thrust(self) -> np.ndarray:
        return np.linalg.norm(self.U, axis=1) * np.exp(self.Y[:-1, 6])


class PDGProblem:
    """Problem 4 for a fixed node count N, compiled once.

    Everything that changes between solves is a cp.Parameter: the initial state, Delta-t = tf/N through
    the ZOH matrices A and B, the Taylor coefficients of (34), the z box (36), and the per-node
    pointing gamma. Node k carries the control held on [t_k, t_{k+1}); constraints (50)-(51) pair
    sigma_k with z_k, as (51) does, and the state constraints (53) are imposed at k = 1..N.
    """

    def __init__(self, N: int, veh: Vehicle, opts: Options = Options()):
        self.N, self.veh, self.opts = N, veh, opts
        p = cp.Parameter
        self.p_y0 = p(7)
        self.p_A = p((7, 7))
        self.p_B = p((7, 4))
        self.p_gin = p(7)          # B [g; 0], the gravity term of (43)
        self.p_dt = p(nonneg=True)
        self.p = {k: p(N) for k in ("lo_c0", "lo_c1", "lo_s", "lo_w", "up_c0", "up_c1", "z_lo", "z_hi")}
        self.p_gamma = p(N) if opts.pointing else None

        Y = cp.Variable((N + 1, 7))
        U = cp.Variable((N, 3))
        S = cp.Variable(N)
        self.Y, self.U, self.S = Y, U, S
        W = cp.hstack([U, cp.reshape(S, (N, 1), order="C")])

        cons = [Y[0] == self.p_y0]
        # Dynamics, (41) written recursively: y_{k+1} = A y_k + B w_k + B [g; 0].
        cons += [Y[1:].T == self.p_A @ Y[:-1].T + self.p_B @ W.T + cp.reshape(self.p_gin, (7, 1), order="C")]
        cons += [cp.norm(U, 2, axis=1) <= S]                                      # (50)
        z = Y[:-1, 6]
        P = self.p
        cons += [P["lo_c0"] + cp.multiply(P["lo_c1"], z)
                 + 0.5 * cp.square(cp.multiply(P["lo_s"], z) - P["lo_w"]) <= S]  # (51) left
        cons += [S <= P["up_c0"] + cp.multiply(P["up_c1"], z)]                    # (51) right
        cons += [z >= P["z_lo"], z <= P["z_hi"]]                                  # (52)
        penalty = 0
        if opts.soft_state:
            s_gs, s_v = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
            penalty = opts.soft_weight * (cp.sum(s_gs) + cp.sum(s_v))
            self.s_gs, self.s_v = s_gs, s_v
        else:
            s_gs = s_v = 0
        if opts.glideslope_deg is not None:
            # (11): ||(r2, r3)|| <= tan(theta_alt) r1, in this repo's 0-based axes.
            tan_t = np.tan(np.deg2rad(opts.glideslope_deg))
            cons += [cp.norm(Y[1:, 1:3], 2, axis=1) <= tan_t * Y[1:, 0] + s_gs]
        if opts.subsurface:
            cons += [Y[1:, 0] >= 0]                                               # (7)
        if opts.vmax is not None:
            cons += [cp.norm(Y[1:, 3:6], 2, axis=1) <= opts.vmax + s_v]           # (10)
        cons += [Y[N, 0:6] == 0]                                                  # (5)
        if opts.final_thrust_dir is not None:
            # (37) second equality, u(tf) = sigma(tf) n_f, imposed on the last held control.
            nf = np.asarray(opts.final_thrust_dir, float)
            cons += [U[N - 1] == S[N - 1] * nf]
        if opts.pointing:
            v = np.asarray(opts.pointing_axis, float)
            # (61) v^T u_k >= gamma sigma_k; the heuristic (62) swaps gamma for kappa_2k per node.
            cons += [U @ v >= cp.multiply(self.p_gamma, S)]

        self.prob = cp.Problem(cp.Minimize(self.p_dt * cp.sum(S) + penalty), cons)  # (47)
        assert self.prob.is_dpp()
        self.compile_s: dict[str, float] = {}

    def set_parameters(self, r0, v0, m0: float, tf: float, rho1: float | None = None,
                       rho2: float | None = None, gamma=None) -> None:
        veh, N = self.veh, self.N
        rho1 = veh.rho1 if rho1 is None else rho1
        rho2 = veh.rho2 if rho2 is None else rho2
        dt = tf / N
        A, B = discretize(dt, veh.alpha, veh.omega)
        self.p_A.value, self.p_B.value = A, B
        self.p_gin.value = B @ np.concatenate([veh.g, [0.0]])
        self.p_dt.value = dt
        self.p_y0.value = np.concatenate([r0, v0, [np.log(m0)]])
        coef = taylor_coefficients(m0, veh.alpha, rho1, rho2, dt * np.arange(N))
        for k, par in self.p.items():
            par.value = coef[k]
        if self.p_gamma is not None:
            g = -1.0 if gamma is None else gamma
            self.p_gamma.value = np.broadcast_to(np.asarray(g, float), (N,)).copy()

    def solve(self, r0, v0, m0: float, tf: float, rho1=None, rho2=None, gamma=None,
              backend: str = "CLARABEL") -> Solution:
        self.set_parameters(r0, v0, m0, tf, rho1, rho2, gamma)
        out = self._solve_with(backend, tf)
        if out.status == "solver_error" and backend != "ECOS":
            # Cross-check a numerical failure with a second interior-point code. The first status is
            # kept in extra so a failure is never silently replaced.
            alt = self._solve_with("ECOS", tf)
            alt.extra["first_backend"] = (backend, out.raw_status)
            return alt
        return out

    def _solve_with(self, backend: str, tf: float) -> Solution:
        first = backend not in self.compile_s
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=backend, warm_start=False)
            raw = self.prob.status
        except cp.error.SolverError as e:
            raw = f"solver_error: {e}"[:200]
        wall = time.perf_counter() - t0
        if first:
            # The first call includes DPP canonicalization; it is reported as compile time, not as a
            # solve time.
            self.compile_s[backend] = wall
        status = {"optimal": "optimal", "infeasible": "infeasible",
                  "optimal_inaccurate": "inaccurate",
                  "infeasible_inaccurate": "infeasible_inaccurate"}.get(raw, "solver_error")
        sol = Solution(status=status, tf=tf, dt=tf / self.N, backend=backend, raw_status=raw,
                       wall_s=wall)
        stats = self.prob.solver_stats if status != "solver_error" else None
        if stats is not None and stats.solve_time is not None:
            sol.solver_s = stats.solve_time
        if status in ("optimal", "inaccurate"):
            sol.Y = self.Y.value.copy()
            sol.U = self.U.value.copy()
            sol.S = self.S.value.copy()
            sol.cost = float(self.p_dt.value * sol.S.sum())
            sol.extra["objective"] = float(self.prob.value)
            if self.opts.soft_state:
                sol.extra["soft_slack_max"] = float(max(self.s_gs.value.max(), self.s_v.value.max()))
        sol.extra["first_call"] = first
        return sol
