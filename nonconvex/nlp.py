"""Problem 1 solved directly by a general NLP (SLSQP), on the same ZOH grid as Problem 4.

Problem 1 is written in the variables of (25) and (30): u = Tc/m and z = ln m, held piecewise constant
like Problem 4. That is a change of variables, not an approximation. The nonconvexity is restored by
setting sigma_k = ||u_k|| (no slack), so the thrust bound acts on ||u|| itself:
  variant "taylor": mu1(...)(z_k) <= ||u_k|| <= mu2(...)(z_k), the (34) bounds, so the only difference
                    from Problem 4 is the relaxation (14). This isolates Lemma 1;
  variant "exact":  rho1 e^{-z_k} <= ||u_k|| <= rho2 e^{-z_k}, exact (32). Its feasible set is larger than
                    Problem 4's by Lemma 2, so it may legitimately beat the convex optimum; the margin is
                    the price of the Taylor approximation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dynamics.model import Vehicle, discretize
from problem.pdg import Options
from problem.taylor import taylor_coefficients


@dataclass
class NLPResult:
    beta: float
    feasible: bool
    max_violation: float
    success: bool
    message: str
    nit: int
    U: np.ndarray


class NonconvexProblem1:
    def __init__(self, N: int, tf: float, veh: Vehicle, opts: Options, r0, v0, m0: float, variant: str):
        self.N, self.tf, self.veh, self.opts = N, tf, veh, opts
        self.dt = tf / N
        A, B = discretize(self.dt, veh.alpha)
        y0 = np.concatenate([r0, v0, [np.log(m0)]])
        gw = np.concatenate([veh.g, [0.0]])
        # Y_k = A^k y0 + sum_{j<k} A^{k-1-j} B (w_j + [g;0]): affine in the stacked w.
        self.c = np.zeros((N + 1, 7))
        self.L = np.zeros((N + 1, 7, N, 4))
        self.c[0] = y0
        for k in range(N):
            self.c[k + 1] = A @ self.c[k] + B @ gw
            self.L[k + 1] = np.einsum("ij,jlm->ilm", A, self.L[k])
            self.L[k + 1, :, k, :] = B
        self.Lf = self.L.reshape((N + 1) * 7, N * 4)
        t = self.dt * np.arange(N)
        self.coef = taylor_coefficients(m0, veh.alpha, veh.rho1, veh.rho2, t)
        self.variant = variant
        self.tan_t = np.tan(np.deg2rad(opts.glideslope_deg)) if opts.glideslope_deg is not None else None

    def states(self, x: np.ndarray) -> np.ndarray:
        u = x.reshape(self.N, 3)
        w = np.hstack([u, np.linalg.norm(u, axis=1, keepdims=True)])
        return self.c + (self.Lf @ w.ravel()).reshape(self.N + 1, 7)

    def cost(self, x: np.ndarray) -> float:
        return self.dt * np.linalg.norm(x.reshape(self.N, 3), axis=1).sum()

    def eq(self, x: np.ndarray) -> np.ndarray:
        Y = self.states(x)
        parts = [Y[-1, :6]]
        if self.opts.final_thrust_dir is not None:
            # (37) with n_f = e_1: the last control has no horizontal component.
            parts.append(x.reshape(self.N, 3)[-1, 1:])
        return np.concatenate(parts)

    def ineq(self, x: np.ndarray) -> np.ndarray:
        u = x.reshape(self.N, 3)
        n = np.linalg.norm(u, axis=1)
        Y = self.states(x)
        z = Y[:-1, 6]
        c = self.coef
        if self.variant == "taylor":
            lo = c["lo_c0"] + c["lo_c1"] * z + 0.5 * (c["lo_s"] * z - c["lo_w"]) ** 2
            up = c["up_c0"] + c["up_c1"] * z
        else:
            lo = self.veh.rho1 * np.exp(-z)
            up = self.veh.rho2 * np.exp(-z)
        parts = [n - lo, up - n, z - c["z_lo"], c["z_hi"] - z]
        if self.tan_t is not None:
            parts.append(self.tan_t * Y[1:-1, 0] - np.linalg.norm(Y[1:-1, 1:3], axis=1))
        if self.opts.vmax is not None:
            parts.append(self.opts.vmax - np.linalg.norm(Y[1:, 3:6], axis=1))
        if self.opts.final_thrust_dir is not None:
            parts.append(u[-1, :1])
        return np.concatenate(parts)

    def violation(self, x: np.ndarray) -> float:
        return float(max(np.abs(self.eq(x)).max(), max(0.0, -self.ineq(x).min())))

    def solve(self, x0: np.ndarray, maxiter: int = 1000, feas_tol: float = 1e-5) -> NLPResult:
        r = minimize(self.cost, x0, method="SLSQP",
                     constraints=[{"type": "eq", "fun": self.eq}, {"type": "ineq", "fun": self.ineq}],
                     options={"maxiter": maxiter, "ftol": 1e-12})
        viol = self.violation(r.x)
        return NLPResult(beta=float(self.cost(r.x)), feasible=viol <= feas_tol, max_violation=viol,
                         success=bool(r.success), message=str(r.message), nit=int(r.nit),
                         U=r.x.reshape(self.N, 3))


def random_start(rng: np.random.Generator, N: int, veh: Vehicle, m0: float) -> np.ndarray:
    # Uniform direction on the sphere, magnitude uniform inside the thrust annulus at the initial mass.
    d = rng.normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    mag = rng.uniform(veh.rho1 / m0, veh.rho2 / m0, size=(N, 1))
    return (d * mag).ravel()
