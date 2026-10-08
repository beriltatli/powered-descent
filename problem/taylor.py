import numpy as np


def z0_profile(m0: float, alpha: float, rho: float, t: np.ndarray) -> np.ndarray:
    # (35) with rho = rho2: ln(m0 - alpha rho2 t), the lightest the vehicle can be at time t.
    # With rho = rho1 this is the upper end of the z box (36). m0 is the mass at the start of this
    # solve, which is m_wet at t = 0 and the current estimated mass on a closed-loop re-solve.
    return np.log(np.maximum(m0 - alpha * rho * t, 1e-9))


def taylor_coefficients(m0: float, alpha: float, rho1: float, rho2: float, t: np.ndarray) -> dict:
    """Coefficients that put (34) in DPP form.

    Lower bound, (34) left:  mu1 [1 - (z - z0) + (z - z0)^2 / 2] <= sigma.
    Expanded as  lo_c0 + lo_c1 z + 0.5 (lo_s z - lo_w)^2 <= sigma  with
      lo_c0 = mu1 (1 + z0), lo_c1 = -mu1, lo_s = sqrt(mu1), lo_w = sqrt(mu1) z0.
    cvxpy's DPP rules forbid parameter * parameter and a parameter inside square(z - z0) multiplied by
    another parameter, so every product of mu1 with z0 is precomputed here.

    Upper bound, (34) right: sigma <= mu2 [1 - (z - z0)] = up_c0 + up_c1 z,
      up_c0 = mu2 (1 + z0), up_c1 = -mu2.

    mu_i = rho_i exp(-z0), (33). The direction of each inequality is what Lemma 2 needs: the quadratic
    expansion over-estimates exp(-z) for z >= z0 and the linear one under-estimates it, so this
    interval lies inside the exact (32) interval. Swapping either sign would let the solver use thrust
    the engine does not have.
    """
    z0 = z0_profile(m0, alpha, rho2, t)
    mu1 = rho1 * np.exp(-z0)
    mu2 = rho2 * np.exp(-z0)
    return {
        "z0": z0,
        "lo_c0": mu1 * (1.0 + z0),
        "lo_c1": -mu1,
        "lo_s": np.sqrt(mu1),
        "lo_w": np.sqrt(mu1) * z0,
        "up_c0": mu2 * (1.0 + z0),
        "up_c1": -mu2,
        # (36): z0(t) <= z <= ln(m0 - alpha rho1 t)
        "z_lo": z0,
        "z_hi": z0_profile(m0, alpha, rho1, t),
    }
