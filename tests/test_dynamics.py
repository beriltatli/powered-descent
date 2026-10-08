import numpy as np
from scipy.integrate import solve_ivp

from dynamics.model import (
    continuous_rhs, discretize, interval_states, load_config, propagate, vehicle_from_config,
)

CFG = load_config()
VEH = vehicle_from_config(CFG)


def test_vehicle_constants_match_paper_example():
    # rho = 6 * {0.3, 0.8} * 3100 * cos 27deg; alpha = 1/(225 * 9.807 * cos 27deg)
    assert np.isclose(VEH.rho1, 4971.5, atol=1.0)
    assert np.isclose(VEH.rho2, 13257.5, atol=1.0)
    assert np.isclose(VEH.alpha, 5.0863e-4, rtol=1e-3)


def test_zoh_matches_solve_ivp_to_1e9():
    rng = np.random.default_rng(0)
    N, dt = 12, 1.7
    w = np.column_stack([rng.uniform(-4, 4, (N, 3)), rng.uniform(2, 6, N)])
    y0 = np.array([1500.0, 10.0, 2000.0, -75.0, 3.0, 100.0, np.log(1905.0)])
    Y = propagate(y0, w, dt, VEH)
    t_nodes = dt * np.arange(N + 1)

    def w_of_t(t):
        return w[min(int(t // dt), N - 1)]

    sol = solve_ivp(lambda t, y: continuous_rhs(t, y, w_of_t, VEH), (0, t_nodes[-1]), y0,
                    t_eval=t_nodes, rtol=1e-13, atol=1e-12, method="DOP853",
                    max_step=dt / 4, first_step=1e-3)
    # max_step keeps the integrator from stepping across a ZOH discontinuity unseen.
    err = np.max(np.abs(sol.y.T - Y) / np.maximum(1.0, np.abs(Y)))
    assert err < 1e-9, err


def test_ballistic_analytic():
    # u = 0, sigma = 0: free fall from rest-relative state; r(t) = r0 + v0 t + g t^2 / 2.
    y0 = np.array([1000.0, 0.0, 0.0, 5.0, 1.0, -2.0, np.log(1800.0)])
    N, dt = 20, 0.5
    Y = propagate(y0, np.zeros((N, 4)), dt, VEH)
    t = dt * N
    r_exact = y0[:3] + y0[3:6] * t + 0.5 * VEH.g * t**2
    v_exact = y0[3:6] + VEH.g * t
    assert np.allclose(Y[-1, :3], r_exact, atol=1e-9, rtol=0)
    assert np.allclose(Y[-1, 3:6], v_exact, atol=1e-12, rtol=0)
    assert Y[-1, 6] == y0[6]


def test_B_is_not_euler():
    A, B = discretize(2.0, VEH.alpha)
    # Exact position-from-acceleration gain is dt^2/2; Euler's B = Bc dt would give 0.
    assert np.isclose(B[0, 0], 2.0, rtol=1e-14)
    assert np.isclose(B[3, 0], 2.0, rtol=1e-14)
    assert np.isclose(B[6, 3], -VEH.alpha * 2.0, rtol=1e-14)
    assert np.isclose(A[0, 3], 2.0, rtol=1e-14)


def test_interval_states_closed_form():
    y0 = np.array([100.0, 1.0, 2.0, -3.0, 0.5, 4.0, np.log(1700.0)])
    wk = np.array([1.0, -0.5, 2.0, 3.0])
    Y = propagate(y0, wk[None, :], 0.8, VEH)
    assert np.allclose(interval_states(y0, wk, VEH, [0.8])[0], Y[1], atol=1e-11)
