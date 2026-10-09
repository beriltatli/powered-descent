import dataclasses

import numpy as np
from scipy.integrate import solve_ivp

from dynamics.model import continuous_rhs, load_config, propagate, vehicle_from_config
from verify.lcvx_conditions import condition2, condition3, glideslope_force_balance

CFG = load_config()
VEH = vehicle_from_config(CFG)
W_TUT = np.deg2rad(np.asarray(CFG["lcvx_conditions"]["omega_tutorial_deg_s"], float))
UP = np.array([1.0, 0.0, 0.0])


def test_condition3_needs_rotation_off_the_cone_axis():
    # Malyuta et al. p.46: with n_hat vertical, Condition 3 holds iff w x n_hat != 0.
    assert condition3(np.zeros(3), UP, 75.0)["rank"] == 4
    assert condition3(1e-4 * UP, UP, 75.0)["rank"] == 4
    assert condition3(W_TUT, UP, 75.0)["holds"]


def test_condition2_is_sigma_tf_nonzero():
    assert condition2(1.0)["holds"] and not condition2(0.0)["holds"]


def test_glideslope_force_balance_fails_only_for_the_90deg_cone():
    gs = CFG["scenario"]["glideslope_deg"]
    a45, a90 = glideslope_force_balance(VEH, gs, 45.0), glideslope_force_balance(VEH, gs, 90.0)
    assert a45["a_holds"] and a45["b_holds"]
    assert a90["a_holds"] and not a90["b_holds"]


def test_rotating_zoh_matches_continuous_propagation():
    veh = dataclasses.replace(VEH, omega=100 * W_TUT)
    rng = np.random.default_rng(0)
    w = np.column_stack([rng.normal(0, 3, (5, 3)), rng.uniform(3, 6, 5)])
    dt, y0 = 2.0, np.array([1500.0, 100.0, -200.0, -50.0, 5.0, 20.0, np.log(VEH.m_wet)])
    Y = propagate(y0, w, dt, veh)
    sol = solve_ivp(continuous_rhs, (0, dt * len(w)), y0, args=(lambda t: w[min(int(t // dt), len(w) - 1)], veh),
                    rtol=1e-12, atol=1e-10, max_step=dt / 4, t_eval=dt * np.arange(len(w) + 1))
    assert np.max(np.abs(sol.y.T - Y)) < 1e-6
