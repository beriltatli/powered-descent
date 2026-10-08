import numpy as np

from baselines.polynomial import fly_polynomial
from dynamics.model import load_config, vehicle_from_config
from guidance.config import guidance_kwargs, guidance_problems
from guidance.mpc import fly
from guidance.plant import PlantState, step_rk4, thrust_command

CFG = load_config()
VEH = vehicle_from_config(CFG)


def test_thrust_command_saturates_into_annulus():
    T, sat = thrust_command(np.array([0.1, 0, 0]), 1800.0, VEH)
    assert sat and np.isclose(np.linalg.norm(T), VEH.rho1)
    T, sat = thrust_command(np.array([0, 20.0, 0]), 1800.0, VEH)
    assert sat and np.isclose(np.linalg.norm(T), VEH.rho2)


def test_plant_mass_flow_and_fuel_exhaustion():
    s = PlantState(0.0, np.array([1000.0, 0, 0]), np.zeros(3), VEH.m_dry + 1.0)
    T = np.array([VEH.rho2, 0, 0])
    s2 = step_rk4(s, T, 10.0, VEH)
    assert np.isclose(s2.m, VEH.m_dry)          # engine stops exactly at the dry mass
    s3 = step_rk4(PlantState(0.0, np.zeros(3), np.zeros(3), 1800.0), T, 1.0, VEH)
    assert np.isclose(s3.m, 1800.0 - VEH.alpha * VEH.rho2 * 1.0)


def test_closed_loop_lands_without_error():
    P, R = guidance_problems(CFG)
    kw = guidance_kwargs(CFG)
    kw["plan_mass_margin"] = 0.0
    f = fly(P, VEH, CFG["montecarlo"]["r0"], CFG["montecarlo"]["v0"], VEH.m_wet, 0.0, R=R, **kw)
    assert f.status == "landed_at_tf" and f.landing_error < 0.1 and f.touchdown_speed < 0.1
    assert f.recursively_feasible


def test_polynomial_ideal_actuator_hits_target():
    p = fly_polynomial(VEH, [1500.0, 0, 500.0], [-50.0, 0, -10.0], VEH.m_wet, 0.0, 60.0, ideal_actuator=True)
    assert p.landing_error < 0.05 and p.touchdown_speed < 0.05
