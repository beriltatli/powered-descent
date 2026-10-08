import numpy as np

from dynamics.model import load_config, vehicle_from_config
from problem.setup import make_problem
from tof.search import algorithm1, tf_bounds
from tof.sweep import unimodality

CFG = load_config()
VEH = vehicle_from_config(CFG)
P = make_problem(CFG, N=30)


def test_tf_bounds_eq55():
    lo, hi = tf_bounds(VEH, np.array([-75.0, 0, 100]), VEH.m_wet, VEH.m_fuel)
    assert np.isclose(lo, VEH.m_dry * 125.0 / VEH.rho2)
    assert np.isclose(hi, VEH.m_fuel / (VEH.alpha * VEH.rho1))


def test_insufficient_thrust_is_classified():
    # 150 m up, falling at 110 m/s: max deceleration rho2/m - g ~ 3.2 m/s^2 needs ~1.9 km to stop.
    res = algorithm1(P, [150.0, 0, 0], [-110.0, 0, 0], VEH.m_wet, VEH.m_fuel)
    assert res.classification == "insufficient_thrust"
    assert res.sol is None and all(s == "infeasible" for _, _, s in res.probes)


def test_insufficient_fuel_is_classified():
    # The paper's own Fig. 7 case needs ~403 kg at N = 50 against 400 kg on board (results/convergence.json).
    sc = CFG["scenario"]
    res = algorithm1(make_problem(CFG, N=50), sc["r0"], sc["v0"], VEH.m_wet, VEH.m_fuel)
    assert res.classification == "insufficient_fuel"
    assert np.isfinite(res.beta) and res.fuel > VEH.m_fuel


def test_valid_case():
    res = algorithm1(P, [1500.0, 0, 500.0], [-50.0, 0, -20.0], VEH.m_wet, VEH.m_fuel)
    assert res.classification == "valid" and res.fuel < VEH.m_fuel


def test_initial_state_violating_glideslope_is_its_own_class():
    res = algorithm1(P, [10.0, 0, 1000.0], [0.0, 0, 0], VEH.m_wet, VEH.m_fuel)
    assert res.classification == "initial_state_infeasible"


def test_unimodality_detector():
    tf = np.arange(10.0)
    assert unimodality(tf, np.array([np.inf, 5, 3, 2, 2.5, 4, 6, 7, 8, 9]))["unimodal"]
    assert not unimodality(tf, np.array([np.inf, 5, 3, 2, 2.5, 2.2, 6, 7, 8, 9]))["unimodal"]
    assert not unimodality(tf, np.array([5, 3, np.inf, 2, 2.5, 4, 6, 7, 8, 9]))["unimodal"]
