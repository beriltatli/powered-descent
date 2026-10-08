import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynamics.model import continuous_rhs, load_config, vehicle_from_config
from problem.setup import make_problem
from problem.taylor import taylor_coefficients
from tof.search import algorithm1
from verify.exact32 import exact32_margins
from verify.internode import continuous_trajectory, internode_violation
from verify.slack import mass_tightness, slack_report

CFG = load_config()
VEH = vehicle_from_config(CFG)
SC = CFG["scenario"]


@pytest.fixture(scope="module")
def nominal():
    P = make_problem(CFG)
    res = algorithm1(P, SC["r0"], SC["v0"], VEH.m_wet, VEH.m_fuel)
    return P, res


def test_taylor_bounds_are_inside_exact_32():
    # Lemma 2: for z in [z0, zu] the (34) interval lies inside the exact (32) interval.
    t = np.linspace(0, 150, 60)
    c = taylor_coefficients(VEH.m_wet, VEH.alpha, VEH.rho1, VEH.rho2, t)
    for frac in np.linspace(0, 1, 11):
        z = c["z_lo"] + frac * (c["z_hi"] - c["z_lo"])
        lo = c["lo_c0"] + c["lo_c1"] * z + 0.5 * (c["lo_s"] * z - c["lo_w"]) ** 2
        up = c["up_c0"] + c["up_c1"] * z
        assert np.all(lo >= VEH.rho1 * np.exp(-z) * (1 - 1e-12))
        assert np.all(up <= VEH.rho2 * np.exp(-z) * (1 + 1e-12))


def test_dpp_compiles_once_and_resolves():
    P = make_problem(CFG, N=20)
    s1 = P.solve(SC["r0"], SC["v0"], VEH.m_wet, 85.0)
    s2 = P.solve(np.array(SC["r0"]) + [50, 0, -100], SC["v0"], VEH.m_wet, 86.0)
    assert P.prob.is_dpp()
    assert s1.extra["first_call"] and not s2.extra["first_call"]
    assert s1.ok and s2.ok and s1.cost != s2.cost


def test_nominal_solution_constraints(nominal):
    P, res = nominal
    s = res.sol
    assert s.ok
    assert np.allclose(s.Y[-1, :6], 0, atol=1e-5)
    tan_t = np.tan(np.deg2rad(SC["glideslope_deg"]))
    assert np.all(np.linalg.norm(s.Y[1:, 1:3], axis=1) <= tan_t * s.Y[1:, 0] + 1e-4)
    assert slack_report(s, CFG["solver"]["slack_tol"])["loose_nodes"] == []
    assert exact32_margins(s, VEH)["node_ok"]
    assert abs(mass_tightness(s, VEH.alpha)["difference_kg"]) < 1e-3


def test_reproduces_paper_fig7_at_fine_grid():
    # Fig. 7, p.1363: tf* = 81 s, 399.5 kg. The paper's Delta-t is not stated; at N = 200 (about 0.4 s)
    # this repo gets within 1 kg and 3.5 s.
    P = make_problem(CFG, N=200)
    res = algorithm1(P, SC["r0"], SC["v0"], VEH.m_wet, VEH.m_fuel, tol=0.1)
    assert abs(res.fuel - 399.5) < 1.0
    assert abs(res.tf - 81.0) < 3.5


def test_internode_closed_form_matches_solve_ivp(nominal):
    P, res = nominal
    s = res.sol
    t, Y = continuous_trajectory(s, VEH, per_interval=8)

    def w_of_t(tt):
        k = min(int(tt // s.dt), P.N - 1)
        return np.concatenate([s.U[k], [s.S[k]]])

    sol = solve_ivp(lambda tt, y: continuous_rhs(tt, y, w_of_t, VEH), (0, s.tf), s.Y[0], t_eval=t,
                    rtol=1e-12, atol=1e-10, max_step=s.dt / 4, method="DOP853")
    assert np.max(np.abs(sol.y.T - Y)[:, :6]) < 1e-5


def test_internode_violation_is_measured_not_zero(nominal):
    P, res = nominal
    v = internode_violation(res.sol, VEH, SC["glideslope_deg"], SC["vmax"])
    # The glide slope is touched at nodes, so between nodes the continuous path can poke through.
    assert v["glideslope_violation_m"] > 0
    assert v["glideslope_violation_m"] < 20
