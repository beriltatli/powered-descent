import numpy as np

from dynamics.model import load_config, vehicle_from_config
from nonconvex.nlp import NonconvexProblem1
from pointing.constraint import heuristic_gamma
from problem.setup import make_problem
from tof.search import algorithm1
from verify.slack import classify_loose, slack_gaps

CFG = load_config()
VEH = vehicle_from_config(CFG)


def test_heuristic_rule_is_a_no_op_at_gamma_zero():
    v = np.array([1.0, 0, 0])
    u = np.array([[-3.0, 0, 0], [0.0, 2.0, 0], [2.0, 1.0, 0]])
    g, k1, k2 = heuristic_gamma(u, v, 0.0)
    # kappa_2 is a norm, so kappa_2 >= gamma = 0 always holds and case 1 (exact) is always taken.
    assert np.all(g == 0.0)
    g45, _, _ = heuristic_gamma(u, v, np.cos(np.pi / 4))
    assert g45[0] == 0.0 and g45[2] == np.cos(np.pi / 4)   # node 0 relaxed to kappa_2 = 0; node 2 exact


def test_pointing_makes_slack_loose_on_section_vi_example():
    pc = CFG["pointing"]
    P = make_problem(CFG, pointing=True)
    s = P.solve(pc["r0"], pc["v0"], VEH.m_wet, 76.5, gamma=pc["gamma"])
    assert s.ok
    gaps = slack_gaps(s)
    assert gaps[:10].min() > 1.0          # u ~ 0 while sigma >= rho1 e^{-z}: the Section VI mechanism
    assert set(classify_loose(s, 1e-4).values()) & {"zero", "other"}
    off = P.solve(pc["r0"], pc["v0"], VEH.m_wet, 70.4, gamma=None)
    assert "reversal" in classify_loose(off, 1e-4).values()   # ZOH across the lambda2 zero crossing


def test_convex_optimum_is_a_local_optimum_of_problem1():
    P = make_problem(CFG, N=12)
    r0, v0 = [1200.0, 100.0, -600.0], [-40.0, 5.0, 20.0]
    res = algorithm1(P, r0, v0, VEH.m_wet, VEH.m_fuel)
    nlp = NonconvexProblem1(P.N, res.tf, VEH, P.opts, r0, v0, VEH.m_wet, "taylor")
    assert nlp.violation(res.sol.U.ravel()) < 1e-4
    warm = nlp.solve(res.sol.U.ravel())
    assert warm.feasible and warm.beta >= res.sol.cost * (1 - 1e-6)
