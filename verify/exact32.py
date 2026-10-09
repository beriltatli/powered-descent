import numpy as np

from dynamics.model import Vehicle
from problem.pdg import Solution


def exact32_margins(sol: Solution, veh: Vehicle, rho1=None, rho2=None) -> dict:
    """Exact (32): rho1 e^{-z} <= sigma <= rho2 e^{-z}. Margins are in thrust units (N), positive means
    satisfied: lower = sigma m - rho1, upper = rho2 - sigma m.

    At the node (z_k), which is where the paper imposes (51), and at the end of the interval (z_{k+1}).
    Inside a ZOH interval sigma is constant and m falls, so Gamma = sigma m decreases: the lower bound
    is hardest at the interval end, the upper bound at its start. The node check is what Lemma 2
    guarantees; the interval-end check measures what the node-only formulation leaves uncovered.
    """
    rho1 = veh.rho1 if rho1 is None else rho1
    rho2 = veh.rho2 if rho2 is None else rho2
    m_start = np.exp(sol.Y[:-1, 6])
    m_end = np.exp(sol.Y[1:, 6])
    lo_node = sol.S * m_start - rho1
    up_node = rho2 - sol.S * m_start
    lo_end = sol.S * m_end - rho1
    return {"min_lower_margin_node_N": float(lo_node.min()),
            "min_upper_margin_node_N": float(up_node.min()),
            "min_lower_margin_interval_end_N": float(lo_end.min()),
            "node_ok": bool(lo_node.min() >= -1e-6 * rho1 and up_node.min() >= -1e-6 * rho2)}
