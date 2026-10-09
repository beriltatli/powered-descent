import numpy as np

from problem.pdg import Solution


def slack_gaps(sol: Solution) -> np.ndarray:
    # sigma_k - ||u_k||, in m/s^2. Lemma 1 / Remark 5 say this is zero at the optimum.
    return sol.S - np.linalg.norm(sol.U, axis=1)


def slack_report(sol: Solution, tol: float) -> dict:
    """tol is in m/s^2. At the example's scale (sigma ~ 3-9 m/s^2) Clarabel's default 1e-8 relative
    tolerances put honest gaps around 1e-7; 1e-4 is three orders above that noise and four below the
    smallest physically meaningful thrust change (rho1/m_wet ~ 2.6 m/s^2). Every node above tol is
    listed, never averaged."""
    gap = slack_gaps(sol)
    loose = np.flatnonzero(gap > tol)
    return {"max_gap": float(gap.max()), "min_gap": float(gap.min()),
            "loose_nodes": loose.tolist(), "loose_gaps": gap[loose].tolist()}


def mass_tightness(sol: Solution, alpha: float) -> dict:
    """Constraint (13) is the equality m-dot = -alpha Gamma; what the relaxation loosens is the thrust,
    so the true mass law m-dot = -alpha ||Tc|| holds iff ||u|| = sigma. Fuel booked by the model
    (through sigma) vs fuel the engine would actually burn (through ||u||), both over the ZOH intervals
    with the mass at each interval start."""
    m = np.exp(sol.Y[:-1, 6])
    booked = alpha * sol.dt * np.sum(sol.S * m)
    burned = alpha * sol.dt * np.sum(np.linalg.norm(sol.U, axis=1) * m)
    return {"fuel_booked": float(booked), "fuel_from_thrust": float(burned),
            "difference_kg": float(booked - burned)}


def classify_loose(sol: Solution, tol: float, zero_accel: float = 1e-2, tight_gap: float = 1e-3) -> dict:
    """Why each loose node is loose.

    reversal  - u turns by more than 90 degrees across the node (angle between u_{k-1} and u_{k+1}). In
                continuous time Lemma 1 lets R2 = lambda2/m vanish at one instant, where the thrust may
                flip; ZOH averages the flip over a whole interval, shortening ||u|| below sigma;
    zero      - ||u_k|| below zero_accel m/s^2 with sigma_k > 0: the Section VI mechanism, where the
                Hamiltonian maximizer under a pointing cone is u = 0;
    tolerance - gap below tight_gap m/s^2 and none of the above: interior-point residual, confirmed for
                the one such case in results/lemma1_battery.json by re-solving at 1e-10 tolerances;
    other     - none of the above.
    """
    gap = slack_gaps(sol)
    n = np.linalg.norm(sol.U, axis=1)
    unit = sol.U / np.maximum(n[:, None], 1e-12)
    out = {}
    for k in np.flatnonzero(gap > tol):
        a, b = max(k - 1, 0), min(k + 1, len(n) - 1)
        if n[k] < zero_accel:
            r = "zero"
        elif a != b and unit[a] @ unit[b] < 0:
            r = "reversal"
        elif gap[k] < tight_gap:
            r = "tolerance"
        else:
            r = "other"
        out[int(k)] = r
    return out
