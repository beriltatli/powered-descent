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
