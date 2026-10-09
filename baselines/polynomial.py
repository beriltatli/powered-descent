"""Apollo-style polynomial guidance (linear-acceleration / quadratic-velocity / cubic-position family).

With a linear total acceleration a(tau) = a0 + a1 tau that reaches r(tgo) = 0, rdot(tgo) = 0, solving the
two boundary conditions gives the commanded total acceleration
    a0 = -6 r / tgo^2 - 4 rdot / tgo,
re-evaluated every step (it is a feedback law). Thrust acceleration is a0 - g. No constraint of the
convex problem is known to it: not the thrust annulus (3), not the glide slope (11). The engine
saturates the command into [rho1, rho2]; how often it has to is the statistic of interest.
"""
from dataclasses import dataclass

import numpy as np

from dynamics.model import Vehicle
from guidance.plant import PlantState, step_rk4


@dataclass
class PolyResult:
    status: str
    r_end: np.ndarray
    v_end: np.ndarray
    fuel_used: float
    time_above_rho2: float
    time_below_rho1: float
    t_end: float
    glideslope_violation_m: float
    subsurface_depth_m: float = 0.0

    @property
    def landing_error(self):
        return float(np.linalg.norm(self.r_end))

    @property
    def touchdown_speed(self):
        return float(np.linalg.norm(self.v_end))


def fly_polynomial(veh: Vehicle, r0, v0, m0: float, eps: float, tgo0: float, t_hold: float = 0.25,
                   sim_dt: float = 0.02, glideslope_deg: float | None = 86.0,
                   ideal_actuator: bool = False) -> PolyResult:
    """ideal_actuator=True skips the [rho1, rho2] saturation and lets the plant fly below ground, so
    the law's own trajectory is measured (fuel, time outside the annulus, subsurface depth); with False
    the engine saturates the command and ground contact ends the flight."""
    s = PlantState(0.0, np.asarray(r0, float), np.asarray(v0, float), m0)
    above = below = 0.0
    gs = 0.0
    tan_gs = np.tan(np.deg2rad(glideslope_deg)) if glideslope_deg is not None else None
    a_th = None
    depth = 0.0
    while s.t < tgo0 - 1e-9:
        tgo = tgo0 - s.t
        if tgo > t_hold or a_th is None:
            # Below t_hold the 1/tgo^2 gain blows up; the last command is held.
            a_th = -6.0 * s.r / tgo**2 - 4.0 * s.v / tgo - veh.g
        T = a_th * (1.0 + eps) * s.m          # commanded with the believed mass
        n = np.linalg.norm(T)
        dt = min(sim_dt, tgo)
        if n > veh.rho2:
            above += dt
        elif n < veh.rho1:
            below += dt
        if not ideal_actuator:
            T = T * (np.clip(n, veh.rho1, veh.rho2) / max(n, 1e-12))
        new = step_rk4(s, T, dt, veh)
        if tan_gs is not None and tgo > 1.0:
            gs = max(gs, np.linalg.norm(new.r[1:3]) - tan_gs * new.r[0])
        depth = max(depth, -new.r[0]) if tgo > 1.0 else depth
        if new.r[0] < 0 and not ideal_actuator:
            a = s.r[0] / (s.r[0] - new.r[0])
            return PolyResult("ground_contact", s.r + a * (new.r - s.r), s.v + a * (new.v - s.v),
                              m0 - (s.m + a * (new.m - s.m)), above, below, s.t + a * dt, max(gs, 0.0), depth)
        s = new
    return PolyResult("landed_at_tf", s.r, s.v, m0 - s.m, above, below, s.t, max(gs, 0.0), depth)
