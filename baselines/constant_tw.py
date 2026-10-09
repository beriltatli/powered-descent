"""Constant thrust-to-weight gravity turn: no targeting and no optimization of the trajectory.

Thrust points against the velocity vector with a fixed magnitude TW * m-hat * |g|. The single scalar TW
is calibrated once, on the error-free plant, so that the vehicle comes to rest at zero altitude. That is
the only tuning; the miss distance is whatever the gravity turn produces. It calibrates the fuel scale:
roughly the least fuel a soft touchdown needs when landing position is ignored.
"""
from dataclasses import dataclass

import numpy as np

from dynamics.model import Vehicle
from guidance.plant import PlantState, step_rk4


@dataclass
class TWResult:
    status: str
    r_end: np.ndarray
    v_end: np.ndarray
    fuel_used: float
    tw: float
    t_end: float

    @property
    def landing_error(self):
        return float(np.linalg.norm(self.r_end))

    @property
    def horizontal_miss(self):
        return float(np.linalg.norm(self.r_end[1:3]))

    @property
    def touchdown_speed(self):
        return float(np.linalg.norm(self.v_end))


def fly_constant_tw(veh: Vehicle, r0, v0, m0: float, eps: float, tw: float, sim_dt: float = 0.02,
                    t_max: float = 400.0, v_stop: float = 0.05) -> TWResult:
    s = PlantState(0.0, np.asarray(r0, float), np.asarray(v0, float), m0)
    gmag = np.linalg.norm(veh.g)
    while s.t < t_max:
        sp = np.linalg.norm(s.v)
        if sp < v_stop:
            return TWResult("stopped_above_ground", s.r, s.v, m0 - s.m, tw, s.t)
        mag = np.clip(tw * (1.0 + eps) * s.m * gmag, veh.rho1, veh.rho2)
        T = -s.v / sp * mag
        # Shorten the step near rest so the gravity turn's direction singularity is not stepped across.
        dt = min(sim_dt, 0.25 * sp / (mag / s.m + gmag))
        new = step_rk4(s, T, dt, veh)
        if new.r[0] < 0:
            a = s.r[0] / (s.r[0] - new.r[0])
            return TWResult("ground_contact", s.r + a * (new.r - s.r), s.v + a * (new.v - s.v),
                            m0 - (s.m + a * (new.m - s.m)), tw, s.t + a * dt)
        if new.v[0] > 0 and new.r[0] > 0 and np.linalg.norm(new.v) > np.linalg.norm(s.v):
            # Climbing and speeding up: thrust exceeds what is needed to stop; record and stop.
            return TWResult("stopped_above_ground", s.r, s.v, m0 - s.m, tw, s.t)
        s = new
    return TWResult("timeout", s.r, s.v, m0 - s.m, tw, s.t)


def calibrate_tw(veh: Vehicle, r0, v0, m0: float, lo: float = 0.5, hi: float = 4.0, iters: int = 50) -> float:
    """Bisection on TW: too low hits the ground moving, too high stops above it."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = fly_constant_tw(veh, r0, v0, m0, 0.0, mid)
        if r.status == "ground_contact":
            lo = mid
        else:
            hi = mid
    return hi
