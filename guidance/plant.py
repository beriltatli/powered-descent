from dataclasses import dataclass

import numpy as np

from dynamics.model import Vehicle


@dataclass
class PlantState:
    t: float
    r: np.ndarray
    v: np.ndarray
    m: float


def thrust_command(u: np.ndarray, m_believed: float, veh: Vehicle) -> tuple[np.ndarray, bool]:
    """Guidance outputs an acceleration u = Tc/m (25); the engine is commanded Tc = u m-hat with the
    guidance's mass belief. The engine can only produce rho1 <= ||Tc|| <= rho2 (3), so the command is
    saturated; the flag records whether saturation acted (beyond 0.1 %)."""
    T = u * m_believed
    n = np.linalg.norm(T)
    if n < 1e-9:
        # The engine cannot go below rho1 (3) and u gives no direction: fire along +altitude.
        return np.array([veh.rho1, 0.0, 0.0]), True
    clipped = np.clip(n, veh.rho1, veh.rho2)
    return T * (clipped / n), abs(clipped - n) > 1e-3 * n


def step_rk4(s: PlantState, T: np.ndarray, dt: float, veh: Vehicle) -> PlantState:
    """True dynamics (1)-(2) with constant commanded thrust over dt. Fuel exhaustion: below m_dry the
    engine produces nothing (the step is split so the last bit of fuel is used exactly)."""
    if s.m <= veh.m_dry + 1e-9:
        T = np.zeros(3)
    mdot = -veh.alpha * np.linalg.norm(T)
    if mdot < 0 and s.m + mdot * dt < veh.m_dry:
        dt1 = (veh.m_dry - s.m) / mdot
        mid = step_rk4(s, T, dt1, veh)
        mid.m = veh.m_dry
        return step_rk4(mid, np.zeros(3), dt - dt1, veh)

    def f(y):
        return np.concatenate([y[3:6], T / y[6] + veh.g, [mdot]])

    y = np.concatenate([s.r, s.v, [s.m]])
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return PlantState(s.t + dt, y[:3], y[3:6], float(y[6]))
