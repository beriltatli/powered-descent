# Frame (Açıkmeşe & Ploen 2007, Fig. 1): surface-fixed, origin at the target.
# Axis 0 is altitude, pointing opposite to gravity, so g = [-|g|, 0, 0]; axes 1, 2 are horizontal.
# The paper's r1, r2, r3 are r[0], r[1], r[2] here. Every module uses this frame.
#
# State for the convex problem is y = [r (3), rdot (3), z (1)] with z = ln m, eq. (30).
# Control is w = [u (3), sigma (1)] with u = Tc/m, sigma = Gamma/m, eq. (25).
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Vehicle:
    g: np.ndarray
    m_dry: float
    m_wet: float
    rho1: float
    rho2: float
    alpha: float

    @property
    def m_fuel(self) -> float:
        return self.m_wet - self.m_dry


def vehicle_from_config(cfg: dict) -> Vehicle:
    v = cfg["vehicle"]
    phi = np.deg2rad(v["cant_deg"])
    n, tbar = v["n_thrusters"], v["thrust_per_engine"]
    # rho_i = n T_i cos(phi), below eq. (3); alpha = 1/(Isp ge cos(phi)), below eq. (2).
    return Vehicle(
        g=np.asarray(cfg["planet"]["g"], float),
        m_dry=float(v["m_dry"]),
        m_wet=float(v["m_wet"]),
        rho1=n * v["throttle_min"] * tbar * np.cos(phi),
        rho2=n * v["throttle_max"] * tbar * np.cos(phi),
        alpha=1.0 / (v["isp"] * cfg["planet"]["ge"] * np.cos(phi)),
    )


def continuous_matrices(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    # Ac, Bc of Section IV, p.1360: rddot = u (+ g, added through B), zdot = -alpha sigma, eq. (31).
    Ac = np.zeros((7, 7))
    Ac[0:3, 3:6] = np.eye(3)
    Bc = np.zeros((7, 4))
    Bc[3:6, 0:3] = np.eye(3)
    Bc[6, 3] = -alpha
    return Ac, Bc


def discretize(dt: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    # Exact zero-order hold, p.1360: A = expm(Ac dt), B = int_0^dt expm(Ac s) ds Bc.
    # Both blocks come from one exponential of the augmented generator [[Ac, Bc], [0, 0]] dt
    # (Van Loan): its top-right block is exactly the integral, with no quadrature error.
    Ac, Bc = continuous_matrices(alpha)
    M = np.zeros((11, 11))
    M[:7, :7] = Ac
    M[:7, 7:] = Bc
    E = expm(M * dt)
    return E[:7, :7], E[:7, 7:]


def propagate(y0: np.ndarray, w: np.ndarray, dt: float, veh: Vehicle) -> np.ndarray:
    """Discrete propagation y_{k+1} = A y_k + B (w_k + [g; 0]); w has shape (N, 4). Returns (N+1, 7)."""
    A, B = discretize(dt, veh.alpha)
    gw = np.concatenate([veh.g, [0.0]])
    Y = np.empty((len(w) + 1, 7))
    Y[0] = y0
    for k, wk in enumerate(w):
        Y[k + 1] = A @ Y[k] + B @ (wk + gw)
    return Y


def continuous_rhs(t: float, y: np.ndarray, w_of_t, veh: Vehicle) -> np.ndarray:
    w = w_of_t(t)
    return np.concatenate([y[3:6], w[:3] + veh.g, [-veh.alpha * w[3]]])


def interval_states(yk: np.ndarray, wk: np.ndarray, veh: Vehicle, taus: np.ndarray) -> np.ndarray:
    """Closed-form continuous state inside one ZOH interval at offsets taus (the dynamics are a double
    integrator with constant input, so this is exact, not an approximation)."""
    a = wk[:3] + veh.g
    taus = np.asarray(taus)[:, None]
    r = yk[:3] + yk[3:6] * taus + 0.5 * a * taus**2
    v = yk[3:6] + a * taus
    z = yk[6] - veh.alpha * wk[3] * taus
    return np.hstack([r, v, z])
