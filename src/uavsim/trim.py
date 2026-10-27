"""
Wings-level, straight, constant-altitude trim.

Free variables: alpha, elevator de, throttle dt.
Constraints: V_body_dot = 0 in x and z, pitching moment = 0,
with phi = 0, psi = 0, theta = alpha (gamma = 0), p = q = r = 0,
beta = 0, ailerons & rudder = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from .aero import aero_forces_moments, thrust_body
from .aircraft import Aircraft
from .dynamics import pack_state
from .quaternion import quat_from_euler


@dataclass
class TrimState:
    x:      np.ndarray   # full 13-state
    u:      np.ndarray   # 4-control [de, da, dr, dt]
    Va:     float
    alt:    float
    alpha:  float
    theta:  float


def _residuals(z: np.ndarray, ac: Aircraft, V0: float) -> np.ndarray:
    alpha, de, dt = z
    V_body = np.array([V0 * np.cos(alpha), 0.0, V0 * np.sin(alpha)])
    omega  = np.zeros(3)
    delta  = np.array([de, 0.0, 0.0])

    aero = aero_forces_moments(ac, V_body, omega, delta)
    F_thrust = thrust_body(ac, dt)

    # Gravity in body frame, with theta = alpha (gamma = 0):
    theta = alpha
    g_body = np.array([
        -ac.g * np.sin(theta),
         0.0,
         ac.g * np.cos(theta),
    ])
    F_grav = ac.mass * g_body

    F = aero.F_body + F_thrust + F_grav
    M = aero.M_body
    return np.array([F[0], F[2], M[1]])


def trim(ac: Aircraft, Va: float | None = None, alt: float | None = None,
         z0: tuple[float, float, float] = (0.05, -0.05, 0.5)) -> TrimState:
    if Va  is None: Va  = ac.U0
    if alt is None: alt = ac.alt0

    z, info, ier, msg = fsolve(_residuals, z0, args=(ac, Va), full_output=True, xtol=1e-10)
    if ier != 1:
        raise RuntimeError(f"trim did not converge: {msg}")

    alpha, de, dt = z
    theta = alpha
    V_body = np.array([Va * np.cos(alpha), 0.0, Va * np.sin(alpha)])
    q = quat_from_euler(0.0, theta, 0.0)
    omega = np.zeros(3)
    p_ned = np.array([0.0, 0.0, -alt])   # NED z is down: altitude = -p_d
    x = pack_state(p_ned, V_body, q, omega)
    u = np.array([de, 0.0, 0.0, dt])
    return TrimState(x=x, u=u, Va=Va, alt=alt, alpha=alpha, theta=theta)
