"""
6-DOF rigid-body equations of motion with quaternion attitude.

State vector x (13):
   0:3   p_ned   = [N, E, D]      position in NED (m)
   3:6   V_body  = [u, v, w]      body-frame velocity (m/s)
   6:10  q       = [qw,qx,qy,qz]  body-to-NED quaternion
  10:13  omega   = [p, q, r]      body-frame angular rate (rad/s)

Control vector u (4):
   [de, da, dr, dt]
"""
from __future__ import annotations

import numpy as np

from .aero import aero_forces_moments, thrust_body
from .aircraft import Aircraft
from .quaternion import quat_kinematics, quat_normalize, quat_to_rot


STATE_DIM = 13
CTRL_DIM  = 4


def pack_state(p_ned, V_body, q, omega) -> np.ndarray:
    return np.concatenate([
        np.asarray(p_ned, dtype=float),
        np.asarray(V_body, dtype=float),
        np.asarray(q, dtype=float),
        np.asarray(omega, dtype=float),
    ])


def unpack_state(x: np.ndarray):
    p_ned  = x[0:3]
    V_body = x[3:6]
    q      = x[6:10]
    omega  = x[10:13]
    return p_ned, V_body, q, omega


def _inertia_inv(ac: Aircraft) -> np.ndarray:
    I = np.array([
        [ac.Ix,   0.0,   -ac.Ixz],
        [0.0,     ac.Iy,  0.0   ],
        [-ac.Ixz, 0.0,    ac.Iz ],
    ])
    return np.linalg.inv(I)


def state_derivative(
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    ac: Aircraft,
    wind_ned: np.ndarray | None = None,
) -> np.ndarray:
    p_ned, V_body, q, omega = unpack_state(x)
    q = quat_normalize(q)
    R_b2n = quat_to_rot(q)

    if wind_ned is None:
        V_air_body = V_body
    else:
        V_air_body = V_body - R_b2n.T @ np.asarray(wind_ned, dtype=float)

    delta = np.array(u[:3], dtype=float)
    throttle = float(u[3])

    aero = aero_forces_moments(ac, V_air_body, omega, delta)
    F_thrust = thrust_body(ac, throttle)
    F_aero   = aero.F_body
    M_aero   = aero.M_body

    # Gravity: rotate NED [0,0,g] into body frame.
    g_body = R_b2n.T @ np.array([0.0, 0.0, ac.g])
    F_grav = ac.mass * g_body

    F_total = F_aero + F_thrust + F_grav

    # Translational: m * (Vdot + omega x V) = F_total
    Vdot = F_total / ac.mass - np.cross(omega, V_body)

    # Rotational: I * omegadot + omega x (I * omega) = M
    I = np.array([
        [ac.Ix,   0.0,   -ac.Ixz],
        [0.0,     ac.Iy,  0.0   ],
        [-ac.Ixz, 0.0,    ac.Iz ],
    ])
    omega_dot = np.linalg.solve(I, M_aero - np.cross(omega, I @ omega))

    pdot_ned = R_b2n @ V_body
    qdot     = quat_kinematics(q, omega)

    return np.concatenate([pdot_ned, Vdot, qdot, omega_dot])
