"""
Aerodynamic forces and moments from stability derivatives.

State conventions:
  V_body = [u, v, w]   body-frame velocity (m/s) relative to air
  omega  = [p, q, r]   body-frame angular rate (rad/s)
  delta  = [de, da, dr] elevator, aileron, rudder (rad)

Wind-frame angles:
  Va    = |V_body|              true airspeed
  alpha = atan2(w, u)            angle of attack
  beta  = asin(v / Va)           sideslip
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import Aircraft


@dataclass
class AeroOutput:
    F_body:  np.ndarray   # aerodynamic force in body frame (N)
    M_body:  np.ndarray   # aerodynamic moment in body frame (N m)
    Va:      float
    alpha:   float
    beta:    float
    qbar:    float        # dynamic pressure (Pa)


def airspeed_angles(V_body: np.ndarray) -> tuple[float, float, float]:
    u, v, w = V_body
    Va = float(np.linalg.norm(V_body))
    if Va < 1e-3:
        return Va, 0.0, 0.0
    alpha = float(np.arctan2(w, u))
    beta  = float(np.arcsin(np.clip(v / Va, -1.0, 1.0)))
    return Va, alpha, beta


def aero_forces_moments(
    ac: Aircraft,
    V_body: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
) -> AeroOutput:
    """Return body-frame aerodynamic force + moment."""
    Va, alpha, beta = airspeed_angles(V_body)
    if Va < 1e-3:
        return AeroOutput(
            F_body=np.zeros(3), M_body=np.zeros(3),
            Va=Va, alpha=alpha, beta=beta, qbar=0.0,
        )

    p, q, r = omega
    de, da, dr = delta

    qbar = 0.5 * ac.rho * Va * Va
    qS   = qbar * ac.S
    bhat = ac.b / (2.0 * Va)
    chat = ac.c / (2.0 * Va)

    # Longitudinal coefficients (alpha-dot effect omitted: small at trim,
    # and including it requires alpha_dot which we don't track explicitly).
    CL = ac.CL0 + ac.CL_a * alpha + ac.CL_q * q * chat + ac.CL_de * de
    CD = ac.CD0 + ac.CD_a * alpha + ac.CD_de * de
    Cm = ac.Cm0 + ac.Cm_a * alpha + ac.Cm_q * q * chat + ac.Cm_de * de

    # Lateral / directional coefficients
    CY = ac.CY_b * beta + ac.CY_p * p * bhat + ac.CY_r * r * bhat \
       + ac.CY_da * da + ac.CY_dr * dr
    Cl = ac.Cl_b * beta + ac.Cl_p * p * bhat + ac.Cl_r * r * bhat \
       + ac.Cl_da * da + ac.Cl_dr * dr
    Cn = ac.Cn_b * beta + ac.Cn_p * p * bhat + ac.Cn_r * r * bhat \
       + ac.Cn_da * da + ac.Cn_dr * dr

    # Stability-frame forces -> body frame via rotation by alpha about y_body
    # Drag along -x_stab, Lift along -z_stab; side force along y_body.
    L = qS * CL
    D = qS * CD
    Y = qS * CY
    ca, sa = np.cos(alpha), np.sin(alpha)
    Fx_body = -ca * D + sa * L
    Fz_body = -sa * D - ca * L
    Fy_body =  Y

    # Moments: rolling, pitching, yawing about body axes.
    M_x = qS * ac.b * Cl
    M_y = qS * ac.c * Cm
    M_z = qS * ac.b * Cn

    return AeroOutput(
        F_body=np.array([Fx_body, Fy_body, Fz_body]),
        M_body=np.array([M_x, M_y, M_z]),
        Va=Va, alpha=alpha, beta=beta, qbar=qbar,
    )


def thrust_body(ac: Aircraft, throttle: float) -> np.ndarray:
    """Thrust along +x_body, scaled by throttle in [0,1]."""
    t = float(np.clip(throttle, ac.dt_min, ac.dt_max))
    return np.array([ac.T_max * t, 0.0, 0.0])
