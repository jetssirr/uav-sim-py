"""
Small fixed-wing UAV parameters using Cessna-172-class nondimensional
stability derivatives (Nelson, "Flight Stability and Automatic Control",
2nd ed., Appendix B and Table B.1), scaled to a small ~4.5 kg UAV airframe.

Sign conventions follow Nelson Table B.1 directly:
  elevator δe + : pitches nose down  (Cm_δe < 0, CL_δe > 0)
  aileron  δa + : rolls left          (Cl_δa < 0)
  rudder   δr + : yaws nose left      (Cn_δr < 0)
Throttle dt in [0, 1] scales max thrust.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Aircraft:
    # Mass / geometry (small fixed-wing UAV)
    mass: float = 4.5            # kg
    S:    float = 0.55           # wing area, m^2
    b:    float = 2.90           # span, m
    c:    float = 0.19           # mean aerodynamic chord, m

    # Inertias about body axes (kg m^2). Ixz coupling kept for completeness.
    Ix:   float = 0.824
    Iy:   float = 1.135
    Iz:   float = 1.759
    Ixz:  float = 0.120

    # Propulsion
    T_max: float = 15.0          # N at sea level, full throttle

    # Environment
    rho:   float = 1.225         # kg/m^3 at sea level
    g:     float = 9.80665

    # Control deflection limits (rad) and throttle limits
    de_min: float = -np.deg2rad(25);  de_max: float =  np.deg2rad(25)
    da_min: float = -np.deg2rad(20);  da_max: float =  np.deg2rad(20)
    dr_min: float = -np.deg2rad(25);  dr_max: float =  np.deg2rad(25)
    dt_min: float = 0.0;              dt_max: float = 1.0

    # Trim reference (used as initial guess and nominal cruise)
    U0:   float = 18.0           # m/s
    alt0: float = 100.0          # m

    # ---- Longitudinal stability derivatives (Nelson Table B.1) -----------
    CL0:    float =  0.28
    CL_a:   float =  4.44
    CL_q:   float =  3.80
    CL_de:  float =  0.355
    CL_adot: float =  1.70

    CD0:    float =  0.036
    CD_a:   float =  0.121       # linearized about trim AoA
    CD_de:  float =  0.0

    Cm0:    float =  0.045       # positive: cambered airfoil zero-α moment
    Cm_a:   float = -0.613       # must be <0 for static stability
    Cm_q:   float = -12.4
    Cm_de:  float = -1.122
    Cm_adot: float = -7.27

    # ---- Lateral / directional stability derivatives ---------------------
    CY_b:   float = -0.31
    CY_p:   float = -0.037
    CY_r:   float =  0.21
    CY_da:  float =  0.0
    CY_dr:  float =  0.187

    Cl_b:   float = -0.089
    Cl_p:   float = -0.47
    Cl_r:   float =  0.096
    Cl_da:  float = -0.178       # Nelson sign convention: +δa rolls left
    Cl_dr:  float =  0.0147

    Cn_b:   float =  0.065       # weathervane stability (must be >0)
    Cn_p:   float = -0.030
    Cn_r:   float = -0.099
    Cn_da:  float = -0.053       # adverse yaw
    Cn_dr:  float = -0.0657

    # State / control limits used for sanity checks (rad)
    alpha_max: float = np.deg2rad(15.0)
    beta_max:  float = np.deg2rad(15.0)


def default_uav() -> Aircraft:
    return Aircraft()
