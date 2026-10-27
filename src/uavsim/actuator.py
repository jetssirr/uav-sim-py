"""
First-order actuator dynamics with rate limiting.

For each control surface i:
  u_dot_i = clip( (u_cmd_i - u_i) / tau_i,  -rate_max_i, +rate_max_i )

The throttle uses the same form (first-order toward commanded value with a
slower response and a smaller rate limit, since real engines spool slowly).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import Aircraft


@dataclass
class ActuatorParams:
    # Time constants (s) for each channel: de, da, dr, dt
    tau:      np.ndarray = None
    rate_max: np.ndarray = None
    umin:     np.ndarray = None
    umax:     np.ndarray = None

    @classmethod
    def from_aircraft(cls, ac: Aircraft) -> "ActuatorParams":
        tau      = np.array([0.05, 0.05, 0.05, 0.30])
        rate_max = np.array([np.deg2rad(120),
                             np.deg2rad(120),
                             np.deg2rad(120),
                             1.0])
        umin = np.array([ac.de_min, ac.da_min, ac.dr_min, ac.dt_min])
        umax = np.array([ac.de_max, ac.da_max, ac.dr_max, ac.dt_max])
        return cls(tau=tau, rate_max=rate_max, umin=umin, umax=umax)


def actuator_derivative(u_act: np.ndarray, u_cmd: np.ndarray,
                        ap: ActuatorParams) -> np.ndarray:
    u_cmd_clipped = np.clip(u_cmd, ap.umin, ap.umax)
    raw = (u_cmd_clipped - u_act) / ap.tau
    return np.clip(raw, -ap.rate_max, ap.rate_max)
