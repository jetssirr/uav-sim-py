"""
PID controller with derivative-on-measurement, output saturation,
back-calculation anti-windup, and a first-order setpoint prefilter.

Sign convention: error e = setpoint - measurement. Choose the sign of the
gains so that a positive error produces an output that drives the plant
toward the setpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PID:
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    umin: float = -np.inf
    umax: float =  np.inf
    Kbc: float = 0.0          # back-calculation anti-windup gain (1/Tt)
    tau_d: float = 0.0        # derivative low-pass time constant (s)
    prefilter_tau: float = 0.0  # first-order setpoint prefilter (s)

    integral: float = 0.0
    _prev_meas: float | None = field(default=None, init=False, repr=False)
    _dmeas_filt: float = field(default=0.0, init=False, repr=False)
    _sp_filt: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_meas = None
        self._dmeas_filt = 0.0
        self._sp_filt = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if self.prefilter_tau > 0.0:
            if self._sp_filt is None:
                self._sp_filt = setpoint
            a = dt / (self.prefilter_tau + dt)
            self._sp_filt += a * (setpoint - self._sp_filt)
            sp = self._sp_filt
        else:
            sp = setpoint

        if self._prev_meas is None:
            self._prev_meas = measurement
        dmeas_raw = (measurement - self._prev_meas) / dt if dt > 0 else 0.0
        if self.tau_d > 0.0:
            a = dt / (self.tau_d + dt)
            self._dmeas_filt += a * (dmeas_raw - self._dmeas_filt)
            dmeas = self._dmeas_filt
        else:
            dmeas = dmeas_raw
        self._prev_meas = measurement

        e = sp - measurement
        u_unsat = self.Kp * e + self.Ki * self.integral - self.Kd * dmeas
        u_sat = float(np.clip(u_unsat, self.umin, self.umax))

        # standard back-calculation anti-windup: residual subtracts from
        # the error driving the integrator.
        if self.Ki != 0.0:
            self.integral += dt * (e + self.Kbc * (u_sat - u_unsat))
        return u_sat


@dataclass
class Washout:
    """First-order high-pass: y/u = (tau s) / (tau s + 1). Used as yaw damper."""
    tau: float = 1.0
    _y: float = 0.0
    _u_prev: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._y = 0.0
        self._u_prev = None

    def update(self, u: float, dt: float) -> float:
        if self._u_prev is None:
            self._u_prev = u
        # backward-Euler discretization of tau*y' + y = tau*u'
        a = self.tau / (self.tau + dt)
        self._y = a * (self._y + (u - self._u_prev))
        self._u_prev = u
        return self._y


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, +pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
