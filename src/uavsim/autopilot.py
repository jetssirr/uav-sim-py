"""
Cascaded PID autopilot for fixed-wing UAV. Seven loops:

  outer (slow):
    1. altitude    h    -> theta_cmd
    2. heading     psi  -> phi_cmd
    3. airspeed    Va   -> throttle increment

  inner (fast):
    4. pitch       theta -> elevator
    5. roll        phi   -> aileron
    6. sideslip    beta  -> rudder
    7. yaw damper  r     -> rudder (washed-out)

Inputs are commanded altitude / heading / airspeed (constants or callables of
t). Outputs are the four control commands [de, da, dr, dt] passed downstream
to the actuator model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .aircraft import Aircraft
from .controllers import PID, Washout, wrap_pi
from .dynamics import unpack_state
from .quaternion import quat_to_euler
from .trim import TrimState

CmdLike = float | Callable[[float], float]


def _as_callable(c: CmdLike) -> Callable[[float], float]:
    if callable(c):
        return c
    val = float(c)
    return lambda t, val=val: val


@dataclass
class APGains:
    # 1. altitude (m -> rad). e>0 (too low) wants pitch up so theta_cmd>0; Kp>0.
    #    Small Ki with strong back-calculation removes the steady-state offset
    #    that appears when Va changes (trim feedforward is for one Va only).
    #    Kbc much larger than 1/Kp so the integrator unwinds aggressively while
    #    saturated on large steps.
    alt:    PID = field(default_factory=lambda: PID(
        Kp=0.018, Ki=0.0008, Kd=0.012, umin=-np.deg2rad(10), umax=np.deg2rad(10),
        Kbc=100.0, prefilter_tau=3.0, tau_d=0.8))
    # 2. heading (rad -> rad). e>0 (need to turn right) wants φ_cmd>0; Kp>0.
    hdg:    PID = field(default_factory=lambda: PID(
        Kp=0.7, Ki=0.0, Kd=0.0, umin=-np.deg2rad(30), umax=np.deg2rad(30),
        Kbc=1.5, prefilter_tau=2.0))
    # 3. airspeed (m/s -> throttle increment). e>0 too slow so dt up; Kp>0.
    spd:    PID = field(default_factory=lambda: PID(
        Kp=0.10, Ki=0.025, Kd=0.0, umin=-1.0, umax=1.0, Kbc=10.0,
        prefilter_tau=2.0, tau_d=0.05))
    # 4. pitch (rad -> rad elev). e>0 needs nose up so de<0 -> Kp<0; q>0
    #    (nose-up rate) needs de>0 so Kd<0 (since output -= Kd*q).
    pitch:  PID = field(default_factory=lambda: PID(
        Kp=-2.0, Ki=-0.3, Kd=-0.5, umin=None, umax=None, Kbc=0.5, tau_d=0.02))
    # 5. roll (rad -> rad ail). With Cl_δa<0 (Nelson), +δa rolls left.
    #    e>0 needs roll right so δa<0 -> Kp<0; p>0 (rolling right) wants δa>0 to
    #    damp so output -= Kd*p with Kd<0.
    roll:   PID = field(default_factory=lambda: PID(
        Kp=-0.6, Ki=-0.03, Kd=-0.10, umin=None, umax=None, Kbc=2.0, tau_d=0.02))
    # 6. sideslip (rad -> rad rud). meas=β, sp=0 so e=-β. β>0 wants dr<0 (rudder
    #    right). For e<0 (when β>0), output<0 needs Kp>0.
    beta:   PID = field(default_factory=lambda: PID(
        Kp=2.0, Ki=0.5, Kd=0.0, umin=None, umax=None, Kbc=0.5))
    # 7. yaw damper: washed-out r -> rud. r>0 (yawing right) wants dr>0 to damp.
    yawdamp_K:   float = 1.5
    yawdamp_tau: float = 1.0


@dataclass
class CommandSchedule:
    """Time-only commands. Ignores state. Use as the simplest command source."""
    altitude: Callable[[float], float]
    heading:  Callable[[float], float]
    airspeed: Callable[[float], float]

    @classmethod
    def from_dict(cls, d: Mapping[str, CmdLike]) -> "CommandSchedule":
        return cls(
            altitude=_as_callable(d["altitude"]),
            heading =_as_callable(d["heading"]),
            airspeed=_as_callable(d["airspeed"]),
        )

    def update(self, t: float, x: np.ndarray) -> tuple[float, float, float]:
        return self.altitude(t), self.heading(t), self.airspeed(t)

    def reset(self) -> None:
        pass


class Autopilot:
    """7-loop cascaded PID controller. Stateful: integrators + washout filter."""

    def __init__(self, ac: Aircraft, trim_state: TrimState,
                 commands: CommandSchedule, gains: APGains | None = None):
        self.ac = ac
        self.trim = trim_state
        self.cmd = commands
        self.g = gains if gains is not None else APGains()
        self.washout = Washout(tau=self.g.yawdamp_tau)

        # Inner-loop limits set from aircraft control range.
        self.g.pitch.umin = ac.de_min - trim_state.u[0]
        self.g.pitch.umax = ac.de_max - trim_state.u[0]
        self.g.roll.umin  = ac.da_min
        self.g.roll.umax  = ac.da_max
        self.g.beta.umin  = ac.dr_min * 0.7   # leave room for yaw damper
        self.g.beta.umax  = ac.dr_max * 0.7

        self._theta_cmd_last = trim_state.theta
        self._phi_cmd_last   = 0.0

    def reset(self) -> None:
        for p in (self.g.alt, self.g.hdg, self.g.spd,
                  self.g.pitch, self.g.roll, self.g.beta):
            p.reset()
        self.washout.reset()
        if hasattr(self.cmd, "reset"):
            self.cmd.reset()
        self._theta_cmd_last = self.trim.theta
        self._phi_cmd_last   = 0.0

    def step(self, t: float, x: np.ndarray, dt: float) -> np.ndarray:
        pos, V_body, q, omega = unpack_state(x)
        phi, theta, psi = quat_to_euler(q)
        Va = float(np.linalg.norm(V_body))
        beta = float(np.arcsin(np.clip(V_body[1] / max(Va, 1e-3), -1.0, 1.0)))
        h = -pos[2]
        p, qrate, r = omega

        h_cmd, psi_cmd, Va_cmd = self.cmd.update(t, x)

        # outer loops
        # alt loop produces a pitch *increment* about trim theta. trim theta is
        # the level-flight pitch, so this gives correct steady-state with no
        # integrator wind-up risk on saturated climbs.
        theta_inc = self.g.alt.update(h_cmd, h, dt)
        theta_cmd = self.trim.theta + theta_inc
        # heading: feed wrapped error as setpoint=0, meas=-wrap_pi(...)
        psi_err = wrap_pi(psi_cmd - psi)
        phi_cmd = self.g.hdg.update(psi_err, 0.0, dt)
        dt_inc  = self.g.spd.update(Va_cmd, Va, dt)

        # inner loops (note pitch error uses theta_cmd directly, not theta-trim:
        # the loop output is *increment* about trim elevator)
        de_inc = self.g.pitch.update(theta_cmd, theta, dt)
        da_cmd = self.g.roll.update(phi_cmd, phi, dt)
        dr_beta = self.g.beta.update(0.0, beta, dt)
        dr_yaw  = self.g.yawdamp_K * self.washout.update(r, dt)

        de_cmd = self.trim.u[0] + de_inc
        dr_cmd = float(np.clip(dr_beta + dr_yaw, self.ac.dr_min, self.ac.dr_max))
        dt_cmd = float(np.clip(self.trim.u[3] + dt_inc, self.ac.dt_min, self.ac.dt_max))

        self._theta_cmd_last = theta_cmd
        self._phi_cmd_last   = phi_cmd
        return np.array([de_cmd, da_cmd, dr_cmd, dt_cmd])
