"""
Closed-loop simulation: hybrid integration of continuous-time plant +
continuous-time actuator + discrete-time autopilot.

Strategy:
  - State is augmented with the four actuator positions: dim 17.
  - Outer loop steps in fixed dt_ctrl intervals (controller sample period).
    At each step, the autopilot samples state, computes u_cmd, then we run
    solve_ivp from t to t+dt_ctrl with u_cmd held constant.
  - Inside the ODE, the actuator state slews toward u_cmd via first-order
    dynamics with rate limiting; the plant uses the *actuator* output, not
    the command.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .actuator import ActuatorParams, actuator_derivative
from .aircraft import Aircraft
from .autopilot import Autopilot
from .dynamics import state_derivative, unpack_state
from .quaternion import quat_to_euler
from .sim import SimResults
from .trim import TrimState


@dataclass
class ClosedLoopLog:
    res:     SimResults
    u_cmd:   np.ndarray   # (N, 4) controller commands
    u_act:   np.ndarray   # (N, 4) actuator-state-output (delivered to plant)


def simulate_closed_loop(
    ac: Aircraft,
    trim: TrimState,
    autopilot: Autopilot,
    t_end: float,
    dt_ctrl: float = 0.02,
    dt_log: float | None = None,
    wind=None,
    ap: ActuatorParams | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> ClosedLoopLog:
    if dt_log is None:
        dt_log = dt_ctrl
    if ap is None:
        ap = ActuatorParams.from_aircraft(ac)

    # Augmented state init: plant from trim, actuators at trim controls.
    x = np.concatenate([trim.x, trim.u.copy()])
    autopilot.reset()

    # Output buffers
    n_steps = int(np.round(t_end / dt_ctrl)) + 1
    t_log     = np.zeros(n_steps)
    x_log     = np.zeros((n_steps, 13))
    u_cmd_log = np.zeros((n_steps, 4))
    u_act_log = np.zeros((n_steps, 4))

    t = 0.0
    u_cmd = trim.u.copy()
    for k in range(n_steps):
        t_log[k]     = t
        x_log[k]     = x[:13]
        u_act_log[k] = x[13:17]

        # advance any stateful wind source once per control step
        if wind is not None and hasattr(wind, "step"):
            wind.step(t, x[:13], dt_ctrl)

        # autopilot samples plant state (not actuator state) and produces u_cmd
        u_cmd = autopilot.step(t, x[:13], dt_ctrl)
        u_cmd_log[k] = u_cmd

        if k == n_steps - 1:
            break

        def rhs(_t, _x, _u_cmd=u_cmd):
            x_p   = _x[:13]
            u_act = _x[13:17]
            wind_ned = None if wind is None else wind(_t, x_p)
            dxp = state_derivative(_t, x_p, u_act, ac, wind_ned=wind_ned)
            dua = actuator_derivative(u_act, _u_cmd, ap)
            return np.concatenate([dxp, dua])

        sol = solve_ivp(rhs, (t, t + dt_ctrl), x, rtol=rtol, atol=atol,
                        method="RK45", max_step=dt_ctrl)
        if not sol.success:
            raise RuntimeError(f"integration failed at t={t:.3f}: {sol.message}")
        x = sol.y[:, -1]
        t = t + dt_ctrl

    # Build SimResults from logs.
    pos = x_log[:, 0:3]
    V   = x_log[:, 3:6]
    Q   = x_log[:, 6:10]
    W   = x_log[:, 10:13]
    euler = np.array([quat_to_euler(q) for q in Q])
    Va = np.linalg.norm(V, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.arctan2(V[:, 2], V[:, 0])
        beta  = np.arcsin(np.clip(V[:, 1] / np.where(Va > 1e-6, Va, 1.0), -1.0, 1.0))
    altitude = -pos[:, 2]

    res = SimResults(
        t=t_log, x=x_log, u=u_act_log,
        pos_ned=pos, V_body=V, euler=euler, omega=W,
        Va=Va, alpha=alpha, beta=beta, altitude=altitude,
    )
    return ClosedLoopLog(res=res, u_cmd=u_cmd_log, u_act=u_act_log)
