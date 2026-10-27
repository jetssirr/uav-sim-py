"""
Closed-loop autopilot demo: trim, then command an altitude step at t=5s,
a heading change at t=20s, and an airspeed change at t=40s. Plot tracking.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from uavsim import (Autopilot, CommandSchedule, default_uav, simulate_closed_loop, trim)
from uavsim.viz import plot_controls, plot_state_history


def main() -> None:
    ac = default_uav()
    ts = trim(ac)

    h0   = ts.alt
    psi0 = 0.0
    Va0  = ts.Va

    cmd = CommandSchedule.from_dict({
        "altitude": lambda t: h0 + 50.0 * (t >= 5.0),
        "heading":  lambda t: psi0 + np.deg2rad(30.0) * (t >= 20.0),
        "airspeed": lambda t: Va0 + 2.0 * (t >= 40.0),
    })
    ap_ctrl = Autopilot(ac, ts, cmd)
    log = simulate_closed_loop(ac, ts, ap_ctrl, t_end=60.0, dt_ctrl=0.02)

    res = log.res
    print(f"final altitude = {res.altitude[-1]:.2f} m  (cmd {h0+50:.2f})")
    print(f"final heading  = {np.rad2deg(res.euler[-1,2]):+.2f} deg  (cmd 30.0)")
    print(f"final Va       = {res.Va[-1]:.2f} m/s  (cmd {Va0+2:.2f})")

    plot_state_history(res, title="Closed-loop autopilot demo")

    # Overlay command vs measurement
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = res.t
    h_cmd_arr   = np.array([cmd.altitude(ti) for ti in t])
    psi_cmd_arr = np.array([cmd.heading(ti)  for ti in t])
    Va_cmd_arr  = np.array([cmd.airspeed(ti) for ti in t])
    ax[0].plot(t, h_cmd_arr, "k--", label="cmd"); ax[0].plot(t, res.altitude, label="meas")
    ax[0].set_ylabel("altitude [m]"); ax[0].legend()
    ax[1].plot(t, np.rad2deg(psi_cmd_arr), "k--"); ax[1].plot(t, np.rad2deg(res.euler[:, 2]))
    ax[1].set_ylabel("heading [deg]")
    ax[2].plot(t, Va_cmd_arr, "k--"); ax[2].plot(t, res.Va)
    ax[2].set_ylabel("Va [m/s]"); ax[2].set_xlabel("t [s]")
    fig.suptitle("Command tracking"); fig.tight_layout()

    plot_controls(res, title="Actuator outputs")
    plt.show()


if __name__ == "__main__":
    main()
