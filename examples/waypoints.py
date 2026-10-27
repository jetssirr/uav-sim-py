"""Three-waypoint mission with loiter at the final waypoint."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from uavsim import Autopilot, WaypointCommands, default_uav, simulate_closed_loop, trim
from uavsim.viz import plot_state_history


def main() -> None:
    ac = default_uav()
    ts = trim(ac)
    wps = np.array([
        [300.0, 200.0, 110.0],
        [300.0, 600.0, 120.0],
        [600.0, 600.0, 110.0],
    ])
    cmd = WaypointCommands(waypoints=wps, airspeed=ts.Va,
                           arrival_radius=40.0, loiter_radius=80.0)
    ap = Autopilot(ac, ts, cmd)
    log = simulate_closed_loop(ac, ts, ap, t_end=300.0, dt_ctrl=0.02)
    r = log.res

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(r.pos_ned[:, 1], r.pos_ned[:, 0], lw=1.5)
    ax.scatter(wps[:, 1], wps[:, 0], c="r", s=50, zorder=3, label="waypoints")
    for i, w in enumerate(wps):
        ax.annotate(f"  WP{i+1}", (w[1], w[0]))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(wps[-1, 1] + 80 * np.cos(theta), wps[-1, 0] + 80 * np.sin(theta),
            "k--", alpha=0.4, label="loiter target")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.4); ax.legend()
    ax.set_title("Waypoint mission with loiter")
    fig.tight_layout()

    plot_state_history(r, title="Waypoint mission")
    plt.show()


if __name__ == "__main__":
    main()
