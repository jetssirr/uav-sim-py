"""Run a 3-waypoint mission and animate the resulting flight."""
from __future__ import annotations

import numpy as np

from uavsim import (Autopilot, FlightAnimation, WaypointCommands, default_uav,
                    simulate_closed_loop, trim)


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
    log = simulate_closed_loop(ac, ts, ap, t_end=240.0, dt_ctrl=0.02, dt_log=0.05)

    anim = FlightAnimation(log.res, waypoints=wps, time_scaling=10.0,
                           model_scale=10.0)
    anim.show()


if __name__ == "__main__":
    main()
