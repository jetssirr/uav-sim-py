"""Waypoint guidance reaches all waypoints and loiters near final radius."""
from __future__ import annotations

import numpy as np

from uavsim import Autopilot, WaypointCommands, default_uav, simulate_closed_loop, trim


def test_waypoint_visit_and_loiter():
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

    assert cmd.idx == len(wps) - 1, "should reach final waypoint"

    # During final 100 seconds the loiter radius should stay within reasonable
    # bounds of the target loiter radius.
    final_mask = r.t > 200.0
    pn = r.pos_ned[final_mask, 0]
    pe = r.pos_ned[final_mask, 1]
    radii = np.hypot(pn - wps[-1, 0], pe - wps[-1, 1])
    assert 40.0 < radii.mean() < 160.0, f"loiter radius mean = {radii.mean():.1f}"


if __name__ == "__main__":
    test_waypoint_visit_and_loiter()
    print("OK")
