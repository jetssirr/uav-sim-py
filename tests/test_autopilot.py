"""Closed-loop autopilot tracks altitude / heading / airspeed step commands."""
from __future__ import annotations

import numpy as np

from uavsim import Autopilot, CommandSchedule, default_uav, simulate_closed_loop, trim


def test_altitude_step():
    ac = default_uav()
    ts = trim(ac)
    cmd = CommandSchedule.from_dict({
        "altitude": lambda t: ts.alt + 50.0 * (t >= 5.0),
        "heading":  lambda t: 0.0,
        "airspeed": lambda t: ts.Va,
    })
    ap = Autopilot(ac, ts, cmd)
    log = simulate_closed_loop(ac, ts, ap, t_end=120.0, dt_ctrl=0.02)
    r = log.res
    assert abs(r.altitude[-1] - (ts.alt + 50.0)) < 3.0
    assert abs(r.Va[-1] - ts.Va)                 < 1.0


def test_heading_step():
    ac = default_uav()
    ts = trim(ac)
    cmd = CommandSchedule.from_dict({
        "altitude": lambda t: ts.alt,
        "heading":  lambda t: np.deg2rad(30.0) * (t >= 5.0),
        "airspeed": lambda t: ts.Va,
    })
    ap = Autopilot(ac, ts, cmd)
    log = simulate_closed_loop(ac, ts, ap, t_end=80.0, dt_ctrl=0.02)
    r = log.res
    assert abs(np.rad2deg(r.euler[-1, 2]) - 30.0) < 2.0
    assert abs(r.altitude[-1] - ts.alt)           < 5.0
    assert np.max(np.abs(r.beta))      < np.deg2rad(5.0)


if __name__ == "__main__":
    test_altitude_step()
    test_heading_step()
    print("OK")
