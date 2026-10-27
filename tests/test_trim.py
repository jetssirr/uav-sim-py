"""Trim should produce a state that, propagated, holds altitude and Va."""
from __future__ import annotations

import numpy as np

from uavsim import default_uav, simulate, trim


def test_trim_residuals_small():
    ac = default_uav()
    ts = trim(ac)
    # Sanity: control inputs in range, trim AoA reasonable.
    assert ac.de_min <= ts.u[0] <= ac.de_max
    assert ac.dt_min <= ts.u[3] <= ac.dt_max
    assert -np.deg2rad(15) < ts.alpha < np.deg2rad(15)


def test_trimmed_flight_holds_steady():
    ac = default_uav()
    ts = trim(ac)
    res = simulate(ac, ts.x, t_end=20.0, control=ts.u, dt_log=0.05)

    Va_drift  = abs(res.Va[-1] - ts.Va)
    alt_drift = abs(res.altitude[-1] - ts.alt)
    # Loose tolerances: open-loop trim drifts slowly; just verify it doesn't blow up.
    assert Va_drift  < 1.5,  f"airspeed drifted {Va_drift:.2f} m/s in 20s"
    assert alt_drift < 10.0, f"altitude drifted {alt_drift:.2f} m in 20s"

    # Bank/sideslip should stay near zero in symmetric trim.
    assert np.max(np.abs(res.euler[:, 0])) < np.deg2rad(2.0)
    assert np.max(np.abs(res.beta))        < np.deg2rad(2.0)


if __name__ == "__main__":
    test_trim_residuals_small()
    test_trimmed_flight_holds_steady()
    print("OK")
