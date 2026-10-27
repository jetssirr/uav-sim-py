"""Linearization recovers classical flight modes within textbook ranges."""
from __future__ import annotations

import numpy as np

from uavsim import classify_modes, default_uav, flight_eigenvalues, linearize, trim


def test_jacobian_shapes():
    ac = default_uav()
    ts = trim(ac)
    A, B = linearize(ac, ts.x, ts.u)
    assert A.shape == (13, 13)
    assert B.shape == (13, 4)


def test_short_period_well_damped():
    ac = default_uav()
    ts = trim(ac)
    eigs = flight_eigenvalues(ac, ts)
    modes = {m.name: m for m in classify_modes(eigs)}
    sp = modes["short_period"]
    assert sp.omega_n > 1.0,  f"short period ω_n too low: {sp.omega_n}"
    assert sp.zeta    > 0.3,  f"short period ζ too low: {sp.zeta}"


def test_roll_subsidence_negative_real():
    ac = default_uav()
    ts = trim(ac)
    eigs = flight_eigenvalues(ac, ts)
    modes = {m.name: m for m in classify_modes(eigs)}
    roll = modes["roll"]
    assert roll.eig.real < -1.0, f"roll mode not fast: {roll.eig}"


if __name__ == "__main__":
    test_jacobian_shapes()
    test_short_period_well_damped()
    test_roll_subsidence_negative_real()
    print("OK")
