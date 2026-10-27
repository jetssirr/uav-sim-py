"""Eigenvalue locus and damping vs. airspeed."""
from __future__ import annotations

import numpy as np

from .aircraft import Aircraft
from .linearize import classify_modes, flight_eigenvalues
from .trim import trim


def stability_sweep(ac: Aircraft, V_grid: np.ndarray, alt: float | None = None,
                    tol: float = 1e-3) -> dict:
    """Trim and linearize at each V0, return eigenvalues and classified modes."""
    if alt is None:
        alt = ac.alt0

    eigs_list = []
    modes_per_speed = []
    for V in V_grid:
        ts = trim(ac, Va=float(V), alt=alt)
        eigs = flight_eigenvalues(ac, ts)
        eigs_list.append(eigs[np.abs(eigs) > tol])
        modes_per_speed.append(classify_modes(eigs, tol=tol))

    return {
        "V_grid": np.asarray(V_grid, dtype=float),
        "eigs":   eigs_list,
        "modes":  modes_per_speed,
    }
