"""SimResults round-trip through .npz."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from uavsim import SimResults, default_uav, simulate, trim


def test_save_load_roundtrip():
    ac = default_uav()
    ts = trim(ac)
    res = simulate(ac, ts.x, t_end=8.0, control=ts.u, dt_log=0.1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        res.save(path)
        loaded = SimResults.load(path)
        for k in SimResults._FIELDS:
            assert np.allclose(getattr(res, k), getattr(loaded, k))
    finally:
        os.unlink(path)
