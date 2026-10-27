"""Dryden gust statistics and Monte Carlo scaling sanity checks."""
from __future__ import annotations

import numpy as np

from uavsim import (Autopilot, CommandSchedule, DrydenGust, DrydenWind,
                    default_uav, monte_carlo, simulate_closed_loop, trim)


def test_dryden_steady_state_variance():
    """Variance of a long Dryden run should match sigma² within 20%."""
    d = DrydenGust(sigma_u=2.0, sigma_v=1.5, sigma_w=1.0,
                   Lu=200.0, Lv=200.0, Lw=50.0, seed=7)
    samples = np.array([d.update(0.02, V=18.0) for _ in range(20000)])
    # Skip first 1000 for warmup.
    s = samples[1000:]
    assert abs(s[:, 0].std() - 2.0) < 0.4
    assert abs(s[:, 1].std() - 1.5) < 0.4
    assert abs(s[:, 2].std() - 1.0) < 0.3


def test_monte_carlo_scaling():
    """Tracking error should grow roughly with gust intensity."""
    ac = default_uav()
    ts = trim(ac)

    def build_ap():
        cmd = CommandSchedule.from_dict({
            "altitude": lambda t: ts.alt,
            "heading":  lambda t: 0.0,
            "airspeed": lambda t: ts.Va,
        })
        return Autopilot(ac, ts, cmd)

    res = monte_carlo(ac, ts, build_ap, intensities=[0.5, 2.0],
                      n_runs=3, t_end=20.0, dt_ctrl=0.02)
    low  = res[0].summary()["alt_rms_mean"]
    high = res[1].summary()["alt_rms_mean"]
    assert high > low * 1.5, f"alt RMS did not grow with intensity: {low} -> {high}"


if __name__ == "__main__":
    test_dryden_steady_state_variance()
    test_monte_carlo_scaling()
    print("OK")
