"""Trim the UAV at 18 m/s, 100 m and propagate open-loop for 30 s."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from uavsim import default_uav, simulate, trim
from uavsim.viz import plot_controls, plot_state_history


def main() -> None:
    ac = default_uav()
    ts = trim(ac)
    print(f"Trim solution at Va = {ts.Va:.2f} m/s, alt = {ts.alt:.1f} m:")
    print(f"  alpha = {np.rad2deg(ts.alpha):+.3f} deg")
    print(f"  de    = {np.rad2deg(ts.u[0]):+.3f} deg")
    print(f"  dt    = {ts.u[3]:.3f}")

    res = simulate(ac, ts.x, t_end=30.0, control=ts.u, dt_log=0.02)
    plot_state_history(res, title="Open-loop trimmed flight")
    plot_controls(res, title="Open-loop trimmed flight (controls)")
    plt.show()


if __name__ == "__main__":
    main()
