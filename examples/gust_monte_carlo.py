"""Monte Carlo gust sensitivity sweep, plotted as box plots vs intensity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from uavsim import Autopilot, CommandSchedule, default_uav, monte_carlo, trim


def main() -> None:
    ac = default_uav()
    ts = trim(ac)

    def build_ap():
        cmd = CommandSchedule.from_dict({
            "altitude": lambda t: ts.alt,
            "heading":  lambda t: 0.0,
            "airspeed": lambda t: ts.Va,
        })
        return Autopilot(ac, ts, cmd)

    intensities = [0.25, 0.5, 1.0, 1.5, 2.0]
    results = monte_carlo(ac, ts, build_ap, intensities=intensities,
                          n_runs=20, t_end=30.0, dt_ctrl=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].boxplot([r.altitude_rms for r in results], labels=intensities)
    axes[0].set_xlabel("gust intensity"); axes[0].set_ylabel("altitude RMS [m]")
    axes[1].boxplot([np.rad2deg(r.heading_rms) for r in results], labels=intensities)
    axes[1].set_xlabel("gust intensity"); axes[1].set_ylabel("heading RMS [deg]")
    axes[2].boxplot([r.Va_rms for r in results], labels=intensities)
    axes[2].set_xlabel("gust intensity"); axes[2].set_ylabel("Va RMS [m/s]")
    fig.suptitle("Tracking error vs. gust intensity (Dryden, 20 runs each)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
