"""
Trim and linearize across a range of airspeeds. Plot eigenvalue locus and
damping ratio vs. airspeed for each classical mode.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from uavsim import default_uav, stability_sweep


def main() -> None:
    ac = default_uav()
    V_grid = np.linspace(13.0, 30.0, 18)
    out = stability_sweep(ac, V_grid)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.viridis
    for V, eigs in zip(out["V_grid"], out["eigs"]):
        c = cmap((V - V_grid.min()) / (V_grid.max() - V_grid.min()))
        ax.scatter(eigs.real, eigs.imag, c=[c], s=18, alpha=0.85)
    ax.axvline(0, color="r", ls="--", alpha=0.4)
    ax.set_xlabel("Re(λ) [1/s]"); ax.set_ylabel("Im(λ) [rad/s]")
    ax.set_title("Eigenvalue locus vs. airspeed (color = V)")
    ax.grid(alpha=0.4)

    fig2, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    mode_names = ["short_period", "phugoid", "dutch_roll", "roll", "spiral"]
    series = {n: ([], []) for n in mode_names}   # (omega_n list, zeta list)
    for modes in out["modes"]:
        for n in mode_names:
            m = next((mm for mm in modes if mm.name == n), None)
            series[n][0].append(m.omega_n if m else np.nan)
            series[n][1].append(m.zeta    if m else np.nan)

    for n in mode_names:
        axes[0].plot(out["V_grid"], series[n][0], label=n, marker="o", ms=3)
        axes[1].plot(out["V_grid"], series[n][1], label=n, marker="o", ms=3)
    axes[0].set_ylabel("ω_n [rad/s]"); axes[0].grid(alpha=0.4); axes[0].legend()
    axes[1].set_ylabel("damping ζ");  axes[1].grid(alpha=0.4)
    axes[1].axhline(0, color="r", ls="--", alpha=0.4)
    axes[1].set_xlabel("V [m/s]")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
