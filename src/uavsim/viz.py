"""Plot helpers for SimResults."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .sim import SimResults


def plot_state_history(res: SimResults, title: str = "") -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    t = res.t

    axes[0, 0].plot(t, res.altitude); axes[0, 0].set_ylabel("altitude [m]")
    axes[1, 0].plot(t, res.Va);       axes[1, 0].set_ylabel("Va [m/s]")
    axes[2, 0].plot(t, np.rad2deg(res.alpha), label="α")
    axes[2, 0].plot(t, np.rad2deg(res.beta),  label="β")
    axes[2, 0].set_ylabel("α, β [deg]"); axes[2, 0].legend(loc="best")
    axes[3, 0].plot(t, res.pos_ned[:, 0], label="N")
    axes[3, 0].plot(t, res.pos_ned[:, 1], label="E")
    axes[3, 0].set_ylabel("N, E [m]"); axes[3, 0].legend(loc="best")
    axes[3, 0].set_xlabel("t [s]")

    eul_deg = np.rad2deg(res.euler)
    axes[0, 1].plot(t, eul_deg[:, 0]); axes[0, 1].set_ylabel("φ [deg]")
    axes[1, 1].plot(t, eul_deg[:, 1]); axes[1, 1].set_ylabel("θ [deg]")
    axes[2, 1].plot(t, eul_deg[:, 2]); axes[2, 1].set_ylabel("ψ [deg]")

    omega_deg = np.rad2deg(res.omega)
    axes[3, 1].plot(t, omega_deg[:, 0], label="p")
    axes[3, 1].plot(t, omega_deg[:, 1], label="q")
    axes[3, 1].plot(t, omega_deg[:, 2], label="r")
    axes[3, 1].set_ylabel("ω [deg/s]"); axes[3, 1].legend(loc="best")
    axes[3, 1].set_xlabel("t [s]")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_controls(res: SimResults, title: str = "") -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    t = res.t
    de, da, dr, dt = res.u[:, 0], res.u[:, 1], res.u[:, 2], res.u[:, 3]
    axes[0, 0].plot(t, np.rad2deg(de)); axes[0, 0].set_ylabel("elevator [deg]")
    axes[0, 1].plot(t, np.rad2deg(da)); axes[0, 1].set_ylabel("aileron [deg]")
    axes[1, 0].plot(t, np.rad2deg(dr)); axes[1, 0].set_ylabel("rudder [deg]")
    axes[1, 0].set_xlabel("t [s]")
    axes[1, 1].plot(t, dt);             axes[1, 1].set_ylabel("throttle")
    axes[1, 1].set_xlabel("t [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_ground_track(res: SimResults, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(res.pos_ned[:, 1], res.pos_ned[:, 0])
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.4)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
