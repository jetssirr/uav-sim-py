"""
Comprehensive end-to-end demo: trim → linearize → autopilot tracking →
waypoint mission with L1 loiter → Monte Carlo gust sweep → eigenvalue
locus. Saves all figures and an animated GIF to docs/figures/.

Run from the repo root:
    python examples/demo.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from uavsim import (Autopilot, CommandSchedule, DrydenGust, DrydenWind,
                    WaypointCommands, classify_modes, default_uav,
                    flight_eigenvalues, monte_carlo, simulate,
                    simulate_closed_loop, stability_sweep, trim)
from uavsim.quaternion import quat_from_euler, quat_to_rot
from uavsim.viz import plot_state_history

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LINE = "-" * 64


def banner(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def section_trim_and_modes():
    banner("Trim and linearization")
    ac = default_uav()
    ts = trim(ac)
    print(f"trim @ Va={ts.Va:.2f} m/s, alt={ts.alt:.1f} m")
    print(f"  α = {np.rad2deg(ts.alpha):+6.3f} deg")
    print(f"  θ = {np.rad2deg(ts.theta):+6.3f} deg")
    print(f"  δe= {np.rad2deg(ts.u[0]):+6.3f} deg")
    print(f"  δt= {ts.u[3]:6.3f}")

    eigs = flight_eigenvalues(ac, ts)
    modes = classify_modes(eigs)
    print(f"\n{'mode':14s}  {'eigenvalue':>22s}   ω_n [rad/s]   ζ")
    for m in modes:
        if m.is_oscillatory:
            print(f"  {m.name:12s}  {m.eig.real:+8.4f}{m.eig.imag:+8.4f}j  "
                  f"{m.omega_n:8.3f}     {m.zeta:+.3f}")
        else:
            print(f"  {m.name:12s}  {m.eig.real:+8.4f}             "
                  f"{m.omega_n:8.3f}    (real)")
    return ac, ts


def section_open_loop_trim(ac, ts):
    banner("Open-loop trimmed flight")
    res = simulate(ac, ts.x, t_end=30.0, control=ts.u, dt_log=0.05)
    print(f"  Va drift  in 30s = {abs(res.Va[-1] - ts.Va):.4f} m/s")
    print(f"  alt drift in 30s = {abs(res.altitude[-1] - ts.alt):.4f} m")
    f = plot_state_history(res, title="Open-loop trimmed flight")
    f.savefig(OUT / "basic_trim.png", dpi=110); plt.close(f)


def section_autopilot(ac, ts):
    banner("Closed-loop autopilot: altitude / heading / airspeed steps")
    h0, psi0, Va0 = ts.alt, 0.0, ts.Va
    cmd = CommandSchedule.from_dict({
        "altitude": lambda t: h0 + 50.0 * (t >= 5.0),
        "heading":  lambda t: psi0 + np.deg2rad(30.0) * (t >= 60.0),
        "airspeed": lambda t: Va0 + 2.0 * (t >= 100.0),
    })
    ap = Autopilot(ac, ts, cmd)
    log = simulate_closed_loop(ac, ts, ap, t_end=160.0, dt_ctrl=0.02)
    r = log.res

    err_alt = abs(r.altitude[-1] - (h0 + 50.0))
    err_hdg = abs(np.rad2deg(r.euler[-1, 2]) - 30.0)
    err_va  = abs(r.Va[-1] - (Va0 + 2.0))
    print(f"  altitude tracking error at t=160s: {err_alt:.3f} m")
    print(f"  heading  tracking error at t=160s: {err_hdg:.3f} deg")
    print(f"  airspeed tracking error at t=160s: {err_va:.3f} m/s")

    # 3-panel commanded vs measured
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = r.t
    h_cmd_arr   = np.array([cmd.altitude(ti) for ti in t])
    psi_cmd_arr = np.array([cmd.heading(ti)  for ti in t])
    Va_cmd_arr  = np.array([cmd.airspeed(ti) for ti in t])
    ax[0].plot(t, h_cmd_arr, "k--", lw=1, label="cmd")
    ax[0].plot(t, r.altitude, label="meas")
    ax[0].set_ylabel("altitude [m]"); ax[0].legend(loc="lower right")
    ax[1].plot(t, np.rad2deg(psi_cmd_arr), "k--", lw=1)
    ax[1].plot(t, np.rad2deg(r.euler[:, 2]))
    ax[1].set_ylabel("heading [deg]")
    ax[2].plot(t, Va_cmd_arr, "k--", lw=1)
    ax[2].plot(t, r.Va)
    ax[2].set_ylabel("Va [m/s]"); ax[2].set_xlabel("t [s]")
    fig.suptitle("Autopilot tracking: altitude / heading / airspeed steps")
    fig.tight_layout(); fig.savefig(OUT / "autopilot_tracking.png", dpi=110)
    plt.close(fig)


def section_waypoints(ac, ts):
    banner("Waypoint mission with L1 loiter")
    wps = np.array([
        [300.0, 200.0, 110.0],
        [300.0, 600.0, 120.0],
        [600.0, 600.0, 110.0],
    ])
    cmd = WaypointCommands(waypoints=wps, airspeed=ts.Va,
                           arrival_radius=40.0, loiter_radius=80.0,
                           loiter_mode="l1")
    print(f"  L1 lookahead = {cmd.L1_distance:.1f} m  "
          f"(auto: 1.25 × loiter_radius={cmd.loiter_radius:.0f})")
    ap = Autopilot(ac, ts, cmd)
    log = simulate_closed_loop(ac, ts, ap, t_end=400.0, dt_ctrl=0.02)
    r = log.res

    mask = r.t > 250
    radii = np.hypot(r.pos_ned[mask, 0] - wps[-1, 0],
                     r.pos_ned[mask, 1] - wps[-1, 1])
    print(f"  steady-state loiter radius: mean = {radii.mean():.2f}  "
          f"std = {radii.std():.4f}  target = 80")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(r.pos_ned[:, 1], r.pos_ned[:, 0], lw=1.2)
    ax.scatter(wps[:, 1], wps[:, 0], c="r", s=50, zorder=3, label="waypoints")
    for i, w in enumerate(wps):
        ax.annotate(f"  WP{i+1}", (w[1], w[0]))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(wps[-1, 1] + 80 * np.cos(th), wps[-1, 0] + 80 * np.sin(th),
            "k--", alpha=0.4, label="loiter target (R=80)")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.4)
    ax.legend(); ax.set_title("Waypoint mission with L1 loiter")
    fig.tight_layout(); fig.savefig(OUT / "waypoint_track.png", dpi=110)
    plt.close(fig)
    return wps, log


def section_dryden_mission(ac, ts, wps):
    banner("Mission under Dryden turbulence")
    cmd = WaypointCommands(waypoints=wps, airspeed=ts.Va,
                           arrival_radius=40.0, loiter_radius=80.0,
                           loiter_mode="l1")
    ap = Autopilot(ac, ts, cmd)
    dryden = DrydenGust(sigma_u=1.5, sigma_v=1.5, sigma_w=1.0,
                        Lu=200, Lv=200, Lw=50, order=2, seed=99)
    wind = DrydenWind(dryden=dryden)
    log = simulate_closed_loop(ac, ts, ap, t_end=400.0, dt_ctrl=0.02, wind=wind)
    r = log.res
    mask = r.t > 250
    radii = np.hypot(r.pos_ned[mask, 0] - wps[-1, 0],
                     r.pos_ned[mask, 1] - wps[-1, 1])
    print(f"  loiter radius under turbulence (σ_w=1.0): "
          f"mean={radii.mean():.2f}  std={radii.std():.2f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(r.pos_ned[:, 1], r.pos_ned[:, 0], lw=1.0, alpha=0.8)
    ax.scatter(wps[:, 1], wps[:, 0], c="r", s=50, zorder=3)
    for i, w in enumerate(wps):
        ax.annotate(f"  WP{i+1}", (w[1], w[0]))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(wps[-1, 1] + 80 * np.cos(th), wps[-1, 0] + 80 * np.sin(th),
            "k--", alpha=0.4)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.4)
    ax.set_title("Mission with Dryden turbulence")
    fig.tight_layout(); fig.savefig(OUT / "mission_with_gusts.png", dpi=110)
    plt.close(fig)
    return log


def section_monte_carlo(ac, ts):
    banner("Monte Carlo gust sensitivity sweep")

    def build_ap():
        cmd = CommandSchedule.from_dict({
            "altitude": lambda t: ts.alt,
            "heading":  lambda t: 0.0,
            "airspeed": lambda t: ts.Va,
        })
        return Autopilot(ac, ts, cmd)

    intensities = [0.25, 0.5, 1.0, 1.5, 2.0]
    n_runs = 15
    results = monte_carlo(ac, ts, build_ap, intensities=intensities,
                          n_runs=n_runs, t_end=30.0, dt_ctrl=0.02)
    print(f"  {n_runs} runs per intensity")
    print(f"  {'intensity':>10}  {'alt RMS':>10}  {'hdg RMS [deg]':>15}  {'Va RMS':>8}")
    for s in results:
        z = s.summary()
        print(f"  {z['intensity']:>10.2f}  {z['alt_rms_mean']:>10.3f}  "
              f"{np.rad2deg(z['hdg_rms_mean']):>15.3f}  {z['Va_rms_mean']:>8.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [str(i) for i in intensities]
    axes[0].boxplot([r.altitude_rms for r in results], tick_labels=labels)
    axes[0].set_xlabel("gust intensity"); axes[0].set_ylabel("altitude RMS [m]")
    axes[1].boxplot([np.rad2deg(r.heading_rms) for r in results], tick_labels=labels)
    axes[1].set_xlabel("gust intensity"); axes[1].set_ylabel("heading RMS [deg]")
    axes[2].boxplot([r.Va_rms for r in results], tick_labels=labels)
    axes[2].set_xlabel("gust intensity"); axes[2].set_ylabel("Va RMS [m/s]")
    fig.suptitle(f"Tracking error vs Dryden gust intensity ({n_runs} runs each)")
    fig.tight_layout(); fig.savefig(OUT / "monte_carlo.png", dpi=110)
    plt.close(fig)


def section_stability_sweep(ac):
    banner("Eigenvalue locus over an airspeed sweep")
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
    ax.grid(alpha=0.4); fig.tight_layout()
    fig.savefig(OUT / "stability_locus.png", dpi=110); plt.close(fig)


# ---- 3D animation -> GIF -----------------------------------------------------
def _airframe_segments(scale=1.0):
    L = 1.0 * scale; bspan = 1.6 * scale; th = 0.4 * scale
    return [
        np.array([[ 0.5*L, 0, 0], [-0.5*L, 0, 0]]),
        np.array([[-0.1*L, -0.5*bspan, 0], [-0.1*L, 0.5*bspan, 0]]),
        np.array([[-0.4*L, -0.2*bspan, 0], [-0.4*L, 0.2*bspan, 0]]),
        np.array([[-0.5*L, 0, 0], [-0.5*L, 0, -th]]),
    ]


def section_animation(log, wps):
    banner("Saving flight animation (GIF)")
    res = log.res
    # Downsample aggressively. Take a 80s window covering wp2 -> wp3 -> a few
    # full loiter orbits. This keeps the GIF under ~2 MB.
    t_win_start, t_win_end = 130.0, 200.0
    in_win = (res.t >= t_win_start) & (res.t <= t_win_end)
    sub_t   = res.t[in_win]
    sub_pos = res.pos_ned[in_win]
    sub_eul = res.euler[in_win]
    sub_Va  = res.Va[in_win]
    stride  = 20                  # 0.02 * 20 = 400 ms per kept sample
    t   = sub_t[::stride]
    pos = sub_pos[::stride]
    eul = sub_eul[::stride]
    Va  = sub_Va[::stride]
    Q   = np.array([quat_from_euler(*e) for e in eul])
    n_frames = len(t)
    print(f"  {n_frames} frames  window=[{t_win_start:.0f}, {t_win_end:.0f}]s")

    fig = plt.figure(figsize=(4.5, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    # Full path under window, lighter
    ax.plot(res.pos_ned[:, 1], res.pos_ned[:, 0], -res.pos_ned[:, 2],
            c="tab:blue", lw=0.6, alpha=0.25, label="full path")
    ax.scatter(wps[:, 1], wps[:, 0], wps[:, 2],
               c="r", s=40, marker="o", zorder=5, label="waypoints")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(wps[-1, 1] + 80 * np.cos(th), wps[-1, 0] + 80 * np.sin(th),
            wps[-1, 2] * np.ones_like(th), "k--", alpha=0.4, label="loiter target")
    ax.set_xlabel("E [m]"); ax.set_ylabel("N [m]"); ax.set_zlabel("alt [m]")
    ax.view_init(elev=25, azim=-60)

    airframe_lines = [ax.plot([], [], [], c="k", lw=2)[0] for _ in range(4)]
    trail_line, = ax.plot([], [], [], c="tab:orange", lw=1.5)
    title = ax.set_title("")

    segs0 = _airframe_segments(scale=15.0)

    def init():
        ax.set_xlim(200, 750); ax.set_ylim(200, 750); ax.set_zlim(95, 130)
        ax.legend(loc="upper left", fontsize=7)
        return (*airframe_lines, trail_line, title)

    def update(i):
        R = quat_to_rot(Q[i])
        pN, pE, pD = pos[i]
        for line, seg in zip(airframe_lines, segs0):
            sw = (R @ seg.T).T
            line.set_data(pE + sw[:, 1], pN + sw[:, 0])
            line.set_3d_properties(-pD - sw[:, 2])
        trail_pts = pos[max(0, i - 80):i + 1]
        trail_line.set_data(trail_pts[:, 1], trail_pts[:, 0])
        trail_line.set_3d_properties(-trail_pts[:, 2])
        title.set_text(f"t={t[i]:5.1f}s  alt={-pD:5.1f}m  Va={Va[i]:4.1f}m/s")
        return (*airframe_lines, trail_line, title)

    ani = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        blit=False, interval=80)
    gif_path = OUT / "flight_animation.gif"
    t0 = time.time()
    ani.save(gif_path, writer=PillowWriter(fps=10), dpi=70)
    plt.close(fig)
    size_mb = gif_path.stat().st_size / 1024 / 1024
    print(f"  wrote {gif_path}  ({time.time() - t0:.1f}s, {size_mb:.1f} MB)")
    return gif_path


def main() -> None:
    t0 = time.time()
    ac, ts = section_trim_and_modes()
    section_open_loop_trim(ac, ts)
    section_autopilot(ac, ts)
    wps, mission_log = section_waypoints(ac, ts)
    section_dryden_mission(ac, ts, wps)
    section_monte_carlo(ac, ts)
    section_stability_sweep(ac)
    section_animation(mission_log, wps)
    banner("Done")
    print(f"  figures in {OUT}")
    print(f"  total demo time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
