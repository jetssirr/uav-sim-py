"""
Interactive 3D flight animation with playback controls.

Layout:
  - main 3D axis: ground plane, flight path trail, aircraft model
  - right side panel: altitude, airspeed, Euler angles, control surfaces
  - bottom: play/pause button, speed selector, time slider, track toggle

Run inside an interactive matplotlib backend (TkAgg / QtAgg / MacOSX).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons

from .quaternion import quat_to_rot
from .sim import SimResults


# Body-frame airframe geometry (line segments). x forward, y right, z down.
def _airframe_segments(scale: float = 1.0) -> list[np.ndarray]:
    L = 1.0 * scale       # fuselage length
    bspan = 1.6 * scale   # wing span
    th = 0.4 * scale      # tail height (down is +z; we draw up so use -z)
    fuselage = np.array([[ 0.5 * L, 0, 0], [-0.5 * L, 0, 0]])
    wing     = np.array([[-0.1 * L, -0.5 * bspan, 0],
                         [-0.1 * L,  0.5 * bspan, 0]])
    htail    = np.array([[-0.4 * L, -0.2 * bspan, 0],
                         [-0.4 * L,  0.2 * bspan, 0]])
    vtail    = np.array([[-0.5 * L, 0, 0], [-0.5 * L, 0, -th]])
    return [fuselage, wing, htail, vtail]


@dataclass
class FlightAnimation:
    results:      SimResults
    waypoints:    np.ndarray | None = None    # (M, 3) NED
    time_scaling: float = 1.0                 # 1.0 = real time
    downsample:   int = 1
    model_scale:  float = 8.0
    trail_window: float = 30.0                # seconds of past trail shown

    # Internal
    _t: np.ndarray = field(init=False, repr=False)
    _frame: int = field(default=0, init=False, repr=False)
    _playing: bool = field(default=True, init=False, repr=False)
    _track: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self._t = self.results.t[::self.downsample]
        self._pos = self.results.pos_ned[::self.downsample]
        self._eul = self.results.euler[::self.downsample]
        self._Va  = self.results.Va[::self.downsample]
        self._alt = self.results.altitude[::self.downsample]
        self._u   = self.results.u[::self.downsample]
        # quaternion approximation from Euler for drawing (simpler than re-storing):
        # we can use quat_from_euler, but we already have euler — convert each.
        from .quaternion import quat_from_euler
        self._q   = np.array([quat_from_euler(*e) for e in self._eul])

    def show(self) -> None:
        fig = plt.figure(figsize=(13, 8))
        # Main 3D axis
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        # Right column: 4 metric strip plots
        gs = fig.add_gridspec(4, 2, width_ratios=[1.4, 1.0])
        m1 = fig.add_subplot(gs[0, 1]); m2 = fig.add_subplot(gs[1, 1])
        m3 = fig.add_subplot(gs[2, 1]); m4 = fig.add_subplot(gs[3, 1])

        # 3D path (full)
        ax3d.plot(self._pos[:, 1], self._pos[:, 0], -self._pos[:, 2],
                  c="tab:blue", lw=0.8, alpha=0.4)
        if self.waypoints is not None:
            wp = np.asarray(self.waypoints)
            ax3d.scatter(wp[:, 1], wp[:, 0], wp[:, 2],
                         c="r", s=40, marker="o", zorder=5)
        ax3d.set_xlabel("E [m]"); ax3d.set_ylabel("N [m]"); ax3d.set_zlabel("alt [m]")

        # Animated handles
        airframe_lines = [ax3d.plot([], [], [], c="k", lw=2)[0] for _ in range(4)]
        trail_line, = ax3d.plot([], [], [], c="tab:orange", lw=1.5)
        title = ax3d.set_title("")

        # Strip plots
        m1.plot(self._t, self._alt, c="tab:blue"); m1.set_ylabel("alt [m]")
        m2.plot(self._t, self._Va,  c="tab:green"); m2.set_ylabel("Va [m/s]")
        m3.plot(self._t, np.rad2deg(self._eul[:, 0]), label="φ")
        m3.plot(self._t, np.rad2deg(self._eul[:, 1]), label="θ")
        m3.plot(self._t, np.rad2deg(self._eul[:, 2]), label="ψ")
        m3.legend(loc="best", fontsize=7); m3.set_ylabel("Euler [deg]")
        m4.plot(self._t, np.rad2deg(self._u[:, 0]), label="δe")
        m4.plot(self._t, np.rad2deg(self._u[:, 1]), label="δa")
        m4.plot(self._t, np.rad2deg(self._u[:, 2]), label="δr")
        m4b = m4.twinx()
        m4b.plot(self._t, self._u[:, 3], "k--", label="δt", lw=0.8)
        m4.set_ylabel("ctrl [deg]"); m4b.set_ylabel("throttle"); m4.set_xlabel("t [s]")
        m4.legend(loc="upper left", fontsize=7)

        cursors = [ax_.axvline(self._t[0], c="r", lw=0.8) for ax_ in (m1, m2, m3, m4)]

        # Bottom controls
        ax_play  = fig.add_axes([0.05, 0.02, 0.08, 0.04])
        ax_speed = fig.add_axes([0.16, 0.01, 0.10, 0.06])
        ax_track = fig.add_axes([0.28, 0.02, 0.08, 0.04])
        ax_time  = fig.add_axes([0.40, 0.025, 0.55, 0.03])

        b_play  = Button(ax_play, "Pause")
        radio   = RadioButtons(ax_speed, ("1x", "2x", "5x", "10x"), active=0)
        b_track = Button(ax_track, "Track: ON")
        sld     = Slider(ax_time, "t", self._t[0], self._t[-1], valinit=self._t[0])

        def on_play(_):
            self._playing = not self._playing
            b_play.label.set_text("Pause" if self._playing else "Play")

        def on_speed(label):
            self.time_scaling = float(label.rstrip("x"))

        def on_track(_):
            self._track = not self._track
            b_track.label.set_text(f"Track: {'ON' if self._track else 'OFF'}")

        def on_slider(val):
            self._frame = int(np.argmin(np.abs(self._t - val)))

        b_play.on_clicked(on_play)
        radio.on_clicked(on_speed)
        b_track.on_clicked(on_track)
        sld.on_changed(on_slider)

        segs0 = _airframe_segments(self.model_scale)

        def render(frame):
            R = quat_to_rot(self._q[frame])
            pN, pE, pD = self._pos[frame]
            for line, seg in zip(airframe_lines, segs0):
                seg_world = (R @ seg.T).T
                xs = pE + seg_world[:, 1]
                ys = pN + seg_world[:, 0]
                zs = -pD - seg_world[:, 2]
                line.set_data(xs, ys); line.set_3d_properties(zs)

            # Trail (recent)
            t_now = self._t[frame]
            mask = (self._t >= t_now - self.trail_window) & (self._t <= t_now)
            tp = self._pos[mask]
            trail_line.set_data(tp[:, 1], tp[:, 0]); trail_line.set_3d_properties(-tp[:, 2])

            for c in cursors:
                c.set_xdata([t_now, t_now])

            if self._track:
                pad = 60.0
                ax3d.set_xlim(pE - pad, pE + pad)
                ax3d.set_ylim(pN - pad, pN + pad)
                ax3d.set_zlim(max(0, -pD - 40), -pD + 40)

            title.set_text(
                f"t = {t_now:6.1f}s   alt = {-pD:6.1f} m   Va = {self._Va[frame]:5.1f} m/s\n"
                f"φ={np.rad2deg(self._eul[frame,0]):+5.1f}  "
                f"θ={np.rad2deg(self._eul[frame,1]):+5.1f}  "
                f"ψ={np.rad2deg(self._eul[frame,2]):+5.1f}"
            )
            return (*airframe_lines, trail_line, *cursors, title)

        # Render initial frame
        render(0)

        dt_frame = (self._t[1] - self._t[0]) / self.time_scaling

        def step(_):
            if self._playing:
                self._frame = (self._frame + 1) % len(self._t)
                # update slider without firing callback
                sld.eventson = False
                sld.set_val(self._t[self._frame])
                sld.eventson = True
            return render(self._frame)

        ani = FuncAnimation(fig, step, interval=max(20, int(1000 * dt_frame)),
                            blit=False, cache_frame_data=False)

        fig.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()
        return ani
