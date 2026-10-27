"""
Waypoint guidance with arrival switching and loiter at the final waypoint.

Two loiter strategies are supported:

  - "vector_field": blend an inward-radial unit vector with the orbit tangent
    weighted by tanh(radial_err). Smooth and stable but lazy — the orbit
    overshoots the target radius because the tangent is the desired direction
    only at the target radius (and the heading PID lags).

  - "l1" (default): Park / Deyst / How (2004) lookahead guidance for circles.
    Aim the aircraft at a point on the circle ahead of its current angular
    position by L1/R radians. A short L1 produces a tight, circular orbit
    even with the heading PID's lag.

Output of update(t, x) is (altitude_cmd, heading_cmd, airspeed_cmd) suitable
for the autopilot. Airspeed is constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def _bearing(dN: float, dE: float) -> float:
    """Compass heading in radians: 0 = +N, +π/2 = +E."""
    return float(np.arctan2(dE, dN))


@dataclass
class WaypointCommands:
    waypoints:      np.ndarray                       # (M, 3) NED + alt
    airspeed:       float                            # m/s, constant
    arrival_radius: float = 40.0                     # m, switch threshold
    loiter_radius:  float = 100.0                    # m, final-wp orbit radius
    loiter_dir:     Literal["ccw", "cw"] = "ccw"     # viewed from above
    loiter_mode:    Literal["l1", "vector_field"] = "l1"
    # L1 lookahead distance. If <= 0, auto-set to 1.25 * loiter_radius — this
    # closes the orbit on the target radius almost exactly with the default
    # heading-PID gains.
    L1_distance:    float = 0.0

    idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float).reshape(-1, 3)
        if self.L1_distance <= 0.0:
            self.L1_distance = 1.25 * self.loiter_radius

    def reset(self) -> None:
        self.idx = 0

    @property
    def is_final(self) -> bool:
        return self.idx >= len(self.waypoints) - 1

    def _loiter_l1(self, plane_N: float, plane_E: float,
                   wp_N: float, wp_E: float) -> float:
        """L1-style lookahead heading for a circular orbit."""
        sign = +1.0 if self.loiter_dir == "ccw" else -1.0
        R    = self.loiter_radius
        L1   = self.L1_distance

        rN, rE = plane_N - wp_N, plane_E - wp_E
        dist = float(np.hypot(rN, rE))
        if dist < 1e-3:
            # at center: pick any tangent direction
            return _bearing(0.0, sign)

        # If too far from the circle to track meaningfully, transit direct.
        if dist > 2.0 * R:
            return _bearing(wp_N - plane_N, wp_E - plane_E)

        # Aircraft angular position (heading-convention angle from wp toward plane).
        phi    = float(np.arctan2(rE, rN))
        # Advance angular position by L1/R in the orbit direction.
        # CCW (top-down with N up) corresponds to decreasing heading angle, so we
        # subtract sign*(L1/R) for sign=+1=CCW.
        phi_la = phi - sign * (L1 / R)

        # Lookahead point on the circle.
        la_N = wp_N + R * np.cos(phi_la)
        la_E = wp_E + R * np.sin(phi_la)
        return _bearing(la_N - plane_N, la_E - plane_E)

    def _loiter_vector_field(self, plane_N: float, plane_E: float,
                             wp_N: float, wp_E: float) -> float:
        """Vector-field blend of inward-radial and orbit-tangent."""
        sign = +1.0 if self.loiter_dir == "ccw" else -1.0
        dN, dE = wp_N - plane_N, wp_E - plane_E
        dist = float(np.hypot(dN, dE))
        if dist < 1e-3:
            return _bearing(0.0, sign)
        iN, iE = dN / dist, dE / dist
        tN, tE = -sign * iE, sign * iN
        radial_err = (dist - self.loiter_radius) / self.loiter_radius
        w_r = float(np.tanh(radial_err))
        vN = w_r * iN + (1.0 - abs(w_r)) * tN
        vE = w_r * iE + (1.0 - abs(w_r)) * tE
        return float(np.arctan2(vE, vN))

    def update(self, t: float, x: np.ndarray) -> tuple[float, float, float]:
        N, E, _D = x[0:3]
        wp = self.waypoints[self.idx]
        dN, dE = wp[0] - N, wp[1] - E
        dist = float(np.hypot(dN, dE))

        # Advance through intermediate waypoints on arrival.
        while not self.is_final and dist < self.arrival_radius:
            self.idx += 1
            wp = self.waypoints[self.idx]
            dN, dE = wp[0] - N, wp[1] - E
            dist = float(np.hypot(dN, dE))

        wp_alt = float(wp[2])

        if self.is_final:
            if self.loiter_mode == "l1":
                heading = self._loiter_l1(N, E, wp[0], wp[1])
            else:
                heading = self._loiter_vector_field(N, E, wp[0], wp[1])
        else:
            heading = _bearing(dN, dE)

        return wp_alt, float(heading), self.airspeed
