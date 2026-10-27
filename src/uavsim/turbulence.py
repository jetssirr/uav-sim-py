"""
Dryden turbulence: band-limited filtered Gaussian noise approximating the
MIL-F-8785C continuous gust spectra.

Body-frame gust velocity components are produced as Markov processes:

  * u_g  - 1st-order  (matches the MIL-F-8785C u-component spectrum exactly)
  * v_g  - 2nd-order  (1+√3 τ s)/(1+τ s)² shaping, τ = L_v / V
  * w_g  - 2nd-order  same form, with L_w

The state-space realization is the controllable canonical form

    ẋ = [ 0       1   ] x + [0] ξ(t),   y = [K/τ²  K·√3/τ] x
        [ -1/τ²   -2/τ ]     [1]

with white noise intensity Q = π chosen so that the stationary output
variance C·P·Cᵀ = σ² (verified via the Lyapunov solution
P = diag(π τ³/4, π τ/4)).

The Markov processes integrate via Euler-Maruyama with dt of the controller
sample step (typically 0.02 s). For τ ≳ 1 s and dt ≲ 0.05 s the discretization
error is negligible; angular-rate gusts (p_g, q_g, r_g) are not modeled — they
are second-order corrections that mostly affect handling-quality work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from .quaternion import quat_to_rot


@dataclass
class DrydenGust:
    sigma_u: float = 1.06
    sigma_v: float = 1.06
    sigma_w: float = 0.70
    Lu:      float = 200.0
    Lv:      float = 200.0
    Lw:      float = 50.0
    order:   Literal[1, 2] = 2          # filter order for v_g and w_g
    seed:    Optional[int] = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _x_u: float = field(default=0.0, init=False, repr=False)
    _x_v: np.ndarray = field(init=False, repr=False)
    _x_w: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._x_v = np.zeros(2)
        self._x_w = np.zeros(2)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x_u = 0.0
        self._x_v = np.zeros(2)
        self._x_w = np.zeros(2)

    @staticmethod
    def _step_first_order(x: float, sigma: float, L: float, V: float,
                          dt: float, w: float) -> float:
        """Exact 1st-order Markov update with stationary variance σ²."""
        a = float(np.exp(-V * dt / L))
        return a * x + sigma * float(np.sqrt(max(1.0 - a * a, 0.0))) * w

    def _step_second_order(self, x: np.ndarray, sigma: float, L: float, V: float,
                           dt: float, w: float) -> tuple[np.ndarray, float]:
        """One Euler-Maruyama step on the 2nd-order Dryden filter for v or w."""
        tau = L / V
        K   = sigma * np.sqrt(L / (np.pi * V))
        x1, x2 = x
        # ẋ1 = x2,  ẋ2 = -x1/τ² - 2 x2/τ + ξ(t),  Q=π
        x1_new = x1 + dt * x2
        x2_new = x2 + dt * (-x1 / (tau * tau) - 2.0 * x2 / tau) \
                    + float(np.sqrt(np.pi * dt)) * w
        # output: y = K (x1/τ² + √3/τ · x2)
        y = K * (x1_new / (tau * tau) + np.sqrt(3.0) / tau * x2_new)
        return np.array([x1_new, x2_new]), float(y)

    def update(self, dt: float, V: float) -> np.ndarray:
        V = max(V, 1.0)
        wn = self._rng.standard_normal(3)

        self._x_u = self._step_first_order(self._x_u, self.sigma_u, self.Lu,
                                           V, dt, wn[0])
        u_g = self._x_u

        if self.order == 2:
            self._x_v, v_g = self._step_second_order(self._x_v, self.sigma_v,
                                                    self.Lv, V, dt, wn[1])
            self._x_w, w_g = self._step_second_order(self._x_w, self.sigma_w,
                                                    self.Lw, V, dt, wn[2])
        else:
            v_g = self._step_first_order(self._x_v[0], self.sigma_v, self.Lv,
                                         V, dt, wn[1])
            self._x_v = np.array([v_g, 0.0])
            w_g = self._step_first_order(self._x_w[0], self.sigma_w, self.Lw,
                                         V, dt, wn[2])
            self._x_w = np.array([w_g, 0.0])

        return np.array([u_g, v_g, w_g])


@dataclass
class DrydenWind:
    """Wraps DrydenGust as a wind input for the simulator.

    Required by simulate_closed_loop:
      - step(t, x_plant, dt): advance gust state once per control chunk.
      - __call__(t, x_plant): return wind in NED (mean + rotated gust).
    The body-to-NED rotation is sampled at chunk start and held; this avoids
    quaternion variation inside RK45 substeps from leaking into the gust.
    """
    dryden:   DrydenGust
    mean_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))

    _gust_body: np.ndarray = field(init=False, repr=False)
    _R: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean_ned = np.asarray(self.mean_ned, dtype=float)
        self._gust_body = np.zeros(3)
        self._R = np.eye(3)

    def reset(self, seed: Optional[int] = None) -> None:
        self.dryden.reset(seed)
        self._gust_body = np.zeros(3)

    def step(self, t: float, x_plant: np.ndarray, dt: float) -> None:
        Va = float(np.linalg.norm(x_plant[3:6]))
        self._gust_body = self.dryden.update(dt, V=Va)
        self._R = quat_to_rot(x_plant[6:10])

    def __call__(self, t: float, x_plant: np.ndarray) -> np.ndarray:
        return self.mean_ned + self._R @ self._gust_body
