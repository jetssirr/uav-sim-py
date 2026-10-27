"""
Simulation runner: scipy solve_ivp wrapper around state_derivative,
with optional control law and wind input. Records a clean results dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .aircraft import Aircraft
from .dynamics import state_derivative, unpack_state
from .quaternion import quat_to_euler

ControlLaw = Callable[[float, np.ndarray], np.ndarray]
WindLaw    = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class SimResults:
    t:        np.ndarray
    x:        np.ndarray   # (N, 13)
    u:        np.ndarray   # (N, 4)
    pos_ned:  np.ndarray
    V_body:   np.ndarray
    euler:    np.ndarray   # (N, 3) phi, theta, psi
    omega:    np.ndarray
    Va:       np.ndarray
    alpha:    np.ndarray
    beta:     np.ndarray
    altitude: np.ndarray   # m AGL ( -p_d )

    _FIELDS = ("t", "x", "u", "pos_ned", "V_body", "euler",
               "omega", "Va", "alpha", "beta", "altitude")

    def save(self, path: str) -> None:
        """Save to .npz. Reload with SimResults.load(path)."""
        np.savez_compressed(path, **{k: getattr(self, k) for k in self._FIELDS})

    @classmethod
    def load(cls, path: str) -> "SimResults":
        with np.load(path) as f:
            return cls(**{k: np.asarray(f[k]) for k in cls._FIELDS})


def simulate(
    ac: Aircraft,
    x0: np.ndarray,
    t_end: float,
    control: ControlLaw | np.ndarray,
    wind:    WindLaw | None = None,
    dt_log:  float = 0.02,
    rtol:    float = 1e-6,
    atol:    float = 1e-8,
) -> SimResults:
    if not callable(control):
        u_const = np.asarray(control, dtype=float)
        control_fn: ControlLaw = lambda t, x: u_const
    else:
        control_fn = control

    def rhs(t, x):
        u = control_fn(t, x)
        wind_ned = None if wind is None else wind(t, x)
        return state_derivative(t, x, u, ac, wind_ned=wind_ned)

    t_eval = np.arange(0.0, t_end + dt_log * 0.5, dt_log)
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval, rtol=rtol, atol=atol,
                    method="RK45", max_step=dt_log)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    X = sol.y.T
    N = X.shape[0]
    U = np.zeros((N, 4))
    for i, ti in enumerate(sol.t):
        U[i] = control_fn(ti, X[i])

    pos = X[:, 0:3]
    V   = X[:, 3:6]
    Q   = X[:, 6:10]
    W   = X[:, 10:13]

    euler = np.array([quat_to_euler(q) for q in Q])
    Va    = np.linalg.norm(V, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.arctan2(V[:, 2], V[:, 0])
        beta  = np.arcsin(np.clip(V[:, 1] / np.where(Va > 1e-6, Va, 1.0), -1.0, 1.0))
    altitude = -pos[:, 2]

    return SimResults(
        t=sol.t, x=X, u=U,
        pos_ned=pos, V_body=V, euler=euler, omega=W,
        Va=Va, alpha=alpha, beta=beta, altitude=altitude,
    )
