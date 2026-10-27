"""
Numerical linearization about a trim point and flight-dynamic mode extraction.

We linearize the full 13-state equations of motion (no analytical reduction).
Position and yaw are integrators, so the corresponding eigenvalues lie at the
origin. Quaternion normalization adds a fourth zero eigenvalue. The remaining
non-zero eigenvalues are the classical flight modes:

  * short period   - lightly to moderately damped, ω_n ~ 1-10 rad/s
  * phugoid        - lightly damped, ω_n ~ 0.1-0.5 rad/s
  * dutch roll     - lightly damped, ω_n ~ 1-5 rad/s
  * roll subsidence - real, fast (highly damped)
  * spiral         - real, slow (often unstable)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import Aircraft
from .dynamics import state_derivative
from .trim import TrimState


def linearize(ac: Aircraft, x_trim: np.ndarray, u_trim: np.ndarray,
              eps_x: float = 1e-5, eps_u: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians: A = ∂f/∂x (13x13), B = ∂f/∂u (13x4)."""
    n, m = 13, 4
    A = np.zeros((n, n))
    for i in range(n):
        xp = x_trim.copy(); xp[i] += eps_x
        xm = x_trim.copy(); xm[i] -= eps_x
        A[:, i] = (state_derivative(0.0, xp, u_trim, ac)
                   - state_derivative(0.0, xm, u_trim, ac)) / (2.0 * eps_x)
    B = np.zeros((n, m))
    for j in range(m):
        up = u_trim.copy(); up[j] += eps_u
        um = u_trim.copy(); um[j] -= eps_u
        B[:, j] = (state_derivative(0.0, x_trim, up, ac)
                   - state_derivative(0.0, x_trim, um, ac)) / (2.0 * eps_u)
    return A, B


@dataclass
class Mode:
    name:    str
    eig:     complex
    omega_n: float
    zeta:    float

    @property
    def is_oscillatory(self) -> bool:
        return abs(self.eig.imag) > 1e-6


def _omega_zeta(e: complex) -> tuple[float, float]:
    omega_n = float(abs(e))
    if omega_n < 1e-12:
        return 0.0, 0.0
    zeta = -float(e.real) / omega_n
    return omega_n, float(np.clip(zeta, -1.0, 1.0))


def classify_modes(eigs: np.ndarray, tol: float = 1e-3) -> list[Mode]:
    """Heuristically label the non-trivial eigenvalues as classical modes."""
    eigs = np.asarray(eigs)
    nontrivial = eigs[np.abs(eigs) > tol]

    # Group complex conjugates.
    pairs: list[complex] = []
    reals: list[float] = []
    used = np.zeros(len(nontrivial), dtype=bool)
    for i, e in enumerate(nontrivial):
        if used[i]:
            continue
        if abs(e.imag) < 1e-6:
            reals.append(float(e.real)); used[i] = True
            continue
        for j in range(i + 1, len(nontrivial)):
            if not used[j] and abs(nontrivial[j] - np.conj(e)) < 1e-5:
                pairs.append(e if e.imag > 0 else nontrivial[j])
                used[i] = used[j] = True
                break
        else:
            pairs.append(e); used[i] = True

    # Order pairs by natural frequency. Heuristic mapping:
    #   slowest pair -> phugoid; fastest pair -> short period;
    #   middle pair (if present) -> dutch roll.
    pairs.sort(key=lambda e: abs(e))
    reals.sort(key=lambda r: abs(r))

    out: list[Mode] = []
    if len(pairs) >= 1:
        out.append(Mode("phugoid", pairs[0], *_omega_zeta(pairs[0])))
    if len(pairs) == 2:
        out.append(Mode("short_period", pairs[1], *_omega_zeta(pairs[1])))
    elif len(pairs) >= 3:
        out.append(Mode("dutch_roll",   pairs[1], *_omega_zeta(pairs[1])))
        out.append(Mode("short_period", pairs[-1], *_omega_zeta(pairs[-1])))

    if len(reals) >= 1:
        # closest-to-zero real = spiral; most negative = roll subsidence
        out.append(Mode("spiral", complex(reals[0]), *_omega_zeta(complex(reals[0]))))
    if len(reals) >= 2:
        out.append(Mode("roll", complex(reals[-1]), *_omega_zeta(complex(reals[-1]))))

    return out


def flight_eigenvalues(ac: Aircraft, trim_state: TrimState) -> np.ndarray:
    """Eigenvalues of the linearized 13-state system at a trim point."""
    A, _ = linearize(ac, trim_state.x, trim_state.u)
    return np.linalg.eigvals(A)
