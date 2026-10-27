"""
Hamilton quaternion utilities, scalar-first convention: q = [w, x, y, z].

Body-to-NED rotation: v_ned = R(q) @ v_body.
"""
from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix that rotates body-frame vectors to NED."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX (3-2-1) Euler to quaternion: psi yaw, theta pitch, phi roll."""
    cphi, sphi = np.cos(phi / 2), np.sin(phi / 2)
    cth,  sth  = np.cos(theta / 2), np.sin(theta / 2)
    cpsi, spsi = np.cos(psi / 2), np.sin(psi / 2)
    return np.array([
        cphi * cth * cpsi + sphi * sth * spsi,
        sphi * cth * cpsi - cphi * sth * spsi,
        cphi * sth * cpsi + sphi * cth * spsi,
        cphi * cth * spsi - sphi * sth * cpsi,
    ])


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion to ZYX (3-2-1) Euler: returns (phi, theta, psi)."""
    w, x, y, z = q
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    phi   = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    theta = np.arcsin(sinp)
    psi   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return phi, theta, psi


def quat_kinematics(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """qdot = 0.5 * q ⊗ [0, omega_body]."""
    p, qrate, r = omega_body
    omega_quat = np.array([0.0, p, qrate, r])
    return 0.5 * quat_mul(q, omega_quat)
