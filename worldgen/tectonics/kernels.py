"""Compiled per-point kernels for the tectonic simulation."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .crust import FRONT_ANDEAN, FRONT_ARC


@njit(parallel=True, cache=True)
def rotate_points(points, plate, rotvec):
    """Return the points rotated by their plate's rotation vector (axis × angle)."""
    out = np.empty_like(points)
    for i in prange(points.shape[0]):
        kx, ky, kz = rotvec[plate[i], 0], rotvec[plate[i], 1], rotvec[plate[i], 2]
        theta = np.sqrt(kx * kx + ky * ky + kz * kz)
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        if theta == 0.0:
            out[i, 0], out[i, 1], out[i, 2] = x, y, z
            continue
        kx, ky, kz = kx / theta, ky / theta, kz / theta
        c, s = np.cos(theta), np.sin(theta)
        dot = kx * x + ky * y + kz * z
        cx, cy, cz = ky * z - kz * y, kz * x - kx * z, kx * y - ky * x
        rx = x * c + cx * s + kx * dot * (1.0 - c)
        ry = y * c + cy * s + ky * dot * (1.0 - c)
        rz = z * c + cz * s + kz * dot * (1.0 - c)
        norm = np.sqrt(rx * rx + ry * ry + rz * rz)
        out[i, 0], out[i, 1], out[i, 2] = rx / norm, ry / norm, rz / norm
    return out


@njit(cache=True)
def _closing_speed(xi, xj, wi, wj, radius_m):
    """Return how fast point i approaches point j (m/Myr, positive when converging)."""
    vi = np.array([wi[1] * xi[2] - wi[2] * xi[1], wi[2] * xi[0] - wi[0] * xi[2], wi[0] * xi[1] - wi[1] * xi[0]])
    vj = np.array([wj[1] * xj[2] - wj[2] * xj[1], wj[2] * xj[0] - wj[0] * xj[2], wj[0] * xj[1] - wj[1] * xj[0]])
    d = xj - xi
    dn = np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if dn == 0.0:
        return np.sqrt(((vi - vj) ** 2).sum()) * radius_m
    return ((vi - vj) * d).sum() / dn * radius_m


@njit(cache=True)
def find_contacts(points, neighbours, plate, continental, ocean_age, alive, omega, radius_m, n_plates):
    """Resolve converging point pairs from different plates.

    Returns the points consumed by subduction, the overriding front type and
    speed per point, the collision speed per point, and the number of
    collision contacts between each pair of plates.
    """
    n, k = neighbours.shape
    consumed = np.zeros(n, dtype=np.bool_)
    front_kind = np.zeros(n, dtype=np.int8)
    front_speed = np.zeros(n)
    collision_speed = np.zeros(n)
    pair_contacts = np.zeros((n_plates, n_plates), dtype=np.int64)
    for i in range(n):
        if not alive[i]:
            continue
        pi = plate[i]
        for kk in range(k):
            j = neighbours[i, kk]
            if j >= n:
                break
            pj = plate[j]
            if pj == pi or not alive[j] or consumed[j] or consumed[i]:
                continue
            speed = _closing_speed(points[i], points[j], omega[pi], omega[pj], radius_m)
            if speed <= 0.0:
                continue
            if continental[i] and continental[j]:
                collision_speed[i] = max(collision_speed[i], speed)
                collision_speed[j] = max(collision_speed[j], speed)
                pair_contacts[pi, pj] += 1
                pair_contacts[pj, pi] += 1
            elif continental[i]:
                consumed[j] = True
                front_kind[i] = FRONT_ANDEAN
                front_speed[i] = max(front_speed[i], speed)
            elif continental[j]:
                consumed[i] = True
                front_kind[j] = FRONT_ANDEAN
                front_speed[j] = max(front_speed[j], speed)
            else:
                # The older (denser) oceanic plate sinks.
                if ocean_age[j] > ocean_age[i] or (ocean_age[j] == ocean_age[i] and pj > pi):
                    consumed[j] = True
                    if front_kind[i] == 0:
                        front_kind[i] = FRONT_ARC
                    front_speed[i] = max(front_speed[i], speed)
                else:
                    consumed[i] = True
                    if front_kind[j] == 0:
                        front_kind[j] = FRONT_ARC
                    front_speed[j] = max(front_speed[j], speed)
    return consumed, front_kind, front_speed, collision_speed, pair_contacts
