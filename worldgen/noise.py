"""Seamless procedural noise on the sphere.

Noise is 3D gradient (Perlin) noise evaluated at points on the unit sphere,
so it has no seams or polar distortion. All functions are vectorised over
arrays of unit vectors and are deterministic for a given seed.
"""

from __future__ import annotations

import numpy as np

from .util import named_rng

_GRADIENTS = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
], dtype=float)


def _permutation(seed: int, name: str) -> np.ndarray:
    """Return the doubled permutation table for one noise stream."""
    perm = named_rng(seed, f"noise.{name}").permutation(256)
    return np.concatenate([perm, perm]).astype(np.int64)


def _fade(t: np.ndarray) -> np.ndarray:
    """Return the smooth interpolation weight for fractional coordinates."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(points: np.ndarray, seed: int, name: str = "base") -> np.ndarray:
    """Return 3D gradient noise (roughly −1 to 1) at each point (n×3 array, any scale)."""
    perm = _permutation(seed, name)
    cell = np.floor(points).astype(np.int64)
    frac = points - cell
    cell &= 255
    u, v, w = (_fade(frac[:, k]) for k in range(3))

    def corner(dx: int, dy: int, dz: int) -> np.ndarray:
        """Return the gradient contribution of one lattice corner."""
        h = perm[perm[perm[cell[:, 0] + dx] + cell[:, 1] + dy] + cell[:, 2] + dz] % 12
        offset = frac - np.array([dx, dy, dz], dtype=float)
        return np.einsum("ij,ij->i", _GRADIENTS[h], offset)

    x00 = corner(0, 0, 0) + u * (corner(1, 0, 0) - corner(0, 0, 0))
    x10 = corner(0, 1, 0) + u * (corner(1, 1, 0) - corner(0, 1, 0))
    x01 = corner(0, 0, 1) + u * (corner(1, 0, 1) - corner(0, 0, 1))
    x11 = corner(0, 1, 1) + u * (corner(1, 1, 1) - corner(0, 1, 1))
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return y0 + w * (y1 - y0)


def fbm(unit_vectors: np.ndarray, seed: int, name: str, frequency: float = 2.0, octaves: int = 6,
        lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Return fractal (multi-octave) noise normalised to roughly −1 to 1.

    ``frequency`` is the number of noise cells across one planet radius for
    the first octave.
    """
    total = np.zeros(len(unit_vectors))
    amplitude, norm, f = 1.0, 0.0, frequency
    offsets = named_rng(seed, f"noise.{name}.offsets").uniform(0, 1000, size=(octaves, 3))
    for k in range(octaves):
        total += amplitude * perlin(unit_vectors * f + offsets[k], seed, f"{name}.{k}")
        norm += amplitude
        amplitude *= gain
        f *= lacunarity
    return total / norm / 0.7   # single-octave Perlin rarely exceeds ±0.7; result has std ≈ 0.22


def ridged(unit_vectors: np.ndarray, seed: int, name: str, frequency: float = 2.0, octaves: int = 5) -> np.ndarray:
    """Return ridged fractal noise in 0–1, with sharp crests suited to mountain ranges."""
    total = np.zeros(len(unit_vectors))
    amplitude, norm, f = 1.0, 0.0, frequency
    offsets = named_rng(seed, f"noise.{name}.offsets").uniform(0, 1000, size=(octaves, 3))
    for k in range(octaves):
        r = 1.0 - np.abs(perlin(unit_vectors * f + offsets[k], seed, f"{name}.{k}")) / 0.7
        total += amplitude * np.clip(r, 0, 1) ** 2
        norm += amplitude
        amplitude *= 0.5
        f *= 2.0
    return total / norm


def warp(unit_vectors: np.ndarray, seed: int, name: str, strength: float, frequency: float = 1.5) -> np.ndarray:
    """Return the points displaced by a smooth noise field and projected back onto the sphere.

    ``strength`` is the typical displacement in radians; warping makes
    boundaries between regions irregular.
    """
    displacement = np.column_stack([
        fbm(unit_vectors, seed, f"{name}.x", frequency, octaves=4),
        fbm(unit_vectors, seed, f"{name}.y", frequency, octaves=4),
        fbm(unit_vectors, seed, f"{name}.z", frequency, octaves=4),
    ])
    moved = unit_vectors + strength * displacement
    return moved / np.linalg.norm(moved, axis=1, keepdims=True)
