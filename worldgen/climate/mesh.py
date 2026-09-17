"""Operators on the climate grid and transfer of fields between the climate and surface grids.

The climate models run on a fixed Fibonacci grid (``CLIMATE_GRID_SIZE``
points). Diffusion uses the cotangent Laplacian of its triangulation.
Fields move to the surface grid by inverse-distance weighting of the three
nearest climate points, and back by averaging the surface cells nearest to
each climate point.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from .. import heuristics as h
from ..grid import SphereGrid, build_grid


def climate_grid() -> SphereGrid:
    """Return the grid the climate models run on."""
    return build_grid(h.CLIMATE_GRID_SIZE)


@lru_cache(maxsize=4)
def cotangent_weights(n: int) -> csr_matrix:
    """Return the symmetric edge weights of the cotangent Laplacian on the n-point grid (unit sphere, per cell area)."""
    grid = build_grid(n)
    p, tri = grid.points, grid.triangles
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        i, j, k = tri[:, a], tri[:, b], tri[:, c]
        u, v = p[i] - p[k], p[j] - p[k]
        cot = np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
        rows += [i, j]
        cols += [j, i]
        vals += [0.5 * cot, 0.5 * cot]
    w = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    w.sum_duplicates()
    w.data = np.maximum(w.data, 0.0) / (4.0 * np.pi / n)
    return w


def laplacian(weights: csr_matrix, conductance: np.ndarray | None = None) -> csr_matrix:
    """Return the diffusion operator Σ_j w_ij K_ij (T_j − T_i); ``conductance`` gives K per stored edge."""
    w = weights.copy()
    if conductance is not None:
        w.data = w.data * conductance
    return (w - diags(np.asarray(w.sum(axis=1)).ravel())).tocsr()


def edge_mean(weights: csr_matrix, values: np.ndarray) -> np.ndarray:
    """Return the mean of ``values`` over the two ends of each stored edge, in storage order."""
    rows = np.repeat(np.arange(weights.shape[0]), np.diff(weights.indptr))
    return 0.5 * (values[rows] + values[weights.indices])


@dataclass(frozen=True)
class Regridder:
    """Transfers fields between the climate grid and a surface grid."""

    to_surface_index: np.ndarray     # (n_surface, 3) nearest climate points
    to_surface_weight: np.ndarray    # (n_surface, 3) interpolation weights
    to_climate_index: np.ndarray     # (n_surface,) nearest climate point of each surface cell
    counts: np.ndarray               # surface cells per climate point

    def to_surface(self, values: np.ndarray) -> np.ndarray:
        """Return climate-grid values (last axis) interpolated to the surface grid."""
        return np.sum(values[..., self.to_surface_index] * self.to_surface_weight, axis=-1)

    def to_climate(self, values: np.ndarray) -> np.ndarray:
        """Return surface-grid values averaged onto the climate grid; empty climate cells take the nearest value."""
        sums = np.bincount(self.to_climate_index, weights=values, minlength=self.counts.size)
        out = sums / np.maximum(self.counts, 1)
        empty = self.counts == 0
        if empty.any():
            out[empty] = out[self.fill_from[empty]]
        return out

    @property
    def fill_from(self) -> np.ndarray:
        """Return, for each climate point, the nearest climate point that has surface cells."""
        return _fill_sources(self.counts.size, tuple(np.flatnonzero(self.counts == 0)))


@lru_cache(maxsize=8)
def _fill_sources(size: int, empty: tuple[int, ...]) -> np.ndarray:
    """Return the nearest non-empty climate point for every climate point."""
    from scipy.spatial import cKDTree

    points = climate_grid().points
    mask = np.zeros(size, dtype=bool)
    mask[list(empty)] = True
    filled = np.flatnonzero(~mask)
    _, idx = cKDTree(points[filled]).query(points, k=1)
    return filled[idx]


@lru_cache(maxsize=4)
def regridder(surface_size: int) -> Regridder:
    """Return the cached transfer between the climate grid and the surface grid of the given size."""
    climate = climate_grid()
    surface = build_grid(surface_size)
    dist, idx = climate.tree.query(surface.points, k=3)
    weight = 1.0 / np.maximum(dist, 1e-9) ** 2
    exact = dist[:, 0] < 1e-9
    weight[exact] = [1.0, 0.0, 0.0]
    weight /= weight.sum(axis=1, keepdims=True)
    nearest = idx[:, 0]
    counts = np.bincount(nearest, minlength=climate.size)
    return Regridder(to_surface_index=idx, to_surface_weight=weight, to_climate_index=nearest, counts=counts)
