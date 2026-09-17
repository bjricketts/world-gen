"""Distances across the planet surface, measured along the grid."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..grid import SphereGrid


def distance_from(grid: SphereGrid, sources: np.ndarray, radius_km: float,
                  limit_km: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """Return the surface distance (km) from each cell to the nearest source cell, and that source's index.

    Cells farther than ``limit_km`` get distance ``inf`` and source −1.
    """
    sources = np.unique(np.asarray(sources, dtype=np.int64))
    n = grid.size
    if sources.size == 0:
        return np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
    limit = limit_km / radius_km if np.isfinite(limit_km) else np.inf
    dist, _, nearest = dijkstra(grid.neighbours, directed=False, indices=sources,
                                min_only=True, return_predecessors=True, limit=limit)
    nearest = nearest.astype(np.int64)
    nearest[~np.isfinite(dist)] = -1
    return dist * radius_km, nearest


def smooth(grid: SphereGrid, values: np.ndarray, steps: int) -> np.ndarray:
    """Return the values after ``steps`` passes of averaging each cell with its neighbours."""
    adj = grid.neighbours.copy()
    adj.data[:] = 1.0
    degree = np.asarray(adj.sum(axis=1)).ravel()
    out = values.astype(float)
    for _ in range(steps):
        out = 0.5 * out + 0.5 * (adj @ out) / degree
    return out
