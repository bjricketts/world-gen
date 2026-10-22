"""Samplers from the global grid onto arbitrary points: rasters, lattices and mesh vertices.

A sampler gives each point the three grid cells of the triangle around it with
barycentric weights, so continuous fields are reproduced exactly at the cells
and blend linearly between them, and the nearest cell for categories. It has
the interface of the projected-map sampler, so every map field's colours can be
sampled at any set of points.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..grid import SphereGrid, build_grid
from ..render.raster import Sampler, lonlat_vectors

CHUNK = 2_000_000           # points sampled per batch, which bounds memory
EARTH_RADIUS_M = 6.371e6    # map extents are given in metres on Earth's radius, as cartopy's projections are


@lru_cache(maxsize=2)
def _global_sampler(size: int):
    """Return the barycentric sampler of a grid size (cached)."""
    from ..zoom.inherit import GlobalSampler

    return GlobalSampler(build_grid(size))


def point_sampler(grid: SphereGrid, vectors: np.ndarray, shape: tuple[int, ...] | None = None,
                  extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)) -> Sampler:
    """Return a sampler of grid fields at unit vectors, reshaped to ``shape`` (default: one row of points)."""
    vectors = np.asarray(vectors, dtype=float)
    gs = _global_sampler(grid.size)
    m = len(vectors)
    tris = np.empty((m, 3), dtype=np.int64)
    weights = np.empty((m, 3))
    for start in range(0, m, CHUNK):
        tris[start:start + CHUNK], weights[start:start + CHUNK] = gs.barycentric(vectors[start:start + CHUNK])
    nearest = grid.nearest(vectors, k=1)[1]
    shape = tuple(shape) if shape is not None else (1, m)
    return Sampler(shape=shape, extent=extent, valid=np.ones(m, dtype=bool), nearest=nearest, neighbours=tris,
                   weights=weights)


@lru_cache(maxsize=4)
def equirect_sampler(size: int, width: int) -> Sampler:
    """Return the sampler of a global equirectangular raster ``width`` × ``width/2`` on a grid size (cached).

    Pixel centres sit on a regular latitude–longitude lattice from 180° W and
    90° N; the extent is in metres on Earth's radius, as for map images.
    """
    rows, cols = width // 2, width
    _, _, vectors = lonlat_vectors(rows, cols)
    half = np.pi * EARTH_RADIUS_M
    return point_sampler(build_grid(size), vectors, (rows, cols), (-half, half, -half / 2, half / 2))
