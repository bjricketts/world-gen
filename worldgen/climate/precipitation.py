"""Precipitation on the surface grid: ocean-equivalent rain carried inland by the prevailing winds.

Oceans receive the ocean-equivalent precipitation of the moist energy-balance
model. Over land, moisture is carried from the coast by the surface winds
of each season and is lost with distance and when air rises over high
ground, which leaves rain shadows; windward slopes get extra rain. The wind
pattern follows the circulation cells, shifted with the rising branch
through the year, plus an onshore component toward regions that are warm
for their latitude (monsoons). Synchronously rotating planets have surface
winds converging on the substellar point.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

from .. import heuristics as h
from ..grid import SphereGrid

SEASONS = 4


def tangent_gradient(grid: SphereGrid, values: np.ndarray) -> np.ndarray:
    """Return the gradient of a field along the sphere (per radian) at each point, from neighbour differences."""
    adj = grid.neighbours
    rows = np.repeat(np.arange(grid.size), np.diff(adj.indptr))
    cols = adj.indices
    d = grid.points[cols] - grid.points[rows]
    w = (values[cols] - values[rows]) / np.maximum(adj.data, 1e-12) ** 2
    count = np.maximum(np.diff(adj.indptr), 1)
    grad = np.stack([np.bincount(rows, weights=w * d[:, k], minlength=grid.size) for k in range(3)], axis=1)
    grad *= 2.0 / count[:, None]
    grad -= np.einsum("ij,ij->i", grad, grid.points)[:, None] * grid.points
    return grad


def wind_directions(points: np.ndarray, edge_deg: float, synchronous: bool, rising_deg: float = 0.0,
                    warm_gradient: np.ndarray | None = None) -> np.ndarray:
    """Return unit vectors of the prevailing surface wind at each point.

    Circulation bands are placed relative to the rising branch at
    ``rising_deg``. ``warm_gradient`` (K per radian) adds flow toward
    regions warmer than their latitude: onshore in summer, offshore in winter.
    """
    x, y, z = points.T
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1, 1))
    east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
    north = np.column_stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    if synchronous:
        target = np.array([1.0, 0.0, 0.0])
        wind = target - (points @ target)[:, None] * points
    else:
        rel = np.degrees(lat) - rising_deg
        a = np.abs(rel)
        side = np.where(rel >= 0.0, 1.0, -1.0)[:, None]
        trades = a < edge_deg
        westerlies = (a >= edge_deg) & (a < edge_deg + 30.0)
        zonal = np.where(trades | ~westerlies, -1.0, 1.0)[:, None]
        meridional = np.where(trades, -0.4, np.where(westerlies, 0.3, -0.3))[:, None] * side
        wind = zonal * east + meridional * north
    if warm_gradient is not None:
        wind = wind + warm_gradient * h.MONSOON_WIND_PER_K_RAD
    norm = np.linalg.norm(wind, axis=1, keepdims=True)
    return wind / np.maximum(norm, 1e-12)


@njit(cache=True)
def _transport_weights(points, indptr, indices, lengths, wind, height, decay_rad, rise_m, mixing):
    """Return the share of each neighbour's moisture that reaches each cell, and the upwind rise per cell.

    A cell's moisture is the weighted sum of its neighbours' moisture: mostly
    from upwind, partly from all sides (eddies), reduced by distance and by
    the height the air climbs.
    """
    n = points.shape[0]
    nnz = indptr[n]
    weights = np.zeros(nnz)
    rise = np.zeros(n)
    for i in range(n):
        start, stop = indptr[i], indptr[i + 1]
        count = stop - start
        total_w = 0.0
        lifted = 0.0
        for k in range(start, stop):
            j = indices[k]
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            align = (wind[i, 0] * dx + wind[i, 1] * dy + wind[i, 2] * dz) / norm
            if align > 0.0:
                total_w += align
                lifted += align * max(height[i] - height[j], 0.0)
        advect = (1.0 - mixing) if total_w > 0.0 else 0.0
        eddy = 1.0 - advect
        if total_w > 0.0:
            rise[i] = lifted / total_w
        for k in range(start, stop):
            j = indices[k]
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            align = (wind[i, 0] * dx + wind[i, 1] * dy + wind[i, 2] * dz) / norm
            share = eddy / count
            if align > 0.0:
                share += advect * align / total_w
            up = max(height[i] - height[j], 0.0)
            weights[k] = share * np.exp(-lengths[k] / decay_rad[i]) * np.exp(-up / rise_m)
    return weights, rise


def carry_moisture(grid: SphereGrid, radius_m: float, wind: np.ndarray, height: np.ndarray,
                   ocean: np.ndarray, decay_km: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return the moisture (0–1, ocean = 1) reaching each cell in steady state, and the upwind rise (m).

    ``decay_km`` is the distance over which air loses its moisture, per cell or for all cells.
    """
    adj = grid.neighbours
    decay = np.broadcast_to(np.asarray(decay_km, dtype=float), (grid.size,)) * 1e3 / radius_m
    weights, rise = _transport_weights(grid.points, adj.indptr, adj.indices, adj.data, wind, height,
                                       np.ascontiguousarray(decay), h.RAIN_SHADOW_HEIGHT_M, h.MOISTURE_MIXING)
    moisture = np.ones(grid.size)
    land = np.flatnonzero(~ocean)
    if land.size == 0:
        return moisture, rise
    transfer = csr_matrix((weights, adj.indices, adj.indptr), shape=adj.shape)
    from_land = transfer[land][:, land]
    from_ocean = np.asarray(transfer[land][:, np.flatnonzero(ocean)].sum(axis=1)).ravel()
    system = identity(land.size, format="csr") - from_land
    moisture[land] = np.clip(spsolve(system.tocsc(), from_ocean), 0.0, 1.0)
    return moisture, rise


def rain_terrain(grid: SphereGrid, height_m: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Return the broad high ground that dries the air (m above the rain-shadow base)."""
    from ..surface.distance import smooth   # imported here: the surface package builds on the climate package

    land_height = smooth(grid, np.where(ocean, 0.0, np.maximum(height_m, 0.0)), h.RAIN_TERRAIN_SMOOTHING)
    return np.maximum(land_height - h.RAIN_SHADOW_BASE_M, 0.0)


def land_rain_factor(grid: SphereGrid, radius_m: float, terrain: np.ndarray, ocean: np.ndarray, wind: np.ndarray,
                     decay_km: np.ndarray | float) -> np.ndarray:
    """Return the share of the ocean-equivalent rain that falls on each cell (1 over the ocean)."""
    moisture, rise = carry_moisture(grid, radius_m, wind, terrain, ocean, decay_km)
    uplift = 1.0 + h.OROGRAPHIC_BOOST * (1.0 - np.exp(-rise / h.RAIN_SHADOW_HEIGHT_M))
    return np.where(ocean, 1.0, moisture * uplift)


def seasonal_factors(factors: np.ndarray, months: int = 12) -> np.ndarray:
    """Return monthly values interpolated periodically from evenly spaced seasonal values (axis 0)."""
    seasons = factors.shape[0]
    if seasons == 1:
        return np.repeat(factors, months, axis=0)
    pos = (np.arange(months) + 0.5) / months * seasons - 0.5
    lo = np.floor(pos).astype(int)
    frac = (pos - lo)[:, None]
    return factors[lo % seasons] * (1.0 - frac) + factors[(lo + 1) % seasons] * frac
