"""Inherit the global surface into a zoomed region, and downscale its climate (milestone 7).

A region's coarse fields come from the planet grid by barycentric interpolation
on its triangulation: continuous fields (elevation, climate, the structural
fabric) are the linear blend of the three surrounding cells, so the region
reproduces the global values at the grid points and never drifts from them.
Categorical fields (crust, terrain, biome) take the nearest cell.

The inherited climate is the planet's climate at the coarse elevation. When
detail adds sub-grid relief, ``downscale_climate`` corrects the temperature by
the elevation lapse rate and the precipitation by the local rain shadow of that
added relief, so the large-scale climate is kept and only the sub-grid part is
new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..grid import SphereGrid
from ..state import PlanetState
from ..surface.fabric import tangent_gradient
from .tiles import RegionGrid

CATEGORICAL = ("crust", "terrain", "biome", "koppen")
CONTINUOUS = ("air_temperature", "precipitation", "potential_evaporation", "runoff")


class GlobalSampler:
    """Barycentric and nearest interpolation from a global grid onto arbitrary directions."""

    def __init__(self, grid: SphereGrid):
        self.grid = grid
        self.points = grid.points
        self.tris = grid.triangles
        self.tree = grid.tree
        incident: list[list[int]] = [[] for _ in range(grid.size)]
        for t, (a, b, c) in enumerate(self.tris):
            incident[a].append(t)
            incident[b].append(t)
            incident[c].append(t)
        width = max(len(row) for row in incident)
        self.node_tris = np.full((grid.size, width), -1, dtype=np.int64)
        for i, row in enumerate(incident):
            self.node_tris[i, :len(row)] = row

    @staticmethod
    def _weights(a: np.ndarray, b: np.ndarray, c: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Return the barycentric weights of each direction in its triangle (via the ray-plane hit)."""
        normal = np.cross(b - a, c - a)
        denom = np.einsum("ij,ij->i", q, normal)
        scale = np.einsum("ij,ij->i", a, normal) / np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        point = scale[:, None] * q
        v0, v1, v2 = b - a, c - a, point - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        det = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-30, 1e-30, d00 * d11 - d01 * d01)
        wb = (d11 * d20 - d01 * d21) / det
        wc = (d00 * d21 - d01 * d20) / det
        return np.stack([1.0 - wb - wc, wb, wc], axis=1)

    def barycentric(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return, per direction, the three grid cells and their weights (weights sum to 1)."""
        queries = np.asarray(queries, dtype=float)
        m = len(queries)
        _, near = self.tree.query(queries, k=3)
        candidates = self.node_tris[near].reshape(m, -1)
        verts = np.zeros((m, 3), dtype=np.int64)
        weights = np.zeros((m, 3))
        found = np.zeros(m, dtype=bool)
        for col in range(candidates.shape[1]):
            if found.all():
                break
            active = (~found) & (candidates[:, col] >= 0)
            if not active.any():
                continue
            rows = np.flatnonzero(active)
            tri = self.tris[candidates[rows, col]]
            w = self._weights(self.points[tri[:, 0]], self.points[tri[:, 1]], self.points[tri[:, 2]], queries[rows])
            inside = (w >= -1e-9).all(axis=1)
            hit = rows[inside]
            verts[hit] = tri[inside]
            weights[hit] = w[inside]
            found[hit] = True
        if not found.all():
            # A direction outside every candidate triangle takes its nearest cell.
            miss = ~found
            verts[miss] = near[miss, :1]
            weights[miss] = np.array([1.0, 0.0, 0.0])
        return verts, weights

    def interpolate(self, field: np.ndarray, verts: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return a continuous field blended onto the query directions (scalar or per-cell vector)."""
        field = np.asarray(field)
        if field.ndim == 1:
            return (field[verts] * weights).sum(axis=1)
        return (field[verts] * weights[:, :, None]).sum(axis=1)

    def nearest(self, queries: np.ndarray) -> np.ndarray:
        """Return the index of the nearest grid cell to each direction."""
        return self.tree.query(np.asarray(queries, dtype=float), k=1)[1]


def _interpolate_fabric(sampler: GlobalSampler, fabric: np.ndarray, verts: np.ndarray, weights: np.ndarray,
                        queries: np.ndarray) -> np.ndarray:
    """Return the structural grain blended onto the region, with the undirected sign resolved."""
    vectors = fabric[verts]                                   # (m, 3 cells, xyz)
    reference = vectors[np.arange(len(verts)), weights.argmax(axis=1)]
    sign = np.sign(np.einsum("mvc,mc->mv", vectors, reference))
    sign[sign == 0.0] = 1.0
    grain = (vectors * sign[:, :, None] * weights[:, :, None]).sum(axis=1)
    grain -= np.einsum("ij,ij->i", grain, queries)[:, None] * queries   # into the tangent plane
    magnitude = np.linalg.norm(grain, axis=1, keepdims=True)
    return grain / np.maximum(magnitude, 1.0)               # clip the grain to unit length, keeping zeros zero


@dataclass
class InheritedRegion:
    """The global surface sampled onto a region lattice, before detail is added."""

    grid: RegionGrid
    radius_m: float
    elevation: np.ndarray                        # coarse base elevation, sea level = 0
    ocean: np.ndarray
    fabric: np.ndarray                           # (n, 3) structural grain
    temperature_k: np.ndarray
    wind: np.ndarray                             # (n, 3) prevailing annual surface wind (unit vectors)
    precipitation_m: Optional[np.ndarray] = None
    evaporation_m: Optional[np.ndarray] = None
    categorical: Optional[dict] = None
    lapse_k_per_m: float = h.LAPSE_RATE_K_PER_M


def prevailing_wind(points: np.ndarray, state: PlanetState) -> np.ndarray:
    """Return the annual-mean prevailing surface wind (unit vectors) at each direction."""
    from ..climate import planet_climate
    from ..climate.physics import hadley_edge_deg
    from ..climate.precipitation import wind_directions

    setting, _ = planet_climate(state)
    edge = hadley_edge_deg(setting.rotation_period_s / 3600.0)
    return wind_directions(points, edge, setting.synchronous, 0.0)


def inherit_region(world, region: RegionGrid) -> InheritedRegion:
    """Return the global surface of ``world`` sampled onto ``region``.

    Continuous fields are interpolated barycentrically and categorical fields
    take the nearest cell; the ocean mask follows the interpolated elevation.
    """
    surface = world.surface
    if surface is None:
        raise ValueError("this world has no surface to zoom into")
    sampler = GlobalSampler(world.grid)
    queries = region.points
    verts, weights = sampler.barycentric(queries)

    elevation = sampler.interpolate(surface["elevation"].values, verts, weights)
    fabric = _interpolate_fabric(sampler, surface["fabric"].values.T, verts, weights, queries)
    temperature = sampler.interpolate(surface["air_temperature"].values, verts, weights)
    precipitation = (sampler.interpolate(surface["precipitation"].values, verts, weights)
                     if "precipitation" in surface else None)
    evaporation = (sampler.interpolate(surface["potential_evaporation"].values, verts, weights)
                   if "potential_evaporation" in surface else None)
    nearest = sampler.nearest(queries)
    categorical = {name: surface[name].values[nearest] for name in CATEGORICAL if name in surface}

    return InheritedRegion(
        grid=region,
        radius_m=float(surface.attrs["radius_m"]),
        elevation=elevation,
        ocean=elevation < 0.0,
        fabric=fabric,
        temperature_k=temperature,
        wind=prevailing_wind(queries, world.state),
        precipitation_m=precipitation,
        evaporation_m=evaporation,
        categorical=categorical,
    )


def lapse_temperature(temperature_k: np.ndarray, delta_height_m: np.ndarray,
                      lapse_k_per_m: float = h.LAPSE_RATE_K_PER_M) -> np.ndarray:
    """Return the temperature after descending or climbing ``delta_height_m`` of added relief."""
    return temperature_k - lapse_k_per_m * delta_height_m


def orographic_factor(region: RegionGrid, radius_m: float, wind: np.ndarray,
                      delta_height_m: np.ndarray) -> np.ndarray:
    """Return the local rain-shadow multiplier of the sub-grid relief (windward wetter, lee drier)."""
    slope = np.einsum("ij,ij->i", wind, tangent_gradient(region, delta_height_m)) / radius_m
    factor = np.exp(h.OROGRAPHIC_BOOST * np.tanh(slope / h.ZOOM_OROGRAPHIC_SLOPE))
    return np.clip(factor, 1.0 / h.ZOOM_OROGRAPHIC_CAP, h.ZOOM_OROGRAPHIC_CAP)


def downscale_climate(inherited: InheritedRegion, height_m: np.ndarray
                      ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return the temperature, precipitation and evaporation for a detailed local elevation.

    Temperature falls with the added height by the lapse rate; precipitation
    carries the local rain shadow of the added relief. Evaporation is left as
    inherited.
    """
    delta = height_m - inherited.elevation
    temperature = lapse_temperature(inherited.temperature_k, delta, inherited.lapse_k_per_m)
    precipitation = inherited.precipitation_m
    if precipitation is not None:
        factor = orographic_factor(inherited.grid, inherited.radius_m, inherited.wind, delta)
        precipitation = np.where(inherited.ocean, precipitation, precipitation * factor)
    return temperature, precipitation, inherited.evaporation_m
