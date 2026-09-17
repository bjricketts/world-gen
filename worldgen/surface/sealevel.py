"""Sea level from the ocean volume, or from a land-fraction target.

The ocean fills the basin connected to the lowest point of the planet.
Closed depressions covering at least ``ENCLOSED_SEA_MIN_AREA`` of the
surface are enclosed seas: they fill to the same level as the ocean.
Smaller closed depressions stay dry here; lakes are handled by the
hydrology package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components

from ..grid import SphereGrid
from .. import heuristics as h
from ..hydrology.graph import pass_levels, pass_levels_from
from .fields import Crust, SurfaceFields, Terrain

DEPRESSION_MIN_DEPTH_M = 1.0


@dataclass
class OceanFill:
    """How water fills the ocean basin of one elevation field."""

    elevation: np.ndarray       # elevations the fill was computed for (m)
    pass_level: np.ndarray      # water level at which each cell joins the ocean (m)
    cell_area_m2: float
    _order: np.ndarray = None
    _cum_z: np.ndarray = None

    def __post_init__(self) -> None:
        """Sort cells by the level at which they flood."""
        self._order = np.argsort(self.pass_level, kind="stable")
        self._cum_z = np.concatenate([[0.0], np.cumsum(self.elevation[self._order])])

    def volume(self, level: float) -> float:
        """Return the ocean volume (m³) below a water level."""
        k = int(np.searchsorted(self.pass_level[self._order], level, side="left"))
        return float((k * level - self._cum_z[k]) * self.cell_area_m2)

    def level_for_volume(self, volume_m3: float) -> float:
        """Return the water level that holds a given ocean volume."""
        if volume_m3 <= 0.0:
            return float(self.pass_level.min())
        passes = self.pass_level[self._order]
        n = passes.size
        # Volume is piecewise linear in the level; find the segment by bisection on cell count.
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.volume(passes[mid]) >= volume_m3:
                hi = mid
            else:
                lo = mid + 1
        k = lo
        level = (volume_m3 / self.cell_area_m2 + self._cum_z[k]) / k
        return float(level if k == n else min(level, passes[k]))

    def level_for_land_fraction(self, land_fraction: float) -> float:
        """Return the water level that leaves ``land_fraction`` of the surface dry."""
        land_fraction = float(np.clip(land_fraction, 0.0, 1.0))
        passes = np.sort(self.pass_level)
        k = int(round((1.0 - land_fraction) * passes.size))
        if k <= 0:
            return float(passes[0])
        if k >= passes.size:
            return float(passes[-1] + 1.0)
        # A basin that joins the ocean at one spill level floods all at once:
        # choose the level just below or just above it, whichever is closer.
        lo = int(np.searchsorted(passes, passes[k], side="left"))
        hi = int(np.searchsorted(passes, passes[k], side="right"))
        if k - lo <= hi - k and lo > 0:
            return float(0.5 * (passes[lo - 1] + passes[lo]))
        if hi >= passes.size:
            return float(passes[-1] + 1.0)
        return float(0.5 * (passes[hi - 1] + passes[hi]))


def ocean_fill(grid: SphereGrid, elevation: np.ndarray, radius_m: float) -> OceanFill:
    """Return the ocean fill of an elevation field, flooding from its lowest cell."""
    z = elevation.astype(float)
    adj = grid.neighbours
    lowest = int(np.argmin(z))
    level = pass_levels(z, adj.indptr, adj.indices, lowest)
    seeds = [lowest] + enclosed_sea_floors(grid, z, level)
    if len(seeds) > 1:
        level = pass_levels_from(z, adj.indptr, adj.indices, np.array(seeds, dtype=np.int64))
    return OceanFill(elevation=z, pass_level=level, cell_area_m2=4 * np.pi * radius_m**2 / grid.size)


def enclosed_sea_floors(grid: SphereGrid, z: np.ndarray, pass_level: np.ndarray) -> list[int]:
    """Return the lowest cell of each closed depression large enough to hold an enclosed sea."""
    depression = pass_level > z + DEPRESSION_MIN_DEPTH_M
    if not depression.any():
        return []
    cells = np.flatnonzero(depression)
    adj = grid.neighbours[cells][:, cells]
    count, label = connected_components(adj, directed=False)
    sizes = np.bincount(label, minlength=count)
    floors = []
    for k in np.flatnonzero(sizes >= h.ENCLOSED_SEA_MIN_AREA * z.size):
        members = cells[label == k]
        floors.append(int(members[np.argmin(z[members])]))
    return floors


def apply_sea_level(fields: SurfaceFields, fill: OceanFill | None, level: float) -> np.ndarray:
    """Shift elevations so sea level is zero and return the ocean mask.

    Without a fill (no surface water), ``level`` is the datum subtracted and
    no cell is ocean.
    """
    if fill is None:
        fields.elevation -= level
        return np.zeros(fields.grid.size, dtype=bool)
    ocean = fill.pass_level < level
    fields.elevation -= level
    plain = fields.terrain == Terrain.PLAIN
    fields.terrain[ocean & plain & (fields.crust == Crust.CONTINENTAL)] = Terrain.SHELF
    fields.terrain[ocean & plain & (fields.crust != Crust.CONTINENTAL)] = Terrain.ABYSSAL_PLAIN
    return ocean
