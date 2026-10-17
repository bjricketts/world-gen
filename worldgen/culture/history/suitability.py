"""Suitability of the land for an organic history: six components per cell, each from 0 to 1.

- ``fresh_water``: rivers, lakes and a wet climate.
- ``arable``: land a crop grows on, the fuzzy minimum of growing season, moisture,
  slope and permafrost (after Zabel et al. 2014).
- ``access``: closeness to the coast and to large rivers, the routes of early trade.
- ``resources``: the richest potential among the metals, salt and coal (milestone 8 classes).
- ``climate``: the mean annual temperature people favour.
- ``defensibility``: rugged ground (the terrain ruggedness index of Nunn & Puga
  2012, from the sub-grid slope), sea on the cell's edges and a large river.

The planet does not change during a history, so the components are computed once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.special import erf

from ... import heuristics as h
from ...surface.fields import Terrain
from ..suitability import Covariates

COMPONENTS = ("fresh_water", "arable", "access", "resources", "climate", "defensibility")


def ramp(x, low: float, high: float) -> np.ndarray:
    """Return 0 below ``low``, 1 above ``high`` and a straight line between."""
    return np.clip((np.asarray(x, dtype=float) - low) / (high - low), 0.0, 1.0)


@dataclass
class Suitability:
    """The suitability components of every cell, and the ruggedness index behind defensibility."""

    fresh_water: np.ndarray
    arable: np.ndarray
    access: np.ndarray
    resources: np.ndarray
    climate: np.ndarray
    defensibility: np.ndarray
    ruggedness_index: np.ndarray        # Nunn & Puga's terrain ruggedness index (hundreds of metres)
    arable_limit: np.ndarray            # which factor limits the arable component (index into ARABLE_LIMITS)

    def as_array(self) -> np.ndarray:
        """Return the components stacked (component × cell), in ``COMPONENTS`` order."""
        return np.array([getattr(self, c) for c in COMPONENTS])

    def to_dataarray(self) -> xr.DataArray:
        """Return the components as a (component × cell) layer."""
        return xr.DataArray(self.as_array(), dims=("component", "cell"), coords={"component": list(COMPONENTS)},
                            attrs={"long_name": "suitability components of the organic history (0–1)"})


ARABLE_LIMITS = ("none", "season", "moisture", "slope", "permafrost")


def subgrid_slope(world) -> np.ndarray:
    """Return each cell's typical sub-grid slope (rise over run) from its landform."""
    names = {int(t): t.name for t in Terrain}
    return np.array([h.SUBGRID_SLOPE.get(names.get(int(t), ""), h.SUBGRID_SLOPE_OTHER)
                     for t in world.surface["terrain"].values])


def gentle_share(slope: np.ndarray) -> np.ndarray:
    """Return the share of a cell's ground below the crop slope limit, for half-normal slopes with the given mean."""
    sigma = np.maximum(slope, 1e-6) * np.sqrt(np.pi / 2.0)
    return erf(h.ARABLE_MAX_SLOPE / (sigma * np.sqrt(2.0)))


def arable_land(world, land: np.ndarray, slope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the arable component and the factor that limits it in each cell."""
    s = world.surface
    monthly = s["monthly_temperature"].values - 273.15
    season = ramp((monthly >= h.ARABLE_GROWTH_MIN_C).sum(axis=0), *h.ARABLE_SEASON_MONTHS)
    if "precipitation" in s and "potential_evaporation" in s:
        wetness = s["precipitation"].values / np.maximum(s["potential_evaporation"].values, 1e-6)
        moisture = ramp(wetness, *h.ARABLE_MOISTURE)
    else:
        moisture = np.zeros(land.size)
    gentle = gentle_share(slope)
    thawed = (s["air_temperature"].values - 273.15 > h.ARABLE_PERMAFROST_C).astype(float)
    factors = np.array([season, moisture, gentle, thawed])
    arable = np.where(land, factors.min(axis=0), 0.0)
    limit = np.where(arable >= 1.0, 0, 1 + np.argmin(factors, axis=0))
    return arable, np.where(land, limit, 0).astype(np.int8)


def ruggedness_index(slope: np.ndarray) -> np.ndarray:
    """Return Nunn & Puga's terrain ruggedness index (hundreds of metres) for ground of a given mean slope.

    Their index is √Σ(e_neighbour − e_centre)² over the eight neighbours of a
    30″ grid; with the mean slope standing for every neighbour difference it is
    √8 · slope · spacing.
    """
    return np.sqrt(8.0) * slope * h.RUGGEDNESS_SPACING_M / 100.0


def sea_edge_share(world, land: np.ndarray) -> np.ndarray:
    """Return the share of each cell's neighbours that are sea."""
    nb = world.grid.neighbours.tocsr()
    sea = (~land).astype(float)
    counts = np.diff(nb.indptr)
    total = np.add.reduceat(sea[nb.indices], nb.indptr[:-1]) if nb.indices.size else np.zeros(land.size)
    return np.where(counts > 0, total / np.maximum(counts, 1), 0.0)


def build_suitability(world, cov: Covariates, resource_potential: xr.DataArray | None = None) -> Suitability:
    """Return the suitability components of every cell of a world."""
    land = cov.land & ~cov.ice
    s = world.surface
    slope = subgrid_slope(world)
    lake = s["lake"].values.astype(bool) if "lake" in s else np.zeros(land.size, dtype=bool)
    lake_near = np.asarray(world.grid.neighbours.astype(bool) @ lake.astype(float)).ravel() > 0
    river = ramp(np.log10(np.maximum(cov.discharge, 1e-3)), *np.log10(h.FRESH_WATER_DISCHARGE_M3S))
    if "precipitation" in s and "potential_evaporation" in s:
        wet = ramp(s["precipitation"].values / np.maximum(s["potential_evaporation"].values, 1e-6),
                   *h.FRESH_WATER_WETNESS)
    else:
        wet = np.zeros(land.size)
    fresh = np.maximum.reduce([river, (lake | lake_near).astype(float), wet])
    arable, limit = arable_land(world, land, slope)
    sea = sea_edge_share(world, cov.land)
    coast = np.where(sea > 0, 1.0, np.exp(-cov.coast_km / h.ACCESS_COAST_KM))
    navigable = ramp(np.log10(np.maximum(cov.discharge, 1e-3)), *np.log10(h.ACCESS_RIVER_M3S))
    access = np.maximum(coast, navigable)
    names = [r for r in h.HISTORY_RESOURCES if resource_potential is not None
             and r in resource_potential["resource"].values]
    if names:
        resources = resource_potential.sel(resource=names).values.max(axis=0)
    else:
        resources = np.zeros(land.size)
    t = cov.temperature_c
    cold, warm = h.CLIMATE_COMFORT_C
    climate = np.minimum(ramp(t, h.CLIMATE_COLD_LIMIT_C, cold), 1.0 - ramp(t, warm, h.CLIMATE_HOT_LIMIT_C))
    tri = ruggedness_index(slope)
    rugged = ramp(tri, *h.DEFENCE_RUGGEDNESS)
    big_river = cov.discharge > h.RIVER_CROSSING_M3S
    defence = 1.0 - (1.0 - rugged) * (1.0 - h.DEFENCE_SEA * sea) * (1.0 - h.DEFENCE_RIVER * big_river)

    def on_land(x):
        return np.where(land, np.clip(x, 0.0, 1.0), 0.0)

    return Suitability(fresh_water=on_land(fresh), arable=on_land(arable), access=on_land(access),
                       resources=on_land(resources), climate=on_land(climate), defensibility=on_land(defence),
                       ruggedness_index=np.where(land, tri, 0.0), arable_limit=limit)
