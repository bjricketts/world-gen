"""Geological context for resources: the tectonic setting of each cell and what the planet's past allows.

Settings follow the classes used for the world's sedimentary basins and ore
belts (Mann et al. 2003; Groves & Bierlein 2007): stable craton, sag basin,
passive margin, rift, foreland, strike-slip margin, subduction margin, orogen
and volcanic ground. They are read from the finished surface: plate
boundaries and their kind, crust, terrain, the kind and age of the last
mountain building, and the sediment the rivers laid down.

``DeepTime`` summarises the history that decides which deposits can exist at
all: how long the oceans held life, stayed anoxic, or have been oxygenated,
and how long plants have covered the land.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..surface.distance import distance_from
from ..surface.fields import Boundary, Crust, OrogenyKind, Terrain


class Setting(IntEnum):
    """Tectonic setting of a cell."""

    OCEAN = 0
    CRATON = 1
    SAG_BASIN = 2
    PASSIVE_MARGIN = 3
    RIFT = 4
    FORELAND = 5
    STRIKE_SLIP = 6
    SUBDUCTION_MARGIN = 7
    OROGEN = 8
    VOLCANIC = 9


@dataclass(frozen=True)
class DeepTime:
    """Durations (Gyr, up to the planet's present) that gate deposit types."""

    age_gyr: float
    life_gyr: float            # since life began in the sea or on the surface (0 without life)
    anoxic_ocean_gyr: float    # liquid ocean without free oxygen (iron formations)
    oxygenated_gyr: float      # since the first rise of oxygen (0 if it never rose)
    land_life_gyr: float       # since life covered the land (0 without land life)
    heat_flux_ratio: float     # surface heat flux relative to Earth's


@dataclass
class Geology:
    """Per-cell geological context of a world."""

    setting: np.ndarray        # Setting code
    land: np.ndarray           # above sea level and not ocean
    continental: np.ndarray    # continental crust (land or submerged margin)
    distance_km: dict          # distances to subduction, collision, rift, transform, coast, belts, volcanic ground
    nearest: dict              # for the belts: index of the nearest belt cell
    deep_time: DeepTime


def _first_event_gyr(state, kind: str):
    """Return the planet age (Gyr) of the first logged event of a kind, or None."""
    for e in state.events:
        if e.kind == kind:
            return e.time_s / c.SECONDS_PER_GYR
    return None


def deep_time(state) -> DeepTime:
    """Return the durations that gate deposits, from the history log or the snapshot biosphere rules."""
    age = state.epoch_s / c.SECONDS_PER_GYR
    bio = state.biosphere
    liquid = state.atmosphere.surface_water == "liquid"
    life = bio is not None and bio.life in ("surface", "ocean")
    if state.events:                       # history mode
        origin = _first_event_gyr(state, "origin_of_life")
        oxy = _first_event_gyr(state, "oxygenation")
        land = _first_event_gyr(state, "land_colonisation")
    else:                                  # snapshot mode: Earth-like timing from the biosphere age
        origin = age - bio.age_gyr if life else None
        oxygenic = life and bio.product == "o2"
        oxy = origin + h.OXYGEN_FIRST_RISE[0] if oxygenic and bio.age_gyr > h.OXYGEN_FIRST_RISE[0] else None
        land = (origin + h.LAND_LIFE_DELAY_GYR if bio is not None and bio.life == "surface"
                and bio.age_gyr > h.LAND_LIFE_DELAY_GYR else None)
    life_gyr = max(age - origin, 0.0) if (life and origin is not None) else 0.0
    oxygenated = max(age - oxy, 0.0) if oxy is not None else 0.0
    ocean_start = min(h.OCEAN_FORMED_GYR, age)
    anoxic_end = oxy if oxy is not None else age
    anoxic = max(anoxic_end - ocean_start, 0.0) if (liquid or oxy is not None) else 0.0
    land_gyr = max(age - land, 0.0) if land is not None else 0.0
    q = state.interior.surface_heat_flux_w_m2 / c.EARTH_HEAT_FLUX
    return DeepTime(age, life_gyr, anoxic, oxygenated, land_gyr, float(q))


def _boundary_sides(grid, plate: np.ndarray, continental: np.ndarray) -> np.ndarray:
    """Return, per cell, whether a plate boundary through it has continental crust on both sides."""
    adj = grid.neighbours.tocoo()
    across = plate[adj.row] != plate[adj.col]
    both = np.zeros(grid.size, dtype=bool)
    rows = adj.row[across]
    np.logical_or.at(both, rows, continental[rows] & continental[adj.col[across]])
    return both


def geology(world) -> Geology:
    """Return the tectonic setting of every cell and the distances the resource rules use."""
    s = world.surface
    grid = world.grid
    radius_km = float(s.attrs["radius_m"]) / 1e3
    limit = h.GEOLOGY_DISTANCE_LIMIT_KM
    ocean = s["ocean"].values.astype(bool)
    lake = s["lake"].values.astype(bool) if "lake" in s else np.zeros(grid.size, dtype=bool)
    elevation = s["elevation"].values.astype(float)
    crust = s["crust"].values
    terrain = s["terrain"].values
    boundary = s["boundary"].values
    plate = s["plate"].values
    kind = s["orogeny_kind"].values if "orogeny_kind" in s else np.zeros(grid.size, dtype=np.int8)
    sediment = s["sediment"].values.astype(float) if "sediment" in s else np.zeros(grid.size)
    continental = crust != Crust.OCEANIC
    land = ~ocean

    nearest = {}

    def dist(mask, name=None):
        d, idx = distance_from(grid, np.flatnonzero(mask), radius_km, limit)
        if name:
            nearest[name] = idx
        return d

    conv = boundary == Boundary.CONVERGENT
    both = _boundary_sides(grid, plate, crust == Crust.CONTINENTAL)
    d = {
        "subduction": dist(conv & ~both),
        "collision": dist(conv & both),
        "rift": dist(((boundary == Boundary.DIVERGENT) & continental) | (terrain == Terrain.RIFT)),
        "transform": dist(boundary == Boundary.TRANSFORM),
        "coast": dist(ocean & ~lake),
        "arc_belt": dist(np.isin(kind, (OrogenyKind.ANDEAN, OrogenyKind.ARC)), "arc_belt"),
        "collision_belt": dist(kind == OrogenyKind.COLLISION, "collision_belt"),
        "volcanic": dist(np.isin(terrain, (Terrain.VOLCANO, Terrain.VOLCANIC_PLAIN))),
    }

    setting = np.full(grid.size, Setting.OCEAN, dtype=np.int8)
    setting[continental] = np.where(sediment[continental] >= h.SAG_BASIN_MIN_SEDIMENT_M,
                                    Setting.SAG_BASIN, Setting.CRATON)
    margin = continental & (d["coast"] <= h.PASSIVE_MARGIN_WIDTH_KM) & (elevation < h.PASSIVE_MARGIN_MAX_ELEVATION_M)
    setting[margin] = Setting.PASSIVE_MARGIN
    setting[continental & (d["transform"] <= h.STRIKE_SLIP_WIDTH_KM)] = Setting.STRIKE_SLIP
    setting[continental & (d["rift"] <= h.RIFT_WIDTH_KM)] = Setting.RIFT
    setting[continental & (d["collision"] <= h.FORELAND_WIDTH_KM)] = Setting.FORELAND
    setting[continental & (d["subduction"] <= h.SUBDUCTION_MARGIN_WIDTH_KM)] = Setting.SUBDUCTION_MARGIN
    setting[continental & (terrain == Terrain.MOUNTAIN)] = Setting.OROGEN
    setting[np.isin(terrain, (Terrain.VOLCANO, Terrain.VOLCANIC_PLAIN)) & (continental | land)] = Setting.VOLCANIC
    return Geology(setting=setting, land=land, continental=continental, distance_km=d, nearest=nearest,
                   deep_time=deep_time(world.state))
