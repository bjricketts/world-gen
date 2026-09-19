"""Features left on the surface by the planet's past (history mode).

A planet keeps the marks of an ocean it has lost, of rivers that stopped
running, of ice that has retreated and of volcanism that has died down, but
only for as long as its own erosion takes to remove them. Rain and plate
tectonics rework a surface within tens of Myr, which is why Earth shows no
Hadean shoreline; a dry stagnant lid keeps the record for billions of years,
which is why Mars still shows its valleys.

Each relict is read from the integrated history inside that memory and drawn
on the present surface: a terrace at the old sea level, a valley network in
the dry or frozen lowlands, scoured land above the ice line of the coldest
past epoch, and plains resurfaced when the interior melted faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..climate.physics import FREEZING_K
from ..grid import SphereGrid
from ..hydrology import Rainfall, build_drainage
from ..hydrology.rainfall import fu_runoff, potential_evaporation
from ..noise import fbm
from ..state import PlanetState, Timeline


class Relict(IntEnum):
    """What the past left on a cell; the present landform is unchanged."""

    NONE = 0
    SHORELINE = 1      # terrace at a sea level the planet no longer has
    RIVER = 2          # valley network that no longer carries water
    GLACIAL = 3        # scoured by ice that has since retreated
    VOLCANIC = 4       # plains resurfaced when the interior melted faster


@dataclass
class RelictHistory:
    """What the planet's own past can still be read off its surface."""

    memory_myr: float
    palaeo_ocean_m3: Optional[float] = None       # largest ocean held within the memory
    shoreline_age_myr: Optional[float] = None
    wet_age_myr: Optional[float] = None           # last liquid water, on a planet that has none now
    cooling_k: Optional[float] = None             # how much colder the coldest epoch was than now
    melt_ratio: Optional[float] = None            # peak melting within the memory, relative to now
    resurfaced_share: Optional[float] = None      # share of the surface the melt of that time could bury


def memory_myr(liquid_share: float, activity: float) -> float:
    """Return how long a mark on the surface survives, from the erosion the planet applies to it."""
    wear = 1.0 + h.RELICT_RAIN_EROSION * max(liquid_share, 0.0) + h.RELICT_TECTONIC_EROSION * max(activity, 0.0)
    return h.RELICT_MEMORY_BASE_MYR / wear


def read_history(state: PlanetState, timeline: Optional[Timeline]) -> Optional[RelictHistory]:
    """Return what the recent history left behind, or None without a history to read."""
    if timeline is None or not timeline.series:
        return None
    s = timeline.series
    age = state.star.age_s / c.SECONDS_PER_GYR
    liquid = 1.0 if state.atmosphere.surface_water == "liquid" else 0.0
    window = memory_myr(liquid, state.interior.activity_index)
    time_myr = (np.asarray(s["time_gyr"], dtype=float) - age) * 1e3
    inside = time_myr >= -window
    if inside.sum() < 2:
        inside = np.zeros_like(inside)
        inside[-2:] = True
    out = RelictHistory(memory_myr=window)

    water = np.asarray(s["surface_water_oceans"], dtype=float)[inside]
    present = state.water.ocean_volume_m3 if state.water is not None else 0.0
    if water.size and present > 0.0 and water.max() > water[-1] * (1.0 + h.RELICT_SHORELINE_MIN_RISE):
        out.palaeo_ocean_m3 = present * water.max() / max(water[-1], 1e-12)
        out.shoreline_age_myr = -float(time_myr[inside][int(np.argmax(water))])

    if state.atmosphere.surface_water != "liquid":
        modes = np.asarray(s["climate"])[inside]
        temperature = np.asarray(s["surface_temperature_k"], dtype=float)[inside]
        wet = (modes == "warm") & (temperature > FREEZING_K)
        age_myr = -float(time_myr[inside][np.flatnonzero(wet)[-1]]) if wet.any() else 0.0
        if age_myr > h.RELICT_MIN_AGE_MYR:
            out.wet_age_myr = age_myr

    temperature = np.asarray(s["surface_temperature_k"], dtype=float)[inside]
    if temperature.size and temperature.min() < temperature[-1] - h.RELICT_GLACIAL_COOLING_K:
        out.cooling_k = float(temperature[-1] - temperature.min())

    melt = np.asarray(s["melt"], dtype=float)[inside]
    if melt.size and melt.max() > max(melt[-1], 1e-6) * h.RELICT_RESURFACING_RATIO:
        out.melt_ratio = float(melt.max() / max(melt[-1], 1e-6))
        # Earth buries its whole sea floor in about RESURFACING_EARTH_MYR at its present melt rate,
        # so the melt of the preserved past covers this share of the surface.
        span = min(window, -float(time_myr[inside][0]))
        coverage = melt.mean() * span / h.RESURFACING_EARTH_MYR
        out.resurfaced_share = float(min(1.0 - np.exp(-coverage), h.RELICT_RESURFACED_MAX))
    return out


def paint(grid: SphereGrid, state: PlanetState, past: RelictHistory, elevation: np.ndarray, ocean: np.ndarray,
          fill, temperature_k: np.ndarray, glaciated: Optional[np.ndarray], relief: float,
          seed: int) -> tuple[np.ndarray, dict]:
    """Return the relict of every cell and the summary numbers for the report."""
    relict = np.full(grid.size, Relict.NONE, dtype=np.int8)
    land = ~ocean
    features: dict[str, float] = {}

    if past.palaeo_ocean_m3 is not None and fill is not None:
        level = fill.level_for_volume(past.palaeo_ocean_m3)
        band = h.RELICT_SHORELINE_BAND_M * relief
        # Only a sea that covered a fair part of the planet leaves a coastline worth drawing.
        if float((elevation < level).mean()) >= h.RELICT_SHORELINE_MIN_COVER:
            terrace = land & (np.abs(elevation - level) <= band)
            relict[terrace] = Relict.SHORELINE
            features["relict_shoreline_m"] = float(level)
            features["relict_shoreline_age_myr"] = float(past.shoreline_age_myr or 0.0)

    if past.cooling_k is not None:
        # In the coldest epoch every cell was this much colder; ice covered the ground that froze.
        scoured = land & (temperature_k - past.cooling_k < FREEZING_K)
        if glaciated is not None:
            scoured &= ~glaciated                      # ice that never left leaves no relict
        relict[scoured] = Relict.GLACIAL
        features["relict_glacial_cooling_k"] = past.cooling_k

    if past.wet_age_myr is not None and land.any():
        rivers = _dry_valleys(grid, state, elevation, ocean, temperature_k)
        relict[rivers] = Relict.RIVER
        features["relict_river_age_myr"] = float(past.wet_age_myr)

    if past.resurfaced_share and state.interior.tectonic_regime != "mobile_lid":
        share = past.resurfaced_share
        noise = fbm(grid.points, seed, "relict.resurfacing", frequency=2.5, octaves=4)
        patches = noise <= np.quantile(noise, share)
        relict[patches & (relict == Relict.NONE)] = Relict.VOLCANIC
        features["relict_resurfaced_melt_ratio"] = past.melt_ratio

    for kind in (Relict.SHORELINE, Relict.RIVER, Relict.GLACIAL, Relict.VOLCANIC):
        share = float((relict == kind).mean())
        if share > 0.0:
            features[f"relict_{kind.name.lower()}_fraction"] = share
    if features:
        features["relict_memory_myr"] = past.memory_myr
    return relict, features


def _dry_valleys(grid: SphereGrid, state: PlanetState, elevation: np.ndarray, ocean: np.ndarray,
                 temperature_k: np.ndarray) -> np.ndarray:
    """Return the cells holding a valley network cut when the planet still had rain."""
    rain_m = np.full(grid.size, h.RELICT_PALAEO_RAIN_M)
    evaporation = potential_evaporation(np.maximum(temperature_k, FREEZING_K + 5.0))
    rainfall = Rainfall(precipitation_m=rain_m, evaporation_m=evaporation,
                        runoff_m=fu_runoff(rain_m, evaporation), temperature_k=temperature_k)
    drainage = build_drainage(grid, state.bulk.radius_m, elevation, ocean, rainfall)
    return (~ocean) & (drainage.river_order >= h.RELICT_RIVER_MIN_ORDER)
