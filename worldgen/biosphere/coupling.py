"""Climate, ice sheets and vegetation of a built surface, iterated until they agree.

Each pass solves the seasonal Tier 1 climate for the current ice surface,
land albedo and moisture recycling, then rebuilds the ice sheets and the
vegetation from that climate. The loop stops when nearly all land keeps its
albedo and ice cover, or after ``COUPLING_MAX_PASSES`` passes. If the change
grows from one pass to the next, the vegetation and ice-cover updates are
damped so that oscillating cells settle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..climate import ClimateSetting, SurfaceClimate, TemperatureControl, ZonalClimate, run_climate
from ..climate.ice import IceSheets, build_ice_sheets
from ..grid import SphereGrid
from ..state import BiosphereState
from .vegetation import Vegetation, build_vegetation


@dataclass
class CoupledSurface:
    """The agreed climate, ice and vegetation of a surface."""

    climate: SurfaceClimate
    ice: IceSheets
    vegetation: Vegetation
    passes: int
    converged: bool


def no_ice(size: int) -> IceSheets:
    """Return an empty ice-sheet record."""
    zero = np.zeros(size)
    return IceSheets(thickness_m=zero, surface_rise_m=zero, balance_m=zero, ela_m=np.full(size, np.inf))


def _blend(old: Vegetation, new: Vegetation, weight: float) -> Vegetation:
    """Return vegetation moved from ``old`` toward ``new`` by ``weight``."""
    if weight >= 1.0:
        return new
    return Vegetation(productivity=new.productivity,
                      cover=old.cover + weight * (new.cover - old.cover),
                      land_albedo=old.land_albedo + weight * (new.land_albedo - old.land_albedo),
                      moisture_decay_km=old.moisture_decay_km + weight * (new.moisture_decay_km - old.moisture_decay_km))


def couple_surface(grid: SphereGrid, radius_m: float, height_m: np.ndarray, ocean: np.ndarray,
                   setting: ClimateSetting, zonal: ZonalClimate, gravity_m_s2: float,
                   biosphere: Optional[BiosphereState], instellation_earth: float,
                   control: Optional[TemperatureControl] = None,
                   initial: Optional[CoupledSurface] = None) -> CoupledSurface:
    """Return the climate, ice sheets and vegetation of a surface (``height_m`` is bedrock above sea level).

    ``control`` holds the mean temperature (see ``run_climate``); ``initial``
    is an earlier coupled state of a similar surface to start from.
    """
    ground = np.where(ocean, 0.0, np.maximum(height_m, 0.0))
    icy = setting.has_water and setting.humidity > 0.0
    ice = no_ice(grid.size) if initial is None else initial.ice
    glacier = ice.glaciated.astype(float)    # ice cover seen by the climate (fractional while damped)
    vegetation: Optional[Vegetation] = None if initial is None else initial.vegetation
    converged = False
    passes = 0
    climate = None if initial is None else initial.climate
    weight = 1.0
    last_change = np.inf
    for passes in range(1, h.COUPLING_MAX_PASSES + 1):
        climate = run_climate(grid, radius_m, ground + ice.surface_rise_m, ocean, setting, zonal, seasonal=True,
                              land_albedo=None if vegetation is None else vegetation.land_albedo,
                              glacier=glacier,
                              moisture_decay_km=None if vegetation is None else vegetation.moisture_decay_km,
                              snowline=False, initial=climate, control=control)
        rain = climate.rainfall
        new_ice = (build_ice_sheets(grid, radius_m, ground, ocean, rain.monthly_temperature_k,
                                    rain.monthly_precipitation_m, gravity_m_s2, previous=ice)
                   if icy else ice)
        new_veg = build_vegetation(rain.temperature_k, rain.precipitation_m, rain.evaporation_m, ocean,
                                   new_ice.glaciated, biosphere, instellation_earth)
        cover = new_ice.glaciated.astype(float)
        if vegetation is None:
            glacier = cover
        else:
            albedo_change = _albedo_change(new_veg.land_albedo, vegetation.land_albedo, ocean)
            ice_change = float((new_ice.glaciated != ice.glaciated).mean())
            converged = albedo_change < h.COUPLING_ALBEDO_TOLERANCE and ice_change < h.COUPLING_ICE_TOLERANCE
            change = albedo_change + ice_change
            if change > last_change:
                weight = max(0.5 * weight, h.COUPLING_MIN_DAMPING)
            last_change = change
            new_veg = _blend(vegetation, new_veg, weight)
            glacier = glacier + weight * (cover - glacier)
        ice, vegetation = new_ice, new_veg
        if converged:
            break
    climate.ela_m = ice.ela_m if icy else climate.ela_m
    earlier = 0 if initial is None else initial.passes
    return CoupledSurface(climate=climate, ice=ice, vegetation=vegetation, passes=earlier + passes,
                          converged=converged)


def _albedo_change(new: np.ndarray, old: np.ndarray, ocean: np.ndarray) -> float:
    """Return the ``COUPLING_ALBEDO_QUANTILE`` quantile of the land albedo change between passes."""
    change = np.abs(new - old)[~ocean]
    return float(np.quantile(change, h.COUPLING_ALBEDO_QUANTILE)) if change.size else 0.0
