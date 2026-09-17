"""Tier 1 climate of a surface: temperature, precipitation, evaporation, runoff and sea ice on the surface grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..grid import SphereGrid
from ..hydrology.rainfall import Rainfall, fu_runoff, potential_evaporation
from .ebm import SurfaceState, TemperatureControl, ThermalClimate, solve_thermal
from .ice import equilibrium_line, seasonal_temperatures
from .mesh import climate_grid, regridder
from .moisture import MoistClimate, monthly, solve_moisture
from .physics import ClimateSetting, hadley_edge_deg
from .precipitation import SEASONS, land_rain_factor, rain_terrain, seasonal_factors, tangent_gradient, wind_directions
from .zonal import ZonalClimate

MONTHS = 12


@dataclass
class SurfaceClimate:
    """Climate of a surface on the surface grid, with the climate-grid solution it came from."""

    rainfall: Rainfall
    sea_ice: np.ndarray              # annual mean sea-ice cover (0 on land)
    open_ocean_fraction: float       # share of the ocean free of ice
    mean_k: float
    planetary_albedo: float
    rising_branch_deg: np.ndarray    # (12,) latitude of the rising branch each month
    ela_m: np.ndarray                # snowline (equilibrium-line altitude) above sea level; inf where none
    thermal: ThermalClimate
    moist: MoistClimate
    converged: bool


def run_climate(grid: SphereGrid, radius_m: float, height_m: np.ndarray, ocean: np.ndarray, setting: ClimateSetting,
                zonal: ZonalClimate, seasonal: bool = True, land_albedo: Optional[np.ndarray] = None,
                glacier: Optional[np.ndarray] = None,
                moisture_decay_km: Optional[np.ndarray | float] = None, snowline: bool = True,
                initial: Optional[SurfaceClimate] = None,
                control: Optional[TemperatureControl] = None) -> SurfaceClimate:
    """Return the Tier 1 climate of a surface (``height_m`` above sea level, at the top of any ice).

    ``land_albedo`` (snow-free) and ``glacier`` (ice-sheet cover) are
    surface-grid fields; ``moisture_decay_km`` (per cell or one value) sets
    how far moisture reaches inland, including recycling by vegetation.
    Without seasons, monthly temperatures for the snowline come from the
    Tier 0 seasonal cycle; ``snowline=False`` skips it. ``initial`` is a
    nearby solution to start the energy balance from. ``control`` holds the
    mean surface temperature at a target by adjusting the greenhouse.
    """
    rg = regridder(grid.size)
    land = (~ocean).astype(float)
    ground = np.where(ocean, 0.0, np.maximum(height_m, 0.0))
    albedo_field = np.full(grid.size, h.SURFACE_ALBEDO_LAND) if land_albedo is None else land_albedo
    glacier_field = np.zeros(grid.size) if glacier is None else glacier
    land_c = rg.to_climate(land)
    cell_land_albedo = rg.to_climate(albedo_field * land) / np.maximum(land_c, 1e-6)
    cell_land_albedo = np.where(land_c > 0, cell_land_albedo, h.SURFACE_ALBEDO_LAND)
    state = SurfaceState(land_fraction=land_c, height_m=rg.to_climate(ground),
                         land_albedo=cell_land_albedo,
                         glacier=rg.to_climate(glacier_field * land) / np.maximum(land_c, 1e-6))
    thermal = solve_thermal(setting, state, zonal, seasonal=seasonal,
                            initial=None if initial is None else initial.thermal, control=control)
    moist = solve_moisture(setting, thermal, land_c)

    lapse = h.LAPSE_RATE_K_PER_M * ground
    sea_level = monthly(thermal.sea_level_k, MONTHS)
    temperature = rg.to_surface(sea_level) - lapse
    reference_rain = rg.to_surface(monthly(moist.precipitation_m, MONTHS))
    rising = np.degrees(np.arcsin(np.clip(monthly(moist.itcz_sin[:, None], MONTHS)[:, 0], -1, 1)))

    decay = h.MOISTURE_DECAY_KM if moisture_decay_km is None else moisture_decay_km
    factors = _land_factors(grid, radius_m, height_m, ocean, setting, sea_level, rising, decay,
                            seasonal and not setting.synchronous)
    precipitation = np.maximum(reference_rain * factors, h.RAIN_MINIMUM_M * (moist.global_evaporation_m > 0))

    annual_p = precipitation.mean(axis=0)
    annual_t = temperature.mean(axis=0)
    evaporation = potential_evaporation(temperature).mean(axis=0)
    runoff = np.where(ocean, 0.0, fu_runoff(annual_p, evaporation))
    if glacier is not None:
        runoff = np.where(glacier > 0.5, 0.0, runoff)
    if seasonal or setting.synchronous:
        cycle = temperature
    else:
        cycle = seasonal_temperatures(annual_t, _seasonal_amplitude(zonal, grid), grid.points[:, 2] > 0.0)
    ela = (equilibrium_line(cycle, precipitation, ground, ~ocean) if snowline and moist.global_evaporation_m > 0
           else np.full(grid.size, np.inf))
    rain = Rainfall(precipitation_m=annual_p, evaporation_m=evaporation, runoff_m=runoff, temperature_k=annual_t,
                    monthly_temperature_k=temperature if seasonal else None,
                    monthly_precipitation_m=precipitation if seasonal else None)

    sea_ice = np.where(ocean, np.clip(rg.to_surface(thermal.sea_ice.mean(axis=0)), 0.0, 1.0), 0.0)
    open_ocean = float(1.0 - sea_ice[ocean].mean()) if ocean.any() else 0.0
    return SurfaceClimate(rainfall=rain, sea_ice=sea_ice, open_ocean_fraction=open_ocean,
                          mean_k=float(temperature.mean()), planetary_albedo=thermal.planetary_albedo,
                          rising_branch_deg=rising, ela_m=ela, thermal=thermal, moist=moist,
                          converged=thermal.converged)


def _land_factors(grid: SphereGrid, radius_m: float, height_m: np.ndarray, ocean: np.ndarray,
                  setting: ClimateSetting, sea_level: np.ndarray, rising: np.ndarray, decay_km,
                  seasonal: bool) -> np.ndarray:
    """Return the monthly share of ocean-equivalent rain reaching each surface cell."""
    rg = regridder(grid.size)
    terrain = rain_terrain(grid, height_m, ocean)
    edge = hadley_edge_deg(setting.rotation_period_s / 3600.0)
    seasons = SEASONS if seasonal else 1
    per_season = []
    for s in range(seasons):
        months = np.arange(3 * s, 3 * s + 3) if seasonal else np.arange(MONTHS)
        climate_t = sea_level[months].mean(axis=0)
        anomaly = rg.to_surface(climate_t - _zonal_mean(climate_t, setting.synchronous))
        warm = tangent_gradient(grid, anomaly) if seasonal else None
        wind = wind_directions(grid.points, edge, setting.synchronous, float(rising[months].mean()), warm)
        per_season.append(land_rain_factor(grid, radius_m, terrain, ocean, wind, decay_km))
    return seasonal_factors(np.stack(per_season), MONTHS)


def _seasonal_amplitude(zonal: ZonalClimate, grid: SphereGrid) -> np.ndarray:
    """Return the Tier 0 half-range of land temperature through the year at each cell's latitude."""
    half = 0.5 * (zonal.land_k.max(axis=0) - zonal.land_k.min(axis=0))
    return np.interp(grid.points[:, 2], zonal.coordinate, half)


def _zonal_mean(values: np.ndarray, synchronous: bool) -> np.ndarray:
    """Return the band mean (latitude, or substellar angle) of a climate-grid field at each climate cell."""
    points = climate_grid().points
    x = points[:, 0] if synchronous else points[:, 2]
    bands = h.ZONAL_BANDS
    idx = np.minimum(((x + 1.0) * 0.5 * bands).astype(int), bands - 1)
    means = np.bincount(idx, weights=values, minlength=bands) / np.maximum(np.bincount(idx, minlength=bands), 1)
    return means[idx]
