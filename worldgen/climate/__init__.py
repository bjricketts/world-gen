"""Climate models: Tier 0 (zonal energy balance) and Tier 1 (energy balance on the sphere with moisture).

Tier 0 runs inside the global state and decides the planetary albedo,
mean temperature and water phase. Tier 1 runs on a built surface and
provides temperature, precipitation, evaporation, runoff and sea ice.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from .. import constants as c
from .ebm import SurfaceState, TemperatureControl, ThermalClimate, solve_thermal
from .model import SurfaceClimate, run_climate
from .moisture import MoistClimate, solve_moisture
from .physics import ClimateSetting
from .zonal import COLD_START_K, WARM_START_K, ZonalClimate, solve_zonal

if TYPE_CHECKING:
    from ..grid import SphereGrid
    from ..state import PlanetState

__all__ = ["ClimateSetting", "MoistClimate", "SurfaceClimate", "SurfaceState", "TemperatureControl", "ThermalClimate",
           "ZonalClimate", "band_land_fraction", "planet_control",
           "COLD_START_K", "WARM_START_K", "climate_setting", "planet_climate", "run_climate", "solve_moisture",
           "solve_thermal", "solve_zonal"]


def climate_setting(instellation_earth: float, eccentricity: float, obliquity_rad: float,
                    periapsis_longitude_rad: float, synchronous: bool, pressure_pa: float, gravity_m_s2: float,
                    composition: str, optical_depth: float, rotation_period_s: float, orbital_period_s: float,
                    albedo: float, albedo_fixed: bool, has_water: bool, reference_k: float) -> ClimateSetting:
    """Return the climate setting of a planet from its orbit-averaged instellation and atmosphere."""
    e = min(max(eccentricity, 0.0), 0.95)
    return ClimateSetting(
        flux_w_m2=c.S_EARTH * instellation_earth * math.sqrt(1.0 - e * e),
        eccentricity=e,
        obliquity_rad=obliquity_rad,
        periapsis_longitude_rad=periapsis_longitude_rad,
        synchronous=synchronous,
        pressure_pa=pressure_pa,
        gravity_m_s2=gravity_m_s2,
        composition=composition,
        optical_depth=optical_depth,
        rotation_period_s=rotation_period_s,
        year_s=orbital_period_s,
        albedo=albedo,
        albedo_fixed=albedo_fixed,
        has_water=has_water,
        reference_k=reference_k,
    )


def planet_climate(state: "PlanetState",
                   land_by_band: Optional[np.ndarray] = None) -> tuple[ClimateSetting, ZonalClimate]:
    """Return the climate setting of a generated planet and its Tier 0 solution.

    Without ``land_by_band`` every band has the planet's estimated land fraction.
    """
    a, o, summary = state.atmosphere, state.orbit, state.climate
    water = a.surface_water in ("liquid", "ice")
    setting = climate_setting(
        o.instellation_earth, o.eccentricity, o.obliquity_rad, o.periapsis_longitude_rad,
        o.spin_state == "synchronous", a.surface_pressure_pa, state.bulk.surface_gravity_m_s2, a.composition,
        a.greenhouse_optical_depth, o.rotation_period_s, o.orbital_period_s,
        summary.ice_free_albedo if summary is not None else a.bond_albedo,
        summary.albedo_fixed if summary is not None else True,
        water,
        summary.mean_temperature_k if summary is not None else a.surface_temperature_k,
    )
    from ..biosphere.life import tier0_land_albedo   # imported here: the biosphere package builds on this one

    land = state.water.land_fraction_estimate if water and state.water is not None else 1.0
    if water and land_by_band is not None:
        land = land_by_band
    return setting, solve_zonal(setting, land, land_albedo=tier0_land_albedo(state.biosphere))


def planet_control(state: "PlanetState", zonal: ZonalClimate) -> Optional[TemperatureControl]:
    """Return the mean temperature Tier 1 should hold (weathering or a user value), or None."""
    from ..atmosphere import composition_optical_depth, temperature_control   # the atmosphere builds on this package

    if state.climate is None:
        return None
    user = (state.inputs.get("atmosphere.surface_temperature_k")
            if state.provenance.get("atmosphere.surface_temperature_k") in ("user", "user_range") else None)
    limits = temperature_control(state.atmosphere, composition_optical_depth(state.atmosphere, state.biosphere),
                                 user, zonal.mean_k)
    return None if limits is None else TemperatureControl(*limits)


def band_land_fraction(grid: "SphereGrid", ocean: np.ndarray, synchronous: bool,
                       bands: Optional[int] = None) -> np.ndarray:
    """Return the land share of each Tier 0 band (latitude, or substellar angle) of a surface."""
    from .. import heuristics as h

    n = bands or h.ZONAL_BANDS
    x = grid.points[:, 0] if synchronous else grid.points[:, 2]
    index = np.clip(((x + 1.0) * 0.5 * n).astype(int), 0, n - 1)
    cells = np.bincount(index, minlength=n)
    land = np.bincount(index, weights=(~ocean).astype(float), minlength=n)
    return np.where(cells > 0, land / np.maximum(cells, 1), float((~ocean).mean()))
