"""Planet-specific constants for the history integration, resolved from the spec and the bulk properties."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .. import constants as c
from .. import heuristics as h
from ..interior import silicate_mass
from ..state import BulkState, OrbitState

EARTH_SILICATE_KG = c.M_EARTH * (1.0 - c.EARTH_CMF)
EARTH_CORE_KG = c.M_EARTH * c.EARTH_CMF
EARTH_AREA_M2 = 4.0 * math.pi * c.R_EARTH**2
EARTH_CORE_RADIUS_M = 3.48e6


def core_radius(bulk: BulkState) -> float:
    """Return the core radius (m), scaled from Earth's by core mass."""
    core = max(bulk.core_mass_fraction * bulk.mass_kg, 1.0)
    return EARTH_CORE_RADIUS_M * (core / EARTH_CORE_KG) ** 0.3


def central_pressure_ratio(bulk: BulkState) -> float:
    """Return the central pressure relative to Earth's (∝ M² / R⁴)."""
    return (bulk.mass_kg / c.M_EARTH) ** 2 / (bulk.radius_m / c.R_EARTH) ** 4


@dataclass
class HistoryParams:
    """Everything the history equations need that does not change with time."""

    # Planet
    mass_kg: float
    radius_m: float
    gravity: float
    area_m2: float
    silicate_kg: float
    core_kg: float
    core_radius_m: float
    radiogenic_abundance: float
    tidal_power_w: float
    semi_major_axis_m: float
    star_mass_msun: float
    activity_percentile: float
    tidal_lock_gyr: float          # planet age at which the spin locks (inf if never)
    # Initial state
    water_oceans: float            # total water, Earth oceans
    carbon: float                  # total carbon (10¹⁸ kg CO₂)
    nitrogen: float                # atmospheric N₂ (10¹⁸ kg)
    mantle_temperature_k: float
    # Rates
    outgassing: float
    weathering: float
    weathering_scale_k: float
    weathering_exponent: float
    biotic_weathering: float
    escape_efficiency: float
    activation_energy_j: float
    core_adiabatic: float
    life_delay_gyr: float
    burial: float
    reductant_decay_gyr: float
    land_delay_gyr: float
    # Life settings (spec values or defaults)
    biochemistry: str = "oxygenic"
    product: str = "o2"
    optimum_k: float = 298.0
    tolerance_k: float = 25.0

    @property
    def area_ratio(self) -> float:
        """Return the surface area relative to Earth's."""
        return self.area_m2 / EARTH_AREA_M2

    @property
    def silicate_ratio(self) -> float:
        """Return the silicate mass relative to Earth's."""
        return self.silicate_kg / EARTH_SILICATE_KG

    @property
    def mass_ratio(self) -> float:
        """Return the planet mass relative to Earth's."""
        return self.mass_kg / c.M_EARTH

    def pressure_bar(self, mass_1e18_kg: float) -> float:
        """Return the surface pressure (bar) exerted by a gas column of this mass (10¹⁸ kg)."""
        return mass_1e18_kg * 1e18 * self.gravity / self.area_m2 / c.BAR


def build_params(values, bulk: BulkState, orbit: OrbitState, star_mass_msun: float,
                 tidal_power_w: float, life_settings: Optional[dict] = None) -> HistoryParams:
    """Return the history constants for a planet; ``values`` is a resolved-input getter (path, default)."""
    v = values
    richness = v("atmosphere.volatile_richness", 1.0)
    total_water = v("history.initial_water_mass_fraction", None)
    if total_water is None:
        total_water = v("body.water_mass_fraction", 0.0)
    ocean_fraction = c.EARTH_OCEAN_MASS / bulk.mass_kg
    # Planets that are locked at the target epoch locked when their locking time had passed.
    lock = orbit.tidal_lock_time_s / c.SECONDS_PER_GYR if not orbit.spin_state.startswith("free") else math.inf
    params = HistoryParams(
        mass_kg=bulk.mass_kg,
        radius_m=bulk.radius_m,
        gravity=bulk.surface_gravity_m_s2,
        area_m2=4.0 * math.pi * bulk.radius_m**2,
        silicate_kg=silicate_mass(bulk),
        core_kg=bulk.core_mass_fraction * bulk.mass_kg,
        core_radius_m=core_radius(bulk),
        radiogenic_abundance=v("body.radiogenic_abundance", 1.0),
        tidal_power_w=tidal_power_w,
        semi_major_axis_m=orbit.semi_major_axis_m,
        star_mass_msun=star_mass_msun,
        activity_percentile=v("star.activity_percentile", 0.5),
        tidal_lock_gyr=lock,
        water_oceans=total_water / ocean_fraction,
        carbon=v("history.carbon_inventory", richness) * (h.CARBON_SURFACE_EARTH + h.CARBON_CRUST_EARTH
                                                         + h.CARBON_MANTLE_EARTH) * bulk.mass_kg / c.M_EARTH,
        nitrogen=v("history.nitrogen_inventory", richness) * h.NITROGEN_EARTH * bulk.mass_kg / c.M_EARTH,
        mantle_temperature_k=v("history.initial_mantle_temperature_k", 1750.0),
        outgassing=v("history.outgassing_efficiency", 1.0),
        weathering=v("history.weathering_efficiency", 1.0),
        weathering_scale_k=v("history.weathering_temperature_scale_k", 20.0),
        weathering_exponent=v("history.weathering_co2_exponent", 0.3),
        biotic_weathering=v("history.biotic_weathering_factor", 4.0),
        escape_efficiency=v("history.escape_efficiency", 0.15),
        activation_energy_j=v("history.mantle_activation_energy_kj", 300.0) * 1e3,
        core_adiabatic=v("history.core_adiabatic_heat_flow", 1.0),
        life_delay_gyr=v("history.life_origin_delay_gyr", 0.4),
        burial=v("history.oxygen_burial_efficiency", 1.0),
        reductant_decay_gyr=v("history.reductant_decay_gyr", 3.0),
        land_delay_gyr=v("history.land_colonisation_delay_gyr", 3.4),
    )
    if life_settings:
        for key, value in life_settings.items():
            if value is not None:
                setattr(params, key, value)
    return params
