"""Volatile reservoirs: CO₂ partition and greenhouse, carbon and water cycles, escape to space.

Masses are in 10¹⁸ kg and fluxes in 10¹⁸ kg per Gyr; water is in Earth
oceans. Silicate weathering follows Walker et al. (1981) with a seafloor term
(Krissansen-Totton & Catling 2017) and a supply limit (Foley 2015). Water
cycles between ocean and mantle as in Cowan & Abbot (2014). Escape is
energy-limited (Watson et al. 1981; Erkaev et al. 2007), with hydrogen
capped by the diffusion limit (Hunten 1973).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import constants as c
from .. import heuristics as h
from ..atmosphere import shoreline_ratio
from ..util import logistic
from .params import HistoryParams

MU_CO2, MU_N2, MU_O2, MU_H2O = 44.0, 28.0, 32.0, 18.0
GYR_PER_SECOND_1E18 = c.SECONDS_PER_GYR / 1e18     # kg/s → 10¹⁸ kg per Gyr
EARTH_LAND = 0.29
# Share of Earth's mantle water capacity still free.
EARTH_MANTLE_ROOM = 1.0 - h.MANTLE_WATER_EARTH_OCEANS * c.EARTH_OCEAN_MASS / (
    h.MANTLE_WATER_MAX * c.M_EARTH * (1.0 - c.EARTH_CMF))


@dataclass
class Air:
    """Composition and pressure of the atmosphere at one instant."""

    pressure_bar: float
    co2_bar: float                 # partial pressures
    n2_bar: float
    o2_bar: float
    co2_mass: float                # 10¹⁸ kg in the air
    o2_fraction: float
    co2_fraction: float
    steam_bar: float = 0.0

    @property
    def mean_molecular_weight(self) -> float:
        """Return the mean molecular weight of the dry air."""
        moles = self.co2_bar + self.n2_bar + self.o2_bar
        if moles <= 0:
            return 29.0
        return (self.co2_bar * MU_CO2 + self.n2_bar * MU_N2 + self.o2_bar * MU_O2) / moles


def partition_air(p: HistoryParams, surface_carbon: float, ocean_oceans: float, nitrogen: float,
                  oxygen: float, steam_oceans: float = 0.0) -> Air:
    """Return the air given the surface carbon, the liquid ocean (oceans) and the other gases.

    The ocean holds dissolved carbon ∝ W·pCO₂^exponent; the rest is in the air.
    The split is solved by bisection on the mass of CO₂ in the air.
    """
    surface_carbon = max(surface_carbon, 0.0)
    nitrogen = max(nitrogen, 0.0)
    oxygen = max(oxygen, 0.0)
    per_mass = p.gravity * 1e18 / p.area_m2 / c.BAR      # bar per 10¹⁸ kg of air
    other_moles = nitrogen / MU_N2 + oxygen / MU_O2
    other_mass = nitrogen + oxygen

    def partial(air_co2: float) -> tuple[float, float]:
        """Return the total and CO₂ partial pressure (bar) for a mass of CO₂ in the air."""
        total = (air_co2 + other_mass) * per_mass
        moles = air_co2 / MU_CO2 + other_moles
        return total, (total * air_co2 / MU_CO2 / moles if moles > 0 else 0.0)

    ocean_factor = h.OCEAN_CARBON_EARTH * ocean_oceans
    if ocean_oceans > 0.0 and surface_carbon > 0.0:
        low, high = 0.0, surface_carbon
        for _ in range(h.AIR_PARTITION_STEPS):
            mid = 0.5 * (low + high)
            dissolved = ocean_factor * (partial(mid)[1] / h.CO2_EARTH_BAR) ** h.OCEAN_CARBON_EXPONENT
            if mid + dissolved > surface_carbon:
                high = mid
            else:
                low = mid
        air_co2 = 0.5 * (low + high)
    else:
        air_co2 = surface_carbon
    total, pco2 = partial(air_co2)
    pn2 = total * (nitrogen / MU_N2) / (air_co2 / MU_CO2 + other_moles) if total > 0 else 0.0
    po2 = total * (oxygen / MU_O2) / (air_co2 / MU_CO2 + other_moles) if total > 0 else 0.0
    steam = steam_oceans * c.EARTH_OCEAN_MASS / 1e18 * per_mass if steam_oceans > 0 else 0.0
    return Air(pressure_bar=total, co2_bar=pco2, n2_bar=pn2, o2_bar=po2, co2_mass=air_co2,
               o2_fraction=po2 / total if total > 0 else 0.0,
               co2_fraction=pco2 / total if total > 0 else 0.0, steam_bar=steam)


def optical_depth(air: Air, wet: bool, methane_tau: float = 0.0) -> float:
    """Return the grey greenhouse optical depth from total pressure, CO₂ and water."""
    n = h.GREENHOUSE_PRESSURE_EXPONENT
    total = max(air.pressure_bar, 0.0)
    background = (h.GREENHOUSE_BACKGROUND_WET if wet else h.GREENHOUSE_BACKGROUND_DRY) * total**n
    co2 = air.co2_bar
    log_term = h.GREENHOUSE_CO2_LOG * math.log1p(co2 / h.GREENHOUSE_CO2_LOG_BAR) * min(total, 1.0) ** n
    tau = background + log_term + h.TAU0["co2"] * max(co2, 0.0) ** n
    if air.steam_bar > 0:
        tau += h.GREENHOUSE_STEAM * air.steam_bar**n
    return tau + (methane_tau if total > 0 else 0.0)


@dataclass
class CarbonFluxes:
    """Carbon fluxes (10¹⁸ kg CO₂ per Gyr)."""

    continental: float
    seafloor: float
    volcanic: float
    arc: float
    subduction: float
    recycling: float
    decomposition: float

    @property
    def weathering(self) -> float:
        """Return the total weathering sink."""
        return self.continental + self.seafloor


def _saturating(kinetic: float, supply: float, earth_ratio: float) -> float:
    """Return a rate that follows ``kinetic`` when small and approaches ``supply`` when large.

    ``earth_ratio`` is supply / kinetic rate on present Earth; the kinetic rate
    is rescaled so Earth's weathering keeps its present value.
    """
    if supply <= 0.0:
        return 0.0
    scale = -earth_ratio * math.log1p(-1.0 / earth_ratio)
    return supply * -math.expm1(-kinetic * scale / supply)


def carbon_fluxes(p: HistoryParams, crust: float, mantle: float, co2_bar: float, surface_k: float,
                  land: float, ocean: bool, liquid: float, regime: str, melt: float, spreading: float,
                  heat_ratio: float, land_life: float) -> CarbonFluxes:
    """Return the carbon fluxes between air-ocean, crust and mantle.

    ``liquid`` is the share of the surface water that is open (0 in a
    snowball), ``land_life`` the land biosphere's cover relative to Earth's,
    ``heat_ratio`` the surface heat flow per area relative to Earth's.
    """
    area = p.area_ratio
    earth_continental = h.WEATHERING_EARTH * (1.0 - h.SEAFLOOR_WEATHERING_SHARE)
    earth_seafloor = h.WEATHERING_EARTH * h.SEAFLOOR_WEATHERING_SHARE
    ratio = max(co2_bar, 0.0) / h.CO2_EARTH_BAR
    warm = min(surface_k - c.T_SURFACE_EARTH, 400.0)

    continental = seafloor = 0.0
    if ocean and liquid > 0.0 and surface_k < h.CARBONATE_DECOMPOSITION_K:
        land_factor = (land / (land + 0.01)) / (EARTH_LAND / (EARTH_LAND + 0.01))
        biota = (1.0 + (p.biotic_weathering - 1.0) * land_life) / p.biotic_weathering
        runoff = math.exp(0.65 * h.HYDROLOGY_SENSITIVITY * warm)
        kinetic = (earth_continental * p.weathering * area * land_factor * biota * liquid
                   * ratio**p.weathering_exponent * math.exp(warm / p.weathering_scale_k) * runoff)
        erosion = heat_ratio if regime == "mobile_lid" else 0.1 + melt / max(area, 1e-9)
        supply = (h.WEATHERING_SUPPLY_RATIO * earth_continental * p.weathering * area
                  * (land / EARTH_LAND) * erosion)
        continental = _saturating(kinetic, supply, h.WEATHERING_SUPPLY_RATIO)
        if regime == "mobile_lid" and spreading > 0.0:
            kinetic_sf = (earth_seafloor * p.weathering * area * spreading * liquid
                          * ratio**h.SEAFLOOR_CO2_EXPONENT * math.exp(warm / h.SEAFLOOR_TEMPERATURE_SCALE_K))
            seafloor = _saturating(kinetic_sf, h.SEAFLOOR_CAPACITY_RATIO * earth_seafloor * area * spreading,
                                   h.SEAFLOOR_CAPACITY_RATIO)

    concentration = (mantle / h.CARBON_MANTLE_EARTH) / p.silicate_ratio
    earth_subduction = h.CARBON_CRUST_EARTH / h.CARBONATE_TURNOVER_GYR
    earth_volcanic = h.WEATHERING_EARTH - h.ARC_DEGASSING_SHARE * earth_subduction
    volcanic = earth_volcanic * p.outgassing * melt * concentration
    subduction = crust / h.CARBONATE_TURNOVER_GYR * spreading if regime == "mobile_lid" else 0.0
    arc = h.ARC_DEGASSING_SHARE * subduction * p.outgassing
    recycling = 0.0
    if regime in ("stagnant_lid", "episodic", "heat_pipe"):
        recycling = crust / h.LID_CARBONATE_RECYCLING_GYR * melt / max(area, 1e-9)
    hot = logistic((surface_k - h.CARBONATE_DECOMPOSITION_K) / 20.0)
    decomposition = crust * hot / h.CARBONATE_DECOMPOSITION_GYR
    return CarbonFluxes(continental=continental, seafloor=seafloor, volcanic=volcanic, arc=arc,
                        subduction=subduction, recycling=recycling, decomposition=decomposition)


def water_fluxes(p: HistoryParams, surface: float, mantle: float, regime: str, melt: float,
                 spreading: float, ocean: bool) -> tuple[float, float]:
    """Return (degassing, regassing) of water in oceans per Gyr."""
    concentration = (mantle / h.MANTLE_WATER_EARTH_OCEANS) / p.silicate_ratio
    degassing = h.WATER_DEGASSING_EARTH * p.outgassing * melt * concentration
    regassing = 0.0
    if regime == "mobile_lid" and ocean and surface > 0.0:
        surface_fraction = surface * c.EARTH_OCEAN_MASS / p.mass_kg
        pressure = (p.gravity / c.G_EARTH) ** 2 * surface_fraction / h.SURFACE_WATER_EARTH
        capacity = h.MANTLE_WATER_MAX * p.silicate_kg / c.EARTH_OCEAN_MASS
        room = max(1.0 - mantle / capacity, 0.0)
        regassing = (h.WATER_DEGASSING_EARTH * spreading * p.area_ratio * pressure**h.WATER_PRESSURE_EXPONENT
                     * room / EARTH_MANTLE_ROOM)
    return degassing, regassing


def xuv_flux(luminosity_w: float, xuv_fraction: float, p: HistoryParams) -> float:
    """Return the XUV flux at the planet (W/m²)."""
    return luminosity_w * xuv_fraction / (4.0 * math.pi * p.semi_major_axis_m**2)


def energy_limited(p: HistoryParams, flux_w_m2: float) -> float:
    """Return the energy-limited mass loss rate (10¹⁸ kg per Gyr)."""
    rate = p.escape_efficiency * math.pi * flux_w_m2 * p.radius_m**3 / (c.G * p.mass_kg)
    return rate * GYR_PER_SECOND_1E18


def stratosphere_water(surface_k: float) -> float:
    """Return the water mixing ratio above the cold trap: 3 ppm at 288 K rising to ~10⁻³ at 340 K."""
    slope = (math.log10(3e-3) - math.log10(h.STRATOSPHERE_WATER_COLD)) / (h.MOIST_GREENHOUSE_K - 288.0)
    return min(h.STRATOSPHERE_WATER_COLD * 10 ** (slope * (surface_k - 288.0)), 0.5)


def water_escape(p: HistoryParams, flux_w_m2: float, surface_k: float, steam: bool) -> float:
    """Return the water lost to space (oceans per Gyr): hydrogen escapes, its oxygen stays."""
    limit = energy_limited(p, flux_w_m2)          # hydrogen mass rate if all energy lifts hydrogen
    if not steam:
        mixing = 2.0 * stratosphere_water(surface_k)
        diffusion = h.DIFFUSION_LIMIT_PER_MIXING * mixing * p.area_m2 * c.M_H * GYR_PER_SECOND_1E18
        limit = limit * diffusion / (limit + diffusion) if limit + diffusion > 0 else 0.0
    return 9.0 * limit * 1e18 / c.EARTH_OCEAN_MASS


def bulk_escape(p: HistoryParams, flux_w_m2: float, instellation: float) -> float:
    """Return the loss of heavy gases (10¹⁸ kg per Gyr).

    The loss follows the XUV flux; whether it happens at all depends on where
    the planet lies relative to the cosmic shoreline at present-Sun activity.
    """
    escape_velocity = math.sqrt(2.0 * c.G * p.mass_kg / p.radius_m)
    ratio = shoreline_ratio(instellation, 1.0, escape_velocity)
    switch = logistic(math.log10(max(ratio, 1e-12) / h.BULK_ESCAPE_SHORELINE) / h.BULK_ESCAPE_WIDTH)
    return h.BULK_ESCAPE_SHARE * energy_limited(p, flux_w_m2) * switch
