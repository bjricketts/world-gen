"""Mantle and core thermal evolution, melting and the dynamo (parameterised convection).

Heat flow follows Q ∝ A ΔT^(1+β) η(T)^(−β) scaled from present Earth. Plate
tectonics uses a weak β (Korenaga 2006); stagnant lids use β = 1/3 with a
lid factor (Stevenson et al. 1983; Driscoll & Bercovici 2014). The core
cools through a boundary layer of fixed viscosity; the dynamo runs while the core–mantle heat
flow exceeds the heat conducted along the core adiabat, or a share of it
once an inner core grows.

Calibration: an Earth run from a 1750 K mantle ends at 1620 K with a surface
heat flow of 82 mW/m², its present melt production, a dynamo that never
stops and an inner core that starts to grow at ~3.7 Gyr. A Mars run ends at
18 mW/m² with almost no melting and loses its dynamo after ~0.35 Gyr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import constants as c
from .. import heuristics as h
from ..interior import radiogenic_heat_per_kg
from .params import EARTH_CORE_KG, EARTH_CORE_RADIUS_M, EARTH_SILICATE_KG, HistoryParams

REGIME_FLUX = {
    "mobile_lid": None,
    "stagnant_lid": h.LID_FLUX_FACTOR,
    "episodic": h.EPISODIC_FLUX_FACTOR,
    "heat_pipe": h.HEAT_PIPE_FLUX_FACTOR,
    "inactive": h.INACTIVE_FLUX_FACTOR,
}
SURFACE_REFERENCE_K = 288.0


@dataclass
class ThermalRates:
    """Heat flows (W) and derived quantities at one instant."""

    radiogenic_w: float
    mantle_heating_w: float        # radiogenic heat released in the mantle, plus tides
    surface_w: float               # total surface heat flow
    convective_w: float
    cmb_w: float
    adiabatic_w: float
    inner_core: float              # 0 (none) to 1 (mostly frozen)
    dynamo_margin: float           # > 0 while the dynamo runs (W)
    melt: float                    # melt production relative to present Earth
    spreading: float               # plate creation rate per unit area relative to Earth (0 off plates)
    heat_flux_w_m2: float
    activity: float                # activity index from the heat leaving the surface (1 for Earth)
    budget_flux_w_m2: float        # heat available: radiogenic (at Earth's Urey ratio) plus tides
    budget_activity: float         # activity index of that budget; the tectonic regime follows it

    @property
    def dynamo(self) -> bool:
        """Return whether the core dynamo runs."""
        return self.dynamo_margin > 0.0


def _earth_radiogenic_w() -> float:
    """Return Earth's present radiogenic power in the model (W)."""
    return EARTH_SILICATE_KG * radiogenic_heat_per_kg(c.AGE_SUN_GYR)


def _convective_earth_w() -> float:
    """Return Earth's present convective (mantle) heat flow: surface flow minus crustal heat."""
    return h.SURFACE_HEAT_FLOW_EARTH_W - h.CRUST_RADIOGENIC_SHARE * _earth_radiogenic_w()


def viscosity_ratio(temperature_k: float, reference_k: float, activation_j: float) -> float:
    """Return η(T) / η(reference) for an Arrhenius viscosity."""
    return math.exp(min(activation_j / c.R_GAS * (1.0 / temperature_k - 1.0 / reference_k), 200.0))


def core_gravity_ratio(p: HistoryParams) -> float:
    """Return gravity at the core surface relative to Earth's."""
    return (p.core_kg / EARTH_CORE_KG) / (p.core_radius_m / EARTH_CORE_RADIUS_M) ** 2


def depth_scale(p: HistoryParams) -> float:
    """Return gravity × mantle depth relative to Earth's (sets the adiabatic rise to the core)."""
    depth = p.radius_m - p.core_radius_m
    return (p.gravity * depth) / (c.G_EARTH * (c.R_EARTH - EARTH_CORE_RADIUS_M))


def lower_mantle_factor(p: HistoryParams) -> float:
    """Return the lower-mantle temperature divided by the potential temperature."""
    return 1.0 + (h.LOWER_MANTLE_FACTOR - 1.0) * depth_scale(p)


def inner_core_onset_k(p: HistoryParams) -> float:
    """Return the core–mantle boundary temperature at which an inner core starts to grow."""
    pressure = (p.mass_kg / c.M_EARTH) ** 2 / (p.radius_m / c.R_EARTH) ** 4
    return h.INNER_CORE_ONSET_EARTH_K * pressure**h.INNER_CORE_PRESSURE_EXPONENT


def thermal_rates(p: HistoryParams, t_gyr: float, mantle_k: float, core_k: float, regime: str,
                  surface_k: float) -> ThermalRates:
    """Return heat flows, melting and dynamo state for the current temperatures and regime."""
    radiogenic = p.silicate_kg * radiogenic_heat_per_kg(t_gyr, p.radiogenic_abundance)
    crust_share = h.CRUST_RADIOGENIC_SHARE if regime == "mobile_lid" else 0.0
    heating = radiogenic * (1.0 - crust_share) + p.tidal_power_w

    surface_k = min(surface_k, mantle_k - 1.0)
    d_earth = h.MANTLE_TEMPERATURE_EARTH_K - SURFACE_REFERENCE_K
    d_t = max(mantle_k - surface_k, 1.0) / d_earth
    eta = viscosity_ratio(mantle_k, h.MANTLE_TEMPERATURE_EARTH_K, p.activation_energy_j)
    area = p.area_ratio
    plate_flux = (h.PLATE_FLUX_SCALE * _convective_earth_w() * area * d_t ** (1.0 + h.PLATE_FLUX_BETA)
                  * eta ** (-h.PLATE_FLUX_BETA))
    factor = REGIME_FLUX.get(regime, h.LID_FLUX_FACTOR)
    if factor is None:
        convective = plate_flux
    else:
        g = (p.gravity / c.G_EARTH) ** h.LID_FLUX_BETA
        convective = (factor * _convective_earth_w() * area * d_t ** (1.0 + h.LID_FLUX_BETA)
                      * eta ** (-h.LID_FLUX_BETA) * g)
    surface = convective + radiogenic * crust_share

    # Core–mantle boundary
    lower = lower_mantle_factor(p) * mantle_k
    d_cmb_earth = h.CORE_TEMPERATURE_EARTH_K - h.LOWER_MANTLE_FACTOR * h.MANTLE_TEMPERATURE_EARTH_K
    d_cmb = max(core_k - lower, 0.0) / d_cmb_earth
    core_area = (p.core_radius_m / EARTH_CORE_RADIUS_M) ** 2
    cmb = (h.CMB_HEAT_FLOW_EARTH_W * core_area * d_cmb ** (1.0 + h.CMB_FLUX_BETA)
           * core_gravity_ratio(p) ** h.CMB_FLUX_BETA)
    adiabatic = (h.CORE_ADIABATIC_EARTH_W * p.core_adiabatic * (p.core_kg / EARTH_CORE_KG)
                 * core_k / h.CORE_TEMPERATURE_EARTH_K)
    onset = inner_core_onset_k(p)
    inner = math.sqrt(min(max((onset - core_k) / h.INNER_CORE_RANGE_K, 0.0), 1.0))
    buoyancy = min(inner / 0.1, 1.0)
    needed = adiabatic * (1.0 - (1.0 - h.INNER_CORE_DYNAMO_SHARE) * buoyancy)
    margin = cmb - needed if p.core_kg > 0 else -1.0

    # Melting and plate speed
    raw = max(mantle_k - h.MANTLE_SOLIDUS_K, 0.0) / (h.MANTLE_TEMPERATURE_EARTH_K - h.MANTLE_SOLIDUS_K)
    excess = raw / (1.0 + raw / h.MELT_EXCESS_MAX) * (1.0 + 1.0 / h.MELT_EXCESS_MAX)
    if regime == "mobile_lid":
        spreading = (convective / (h.PLATE_FLUX_SCALE * _convective_earth_w() * area)) ** 2
        melt = spreading * excess * area
    elif regime in ("stagnant_lid", "episodic", "heat_pipe"):
        spreading = 0.0
        scale = {"stagnant_lid": 1.0, "episodic": 3.0, "heat_pipe": 50.0}[regime]
        melt = scale * h.LID_MELT_FACTOR * area * excess**3
    else:
        spreading = 0.0
        melt = 0.0
    if regime != "mobile_lid":
        # Magma carries heat through the lid (on plate planets it is part of the convective flow).
        convective += h.MAGMA_HEAT_EARTH_W * melt
        surface += h.MAGMA_HEAT_EARTH_W * melt
    flux = surface / p.area_m2
    mass_factor = (p.mass_kg / c.M_EARTH) ** h.ACTIVITY_MASS_EXPONENT
    activity = flux / c.EARTH_HEAT_FLUX * mass_factor
    # The regime follows the heat the planet has to lose, not the flux its current lid happens to carry.
    budget_flux = (radiogenic * h.INVERSE_UREY_RATIO + p.tidal_power_w) / p.area_m2
    return ThermalRates(radiogenic_w=radiogenic, mantle_heating_w=heating, surface_w=surface,
                        convective_w=convective, cmb_w=cmb, adiabatic_w=adiabatic, inner_core=inner,
                        dynamo_margin=margin, melt=melt, spreading=spreading, heat_flux_w_m2=flux,
                        activity=activity, budget_flux_w_m2=budget_flux,
                        budget_activity=budget_flux / c.EARTH_HEAT_FLUX * mass_factor)


def temperature_rates(p: HistoryParams, rates: ThermalRates) -> tuple[float, float]:
    """Return dT_mantle/dt and dT_core/dt in K per Gyr."""
    mantle = (rates.mantle_heating_w - rates.convective_w + rates.cmb_w) / (p.silicate_kg * h.MANTLE_HEAT_CAPACITY)
    core = -rates.cmb_w / max(p.core_kg * h.CORE_HEAT_CAPACITY, 1.0)
    return mantle * c.SECONDS_PER_GYR, core * c.SECONDS_PER_GYR


def initial_core_k(p: HistoryParams, mantle_k: float) -> float:
    """Return the core temperature at formation: the lower-mantle temperature plus a superheat from core formation."""
    return lower_mantle_factor(p) * mantle_k + h.CORE_SUPERHEAT_EARTH_K * depth_scale(p) ** h.CORE_SUPERHEAT_EXPONENT
