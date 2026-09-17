"""Atmosphere retention, surface pressure, greenhouse warming, global climate and surface water state.

Retention follows the empirical cosmic shoreline (Zahnle & Catling 2017).
Surface temperature uses a grey-atmosphere greenhouse calibrated to Earth,
Venus, Mars and Titan (see ``heuristics.py``). The zonal climate model
(Tier 0) then sets the planetary albedo, mean temperature and water phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import constants as c
from . import heuristics as h
from .orbit import equilibrium_temperature
from .climate import COLD_START_K, ZonalClimate, climate_setting, solve_zonal
from .climate.physics import optical_depth_for_outgoing
from .state import AtmosphereState, BulkState, ClimateSummary, Issue, OrbitState, StarState, WaterState
from .util import logistic, named_rng
from .water import has_surface_water

if TYPE_CHECKING:
    from .state import PlanetState

NO_SURFACE_CLASSES = ("sub-Neptune", "ice giant", "gas giant", "brown dwarf")
WATER_TRIPLE_POINT_PA = 611.657
WATER_TRIPLE_POINT_K = 273.16
WATER_CRITICAL_K = 647.1
WATER_LATENT_HEAT_J_MOL = 40_650.0
WEATHERING_COMPOSITIONS = ("n2_co2", "n2_o2")


@dataclass
class AtmosphereInputs:
    """User or prior values for the atmosphere; ``None`` means derive."""

    present: Optional[bool] = None
    surface_pressure_bar: Optional[float] = None
    composition: Optional[str] = None
    bond_albedo: Optional[float] = None
    surface_temperature_k: Optional[float] = None
    volatile_richness: float = 1.0
    weathering_target_k: float = h.WEATHERING_TARGET_K
    pressure_is_user: bool = False
    extra_optical_depth: float = 0.0     # greenhouse from gases the composition does not include (biotic methane)
    land_albedo: Optional[float] = None  # snow-free land albedo for the climate model
    land_by_band: Optional[np.ndarray] = None  # land share of each climate band, once the surface is built
    # History mode: the integrated greenhouse, runaway state and climate branch replace the snapshot rules.
    optical_depth: Optional[float] = None
    runaway: Optional[bool] = None
    frozen_branch: bool = False
    regulate: bool = True
    land_fraction: Optional[float] = None


def shoreline_ratio(instellation_earth: float, xuv_relative_sun: float, escape_velocity_m_s: float) -> float:
    """Return effective insolation divided by the cosmic-shoreline value (< 1 favours an atmosphere)."""
    i_eff = instellation_earth * xuv_relative_sun**h.SHORELINE_XUV_EXPONENT
    i_shore = (escape_velocity_m_s / (h.SHORELINE_V0_KMS * c.KM)) ** 4
    return i_eff / i_shore


def retention_probability(ratio: float) -> float:
    """Return the heuristic probability that the planet keeps an atmosphere."""
    return logistic(-math.log10(ratio) / h.SHORELINE_LOG_WIDTH)


def default_surface_pressure_bar(bulk: BulkState, ratio: float, volatile_richness: float, seed: int) -> float:
    """Return a surface pressure (bar) from the volatile inventory and distance to the shoreline."""
    m = bulk.mass_kg / c.M_EARTH
    r = bulk.radius_m / c.R_EARTH
    shore = max(1.0 - ratio, h.PRESSURE_MIN_SHORELINE_FACTOR)
    scatter = 10 ** named_rng(seed, "atmosphere.pressure_scatter").normal(0.0, h.PRESSURE_SCATTER_DEX)
    return volatile_richness * m**2 / r**4 * shore * scatter


def boiling_point(pressure_pa: float) -> float:
    """Return the boiling point of water (K) at a pressure, capped at the critical point."""
    if pressure_pa <= WATER_TRIPLE_POINT_PA:
        return WATER_TRIPLE_POINT_K
    inv_t = 1 / 373.15 - c.R_GAS / WATER_LATENT_HEAT_J_MOL * math.log(pressure_pa / c.P_EARTH)
    return min(1 / inv_t, WATER_CRITICAL_K) if inv_t > 0 else WATER_CRITICAL_K


def greenhouse_optical_depth(composition: str, pressure_pa: float) -> float:
    """Return the grey infrared optical depth for a composition and surface pressure."""
    if pressure_pa <= 0:
        return 0.0
    return h.TAU0[composition] * (pressure_pa / c.BAR) ** h.GREENHOUSE_PRESSURE_EXPONENT


def surface_temperature(t_eq: float, tau: float) -> float:
    """Return the grey-atmosphere surface temperature (K)."""
    return t_eq * (1 + 0.75 * tau) ** 0.25


def optical_depth_for_temperature(t_eq: float, t_surface: float) -> float:
    """Return the optical depth that produces a given surface temperature."""
    return max((4 / 3) * ((t_surface / t_eq) ** 4 - 1), 0.0)


def default_albedo(present: bool, pressure_pa: float, planet_class: str) -> float:
    """Return a default Bond albedo from the atmosphere thickness and planet class."""
    if planet_class in NO_SURFACE_CLASSES:
        return h.ALBEDO_GIANT
    if not present:
        return h.ALBEDO_AIRLESS
    bar = pressure_pa / c.BAR
    if bar < h.THIN_ATMOSPHERE_BAR:
        return h.ALBEDO_THIN
    if bar > h.THICK_ATMOSPHERE_BAR:
        return h.ALBEDO_THICK
    return h.ALBEDO_TEMPERATE


def default_composition(present: bool, planet_class: str, regime: Optional[str], t_surface: float,
                        has_water: bool, runaway: bool) -> str:
    """Return a default composition from planet class, tectonic regime, water and temperature.

    Dry planets and planets past the runaway limit have no weathering sink and
    get CO2-dominated air (Venus, Mars); others get nitrogen with some CO2.
    """
    if planet_class in NO_SURFACE_CLASSES:
        return "h2_he"
    if not present:
        return "none"
    if t_surface < h.COLD_NITROGEN_METHANE_BELOW_K:
        return "n2_ch4"
    if runaway or not has_water:
        return "co2"
    return "n2_co2"


def water_state(has_water: bool, present: bool, pressure_pa: float, t_surface: float, runaway: bool,
                open_water: float = 1.0, open_ocean: float = 1.0) -> str:
    """Return the dominant surface water phase: 'liquid', 'ice', 'vapour' or 'none'.

    ``open_water`` (share of the surface) and ``open_ocean`` (share of the
    ocean) are the ice-free fractions from the climate model; enough open
    water makes the phase 'liquid' even when the mean temperature is below
    freezing, and too little makes it 'ice' even when it is above.
    """
    if not has_water or runaway:
        return "none"
    if not present or pressure_pa < WATER_TRIPLE_POINT_PA:
        return "ice" if t_surface < WATER_TRIPLE_POINT_K else "none"
    if t_surface >= boiling_point(pressure_pa):
        return "vapour"
    if open_water >= h.OPEN_WATER_LIQUID or open_ocean >= h.OPEN_OCEAN_LIQUID:
        return "liquid"
    return "ice"


def weathering_limits(tau: float, pressure_pa: float) -> tuple[float, float]:
    """Return the optical depths weathering can reach from ``tau``: CO₂ can at most make up the whole atmosphere."""
    return min(h.WEATHERING_TAU_MIN, tau), min(max(greenhouse_optical_depth("co2", pressure_pa), tau),
                                               h.WEATHERING_TAU_MAX)


def user_temperature_limits(tau: float) -> tuple[float, float]:
    """Return the optical depths allowed when matching a user-set surface temperature."""
    return 0.0, max(10.0 * tau, 50.0)


def temperature_control(atmosphere: AtmosphereState, composition_tau: float, user_temperature_k: Optional[float],
                        mean_k: float) -> Optional[tuple[float, float, float]]:
    """Return (target, lowest, highest optical depth) for holding a planet's mean temperature, or None.

    Planets whose weathering reached its target, and planets with a user-set
    surface temperature, keep that temperature; ``composition_tau`` is the
    optical depth of the air before any adjustment.
    """
    if user_temperature_k is not None:
        return (user_temperature_k, *user_temperature_limits(atmosphere.greenhouse_optical_depth))
    if atmosphere.weathering_regulated and abs(mean_k - atmosphere.weathering_target_k) <= h.CLIMATE_TARGET_MISS_K:
        return (atmosphere.weathering_target_k, *weathering_limits(composition_tau, atmosphere.surface_pressure_pa))
    return None


def build_atmosphere(
    star: StarState,
    orbit: OrbitState,
    bulk: BulkState,
    inputs: AtmosphereInputs,
    regime: Optional[str],
    water: WaterState,
    seed: int,
) -> tuple[AtmosphereState, Optional[ClimateSummary], Optional[ZonalClimate], list[Issue], dict[str, str]]:
    """Return the atmosphere, the global climate summary and its zonal solution, notes and provenance.

    ``regime`` is the tectonic regime, or ``None`` for a preliminary estimate
    made before the interior is known; the preliminary estimate skips the
    climate model and has no climate summary.
    """
    issues: list[Issue] = []
    prov: dict[str, str] = {}
    s = orbit.instellation_earth
    no_surface = bulk.planet_class in NO_SURFACE_CLASSES
    has_water = has_surface_water(water, bulk)

    # Retention
    ratio = shoreline_ratio(s, star.xuv_fraction_relative_sun, bulk.escape_velocity_m_s)
    p_keep = 1.0 if no_surface else retention_probability(ratio)
    present = inputs.present
    if present is None and inputs.pressure_is_user and inputs.surface_pressure_bar is not None:
        present = inputs.surface_pressure_bar > 0
        prov["atmosphere.present"] = "derived"
    if present is None:
        present = no_surface or bool(named_rng(seed, "atmosphere.retention").random() < p_keep)
        prov["atmosphere.present"] = "derived" if no_surface else "heuristic"
        if not no_surface:
            issues.append(Issue("info", "heuristic", "atmosphere",
                                "atmosphere retention drawn from the cosmic-shoreline heuristic"))
    elif present and p_keep < 0.05:
        issues.append(Issue("warning", "conflict", "atmosphere",
                            f"planet lies well beyond the cosmic shoreline (ratio {ratio:.1f}) "
                            "but an atmosphere was specified"))

    # Pressure
    if not present:
        pressure = 0.0
        prov["atmosphere.surface_pressure_bar"] = "derived"
        if inputs.pressure_is_user and inputs.surface_pressure_bar not in (None, 0):
            issues.append(Issue("warning", "conflict", "atmosphere", "pressure given but atmosphere is absent"))
    elif no_surface:
        pressure = c.BAR
        prov["atmosphere.surface_pressure_bar"] = "derived"
        issues.append(Issue("info", "note", "atmosphere",
                            "no solid surface: 'surface' values refer to the 1 bar level"))
    elif inputs.surface_pressure_bar is not None:
        pressure = inputs.surface_pressure_bar * c.BAR
    else:
        pressure = default_surface_pressure_bar(bulk, ratio, inputs.volatile_richness, seed) * c.BAR
        prov["atmosphere.surface_pressure_bar"] = "heuristic"
        issues.append(Issue("info", "heuristic", "atmosphere",
                            "surface pressure from a heuristic volatile-inventory scaling"))

    # Albedo, equilibrium temperature, composition and greenhouse
    albedo = inputs.bond_albedo
    albedo_fixed = albedo is not None
    if albedo is None:
        albedo = default_albedo(present, pressure, bulk.planet_class)
        prov["atmosphere.bond_albedo"] = "heuristic"
    t_eq = equilibrium_temperature(s, albedo)

    # Runaway greenhouse
    runaway_limit = orbit.habitable_zone.runaway_greenhouse_s
    runaway = bool(present and has_water and not no_surface and s > runaway_limit)
    if inputs.runaway is not None:
        runaway = bool(inputs.runaway and has_water and not no_surface)
    if runaway:
        issues.append(Issue("warning", "note", "atmosphere",
                            f"instellation ({s:.2f}) exceeds the runaway-greenhouse limit ({runaway_limit:.2f}): "
                            "surface water would be lost, as on Venus"))

    composition = inputs.composition
    if composition is None:
        guess = surface_temperature(t_eq, greenhouse_optical_depth("n2_co2", pressure)) if present else t_eq
        composition = default_composition(present, bulk.planet_class, regime, guess, has_water, runaway)
        prov["atmosphere.composition"] = "heuristic"
    elif not present and composition != "none":
        issues.append(Issue("warning", "conflict", "atmosphere", "composition given but atmosphere is absent"))
        composition = "none"

    tau = 0.0 if no_surface else greenhouse_optical_depth(composition, pressure)
    if present and not no_surface:
        tau += inputs.extra_optical_depth
    if inputs.optical_depth is not None and not no_surface:
        tau = inputs.optical_depth
    t_surf = surface_temperature(t_eq, tau)
    if present and not no_surface:
        issues.append(Issue("info", "heuristic", "atmosphere",
                            "greenhouse warming uses a grey-atmosphere calibration"))

    use_climate = regime is not None and not no_surface
    surface_water = has_water and not runaway

    land_share = water.land_fraction_estimate if surface_water else 1.0
    if surface_water and inputs.land_fraction is not None:
        land_share = inputs.land_fraction
    if surface_water and inputs.land_by_band is not None:
        land_share = inputs.land_by_band

    def climate_at(optical_depth: float, reference_k: Optional[float] = None, **options) -> ZonalClimate:
        """Return the Tier 0 climate for a greenhouse optical depth."""
        reference = surface_temperature(t_eq, optical_depth) if reference_k is None else reference_k
        setting = climate_setting(s, orbit.eccentricity, orbit.obliquity_rad, orbit.periapsis_longitude_rad,
                                  orbit.spin_state == "synchronous", pressure, bulk.surface_gravity_m_s2,
                                  composition, optical_depth, orbit.rotation_period_s, orbit.orbital_period_s,
                                  albedo, albedo_fixed, surface_water, reference)
        if inputs.frozen_branch and "initial" not in options:
            options.setdefault("start_k", COLD_START_K)
        return solve_zonal(setting, land_share, land_albedo=inputs.land_albedo, **options)

    def matched(target_k: float, low: float, high: float) -> tuple[float, ZonalClimate]:
        """Return the optical depth (within limits) that brings the Tier 0 mean closest to a target, and its climate."""
        controlled = climate_at(tau, reference_k=target_k, target_mean_k=target_k)
        a, b = controlled.outgoing
        grey = a + b * target_k
        depth = min(max(optical_depth_for_outgoing(grey, target_k), low), high)
        # Check the state holds with the greenhouse fixed; it can fall to another branch if it does not.
        return depth, climate_at(depth, reference_k=target_k, initial=controlled)

    # Carbonate-silicate regulation
    zonal: Optional[ZonalClimate] = None
    weathering = False
    wet = surface_water and water_state(has_water, present, pressure, t_surf, runaway) in ("liquid", "ice")
    waterworld = water.land_fraction_estimate < h.WATERWORLD_LAND_FRACTION
    regulated = (inputs.regulate and inputs.surface_temperature_k is None and regime == "mobile_lid"
                 and composition in WEATHERING_COMPOSITIONS and wet and not runaway)
    if regulated and waterworld:
        issues.append(Issue("info", "note", "atmosphere",
                            "almost no land: continental weathering cannot regulate CO₂"))
    elif regulated:
        # CO2 can at most make up the whole atmosphere at its current pressure.
        tau_min, tau_max = weathering_limits(tau, pressure)
        tau_needed = optical_depth_for_temperature(t_eq, inputs.weathering_target_k)
        tau = min(max(tau_needed, tau_min), tau_max)
        if use_climate:
            tau, zonal = matched(inputs.weathering_target_k, tau_min, tau_max)
        t_surf = surface_temperature(t_eq, tau)
        weathering = True
        issues.append(Issue("info", "heuristic", "atmosphere",
                            "carbonate–silicate cycle adjusts CO₂ toward temperate conditions"))
        if zonal is not None and abs(zonal.mean_k - inputs.weathering_target_k) > h.CLIMATE_TARGET_MISS_K:
            limited = tau in (tau_min, tau_max)
            reason = ("the greenhouse gases available cannot reach it" if limited else
                      "no stable climate exists there (ice-albedo feedback)")
            issues.append(Issue("info", "note", "climate",
                                f"weathering target {inputs.weathering_target_k:.0f} K not reached: {reason}; "
                                f"the planet settles at {zonal.mean_k:.0f} K"))

    # User-fixed surface temperature
    if inputs.surface_temperature_k is not None:
        t_surf = inputs.surface_temperature_k
        tau = optical_depth_for_temperature(t_eq, t_surf)
        if use_climate:
            tau, zonal = matched(t_surf, *user_temperature_limits(tau))
        if t_surf < t_eq:
            issues.append(Issue("info", "note", "atmosphere",
                                "surface temperature below equilibrium implies an anti-greenhouse effect"))
        if not present and abs(t_surf - t_eq) > 5:
            issues.append(Issue("warning", "conflict", "atmosphere",
                                "surface temperature differs from equilibrium but there is no atmosphere"))

    # Global climate: planetary albedo, ice cover and the regional water phase
    climate = None
    open_water = open_ocean = 1.0
    if use_climate:
        if zonal is None:
            zonal = climate_at(tau)
        snowball = zonal.open_water_fraction < h.OPEN_WATER_LIQUID
        if surface_water and zonal.open_ocean_fraction < 0.999 and not snowball and not inputs.frozen_branch:
            cold = climate_at(tau, reference_k=zonal.mean_k, start_k=COLD_START_K)
            snowball = cold.open_water_fraction < h.OPEN_WATER_LIQUID
            if snowball:
                issues.append(Issue("info", "note", "climate",
                                    "a fully frozen (snowball) climate is also stable; the partly open one is used"))
        if inputs.surface_temperature_k is None:
            t_surf = zonal.mean_k
        if not albedo_fixed:
            albedo = zonal.albedo
            t_eq = equilibrium_temperature(s, albedo)
        if surface_water:
            open_water, open_ocean = zonal.open_water_fraction, zonal.open_ocean_fraction
        climate = ClimateSummary(
            mean_temperature_k=zonal.mean_k,
            planetary_albedo=zonal.albedo,
            ice_free_albedo=inputs.bond_albedo if albedo_fixed else default_albedo(present, pressure,
                                                                                    bulk.planet_class),
            albedo_fixed=albedo_fixed,
            open_water_fraction=zonal.open_water_fraction if surface_water else 0.0,
            open_ocean_fraction=zonal.open_ocean_fraction if surface_water else 0.0,
            ice_line_deg=zonal.ice_line_deg,
            dayside_temperature_k=zonal.dayside_k,
            nightside_temperature_k=zonal.nightside_k,
            snowball_stable=bool(surface_water and snowball),
            converged=zonal.converged,
        )
        issues.append(Issue("info", "heuristic", "climate",
                            "global climate from a zonal energy-balance model with seasons and ice albedo"))
        if not zonal.converged:
            issues.append(Issue("warning", "validity", "climate", "the zonal climate model did not converge"))
    elif inputs.bond_albedo is None and has_water and not no_surface and t_surf < h.ICE_COVER_BELOW_K:
        # Preliminary estimate: ice-covered planets are brighter.
        albedo = h.ALBEDO_ICE
        t_eq = equilibrium_temperature(s, albedo)
        t_surf = surface_temperature(t_eq, tau)

    mu = h.MEAN_MOLECULAR_WEIGHT[composition]
    scale_height = c.K_B * t_surf / (mu * c.M_H * bulk.surface_gravity_m_s2) if mu > 0 else 0.0
    wstate = "none" if no_surface else water_state(has_water, present, pressure, t_surf, runaway,
                                                   open_water, open_ocean)

    state = AtmosphereState(
        present=present,
        retention_probability=p_keep,
        shoreline_ratio=ratio,
        surface_pressure_pa=pressure,
        composition=composition,
        mean_molecular_weight=mu,
        bond_albedo=albedo,
        equilibrium_temperature_k=t_eq,
        greenhouse_optical_depth=tau,
        surface_temperature_k=t_surf,
        scale_height_m=scale_height,
        water_boiling_point_k=boiling_point(pressure),
        surface_water=wstate,
        weathering_regulated=weathering,
        weathering_target_k=inputs.weathering_target_k,
        runaway_greenhouse=runaway,
    )
    return state, climate, zonal, issues, prov


def refine_climate(state: "PlanetState", land_by_band: np.ndarray) -> tuple[AtmosphereState, ClimateSummary,
                                                                            ZonalClimate, list[Issue]]:
    """Return the atmosphere and global climate recomputed with the built surface's land share per band.

    Pressure, composition and the presence of air are kept; the greenhouse,
    temperature and ice cover follow the real land distribution. The water
    phase label is kept, with a note if the new climate would change it.
    """
    from .biosphere.life import methane_optical_depth, tier0_land_albedo   # the biosphere builds on this module

    old = state.atmosphere
    values = state.inputs
    life = state.biosphere
    inputs = AtmosphereInputs(
        present=old.present,
        surface_pressure_bar=old.surface_pressure_pa / c.BAR,
        composition=old.composition,
        bond_albedo=values.get("atmosphere.bond_albedo"),
        surface_temperature_k=values.get("atmosphere.surface_temperature_k"),
        weathering_target_k=old.weathering_target_k,
        extra_optical_depth=methane_optical_depth(life.methane_fraction) if life is not None else 0.0,
        land_albedo=tier0_land_albedo(life),
        land_by_band=land_by_band,
    )
    atmosphere, climate, zonal, issues, _ = build_atmosphere(state.star, state.orbit, state.bulk, inputs,
                                                            state.interior.tectonic_regime, state.water,
                                                            state.draw_seed)
    if atmosphere.surface_water != old.surface_water:
        issues.append(Issue("info", "note", "climate",
                            f"with the built surface's land distribution the surface water would be "
                            f"'{atmosphere.surface_water}'; '{old.surface_water}' is kept"))
        atmosphere = replace(atmosphere, surface_water=old.surface_water)
    return atmosphere, climate, zonal, issues


def composition_optical_depth(atmosphere: AtmosphereState, biosphere) -> float:
    """Return the greenhouse optical depth of the air's composition before any climate regulation."""
    from .biosphere.life import methane_optical_depth   # the biosphere builds on this module

    extra = methane_optical_depth(biosphere.methane_fraction) if biosphere is not None else 0.0
    return greenhouse_optical_depth(atmosphere.composition, atmosphere.surface_pressure_pa) + extra
