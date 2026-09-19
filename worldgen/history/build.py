"""Planet states at chosen epochs from an integrated history, and user overrides at the target epoch."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

from .. import atmosphere as atm
from .. import constants as c
from .. import heuristics as h
from .. import orbit as orbit_model
from .. import star as star_model
from ..biosphere.life import LifeInputs, build_biosphere, tier0_land_albedo
from ..priors import score_planet
from ..priors.sampling import ResolvedInputs
from ..spec import PlanetSpec
from ..state import (InteriorState, Issue, Override, PlanetState, TimelineEvent, WaterState)
from .integrate import CK, CM, CS, N2, O2, TC, TM, WM, WS, HistoryModel, HistoryResult
from .volatiles import MU_CO2

# Continuous present-state fields: history computes them, user values replace them at the end.
END_OVERRIDES = (
    "body.water_mass_fraction", "atmosphere.present", "atmosphere.surface_pressure_bar", "atmosphere.composition",
    "atmosphere.bond_albedo", "atmosphere.surface_temperature_k", "atmosphere.weathering_target_k",
    "surface.land_fraction", "biosphere.oxygen_fraction", "biosphere.age_gyr", "interior.magnetic_field",
)
# Discrete fields the history holds while physics allows; a changed value is overridden at the end.
GUIDED = ("interior.tectonic_regime", "biosphere.life")


@dataclass
class HistoryContext:
    """What building a state from the history needs besides the integration result."""

    spec: PlanetSpec
    inputs: ResolvedInputs
    base: PlanetState                    # snapshot build at the target epoch (star, orbit, bulk, draws)
    model: HistoryModel
    result: HistoryResult
    regime_probabilities: dict[str, float]
    issues: list[Issue] = field(default_factory=list)


def composition_label(o2_fraction: float, co2_fraction: float, present: bool, surface_k: float) -> str:
    """Return the composition class of the integrated air."""
    if not present:
        return "none"
    if co2_fraction > 0.5:
        return "co2"
    if surface_k < h.COLD_NITROGEN_METHANE_BELOW_K:
        return "n2_ch4"
    if o2_fraction > h.OXYGEN_COMPOSITION_ABOVE:
        return "n2_o2"
    return "n2_co2"


def _events_until(result: HistoryResult, t_gyr: float) -> list[TimelineEvent]:
    """Return the logged events up to a planet age."""
    return [TimelineEvent(e.time_gyr * c.SECONDS_PER_GYR, e.kind, e.detail, e.flagged)
            for e in result.events if e.time_gyr <= t_gyr + 1e-9]


def build_state(ctx: HistoryContext, t_gyr: float, apply_overrides: bool = False) -> PlanetState:
    """Return the planet state at age ``t_gyr``; with ``apply_overrides`` user end values replace history values."""
    inputs, base, model, result = ctx.inputs, ctx.base, ctx.model, ctx.result
    v = inputs.get
    p = model.p
    seg = result.segment_at(t_gyr)
    y = result.state_at(t_gyr)
    modes = seg.modes
    d = model.diagnostics(t_gyr, y, modes)
    user = {k: inputs.values.get(k) for k in END_OVERRIDES + GUIDED if inputs.is_user(k)} if apply_overrides else {}
    provenance = dict(base.provenance)
    issues = [i for i in ctx.issues]
    overrides: list[Override] = []

    def override(path: str, history_value, user_value) -> None:
        """Record a user value that replaces a different history value."""
        if hasattr(history_value, "item"):
            history_value = history_value.item()
        if history_value == user_value or (isinstance(history_value, float) and isinstance(user_value, float)
                                           and math.isclose(history_value, user_value, rel_tol=1e-3)):
            return
        provenance[path] = inputs.provenance.get(path, "user")
        overrides.append(Override(path, history_value, user_value))

    star, star_issues = star_model.build_star(p.star_mass_msun, t_gyr, v("star.metallicity_feh", 0.0),
                                              p.activity_percentile)
    ocean_kg = c.EARTH_OCEAN_MASS
    surface_w, mantle_w = max(y[WS], 0.0), max(y[WM], 0.0)
    total_fraction = (surface_w + mantle_w) * ocean_kg / p.mass_kg
    if "body.water_mass_fraction" in user:
        wanted = user["body.water_mass_fraction"]
        override("body.water_mass_fraction", total_fraction, wanted)
        scale = wanted / total_fraction if total_fraction > 0 else 0.0
        surface_w, mantle_w = surface_w * scale, mantle_w * scale
        total_fraction = wanted
    bulk = replace(base.bulk, water_mass_fraction=total_fraction)

    rotation_h = v("orbit.rotation_period_h")
    orbit, orbit_issues, _ = orbit_model.build_orbit(
        star=star, bulk=bulk, semi_major_axis_m=p.semi_major_axis_m, instellation_earth=None,
        eccentricity=base.orbit.eccentricity, obliquity_rad=math.radians(v("orbit.obliquity_deg", 0.0)),
        rotation_period_s=None if rotation_h is None else rotation_h * c.SECONDS_PER_HOUR,
        rotation_is_user=inputs.is_user("orbit.rotation_period_h"),
        obliquity_is_user=inputs.is_user("orbit.obliquity_deg"),
        tidal_q=base.orbit.tidal_q, periapsis_longitude_rad=base.orbit.periapsis_longitude_rad)

    rates = d.thermal
    regime = modes.regime
    field_on = rates.dynamo
    if "interior.tectonic_regime" in user and user["interior.tectonic_regime"] != regime:
        override("interior.tectonic_regime", regime, user["interior.tectonic_regime"])
        regime = user["interior.tectonic_regime"]
    if "interior.magnetic_field" in user and user["interior.magnetic_field"] != field_on:
        override("interior.magnetic_field", field_on, user["interior.magnetic_field"])
        field_on = user["interior.magnetic_field"]
    interior = InteriorState(
        radiogenic_power_w=rates.radiogenic_w, tidal_power_w=p.tidal_power_w,
        surface_heat_flux_w_m2=rates.heat_flux_w_m2, activity_index=rates.activity, tectonic_regime=regime,
        regime_probabilities=ctx.regime_probabilities, magnetic_field=bool(field_on),
        dynamo_probability=1.0 if rates.dynamo else 0.0)
    provenance["interior.tectonic_regime"] = inputs.provenance.get("interior.tectonic_regime", "derived") \
        if inputs.is_user("interior.tectonic_regime") else "derived"
    provenance["interior.magnetic_field"] = "derived" if "interior.magnetic_field" not in user else "user"

    runaway = modes.climate == "runaway"
    surface_kg = surface_w * ocean_kg
    land = d.land if d.ocean else 1.0
    if "surface.land_fraction" in user:
        override("surface.land_fraction", land, user["surface.land_fraction"])
    water = WaterState(
        total_mass_fraction=total_fraction, mantle_mass_fraction=mantle_w * ocean_kg / p.mass_kg,
        surface_mass_kg=surface_kg, ocean_volume_m3=surface_kg / h.SEAWATER_DENSITY,
        seafloor_pressure_ratio=(p.gravity / c.G_EARTH) ** 2 * (surface_kg / p.mass_kg) / h.SURFACE_WATER_EARTH,
        plate_cycling=regime == "mobile_lid", land_fraction_estimate=land)

    air = d.air
    pressure = air.pressure_bar + (air.steam_bar if runaway else 0.0)
    present = pressure > h.THIN_AIR_BAR
    composition = composition_label(air.o2_fraction, air.co2_fraction, present, d.surface_k)
    life_mode = modes.life
    if "biosphere.life" in user and user["biosphere.life"] != life_mode:
        override("biosphere.life", life_mode, user["biosphere.life"])
        life_mode = user["biosphere.life"]

    atm_inputs = atm.AtmosphereInputs(
        present=present, surface_pressure_bar=pressure, composition=composition,
        volatile_richness=v("atmosphere.volatile_richness", 1.0),
        weathering_target_k=base.atmosphere.weathering_target_k, pressure_is_user=True,
        optical_depth=d.optical_depth, runaway=runaway, frozen_branch=modes.climate == "snowball",
        regulate=False, land_fraction=land if d.ocean else None)
    for path, name in (("atmosphere.present", "present"), ("atmosphere.surface_pressure_bar", "surface_pressure_bar"),
                       ("atmosphere.composition", "composition"), ("atmosphere.bond_albedo", "bond_albedo"),
                       ("atmosphere.surface_temperature_k", "surface_temperature_k")):
        if path in user:
            history_value = {"present": present, "surface_pressure_bar": pressure, "composition": composition,
                             "bond_albedo": None, "surface_temperature_k": d.surface_k}[name]
            setattr(atm_inputs, name, user[path])
            override(path, history_value, user[path])
    if "atmosphere.weathering_target_k" in user:
        atm_inputs.regulate = True
        atm_inputs.weathering_target_k = user["atmosphere.weathering_target_k"]
        override("atmosphere.weathering_target_k", d.surface_k, user["atmosphere.weathering_target_k"])
    if atm_inputs.surface_pressure_bar != pressure or atm_inputs.composition != composition:
        # A user atmosphere replaces the integrated greenhouse with the snapshot relation.
        atm_inputs.optical_depth = None

    def atmosphere_with(land_albedo):
        """Return the atmosphere and climate for a land albedo."""
        atm_inputs.land_albedo = land_albedo
        return atm.build_atmosphere(star, orbit, bulk, atm_inputs, regime=regime, water=water, seed=base.draw_seed)

    atmosphere, climate, zonal, atm_issues, _ = atmosphere_with(None)
    s = orbit.instellation_earth
    biosphere = None
    if life_mode != "none":
        origin = modes.life_origin if modes.life_origin is not None else t_gyr
        life_inputs = LifeInputs(
            life=life_mode, biochemistry=p.biochemistry, alien_product=v("biosphere.alien_product"),
            age_gyr=user.get("biosphere.age_gyr", max(t_gyr - origin, 0.0)),
            optimum_temperature_k=v("biosphere.optimum_temperature_k"),
            temperature_tolerance_k=v("biosphere.temperature_tolerance_k"),
            pigment_absorption_nm=v("biosphere.pigment_absorption_nm"),
            oxygen_fraction=user.get("biosphere.oxygen_fraction", air.o2_fraction),
            weathering_target_is_user=True, methane_fraction=d.methane_fraction)
        if "biosphere.age_gyr" in user:
            override("biosphere.age_gyr", max(t_gyr - origin, 0.0), user["biosphere.age_gyr"])
        if "biosphere.oxygen_fraction" in user:
            override("biosphere.oxygen_fraction", air.o2_fraction, user["biosphere.oxygen_fraction"])
        biosphere, bio_issues, _ = build_biosphere(life_inputs, star, s, atmosphere, water, zonal, t_gyr)
        issues.extend(i for i in bio_issues if i.kind != "heuristic")
        if biosphere is not None and biosphere.life == "surface":
            atmosphere, climate, zonal, atm_issues, _ = atmosphere_with(tier0_land_albedo(biosphere))
    provenance.update({k: "derived" for k in ("atmosphere.present", "atmosphere.surface_pressure_bar",
                                              "atmosphere.composition", "atmosphere.bond_albedo",
                                              "biosphere.oxygen_fraction", "biosphere.life", "biosphere.age_gyr")
                       if k not in user})
    for o in overrides:
        provenance[o.field] = inputs.provenance.get(o.field, "user")

    issues += [i for i in star_issues + orbit_issues + atm_issues if i.kind != "heuristic" or i.subsystem == "climate"]
    issues.append(Issue("info", "heuristic", "history",
                        "history mode: interior, volatiles, climate and life integrated from formation"))
    for o in overrides:
        issues.append(Issue("warning", "conflict", "history",
                            f"{o.field}: user value {_fmt(o.user_value)} replaces the history value "
                            f"{_fmt(o.history_value)}"))
    occupiability = score_planet(star, bulk, interior, atmosphere, ctx.spec.priors.occupiability_weights, biosphere)
    values = dict(inputs.values)
    values.update({"history.surface_carbon": float(y[CS]), "history.crust_carbon": float(y[CK]),
                   "history.mantle_carbon": float(y[CM]), "history.nitrogen": float(y[N2]),
                   "history.oxygen": float(y[O2]), "history.mantle_temperature_k": float(y[TM]),
                   "history.core_temperature_k": float(y[TC]), "history.co2_bar": float(air.co2_bar),
                   "history.climate_mode": modes.climate})
    return PlanetState(
        name=ctx.spec.name, seed=ctx.spec.seed, draw_seed=base.draw_seed, attempts=base.attempts,
        constraints_met=True, mode="history", archetype=base.archetype, epoch_s=t_gyr * c.SECONDS_PER_GYR,
        star=star, orbit=orbit, bulk=bulk, interior=interior, atmosphere=atmosphere, occupiability=occupiability,
        inputs=values, provenance=provenance, issues=_deduplicate(issues), water=water, climate=climate,
        biosphere=biosphere, events=_events_until(result, t_gyr), overrides=overrides)


def _fmt(value) -> str:
    """Return a short text form of a value for the report."""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _deduplicate(issues: list[Issue]) -> list[Issue]:
    """Return the issues with exact repeats removed, keeping order."""
    seen, out = set(), []
    for issue in issues:
        key = (issue.level, issue.kind, issue.subsystem, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return out


def sample_series(ctx: HistoryContext, points: int = h.HISTORY_OUTPUT_POINTS) -> dict[str, list]:
    """Return history values sampled for figures, including the times of events."""
    result, model = ctx.result, ctx.model
    t0, t1 = result.t_start, result.t_end
    times = sorted(set([t0 + (t1 - t0) * k / (points - 1) for k in range(points)]
                       + [e.time_gyr for e in result.events if t0 <= e.time_gyr <= t1]))
    names = ("time_gyr", "surface_temperature_k", "pressure_bar", "co2_bar", "o2_fraction", "ch4_fraction",
             "surface_water_oceans", "mantle_water_oceans", "mantle_temperature_k", "core_temperature_k",
             "heat_flux_w_m2", "dynamo", "land_fraction", "open_ocean_fraction", "luminosity_lsun",
             "xuv_fraction", "instellation", "runaway_limit", "melt", "spreading", "weathering", "outgassing",
             "productivity", "water_escape", "climate", "regime", "life", "optical_depth", "albedo")
    series = {n: [] for n in names}
    for t in times:
        seg = result.segment_at(t)
        y = result.state_at(t)
        d = model.diagnostics(t, y, seg.modes)
        row = (t, d.surface_k, d.air.pressure_bar + d.air.steam_bar, d.air.co2_bar, d.air.o2_fraction,
               d.methane_fraction, max(y[WS], 0.0), max(y[WM], 0.0), y[TM], y[TC], d.thermal.heat_flux_w_m2,
               bool(d.thermal.dynamo), d.land, d.climate.open_ocean if d.ocean else 0.0,
               d.luminosity_w / c.L_SUN, d.xuv_fraction, d.instellation, d.runaway_limit, d.thermal.melt,
               d.thermal.spreading, d.carbon.weathering,
               d.carbon.volcanic + d.carbon.arc + d.carbon.recycling, d.productivity, d.water_escape, seg.modes.climate, seg.modes.regime, seg.modes.life, d.optical_depth,
               d.climate.albedo)
        for n, value in zip(names, row):
            series[n].append(value if isinstance(value, (str, bool)) else float(value))
    return series


__all__ = ["HistoryContext", "build_state", "sample_series", "composition_label", "END_OVERRIDES", "GUIDED",
           "MU_CO2"]
