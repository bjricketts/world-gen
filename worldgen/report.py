"""Human-readable planet reports and machine-readable state export."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from . import constants as c
from .state import PlanetState

TAGS = {
    "user": "user",
    "user_range": "range",
    "archetype": "archetype",
    "prior": "prior",
    "derived": "derived",
    "heuristic": "heuristic",
    "default": "default",
}
LEVEL_ORDER = {"error": 0, "warning": 1, "info": 2}


def _row(label: str, value: str, tag: str | None = None) -> str:
    """Return one aligned report line with an optional value-source tag."""
    suffix = f"  [{TAGS.get(tag, tag)}]" if tag else ""
    return f"  {label:<28}{value}{suffix}"


def _yes_no(flag: bool) -> str:
    """Return 'yes' or 'no'."""
    return "yes" if flag else "no"


def _water_rows(state: PlanetState) -> list[str]:
    """Return the report lines for the water inventory."""
    w, m = state.water, state.bulk.mass_kg
    if w is None:
        return [_row("inventory", "not computed")]
    derived = state.provenance.get("water.total_mass_fraction")
    rows = []
    if derived:
        rows.append(_row("total water", f"{w.total_mass_fraction:.3g} of planet mass (set by the land fraction)",
                         derived))
    rows += [
        _row("in the mantle", f"{w.mantle_mass_fraction * m / c.EARTH_OCEAN_MASS:.3g} Earth oceans"
             + (" (plate-tectonic water cycle)" if w.plate_cycling else " (no plate recycling)"), "heuristic"
             if not w.plate_cycling else None),
        _row("at the surface", f"{w.surface_mass_kg / c.EARTH_OCEAN_MASS:.3g} Earth oceans "
             f"({w.ocean_volume_m3 / 1e9:.3g} km³)"),
        _row("seafloor pressure", f"{w.seafloor_pressure_ratio:.3g} × Earth"),
        _row("expected land fraction", f"{w.land_fraction_estimate:.2f}", "heuristic"),
    ]
    return rows


def _climate_rows(state: PlanetState) -> list[str]:
    """Return the report lines for the global climate."""
    cl = state.climate
    if cl is None:
        return [_row("model", "not run (no solid surface)")]
    line = "none" if cl.ice_line_deg >= (180.0 if cl.dayside_temperature_k is not None else 90.0) \
        else f"{cl.ice_line_deg:.0f}°" + (" from the substellar point" if cl.dayside_temperature_k is not None else "")
    rows = [
        _row("mean temperature", f"{cl.mean_temperature_k:.0f} K (zonal energy balance)", "heuristic"),
        _row("planetary albedo", f"{cl.planetary_albedo:.2f} (ice-free {cl.ice_free_albedo:.2f})"),
        _row("open water", f"{cl.open_water_fraction:.0%} of the surface, {cl.open_ocean_fraction:.0%} of the ocean"),
        _row("permanent sea ice from", line),
    ]
    if cl.dayside_temperature_k is not None:
        rows.append(_row("day / night side", f"{cl.dayside_temperature_k:.0f} K / {cl.nightside_temperature_k:.0f} K"))
    if cl.snowball_stable:
        rows.append(_row("snowball state", "also stable"))
    return rows


def _biosphere_rows(state: PlanetState) -> list[str]:
    """Return the report lines for life."""
    life = state.biosphere
    p = state.provenance
    if life is None:
        return [_row("life", "none", p.get("biosphere.life"))]
    rows = [
        _row("life", f"{life.life} ({life.biochemistry}, releases {life.product})", p.get("biosphere.life")),
        _row("biosphere age", f"{life.age_gyr:.2f} Gyr", p.get("biosphere.age_gyr")),
        _row("O₂ / CH₄", f"{life.oxygen_fraction:.1%} / {life.methane_fraction * 1e6:.0f} ppm",
             p.get("biosphere.oxygen_fraction")),
    ]
    if life.life == "surface":
        rows += [
            _row("growth optimum", f"{life.optimum_temperature_k:.0f} K, tolerance +{life.temperature_tolerance_k:.0f} K"),
            _row("pigment", f"absorbs at {life.pigment_absorption_nm:.0f} nm, looks {life.pigment_colour}",
                 p.get("biosphere.pigment_absorption_nm")),
            _row("expected productivity", f"{life.productivity:.2f} × Earth", "heuristic"),
        ]
    if life.weathering_cooling_k:
        rows.append(_row("weathering by biota", f"lowers the weathering target by {life.weathering_cooling_k:.0f} K",
                         "heuristic"))
    return rows


def format_report(state: PlanetState) -> str:
    """Return a multi-line text report of the planet with value sources and consistency notes."""
    p = state.provenance
    s, o, b, i, a, occ = state.star, state.orbit, state.bulk, state.interior, state.atmosphere, state.occupiability
    hz = o.habitable_zone
    lines = [
        f"=== {state.name} ===",
        f"mode: {state.mode}   seed: {state.seed}   archetype: {state.archetype or '-'}"
        + (f"   (accepted draw {state.attempts}, draw seed {state.draw_seed})" if state.attempts > 1 else ""),
        "",
        "Star",
        _row("mass", f"{s.mass_kg / c.M_SUN:.3f} M☉", p.get("star.mass_msun")),
        _row("age", f"{s.age_s / c.SECONDS_PER_GYR:.2f} Gyr "
             f"(MS lifetime {s.main_sequence_lifetime_s / c.SECONDS_PER_GYR:.1f} Gyr)", p.get("star.age_gyr")),
        _row("metallicity [Fe/H]", f"{s.metallicity_feh:+.2f}", p.get("star.metallicity_feh")),
        _row("spectral type", s.spectral_type),
        _row("luminosity", f"{s.luminosity_w / c.L_SUN:.3g} L☉"),
        _row("radius", f"{s.radius_m / c.R_SUN:.3f} R☉"),
        _row("effective temperature", f"{s.effective_temperature_k:.0f} K"),
        _row("XUV activity", f"{s.xuv_fraction_relative_sun:.3g} × Sun (fractional)", "heuristic"),
        "",
        "Orbit and spin",
        _row("semi-major axis", f"{o.semi_major_axis_m / c.AU:.4g} AU", p.get("orbit.semi_major_axis_au")),
        _row("instellation", f"{o.instellation_earth:.3g} × Earth", p.get("orbit.instellation_earth")),
        _row("eccentricity", f"{o.eccentricity:.3f}", p.get("orbit.eccentricity")),
        _row("obliquity", f"{math.degrees(o.obliquity_rad):.1f}°", p.get("orbit.obliquity_deg")),
        _row("orbital period", f"{o.orbital_period_s / c.SECONDS_PER_DAY:.4g} days"),
        _row("rotation period", f"{o.rotation_period_s / c.SECONDS_PER_HOUR:.4g} h", p.get("orbit.rotation_period_h")),
        _row("spin state", o.spin_state),
        _row("tidal Q", f"{o.tidal_q:.3g}", p.get("body.tidal_q")),
        _row("tidal locking time", f"{o.tidal_lock_time_s / c.SECONDS_PER_GYR:.3g} Gyr", "heuristic"),
        _row("Hill radius", f"{o.hill_radius_m / 1e3:,.0f} km"),
        _row("habitable zone", hz.position),
        _row("  position", f"{hz.conservative_fraction:.2f} across conservative zone (0 inner, 1 outer)"),
        _row("  conservative", f"{hz.runaway_greenhouse_m / c.AU:.3f}–{hz.maximum_greenhouse_m / c.AU:.3f} AU"),
        _row("  optimistic", f"{hz.recent_venus_m / c.AU:.3f}–{hz.early_mars_m / c.AU:.3f} AU"),
        "",
        "Body",
        _row("class", b.planet_class),
        _row("mass", f"{b.mass_kg / c.M_EARTH:.4g} M⊕", p.get("body.mass_mearth")),
        _row("radius", f"{b.radius_m / c.R_EARTH:.4g} R⊕", p.get("body.radius_rearth")),
        _row("core mass fraction", f"{b.core_mass_fraction:.3f}", p.get("body.core_mass_fraction")),
        _row("water mass fraction", f"{b.water_mass_fraction:.3g} "
             f"(~{b.water_mass_fraction * b.mass_kg / c.EARTH_OCEAN_MASS:.3g} Earth oceans, total)",
             p.get("body.water_mass_fraction")),
        _row("envelope mass fraction", f"{b.envelope_mass_fraction:.3g}", p.get("body.envelope_mass_fraction")),
        _row("density", f"{b.density_kg_m3:,.0f} kg/m³"),
        _row("surface gravity", f"{b.surface_gravity_m_s2:.2f} m/s² ({b.surface_gravity_m_s2 / c.G_EARTH:.2f} g)"),
        _row("escape velocity", f"{b.escape_velocity_m_s / 1e3:.2f} km/s"),
        "",
        "Interior",
        _row("tectonic regime", i.tectonic_regime, p.get("interior.tectonic_regime")),
        _row("  probabilities", ", ".join(f"{k} {v:.2f}" for k, v in i.regime_probabilities.items())),
        _row("activity index", f"{i.activity_index:.3g} (Earth = 1)", "heuristic"),
        _row("surface heat flux", f"{i.surface_heat_flux_w_m2 * 1e3:.1f} mW/m²"),
        _row("radiogenic power", f"{i.radiogenic_power_w / 1e12:.3g} TW", p.get("body.radiogenic_abundance")),
        _row("tidal heating", f"{i.tidal_power_w / 1e12:.3g} TW", p.get("body.tidal_heating_w")),
        _row("magnetic field", f"{_yes_no(i.magnetic_field)} (p = {i.dynamo_probability:.2f})",
             p.get("interior.magnetic_field")),
        "",
        "Water",
        *_water_rows(state),
        "",
        "Atmosphere and surface",
        _row("atmosphere", f"{_yes_no(a.present)} (retention p = {a.retention_probability:.2f}, "
             f"shoreline ratio {a.shoreline_ratio:.3g})", p.get("atmosphere.present")),
        _row("surface pressure", f"{a.surface_pressure_pa / c.BAR:.3g} bar", p.get("atmosphere.surface_pressure_bar")),
        _row("composition", a.composition, p.get("atmosphere.composition")),
        _row("Bond albedo", f"{a.bond_albedo:.2f}", p.get("atmosphere.bond_albedo")),
        _row("equilibrium temperature", f"{a.equilibrium_temperature_k:.0f} K"),
        _row("greenhouse optical depth", f"{a.greenhouse_optical_depth:.3g}", "heuristic"),
        _row("surface temperature", f"{a.surface_temperature_k:.0f} K ({a.surface_temperature_k - 273.15:.0f} °C)",
             p.get("atmosphere.surface_temperature_k")),
        _row("scale height", f"{a.scale_height_m / 1e3:.1f} km"),
        _row("water boiling point", f"{a.water_boiling_point_k:.0f} K"),
        _row("surface water", a.surface_water),
        _row("CO₂ regulated by weathering", _yes_no(a.weathering_regulated)
             + (f" (target {a.weathering_target_k:.0f} K)" if a.weathering_regulated else ""),
             p.get("atmosphere.weathering_target_k") if a.weathering_regulated else None),
        _row("runaway greenhouse", _yes_no(a.runaway_greenhouse)),
        "",
        "Climate",
        *_climate_rows(state),
        "",
        "Biosphere",
        *_biosphere_rows(state),
        "",
        "Occupiability",
        _row("score", f"{occ.score:.2f} (0–1)"),
        _row("habitable unaided", _yes_no(occ.habitable_unaided)),
        _row("factors", ", ".join(f"{k} {v:.2f}" for k, v in occ.factors.items())),
    ]

    if state.surface is not None:
        sf = state.surface
        lines += [
            "",
            "Surface",
            _row("grid", f"{sf.grid_size:,} cells (~{sf.spacing_km:.0f} km spacing)"),
            _row("land fraction", f"{sf.land_fraction:.2f}"
                 + (" (frozen ocean)" if sf.frozen_ocean else "" if sf.has_ocean else " (no ocean)"),
                 p.get("surface.land_fraction")),
            _row("elevation range", f"{sf.min_elevation_m / 1e3:.1f} to {sf.max_elevation_m / 1e3:.1f} km"),
            _row("relief factor", f"{sf.relief_factor:.2f} (Earth = 1)", "heuristic"),
        ]
        features = dict(sf.features)
        if "simulated_myr" in features:
            start = state.inputs.get("surface.tectonics_start")
            lines += [
                _row("tectonics", f"simulated {features.pop('simulated_myr'):.0f} Myr from {start}",
                     p.get("surface.tectonics_start")),
                _row("tectonic events", f"{features.pop('rifts')} rifts, {features.pop('plate_merges')} plate merges, "
                     f"{features.pop('subduction_initiations')} new subduction zones, "
                     f"{features.pop('subducted_points'):,} crust cells subducted"),
                _row("median sea-floor age", f"{features.pop('median_ocean_age_myr'):.0f} Myr"),
            ]
        if "river_outflow_km3_yr" in features:
            lines += [
                _row("rivers and lakes", f"largest river {features.pop('largest_river_m3_s'):,.0f} m³/s, "
                     f"largest basin {features.pop('largest_basin_km2') / 1e6:.2g} million km², "
                     f"total outflow {features.pop('river_outflow_km3_yr'):,.0f} km³/yr", "heuristic"),
                _row("lakes, closed basins", f"{features.pop('lake_fraction_of_land'):.1%} of land under lakes, "
                     f"{features.pop('endorheic_fraction_of_land'):.0%} draining inland"),
            ]
        if "climate_mean_temperature_k" in features:
            lines.append(_row("surface climate", f"{features.pop('climate_mean_temperature_k'):.0f} K mean, "
                              f"albedo {features.pop('climate_planetary_albedo'):.2f}, "
                              f"greenhouse optical depth {features.pop('climate_optical_depth'):.2f}, "
                              f"{features.pop('climate_coupling_passes')} passes with ice and vegetation",
                              "heuristic"))
        if "ocean_heat_transport_35n_pw" in features:
            lines.append(_row("heat transport at 35° N",
                              f"atmosphere {features.pop('atmosphere_heat_transport_35n_pw'):.1f} PW, "
                              f"ocean {features.pop('ocean_heat_transport_35n_pw'):.1f} PW", "heuristic"))
        if "ice_sheet_fraction_of_land" in features:
            lines.append(_row("ice sheets", f"{features.pop('ice_sheet_fraction_of_land'):.1%} of land, "
                              f"up to {features.pop('max_ice_thickness_m'):,.0f} m thick"))
        if "ice_sheet_volume_km3" in features:
            lines.append(_row("water in ice", f"{features.pop('ice_sheet_volume_km3') / 1e6:.3g} million km³, "
                              f"{features.pop('ice_sea_level_equivalent_m'):,.0f} m of sea level; "
                              f"sea level lowered by {features.pop('glacial_sea_level_drop_m'):,.0f} m"))
        lines.append(_row("features", ", ".join(f"{k.replace('_', ' ')} {v:.2f}" if isinstance(v, float)
                                                 else f"{k.replace('_', ' ')} {v}" for k, v in features.items())))

    problems = sorted((x for x in state.issues if x.level != "info" or x.kind == "note"),
                      key=lambda x: LEVEL_ORDER[x.level])
    heuristics = [x for x in state.issues if x.kind == "heuristic"]
    lines += ["", "Consistency report"]
    if problems:
        lines += [f"  {x.level.upper():<8}[{x.subsystem}] {x.message}" for x in problems]
    else:
        lines.append("  no conflicts or validity problems")
    if heuristics:
        lines += ["", "Heuristic relations used"]
        lines += [f"  - [{x.subsystem}] {x.message}" for x in heuristics]
    lines += ["", "Value sources: user, range (drawn from user range), archetype, prior, "
                  "derived (physics), heuristic, default"]
    return "\n".join(lines)


def state_to_plain(state: PlanetState) -> dict:
    """Return the planet state as plain Python types suitable for YAML or JSON."""
    return _plain(state.to_dict())


def _plain(obj):
    """Return a copy of nested data using only built-in Python types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, float):
        return float(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def save_state(state: PlanetState, path: str | Path) -> None:
    """Write the full planet state (SI units) to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(state_to_plain(state), f, sort_keys=False, allow_unicode=True)
