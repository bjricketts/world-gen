"""Archetype presets: parameter distributions for recognisable kinds of world.

Keys are spec field paths (``section.field``). Fields an archetype leaves out
fall back to ``BASE_PRIORS``. Spec values always take precedence.

Every archetype bounds the planet mass so that the weighted mix can exclude
archetypes that contradict a user-fixed mass. ``constraints`` are outcomes a
planet must have to count as the archetype; the generator redraws unset
parameters until they hold (see ``constraints.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import heuristics as h
from .constraints import (
    ICE_SURFACE,
    MOSTLY_LAND,
    MOSTLY_OPEN_OCEAN,
    NOT_LOCKED,
    NOT_RUNAWAY,
    SYNCHRONOUS,
    Constraint,
    dayside_temperature_between,
    surface_temperature_between,
)
from .distributions import Choice, Distribution, Fixed, HZFraction, LogUniform, TruncNormal, Uniform

# Broad defaults used when an archetype does not set a field.
BASE_PRIORS: dict[str, Distribution] = {
    "star.mass_msun": LogUniform(0.3, 1.3),
    "star.age_gyr": Uniform(1.0, 8.0),
    "star.metallicity_feh": TruncNormal(0.0, 0.2, -1.0, 0.5),
    "orbit.instellation_earth": LogUniform(0.3, 2.0),
    "orbit.eccentricity": TruncNormal(0.03, 0.05, 0.0, 0.3),
    "orbit.obliquity_deg": TruncNormal(20.0, 15.0, 0.0, 90.0),
    "orbit.rotation_period_h": LogUniform(10.0, 60.0),
    "orbit.periapsis_longitude_deg": Uniform(0.0, 360.0),
    "body.mass_mearth": LogUniform(0.3, 3.0),
    "body.core_mass_fraction": TruncNormal(0.32, 0.05, 0.1, 0.6),
    "body.water_mass_fraction": LogUniform(1e-4, 2e-3),
    "body.radiogenic_abundance": LogUniform(0.6, 1.7),
    "body.tidal_heating_w": Fixed(0.0),
    "body.tidal_q": LogUniform(*h.TIDAL_Q_ROCKY_RANGE),
    "atmosphere.volatile_richness": LogUniform(0.3, 3.0),
    "atmosphere.weathering_target_k": Uniform(*h.WEATHERING_TARGET_RANGE_K),
    "surface.tectonics": Fixed("simulated"),
    "surface.tectonics_start": Choice(("supercontinent", "cratons")),
    "surface.tectonics_duration_myr": Fixed(h.TECTONIC_DURATION_MYR),
    "star.activity_percentile": Uniform(0.0, 1.0),
    # History mode: initial state and poorly constrained rates (Earth-like values inside each range).
    "history.initial_mantle_temperature_k": Uniform(1650.0, 1850.0),
    "history.outgassing_efficiency": LogUniform(0.5, 2.0),
    "history.weathering_efficiency": LogUniform(0.5, 2.0),
    "history.weathering_temperature_scale_k": Uniform(10.0, 40.0),
    "history.weathering_co2_exponent": Uniform(0.1, 0.5),
    "history.biotic_weathering_factor": LogUniform(2.0, 10.0),
    "history.escape_efficiency": LogUniform(0.05, 0.3),
    "history.mantle_activation_energy_kj": Uniform(250.0, 350.0),
    "history.core_adiabatic_heat_flow": LogUniform(0.5, 2.0),
    "history.life_origin_delay_gyr": LogUniform(0.1, 1.0),
    "history.oxygen_burial_efficiency": LogUniform(0.5, 2.0),
    "history.reductant_decay_gyr": LogUniform(1.5, 6.0),
    "history.land_colonisation_delay_gyr": Uniform(1.5, 4.5),
}


@dataclass(frozen=True)
class Archetype:
    """A named family of worlds."""

    name: str
    description: str
    priors: dict[str, Distribution] = field(default_factory=dict)
    mix_weight: float = 1.0   # weight when no archetype is requested
    constraints: tuple[Constraint, ...] = ()   # outcomes every planet of this type must have


ARCHETYPES: dict[str, Archetype] = {a.name: a for a in [
    Archetype(
        "temperate",
        "Earth-like: temperate instellation, oceans and continents, moderate air pressure.",
        {
            "star.mass_msun": LogUniform(0.6, 1.2),
            "star.age_gyr": Uniform(2.0, 7.0),
            "orbit.instellation_earth": HZFraction(0.02, 0.6),
            "orbit.rotation_period_h": LogUniform(14.0, 40.0),
            "orbit.obliquity_deg": TruncNormal(22.0, 8.0, 5.0, 40.0),
            "body.mass_mearth": LogUniform(0.5, 2.5),
            "body.water_mass_fraction": LogUniform(3.5e-4, 1e-3),
            "atmosphere.surface_pressure_bar": LogUniform(0.6, 2.5),
            "interior.tectonic_regime": Fixed("mobile_lid"),
        },
        mix_weight=0.30,
        constraints=(NOT_LOCKED, MOSTLY_OPEN_OCEAN, NOT_RUNAWAY),
    ),
    Archetype(
        "arid",
        "Desert world: little surface water, cold to hot, thin to moderate air.",
        {
            "orbit.instellation_earth": HZFraction(0.0, 0.4),
            "body.mass_mearth": LogUniform(0.3, 2.0),
            "body.water_mass_fraction": LogUniform(4e-5, 3.5e-4),
            "atmosphere.surface_pressure_bar": LogUniform(0.1, 1.5),
        },
        mix_weight=0.12,
        constraints=(NOT_LOCKED, NOT_RUNAWAY, surface_temperature_between(250.0, 350.0), MOSTLY_LAND),
    ),
    Archetype(
        "ocean",
        "Ocean world: deep global or near-global ocean with few islands.",
        {
            "orbit.instellation_earth": HZFraction(0.0, 0.7),
            "body.mass_mearth": LogUniform(0.8, 5.0),
            "body.water_mass_fraction": LogUniform(2e-3, 5e-2),
            "atmosphere.surface_pressure_bar": LogUniform(0.8, 5.0),
        },
        mix_weight=0.12,
        constraints=(NOT_LOCKED, MOSTLY_OPEN_OCEAN, NOT_RUNAWAY),
    ),
    Archetype(
        "ice",
        "Frozen world: low instellation, ice sheets, possible subsurface oceans.",
        {
            "orbit.instellation_earth": HZFraction(1.0, 2.5),
            "body.mass_mearth": LogUniform(0.1, 4.0),
            "body.water_mass_fraction": LogUniform(4e-4, 1e-3),
            "atmosphere.surface_pressure_bar": LogUniform(0.2, 3.0),
        },
        mix_weight=0.12,
        constraints=(ICE_SURFACE,),
    ),
    Archetype(
        "tidally_locked",
        "Red-dwarf planet with a permanent day side and night side.",
        {
            "star.mass_msun": LogUniform(0.1, 0.5),
            "star.age_gyr": Uniform(2.0, 10.0),
            "orbit.instellation_earth": HZFraction(0.0, 0.8),
            "orbit.eccentricity": TruncNormal(0.01, 0.02, 0.0, 0.08),
            "body.mass_mearth": LogUniform(0.5, 3.0),
            "body.water_mass_fraction": LogUniform(2e-4, 8e-4),
            "atmosphere.surface_pressure_bar": LogUniform(0.5, 5.0),
        },
        mix_weight=0.12,
        constraints=(SYNCHRONOUS, NOT_RUNAWAY, dayside_temperature_between(250.0, 330.0)),
    ),
    Archetype(
        "hot_volcanic",
        "Hot, geologically violent world with lava plains and a thick or stripped atmosphere.",
        {
            "orbit.instellation_earth": LogUniform(1.8, 20.0),
            "body.mass_mearth": LogUniform(0.3, 3.0),
            "body.water_mass_fraction": LogUniform(1e-6, 1e-4),
            "body.radiogenic_abundance": LogUniform(1.5, 4.0),
            "body.tidal_heating_w": LogUniform(1e13, 3e15),   # close-in, eccentric orbits
            "atmosphere.composition": Choice(["co2", "none"], [0.7, 0.3]),
        },
        mix_weight=0.05,
    ),
    Archetype(
        "super_earth",
        "Massive rocky planet: high gravity, thick air, subdued relief.",
        {
            "orbit.instellation_earth": HZFraction(0.0, 0.7),
            "body.mass_mearth": LogUniform(3.0, 10.0),
            "body.envelope_mass_fraction": Fixed(0.0),
            "body.water_mass_fraction": LogUniform(2.5e-4, 1.2e-3),
        },
        mix_weight=0.07,
        constraints=(NOT_LOCKED, NOT_RUNAWAY),
    ),
    Archetype(
        "small_world",
        "Low-gravity world: tall mountains, thin air, Mars-like or smaller.",
        {
            "orbit.instellation_earth": Uniform(0.4, 1.2),
            "body.mass_mearth": LogUniform(0.05, 0.3),
            "body.water_mass_fraction": LogUniform(1e-4, 3e-3),
        },
        mix_weight=0.05,
    ),
    Archetype(
        "giant",
        "Gas or ice giant with no solid surface.",
        {
            "orbit.instellation_earth": LogUniform(0.01, 5.0),
            "body.mass_mearth": LogUniform(12.0, 3000.0),
            "body.envelope_mass_fraction": Uniform(0.1, 0.95),
            "orbit.rotation_period_h": LogUniform(8.0, 20.0),
        },
        mix_weight=0.05,
    ),
]}


def get_archetype(name: str) -> Archetype:
    """Return an archetype by name, with a helpful error for unknown names."""
    try:
        return ARCHETYPES[name]
    except KeyError:
        raise ValueError(f"unknown archetype '{name}'; choose from: {', '.join(ARCHETYPES)}") from None
