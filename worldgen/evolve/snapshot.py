"""Snapshot mode: evaluate the planet's present-day state from closed-form relations."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .. import atmosphere as atm
from .. import bulk as bulk_model
from .. import interior as interior_model
from .. import orbit as orbit_model
from .. import star as star_model
from .. import water as water_model
from ..constants import AU, SECONDS_PER_GYR, SECONDS_PER_HOUR
from .. import heuristics as h
from ..giants import build_appearance, build_parent, build_system, is_giant
from ..giants.parent import apply_parent_energy
from ..biosphere.life import LifeInputs, build_biosphere, methane_optical_depth, tier0_land_albedo
from ..heuristics import WEATHERING_TARGET_K
from ..priors import draw_until_satisfied, resolve, score_planet
from ..priors.archetypes import get_archetype
from ..spec import PlanetSpec
from ..state import Issue, PlanetState, Timeline
from ..surface import relief_factor
from ..util import named_rng


class SnapshotEvolver:
    """Builds a planet at a single epoch; user values are used as given."""

    def evolve(self, spec: PlanetSpec, t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
        """Return the planet state at the target epoch and a one-point timeline."""
        state = draw_until_satisfied(spec, lambda seed: self.build(spec, seed, t_target_gyr))
        timeline = Timeline(times_s=[state.epoch_s], states=[state], events=[])
        return state, timeline

    def build(self, spec: PlanetSpec, seed: int, t_target_gyr: Optional[float] = None) -> PlanetState:
        """Return one planet drawn from the spec with ``seed`` driving every random choice."""
        inputs = resolve(spec, seed)
        v = inputs.get
        provenance = dict(inputs.provenance)
        issues: list[Issue] = [Issue("info", "note", "priors", note) for note in inputs.notes]

        def merge(new_issues: list[Issue], new_prov: dict[str, str]) -> None:
            """Record notes and provenance from a module, keeping user provenance."""
            issues.extend(new_issues)
            for key, source in new_prov.items():
                if not inputs.is_user(key):
                    provenance[key] = source

        age = t_target_gyr if t_target_gyr is not None else spec.target_epoch_gyr
        if age is None:
            age = v("star.age_gyr")
        else:
            provenance["star.age_gyr"] = "user"

        star, new = star_model.build_star(v("star.mass_msun"), age, v("star.metallicity_feh"))
        merge(new, {})

        bulk, new, prov = bulk_model.build_bulk(
            mass_mearth=v("body.mass_mearth"),
            radius_rearth=v("body.radius_rearth"),
            cmf=v("body.core_mass_fraction"),
            wmf=v("body.water_mass_fraction", 0.0),
            envelope=v("body.envelope_mass_fraction"),
            wmf_is_user=inputs.is_user("body.water_mass_fraction"),
        )
        merge(new, prov)

        a_au = v("orbit.semi_major_axis_au")
        rotation_h = v("orbit.rotation_period_h")
        parent = None
        moon_archetype = inputs.archetype is not None and get_archetype(inputs.archetype).moon
        if spec.parent is not None or moon_archetype:
            if v("parent.mass_mearth") is None or v("parent.semi_major_axis_rp") is None:
                raise ValueError("a moon's parent needs mass_mearth and semi_major_axis_rp "
                                 "(or leave them to a moon archetype such as habitable_moon)")
            a_star = (a_au * AU if a_au is not None else orbit_model.semi_major_axis_for_instellation(
                star.luminosity_w, v("orbit.instellation_earth"), v("orbit.eccentricity", 0.0)))
            parent_prov: dict[str, str] = {}
            parent, new = build_parent(v, star, bulk, a_star, v("orbit.eccentricity", 0.0), parent_prov,
                                       spec.parent.name if spec.parent is not None else None, seed)
            merge(new, parent_prov)
        orbit, new, prov = orbit_model.build_orbit(
            star=star,
            bulk=bulk,
            semi_major_axis_m=None if a_au is None else a_au * AU,
            instellation_earth=v("orbit.instellation_earth"),
            eccentricity=v("orbit.eccentricity", 0.0),
            obliquity_rad=math.radians(v("orbit.obliquity_deg", 0.0)),
            rotation_period_s=None if rotation_h is None else rotation_h * SECONDS_PER_HOUR,
            rotation_is_user=inputs.is_user("orbit.rotation_period_h"),
            obliquity_is_user=inputs.is_user("orbit.obliquity_deg"),
            tidal_q=_tidal_q(inputs, bulk),
            periapsis_longitude_rad=math.radians(v("orbit.periapsis_longitude_deg", 0.0)),
            parent=parent,
        )
        merge(new, prov)
        tidal_w = v("body.tidal_heating_w") if inputs.is_user("body.tidal_heating_w") else None
        if parent is not None:
            internal = (interior_model.silicate_mass(bulk) * h.INVERSE_UREY_RATIO
                        * interior_model.radiogenic_heat_per_kg(age, v("body.radiogenic_abundance", 1.0)))
            free = not inputs.is_user("parent.eccentricity")     # drawn moons have no resonance partner
            parent, orbit = apply_parent_energy(parent, orbit, bulk, h.TIDAL_K2_ROCKY, tidal_w, internal,
                                                inputs.is_user("body.tidal_q"), age * SECONDS_PER_GYR, free)
            tidal_w = parent.tidal_heating_w
            merge([], {"body.tidal_heating_w": "heuristic"})          # raised by the parent, not drawn
        else:
            tidal_w = v("body.tidal_heating_w", 0.0)
            orbit = replace(orbit, extra_flux_w_m2=tidal_w / (4.0 * math.pi * bulk.radius_m**2))
        system = None
        if parent is None:
            system, new = build_system(spec, star, orbit, bulk)
            merge(new, {})

        atm_inputs = atm.AtmosphereInputs(
            present=v("atmosphere.present"),
            surface_pressure_bar=v("atmosphere.surface_pressure_bar"),
            composition=v("atmosphere.composition"),
            bond_albedo=v("atmosphere.bond_albedo"),
            surface_temperature_k=v("atmosphere.surface_temperature_k"),
            volatile_richness=v("atmosphere.volatile_richness", 1.0),
            weathering_target_k=v("atmosphere.weathering_target_k", WEATHERING_TARGET_K),
            pressure_is_user=inputs.is_user("atmosphere.surface_pressure_bar"),
        )
        # Preliminary surface conditions decide the tectonic regime,
        # which in turn sets composition defaults and weathering.
        relief = relief_factor(bulk.surface_gravity_m_s2)
        preliminary_water = water_model.build_water(bulk, "mobile_lid", relief)
        preliminary, *_ = atm.build_atmosphere(star, orbit, bulk, atm_inputs, regime=None,
                                               water=preliminary_water, seed=seed)

        interior, new, prov = interior_model.build_interior(
            bulk=bulk,
            age_gyr=age,
            radiogenic_abundance=v("body.radiogenic_abundance", 1.0),
            tidal_power_w=tidal_w,
            surface_temperature_k=preliminary.surface_temperature_k,
            surface_water=preliminary.surface_water in ("liquid", "ice"),
            regime_override=v("interior.tectonic_regime"),
            dynamo_override=v("interior.magnetic_field"),
            regime_rng=named_rng(seed, "interior.tectonic_regime"),
            dynamo_rng=named_rng(seed, "interior.magnetic_field"),
        )
        merge(new, prov)

        water = water_model.build_water(bulk, interior.tectonic_regime, relief)
        atmosphere, climate, zonal, new, prov = atm.build_atmosphere(
            star, orbit, bulk, atm_inputs, regime=interior.tectonic_regime, water=water, seed=seed)
        life_inputs = LifeInputs(
            life=v("biosphere.life"), biochemistry=v("biosphere.biochemistry"),
            alien_product=v("biosphere.alien_product"), age_gyr=v("biosphere.age_gyr"),
            optimum_temperature_k=v("biosphere.optimum_temperature_k"),
            temperature_tolerance_k=v("biosphere.temperature_tolerance_k"),
            pigment_absorption_nm=v("biosphere.pigment_absorption_nm"),
            oxygen_fraction=v("biosphere.oxygen_fraction"),
            weathering_target_is_user=inputs.is_user("atmosphere.weathering_target_k"),
        )
        biosphere, bio_new, bio_prov = build_biosphere(life_inputs, star, orbit.instellation_earth, atmosphere,
                                                       water, zonal, age)
        if biosphere is not None:
            # Life changes the air: rebuild the atmosphere and climate with its effects.
            atm_inputs.weathering_target_k -= biosphere.weathering_cooling_k
            atm_inputs.extra_optical_depth = methane_optical_depth(biosphere.methane_fraction)
            atm_inputs.land_albedo = tier0_land_albedo(biosphere)
            if (atm_inputs.composition is None and biosphere.oxygen_fraction > h.OXYGEN_COMPOSITION_ABOVE
                    and atmosphere.composition == "n2_co2"):
                atm_inputs.composition = "n2_o2"
                provenance["atmosphere.composition"] = "derived"
            atmosphere, climate, zonal, new, prov = atm.build_atmosphere(
                star, orbit, bulk, atm_inputs, regime=interior.tectonic_regime, water=water, seed=seed)
            if biosphere.life == "surface" and atmosphere.surface_water != "liquid":
                bio_new.append(Issue("info", "note", "biosphere",
                                     f"the surface water is '{atmosphere.surface_water}' once life has changed the air"))
        merge(new, prov)
        merge(bio_new, bio_prov)

        appearance = None
        if is_giant(bulk):
            appearance = build_appearance(star, orbit, bulk, atmosphere, seed,
                                          inputs.get("body.atmospheric_metallicity"))
            merge([], {"body.atmospheric_metallicity": "heuristic"})
        ice_shell = None
        if not is_giant(bulk) and atmosphere.surface_water == "ice" and water.surface_mass_kg > 0:
            ice_shell = water_model.build_ice_shell(bulk, water, atmosphere.surface_temperature_k,
                                                    interior.radiogenic_power_w + interior.tidal_power_w)

        occupiability = score_planet(star, bulk, interior, atmosphere, spec.priors.occupiability_weights,
                                     biosphere)

        return PlanetState(
            name=spec.name,
            seed=spec.seed,
            draw_seed=seed,
            attempts=1,
            constraints_met=True,
            mode="snapshot",
            archetype=inputs.archetype,
            epoch_s=age * SECONDS_PER_GYR,
            star=star,
            orbit=orbit,
            bulk=bulk,
            interior=interior,
            atmosphere=atmosphere,
            occupiability=occupiability,
            water=water,
            climate=climate,
            biosphere=biosphere,
            system=system,
            parent=parent,
            appearance=appearance,
            ice_shell=ice_shell,
            inputs=dict(inputs.values),
            provenance=provenance,
            issues=_deduplicate(issues),
        )


def _tidal_q(inputs, bulk) -> Optional[float]:
    """Return the tidal Q to use: the user's value, the drawn value for rocky planets, or None for the default."""
    if inputs.is_user("body.tidal_q"):
        return inputs.get("body.tidal_q")
    if bulk.planet_class in ("ice giant", "gas giant", "brown dwarf", "sub-Neptune"):
        return None
    return inputs.get("body.tidal_q")


def _deduplicate(issues: list[Issue]) -> list[Issue]:
    """Return the issues with exact repeats removed, keeping order."""
    seen = set()
    out = []
    for issue in issues:
        key = (issue.level, issue.kind, issue.subsystem, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return out
