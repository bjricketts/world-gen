"""The planet a moon orbits: its properties, the moon's orbit about it, and checks on that orbit."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .. import bulk as bulk_model
from .. import constants as c
from .. import heuristics as h
from ..state import BulkState, Issue, ParentState, StarState
from .appearance import internal_luminosity
from .limits import moon_hill_radius, orbital_period, roche_limit_fluid, stable_moon_limit
from .moon_energy import (damped_eccentricity, dipole_moment, eccentricity_damping_time, eclipse_fraction,
                          ice_shell_dissipation, ice_shell_world, magnetosphere_fraction, parent_heat_flux,
                          reflected_fraction, standoff_distance, tidal_heating)

GIANT_CLASSES = ("ice giant", "gas giant", "brown dwarf")


def _parent_class(mass_mearth: float) -> str:
    """Return the class of a parent planet from its mass alone (envelope assumed above the rocky limit)."""
    envelope = 0.1 if mass_mearth > h.ENVELOPE_ASSUMED_ABOVE_MEARTH else 0.0
    return bulk_model.classify(mass_mearth, envelope, 0.0)


def build_parent(v, star: StarState, moon: BulkState, a_star_m: float, e_star: float,
                 provenance: dict[str, str], name: Optional[str] = None,
                 seed: int = 0) -> tuple[ParentState, list[Issue]]:
    """Return the parent planet and the moon's orbit about it from resolved ``parent.*`` values.

    ``v`` reads resolved inputs; ``a_star_m`` and ``e_star`` are the parent's
    orbit about the star. ``provenance`` gains entries for defaulted values.
    """
    issues: list[Issue] = []
    mass_mearth = v("parent.mass_mearth")
    giant = _parent_class(mass_mearth) in GIANT_CLASSES
    radius_rearth = v("parent.radius_rearth")
    if radius_rearth is None:
        radius_rearth = (bulk_model.chen_kipping_radius(mass_mearth) if giant
                         else bulk_model.rocky_radius(mass_mearth, bulk_model.default_cmf()))
        provenance["parent.radius_rearth"] = "derived"
    rotation_h = v("parent.rotation_period_h")
    if rotation_h is None:
        rotation_h = h.PARENT_ROTATION_GIANT_H if giant else h.PARENT_ROTATION_ROCKY_H
        provenance["parent.rotation_period_h"] = "default"
    defaults = {
        "parent.bond_albedo": h.ALBEDO_GIANT if giant else h.ALBEDO_TEMPERATE,
        "parent.tidal_q": h.TIDAL_Q_GIANT if giant else h.TIDAL_Q_ROCKY,
        "parent.love_number_k2": h.TIDAL_K2_GIANT if giant else h.TIDAL_K2_ROCKY,
        "parent.obliquity_deg": 0.0,
        "parent.eccentricity": 0.0,
        "parent.inclination_deg": 0.0,
    }
    values = {}
    for key, default in defaults.items():
        value = v(key)
        if value is None:
            value = default
            provenance[key] = "default"
        values[key] = value

    mass = mass_mearth * c.M_EARTH
    radius = radius_rearth * c.R_EARTH
    density = mass / (4.0 / 3.0 * math.pi * radius**3)
    a = v("parent.semi_major_axis_rp") * radius
    e = values["parent.eccentricity"]
    parent_hill = moon_hill_radius(a_star_m, e_star, star.mass_kg, mass)
    retrograde = values["parent.inclination_deg"] > 90.0
    stable = stable_moon_limit(parent_hill, retrograde=retrograde)
    roche = roche_limit_fluid(radius, density, moon.density_kg_m3)

    if a * (1.0 - e) <= radius:
        issues.append(Issue("error", "validity", "parent", "the moon's orbit passes inside the parent planet"))
    elif a < roche:
        issues.append(Issue("error", "validity", "parent",
                            f"the moon orbits inside the parent's fluid Roche limit ({roche / radius:.2f} parent radii) "
                            "and would be torn apart"))
    if a > stable_moon_limit(parent_hill, retrograde=True):
        issues.append(Issue("error", "validity", "parent",
                            f"the moon orbits beyond {h.MOON_STABLE_FRACTION_RETROGRADE:.2f} of the parent's Hill radius "
                            "and is not bound"))
    elif a > stable and not retrograde:
        issues.append(Issue("warning", "validity", "parent",
                            f"the moon orbits beyond {h.MOON_STABLE_FRACTION_PROGRADE:.2f} of the parent's Hill radius, "
                            "where only retrograde moons are stable"))

    # what the parent gives the moon: its glow, eclipses, reflected light and magnetic shelter
    t_eff = v("parent.effective_temperature_k")
    if t_eff is None:
        # equilibrium with the star, plus a giant's own heat (T⁴ = T_eq⁴ + T_int⁴, Heller & Zuluaga 2013)
        t_eq4 = star.luminosity_w * (1.0 - values["parent.bond_albedo"]) / (16.0 * math.pi * c.SIGMA_SB * a_star_m**2)
        t_int4 = 0.0
        if giant:
            ice = mass_mearth < h.JOVIAN_MIN_MEARTH_FOR_GLOW
            t_int4 = internal_luminosity(mass, star.age_s, ice, seed) / (4.0 * math.pi * radius**2 * c.SIGMA_SB)
        t_eff = (t_eq4 + t_int4) ** 0.25
        provenance["parent.effective_temperature_k"] = "derived"
    moment = v("parent.magnetic_moment_am2")
    if moment is None:
        moment = dipole_moment(mass, giant)
        provenance["parent.magnetic_moment_am2"] = "heuristic"
    tilt = math.radians(values["parent.obliquity_deg"]) + math.radians(values["parent.inclination_deg"])
    standoff = standoff_distance(moment, a_star_m)

    state = ParentState(
        name=name or "parent",
        mass_kg=mass, radius_m=radius, density_kg_m3=density,
        rotation_period_s=rotation_h * c.SECONDS_PER_HOUR,
        obliquity_rad=math.radians(values["parent.obliquity_deg"]),
        bond_albedo=values["parent.bond_albedo"],
        tidal_q=values["parent.tidal_q"],
        love_number_k2=values["parent.love_number_k2"],
        semi_major_axis_m=a, eccentricity=e,
        inclination_rad=math.radians(values["parent.inclination_deg"]),
        orbital_period_s=orbital_period(a, mass, moon.mass_kg),
        parent_hill_radius_m=parent_hill, stable_limit_m=stable, roche_limit_m=roche,
        effective_temperature_k=t_eff, magnetic_moment_am2=moment,
        star_instellation_earth=star.luminosity_w / (4.0 * math.pi * a_star_m**2) / c.S_EARTH,
        eclipse_fraction=eclipse_fraction(radius, a, tilt),
        reflected_fraction=reflected_fraction(radius, a, values["parent.bond_albedo"]),
        planet_heat_w_m2=parent_heat_flux(radius, a, t_eff),
        magnetopause_standoff_m=standoff, magnetosphere_fraction=magnetosphere_fraction(standoff, a),
        moon_origin=v("parent.moon_origin") or "",
    )
    return state, issues


def apply_parent_energy(parent: ParentState, orbit, bulk: BulkState, love_number_k2: float,
                        user_tidal_w: Optional[float], internal_heat_w: float = 0.0, user_q: bool = False,
                        age_s: float = 0.0, free_eccentricity: bool = False):
    """Return the parent with the moon's tidal heat, and the moon's orbit with planetshine and parent heat.

    The moon's instellation becomes the starlight it receives after eclipses
    plus the starlight the parent reflects; the parent's absorbed glow and the
    tidal heat per square metre become the orbit's extra absorbed flux.

    Dissipation: a frozen water world dissipates in its ice shell
    (``ice_shell_dissipation``). Otherwise, melt feedback (task 11.8): a moon
    whose tides, at the dissipation of a partially molten interior (Io's
    measured k₂/Q = 0.015), would exceed its other internal heat
    (``internal_heat_w``) is taken to be in that state. A spec that fixes Q
    keeps it.

    A free eccentricity (``free_eccentricity``: drawn, not given and not held
    by a resonance) is damped by the moon's tides over ``age_s``.
    """
    phi = love_number_k2 / orbit.tidal_q
    molten = shell = False
    e = parent.eccentricity
    damping = math.inf
    if user_tidal_w is not None:
        tidal = user_tidal_w
    else:
        if not user_q and ice_shell_world(bulk.water_mass_fraction, orbit.instellation_earth, parent.planet_heat_w_m2):
            phi, shell = ice_shell_dissipation(bulk.mass_kg, bulk.radius_m), True
        elif not user_q:
            hot = tidal_heating(parent.mass_kg, bulk.radius_m, parent.semi_major_axis_m, e,
                                h.TIDAL_MOLTEN_K2_OVER_Q, 1.0)
            molten = phi < h.TIDAL_MOLTEN_K2_OVER_Q and hot > internal_heat_w
            phi = h.TIDAL_MOLTEN_K2_OVER_Q if molten else phi
        if free_eccentricity:
            damping = eccentricity_damping_time(parent.mass_kg, bulk.mass_kg, bulk.radius_m,
                                                parent.semi_major_axis_m, phi)
            e = damped_eccentricity(e, age_s, damping)
            if molten and tidal_heating(parent.mass_kg, bulk.radius_m, parent.semi_major_axis_m, e,
                                        h.TIDAL_MOLTEN_K2_OVER_Q, 1.0) <= internal_heat_w:
                molten, phi = False, love_number_k2 / orbit.tidal_q
        tidal = tidal_heating(parent.mass_kg, bulk.radius_m, parent.semi_major_axis_m, e, phi, 1.0)
    parent = replace(parent, eccentricity=e, initial_eccentricity=parent.eccentricity,
                     eccentricity_damping_time_s=damping, ice_shell_dissipation=shell)
    parent = replace(parent, tidal_heating_w=tidal, star_instellation_earth=orbit.instellation_earth,
                     tidal_k2_over_q=phi, partially_molten=molten)
    area = 4.0 * math.pi * bulk.radius_m**2
    orbit = replace(orbit,
                    instellation_earth=orbit.instellation_earth * (parent.eclipse_fraction + parent.reflected_fraction),
                    extra_flux_w_m2=parent.planet_heat_w_m2 + tidal / area)
    return parent, orbit
