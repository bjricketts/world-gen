"""A planet's system summary (limits, moons, rings) and turning one of its moons into a world.

The system summary is cheap and lives on the planet's state. A moon becomes a
full world only on request: ``moon_spec`` writes a spec with a ``parent``
section from the planet's state, and ``moon_world`` generates it (DESIGN 4.12).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from .. import bulk as bulk_model
from .. import constants as c
from .. import heuristics as h
from ..state import BulkState, Issue, MoonState, OrbitState, StarState, SystemLimits, SystemState
from ..util import stable_hash
from .moons import draw_moons, name_moons
from .rings import build_rings, ring_regime
from .limits import (max_surviving_moon_mass, orbital_period, roche_limit_rubble, stable_moon_limit,
                     synchronous_orbit_radius)

MOONS_FOLDER = "moons"


def _giant(bulk: BulkState) -> bool:
    """Return whether the planet is a giant without a solid surface."""
    return bulk.planet_class in ("ice giant", "gas giant", "brown dwarf")


def system_limits(star: StarState, orbit: OrbitState, bulk: BulkState) -> SystemLimits:
    """Return where moons and rings can orbit the planet."""
    k2 = h.TIDAL_K2_GIANT if _giant(bulk) else h.TIDAL_K2_ROCKY
    return SystemLimits(
        hill_radius_m=orbit.hill_radius_m,
        stable_prograde_m=stable_moon_limit(orbit.hill_radius_m),
        stable_retrograde_m=stable_moon_limit(orbit.hill_radius_m, retrograde=True),
        roche_icy_m=roche_limit_rubble(bulk.mass_kg, h.RING_DENSITY_ICY),
        roche_rocky_m=roche_limit_rubble(bulk.mass_kg, h.RING_DENSITY_ROCKY),
        synchronous_orbit_m=synchronous_orbit_radius(bulk.mass_kg, orbit.rotation_period_s),
        max_surviving_moon_mass_kg=max_surviving_moon_mass(
            bulk.mass_kg, bulk.radius_m, orbit.semi_major_axis_m, star.mass_kg, star.age_s, orbit.tidal_q, k2),
    )


def _user_moons(entries, planet_name: str, bulk: BulkState, limits: SystemLimits) -> tuple[list[MoonState], list[Issue]]:
    """Return the moons the user listed, ordered outward and named, with notes on orbits outside the limits."""
    issues: list[Issue] = []
    moons = []
    for entry in sorted(entries, key=lambda m: m.semi_major_axis_rp):
        mass = entry.mass_mearth * c.M_EARTH
        radius_rearth = entry.radius_rearth or bulk_model.rocky_radius(entry.mass_mearth, bulk_model.default_cmf(),
                                                                       entry.water_mass_fraction)
        radius = radius_rearth * c.R_EARTH
        a = entry.semi_major_axis_rp * bulk.radius_m
        moons.append(MoonState(
            name=entry.name or "", mass_kg=mass, radius_m=radius,
            density_kg_m3=mass / (4.0 / 3.0 * math.pi * radius**3), semi_major_axis_m=a,
            eccentricity=entry.eccentricity, inclination_rad=math.radians(entry.inclination_deg),
            orbital_period_s=orbital_period(a, bulk.mass_kg, mass), water_mass_fraction=entry.water_mass_fraction,
            origin="user", retrograde=entry.retrograde))
    name_moons(moons, planet_name)
    for moon in moons:
        limit = limits.stable_retrograde_m if moon.retrograde else limits.stable_prograde_m
        if moon.semi_major_axis_m > limit:
            issues.append(Issue("warning", "validity", "system", f"{moon.name} orbits beyond the stable limit "
                                f"({limit / bulk.radius_m:,.0f} planet radii)"))
        if moon.semi_major_axis_m < limits.roche_icy_m:
            issues.append(Issue("info", "note", "system", f"{moon.name} orbits inside the ring–moon boundary of "
                                "porous icy material; it must be dense or strong to hold together"))
        if moon.mass_kg > limits.max_surviving_moon_mass_kg:
            issues.append(Issue("info", "note", "system", f"{moon.name} is heavier than a moon that survives tidal "
                                "migration for the planet's age (Barnes & O'Brien 2002)"))
    return moons, issues


def build_system(spec, star: StarState, orbit: OrbitState, bulk: BulkState) -> tuple[SystemState, list[Issue]]:
    """Return the planet's system summary: limits, the user's or drawn moons, and rings."""
    limits = system_limits(star, orbit, bulk)
    moons, issues, source = [], [], "none"
    formed = "none"
    if spec.system.moons is not None:
        moons, issues = _user_moons(spec.system.moons, spec.name, bulk, limits)
        source = "user" if moons else "none"
    else:
        moons, formed, issues = draw_moons(spec.name, star, orbit, bulk, limits, _rings_for_moons(spec, star, orbit, bulk),
                                           _k2(bulk), spec.seed)
        source = "drawn" if moons else "none"
    rings, regime, ring_issues = build_rings(spec.system.rings, star, orbit, bulk, moons, spec.seed)
    return (SystemState(limits=limits, moons=moons, rings=rings, source=source, ring_regime=regime, moon_regime=formed),
            issues + ring_issues)


def _k2(bulk: BulkState) -> float:
    """Return the planet's tidal Love number k₂."""
    return h.TIDAL_K2_GIANT if _giant(bulk) else h.TIDAL_K2_ROCKY


def _rings_for_moons(spec, star: StarState, orbit: OrbitState, bulk: BulkState) -> str:
    """Return the ring regime that sets the ring-born moons (the listed rings count as broad if dense, else dusty)."""
    rings = spec.system.rings
    if isinstance(rings, list):
        return "broad" if any(r.optical_depth > 0.3 for r in rings) else "dusty warm"
    return ring_regime(star, orbit, bulk, spec.seed) if rings == "auto" else rings


def _slug(name: str) -> str:
    """Return a folder-safe version of a name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "moon"


def moon_spec(world, index: int):
    """Return the spec of moon ``index`` of a planet's world, with the planet as its parent."""
    from ..spec import PlanetSpec

    state = world.state
    if state.system is None or not 0 <= index < len(state.system.moons):
        count = 0 if state.system is None else len(state.system.moons)
        raise IndexError(f"{state.name} has {count} moons; there is no moon {index}")
    moon = state.system.moons[index]
    o, b = state.orbit, state.bulk
    parent = {
        "name": state.name,
        "mass_mearth": b.mass_kg / c.M_EARTH,
        "radius_rearth": b.radius_m / c.R_EARTH,
        "rotation_period_h": o.rotation_period_s / c.SECONDS_PER_HOUR,
        "obliquity_deg": math.degrees(o.obliquity_rad),
        "bond_albedo": state.atmosphere.bond_albedo,
        "tidal_q": o.tidal_q,
        "love_number_k2": h.TIDAL_K2_GIANT if _giant(b) else h.TIDAL_K2_ROCKY,
        "semi_major_axis_rp": moon.semi_major_axis_m / b.radius_m,
        "eccentricity": moon.eccentricity,
        "inclination_deg": math.degrees(moon.inclination_rad),
    }
    origin = {"regular": "regular", "ring": "regular", "impact": "impact", "captured": "captured"}.get(moon.origin)
    if origin:
        parent["moon_origin"] = origin
    if state.appearance is not None:
        parent["effective_temperature_k"] = state.appearance.effective_temperature_k
    if not _giant(b) and state.interior is not None and not state.interior.magnetic_field:
        parent["magnetic_moment_am2"] = 0.0
    data = {
        "name": moon.name,
        "seed": stable_hash(world.spec.seed, "moon", index) % 2**31,
        "mode": world.spec.mode,
        "star": {"mass_msun": state.star.mass_kg / c.M_SUN, "age_gyr": state.star.age_s / c.SECONDS_PER_GYR,
                 "metallicity_feh": state.star.metallicity_feh},
        "orbit": {"semi_major_axis_au": o.semi_major_axis_m / c.AU, "eccentricity": o.eccentricity,
                  "periapsis_longitude_deg": math.degrees(o.periapsis_longitude_rad)},
        "body": {"mass_mearth": moon.mass_kg / c.M_EARTH, "radius_rearth": moon.radius_m / c.R_EARTH,
                 "water_mass_fraction": moon.water_mass_fraction},
        "parent": parent,
        "priors": {"enforce_constraints": False},
    }
    return PlanetSpec.model_validate(data)


def moon_folder(planet_folder: str | Path, world, index: int) -> Path:
    """Return the folder a moon's world is kept in inside its planet's saved world."""
    return Path(planet_folder) / MOONS_FOLDER / f"{index:02d}_{_slug(world.state.system.moons[index].name)}"


def moon_world(world, index: int, resolution: int | str = "standard", folder: Optional[str | Path] = None):
    """Return moon ``index`` of a planet as a world; with ``folder`` (the planet's saved world) it is kept there.

    A kept moon is loaded instead of generated when its spec still matches.
    """
    from ..generate import generate_world
    from ..world import load_world, save_world

    spec = moon_spec(world, index)
    if folder is not None:
        path = moon_folder(folder, world, index)
        if (path / "state.yaml").exists():
            kept = load_world(path)
            if kept.spec == spec:
                return kept
    moon = generate_world(spec, resolution)
    if folder is not None:
        save_world(moon, moon_folder(folder, world, index))
    return moon


def moon_index(world, moon: str | int) -> int:
    """Return the index of a planet's moon given by name (any case) or by number (1 for the innermost)."""
    moons = world.state.system.moons if world.state.system is not None else []
    if isinstance(moon, int) or str(moon).isdigit():
        number = int(moon)
        if not 1 <= number <= len(moons):
            raise IndexError(f"{world.state.name} has {len(moons)} moons; there is no moon {number}")
        return number - 1
    for i, m in enumerate(moons):
        if m.name.lower() == str(moon).lower():
            return i
    names = ", ".join(m.name for m in moons) or "none"
    raise KeyError(f"{world.state.name} has no moon '{moon}' (moons: {names})")


def kept_moons(planet_folder: str | Path, world) -> dict[str, Path]:
    """Return the moons kept as worlds in a planet's saved folder, by name."""
    if world.state.system is None:
        return {}
    out = {}
    for i, m in enumerate(world.state.system.moons):
        path = moon_folder(planet_folder, world, i)
        if (path / "state.yaml").exists():
            out[m.name] = path
    return out


def parent_folder(moon_folder_path: str | Path) -> Optional[Path]:
    """Return the saved planet a kept moon's folder sits in, or None for a standalone moon."""
    path = Path(moon_folder_path).resolve()
    planet = path.parent.parent
    if path.parent.name == MOONS_FOLDER and (planet / "state.yaml").exists():
        return planet
    return None


def rebuilt_parent_spec(moon_state, moon_spec_):
    """Return a spec for a standalone moon's parent, rebuilt from its parent section with the moon listed.

    The parent keeps its mass, radius, rotation, obliquity and the moon's
    star and orbit; its envelope is the default for its mass, and it has no
    rings or other moons (they are not known).
    """
    from ..spec import PlanetSpec

    pa, st = moon_state.parent, moon_state
    moon = {"name": st.name, "mass_mearth": st.bulk.mass_kg / c.M_EARTH, "radius_rearth": st.bulk.radius_m / c.R_EARTH,
            "water_mass_fraction": st.bulk.water_mass_fraction, "semi_major_axis_rp": pa.semi_major_axis_m / pa.radius_m,
            "eccentricity": pa.eccentricity, "inclination_deg": math.degrees(min(pa.inclination_rad,
                                                                                 math.pi - pa.inclination_rad)),
            "retrograde": pa.inclination_rad > math.pi / 2}
    o = st.orbit
    return PlanetSpec.model_validate({
        "name": pa.name or "parent", "seed": moon_spec_.seed,
        "star": {"mass_msun": st.star.mass_kg / c.M_SUN, "age_gyr": st.star.age_s / c.SECONDS_PER_GYR,
                 "metallicity_feh": st.star.metallicity_feh},
        "orbit": {"semi_major_axis_au": o.semi_major_axis_m / c.AU, "eccentricity": o.eccentricity,
                  "obliquity_deg": math.degrees(pa.obliquity_rad), "rotation_period_h": pa.rotation_period_s / 3600.0},
        "body": {"mass_mearth": pa.mass_kg / c.M_EARTH, "radius_rearth": pa.radius_m / c.R_EARTH},
        "system": {"moons": [moon], "rings": "none"},
        "priors": {"enforce_constraints": False},
    })
