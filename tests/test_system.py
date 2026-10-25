"""System summary, parent planets and moon worlds (task 11.1)."""

import math

import pytest
import yaml

from worldgen import constants as c
from worldgen import generate, generate_world, load_world, save_world
from worldgen.giants import (max_surviving_moon_mass, moon_folder, moon_spec, moon_world, roche_limit_rubble,
                             stable_moon_limit, synchronous_orbit_radius)
from worldgen.report import format_report
from worldgen.spec import PlanetSpec

JUPITER = {"name": "Jupiter", "seed": 1, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
           "orbit": {"semi_major_axis_au": 5.2, "eccentricity": 0.048, "obliquity_deg": 3.1,
                     "rotation_period_h": 9.93},
           "body": {"mass_mearth": 317.8, "radius_rearth": 11.21, "envelope_mass_fraction": 0.9}}
GALILEAN = [
    {"name": "Io", "mass_mearth": 0.01496, "radius_rearth": 0.286, "semi_major_axis_rp": 6.03, "eccentricity": 0.0041},
    {"name": "Europa", "mass_mearth": 0.00804, "radius_rearth": 0.245, "water_mass_fraction": 0.08,
     "semi_major_axis_rp": 9.6, "eccentricity": 0.0094},
    {"name": "Ganymede", "mass_mearth": 0.0248, "radius_rearth": 0.413, "water_mass_fraction": 0.45,
     "semi_major_axis_rp": 15.3, "eccentricity": 0.0013},
    {"name": "Callisto", "mass_mearth": 0.018, "radius_rearth": 0.378, "water_mass_fraction": 0.5,
     "semi_major_axis_rp": 26.9, "eccentricity": 0.0074},
]


def jupiter_spec(moons=GALILEAN):
    """Return Jupiter with the given moons listed."""
    return PlanetSpec.model_validate({**JUPITER, "system": {"moons": moons}})


@pytest.mark.parametrize("mass_kg, expected_km", [(1.9e27, 210_000), (5.7e26, 140_000), (8.7e25, 74_000),
                                                   (1.0e26, 78_000)])
def test_icy_roche_limit_matches_sicardy_table(mass_kg, expected_km):
    """The icy ring–moon boundary reproduces Sicardy et al. (2024) Table 1 (γ = 1.6, ρ = 400 kg/m³)."""
    assert roche_limit_rubble(mass_kg, 400.0) / 1e3 == pytest.approx(expected_km, rel=0.02)


def test_max_surviving_moon_mass_hd209458b():
    """Barnes & O'Brien (2002) find no moon above ~7e-7 M⊕ survives around HD 209458b."""
    m = max_surviving_moon_mass(0.69 * c.M_JUPITER, 1.35 * 7.1492e7, 0.0468 * c.AU, 1.1 * c.M_SUN,
                                5.0 * c.SECONDS_PER_GYR, 1e5, 0.51)
    assert m / c.M_EARTH == pytest.approx(7e-7, rel=0.5)


def test_solar_system_prograde_moons_inside_stable_limit():
    """The Moon and Callisto lie inside 0.36 of their planet's Hill radius (Barnes & O'Brien Table 1)."""
    for a_over_rh in (0.257, 0.0354, 0.0187, 0.00837):
        assert a_over_rh < stable_moon_limit(1.0)
    assert stable_moon_limit(1.0, retrograde=True) > stable_moon_limit(1.0)


def test_jupiter_synchronous_orbit():
    """Jupiter's synchronous orbit lies at ~2.24 planet radii (Sasaki et al. 2010)."""
    r = synchronous_orbit_radius(c.M_JUPITER, 9.925 * c.SECONDS_PER_HOUR)
    assert r / 7.1492e7 == pytest.approx(2.24, rel=0.03)


def test_system_lists_user_moons_in_order():
    """User moons appear on the planet's state ordered outward, with periods and the system limits."""
    shuffled = [GALILEAN[2], GALILEAN[0], GALILEAN[3], GALILEAN[1]]
    state, _ = generate(jupiter_spec(shuffled))
    names = [m.name for m in state.system.moons]
    assert names == ["Io", "Europa", "Ganymede", "Callisto"]
    io = state.system.moons[0]
    assert io.orbital_period_s / c.SECONDS_PER_DAY == pytest.approx(1.77, rel=0.05)
    assert state.system.source == "user"
    lim = state.system.limits
    assert lim.roche_icy_m < lim.synchronous_orbit_m * 1.5 < lim.stable_prograde_m
    assert "Moons and rings" in format_report(state)


def test_unnamed_moons_get_numerals_and_planets_without_moons_have_limits():
    """Moons without names are numbered outward; a planet listed with no moons still has its limits."""
    state, _ = generate(jupiter_spec([{"mass_mearth": 0.01, "semi_major_axis_rp": 20},
                                      {"mass_mearth": 0.01, "semi_major_axis_rp": 8}]))
    assert [m.name for m in state.system.moons] == ["Jupiter I", "Jupiter II"]
    bare, _ = generate(jupiter_spec([]))
    assert bare.system.moons == [] and bare.system.source == "none"
    assert bare.system.limits.hill_radius_m == bare.orbit.hill_radius_m


def test_moon_beyond_stable_limit_is_flagged():
    """A prograde moon beyond 0.36 of the Hill radius gets a warning."""
    state, _ = generate(jupiter_spec([{"name": "Far", "mass_mearth": 0.001, "semi_major_axis_rp": 300}]))
    assert any("Far orbits beyond the stable limit" in i.message for i in state.issues)


def test_parent_spec_validation():
    """A moon's missing parent orbit comes from the moon archetype, or is an error; a moon cannot list moons."""
    filled, _ = generate(PlanetSpec.model_validate({"body": {"mass_mearth": 0.2}, "parent": {"mass_mearth": 300}}))
    assert filled.archetype == "habitable_moon" and filled.parent.semi_major_axis_m > 0
    with pytest.raises(ValueError):
        generate(PlanetSpec.model_validate({"body": {"mass_mearth": 0.01}, "parent": {"mass_mearth": 300},
                                            "priors": {"archetype": "temperate"}}))
    with pytest.raises(ValueError):
        PlanetSpec.model_validate({"body": {"mass_mearth": 0.01},
                                   "parent": {"mass_mearth": 300, "semi_major_axis_rp": 10},
                                   "system": {"moons": [{"mass_mearth": 1e-4, "semi_major_axis_rp": 5}]}})


def test_moon_spec_carries_parent_and_star():
    """A moon's spec fixes the star, the parent's orbit and the moon's orbit about the parent."""
    state, _ = generate(jupiter_spec())
    world = type("W", (), {"state": state, "spec": jupiter_spec()})()
    spec = moon_spec(world, 1)
    assert spec.name == "Europa" and spec.parent.name == "Jupiter"
    assert spec.parent.semi_major_axis_rp == pytest.approx(9.6)
    assert spec.orbit.semi_major_axis_au == pytest.approx(5.2)
    assert spec.parent.mass_mearth == pytest.approx(317.8)
    with pytest.raises(IndexError):
        moon_spec(world, 4)


def moon_of_jupiter(a_rp, mass=0.008):
    """Return the spec of a moon of Jupiter at ``a_rp`` parent radii."""
    return PlanetSpec.model_validate({
        "name": "Moon", "seed": 3, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
        "orbit": {"semi_major_axis_au": 5.2}, "body": {"mass_mearth": mass, "radius_rearth": 0.245},
        "parent": {"name": "Jupiter", "mass_mearth": 317.8, "radius_rearth": 11.21, "rotation_period_h": 9.93,
                   "obliquity_deg": 3.1, "semi_major_axis_rp": a_rp},
        "priors": {"enforce_constraints": False}})


def test_moon_locks_to_parent():
    """A Europa-like moon rotates once per orbit about its parent and takes the parent's obliquity."""
    state, _ = generate(moon_of_jupiter(9.6))
    pa = state.parent
    assert state.orbit.spin_state == "synchronous with parent"
    assert state.orbit.rotation_period_s == pytest.approx(pa.orbital_period_s)
    assert pa.orbital_period_s / c.SECONDS_PER_DAY == pytest.approx(3.67, rel=0.03)
    assert math.degrees(state.orbit.obliquity_rad) == pytest.approx(3.1)
    assert state.orbit.hill_radius_m / 1e3 == pytest.approx(13_800, rel=0.05)
    assert state.system is None
    assert "Parent planet" in format_report(state)


def test_moon_orbit_checks():
    """Orbits inside the fluid Roche limit or beyond the Hill sphere are errors."""
    inside, _ = generate(moon_of_jupiter(1.5))
    assert any(i.level == "error" and "Roche" in i.message for i in inside.issues)
    outside, _ = generate(moon_of_jupiter(400))
    assert any(i.level == "error" and "not bound" in i.message for i in outside.issues)


def test_moon_world_round_trip_and_cache(tmp_path):
    """A planet with its system and a moon world built from it save as format 8 and load back."""
    planet = generate_world(jupiter_spec(), "preview")
    save_world(planet, tmp_path / "jupiter")
    loaded = load_world(tmp_path / "jupiter")
    assert loaded.state.system == planet.state.system
    moon = moon_world(loaded, 1, "preview", folder=tmp_path / "jupiter")
    path = moon_folder(tmp_path / "jupiter", loaded, 1)
    assert (path / "state.yaml").exists() and path.name == "01_europa"
    again = moon_world(loaded, 1, "preview", folder=tmp_path / "jupiter")
    assert again.state.parent == moon.state.parent
    assert load_world(path).state.parent.name == "Jupiter"
    with open(path / "state.yaml", encoding="utf-8") as f:
        assert yaml.safe_load(f)["format_version"] == 8


def test_format_6_world_loads_without_a_system(tmp_path):
    """Worlds saved before format 7 load with no system summary."""
    planet = generate_world(PlanetSpec.model_validate(JUPITER), "preview")
    save_world(planet, tmp_path / "old")
    path = tmp_path / "old" / "state.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["format_version"] = 6
    data.pop("system"), data.pop("parent")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    assert load_world(tmp_path / "old").state.system is None


def test_moon_in_history_mode_keeps_parent():
    """History mode carries the parent and the lock to it through to the final state."""
    state, _ = generate(moon_of_jupiter(9.6).model_copy(update={"mode": "history"}))
    assert state.parent is not None and state.orbit.spin_state == "synchronous with parent"
