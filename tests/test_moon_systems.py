"""Drawn moon systems: Jovian and Saturnian architectures, ring-born chains, impact moons and limits (task 11.5)."""

import math

import numpy as np
import pytest

from worldgen import constants as c
from worldgen import generate, heuristics as h
from worldgen.giants.moons import (chain_delta, chain_shape, gap_criterion, impact_disc_fraction, impact_moon,
                                   jovian_moons, moon_density, saturnian_moons, tidal_reach)
from worldgen.report import format_report
from worldgen.spec import PlanetSpec
from worldgen.state import PlanetState

GIANTS = {
    "Jupiter": (5.2, 317.8, 11.21, 0.9, 9.93),
    "Saturn": (9.58, 95.2, 9.45, 0.85, 10.56),
    "Uranus": (19.2, 14.5, 4.01, 0.15, 17.24),
    "Neptune": (30.1, 17.1, 3.88, 0.15, 16.11),
}


def giant(name, seed=1, a=None, system=None):
    """Return the state of a Solar System giant analogue with drawn moons."""
    a_au, mass, radius, envelope, rotation = GIANTS[name]
    spec = PlanetSpec.model_validate({
        "name": name, "seed": seed, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
        "orbit": {"semi_major_axis_au": a if a is not None else a_au, "rotation_period_h": rotation},
        "body": {"mass_mearth": mass, "radius_rearth": radius, "envelope_mass_fraction": envelope},
        "system": system or {}})
    return generate(spec)[0]


def rocky(seed, star_mass=1.0, a_au=1.0):
    """Return the state of an Earth-mass planet with a drawn moon, if any."""
    spec = PlanetSpec.model_validate({"name": "Terra", "seed": seed, "star": {"mass_msun": star_mass},
                                      "orbit": {"semi_major_axis_au": a_au},
                                      "body": {"mass_mearth": 1.0, "radius_rearth": 1.0}})
    return generate(spec)[0]


def test_gap_criterion_separates_jupiter_and_saturn():
    """Jupiter opens a gap in the nebula and Saturn does not (Crida et al. 2006, as Sasaki et al. assume)."""
    jupiter, saturn = giant("Jupiter"), giant("Saturn")
    assert gap_criterion(jupiter.star, jupiter.orbit, jupiter.bulk) == pytest.approx(0.68, abs=0.03)
    assert gap_criterion(saturn.star, saturn.orbit, saturn.bulk) == pytest.approx(1.55, abs=0.05)
    assert jupiter.system.moon_regime == "jovian" and saturn.system.moon_regime == "saturnian"
    assert giant("Uranus", seed=4).system.moon_regime == "rings only"     # no impact system in this draw
    assert giant("Uranus", seed=1).system.moon_regime == "uranian"


def test_jovian_architecture_statistics():
    """4–5 large moons in ~80% of draws, the inner three in 2:1 resonances, ~2e-4 of the planet's mass."""
    bulk = giant("Jupiter").bulk
    counts, totals = [], []
    for s in range(300):
        moons = jovian_moons(bulk, True, np.random.default_rng(s))
        counts.append(len(moons))
        totals.append(sum(m.mass_kg for m in moons) / bulk.mass_kg)
        periods = [m.a_m**1.5 for m in moons]
        assert periods[1] / periods[0] == pytest.approx(2.0) and periods[2] / periods[1] == pytest.approx(2.0)
        assert moons[0].water <= 0.1 and (len(moons) < 4 or moons[-1].water >= 0.4)
    share = np.mean([n in (4, 5) for n in counts])
    assert share == pytest.approx(0.80, abs=0.07)
    assert 1e-4 < np.median(totals) < 4e-4


def test_saturnian_architecture_statistics():
    """One dominant icy moon about Titan's mass and place in about two thirds of draws."""
    bulk = giant("Saturn").bulk
    single, largest, orbit = [], [], []
    for s in range(300):
        moons = saturnian_moons(bulk, True, np.random.default_rng(s))
        single.append(len(moons) == 1)
        top = max(moons, key=lambda m: m.mass_kg)
        largest.append(top.mass_kg / bulk.mass_kg)
        orbit.append(top.a_m / bulk.radius_m)
        assert all(m.water >= 0.4 for m in moons)
    assert np.mean(single) == pytest.approx(0.67, abs=0.07)
    assert np.median(largest) == pytest.approx(2.4e-4, rel=0.2)
    assert np.median(orbit) == pytest.approx(20.3, rel=0.1)


def test_no_ice_inside_the_snow_line():
    """A gas giant at 1 AU forms only rocky moons: its nebula was above 160 K."""
    warm = giant("Jupiter", a=1.0)
    regular = [m for m in warm.system.moons if m.origin == "regular"]
    assert regular and all(m.water_mass_fraction <= h.ROCKY_MOON_WATER_FRACTION[1] for m in regular)


def test_chain_shape_is_inverted_and_halves_inward():
    """Q(Δ) and its inverse agree; each ring-born moon is about half the next one out."""
    for delta in (0.03, 0.3, 0.587, 1.2, 6.0):
        assert chain_delta(chain_shape(delta)) == pytest.approx(delta)
    uranus = giant("Uranus", seed=9)          # no impact system and no capture in this draw
    chain = [m for m in uranus.system.moons if m.origin == "ring"]
    assert len(chain) >= 8
    masses = np.array([m.mass_kg for m in chain])
    ratios = masses[1:] / masses[:-1]
    assert np.median(ratios) == pytest.approx(2.0, rel=0.3)
    total = masses.sum() / uranus.bulk.mass_kg
    assert 1e-5 < total < 1e-3    # Uranus's regular moons hold 1.05e-4 of the planet


def test_saturn_ring_born_moons_reach_by_tides():
    """Saturn's ring-born moons run from the Roche limit to the tidal reach of the outermost, inside the dominant moon."""
    saturn = giant("Saturn")
    chain = [m for m in saturn.system.moons if m.origin == "ring"]
    regular = [m for m in saturn.system.moons if m.origin == "regular"]
    assert chain and regular
    assert chain[0].semi_major_axis_m > saturn.system.limits.roche_icy_m
    assert chain[-1].semi_major_axis_m < regular[0].semi_major_axis_m
    outer = chain[-1]
    reach = tidal_reach(outer.mass_kg / saturn.bulk.mass_kg, saturn.bulk, saturn.orbit.tidal_q, h.TIDAL_K2_GIANT,
                        saturn.star.age_s)
    assert outer.semi_major_axis_m == pytest.approx(max(reach, saturn.system.limits.roche_icy_m * 1.587), rel=0.5)


def test_impact_disc_matches_the_moon_forming_impact():
    """Canup's γ = 0.11, b = 0.7 impact puts about a Moon's mass into orbit around the Earth."""
    disc = impact_disc_fraction(0.11, 0.7) * c.M_EARTH
    assert disc == pytest.approx(7.35e22, rel=0.15)


def test_impact_moon_statistics_and_recession():
    """Under half of rocky planets keep an impact moon (tiny discs make none) and ~1 in 10 a large one; Q = 12 takes
    the Moon past 400 000 km."""
    bulk = rocky(1).bulk
    draws = [impact_moon(bulk, 100.0, h.TIDAL_K2_ROCKY, 4.5 * c.SECONDS_PER_GYR, np.random.default_rng(s))
             for s in range(600)]
    have = [d for d in draws if d is not None]
    assert 0.3 < len(have) / len(draws) <= h.IMPACT_MOON_PROBABILITY + 0.05
    large = sum(d.mass_kg > 0.5 * 7.35e22 for d in have) / len(draws)
    assert 1 / 20 < large < 1 / 6
    moon = 7.35e22 / c.M_EARTH
    start = 3.8 * c.R_EARTH
    a = tidal_reach(moon, bulk, 12.0, 0.3, 4.5 * c.SECONDS_PER_GYR, start)
    assert a == pytest.approx(4.5e8, rel=0.1)


def test_limits_remove_moons():
    """A hot Jupiter keeps no moons; an Earth in an M dwarf's habitable zone loses any large moon."""
    assert giant("Jupiter", a=0.05).system.moons == []
    for s in range(1, 20):
        state = rocky(s, star_mass=0.2, a_au=0.08)
        assert all(m.mass_kg <= state.system.limits.max_surviving_moon_mass_kg for m in state.system.moons)
        assert all(m.semi_major_axis_m <= state.system.limits.stable_prograde_m for m in state.system.moons)


def test_dense_planets_move_the_template_out():
    """Around a brown dwarf the innermost regular moon stays three fluid Roche radii out."""
    spec = PlanetSpec.model_validate({"name": "Dwarf", "seed": 1, "star": {"mass_msun": 1.0},
                                      "orbit": {"semi_major_axis_au": 10.0},
                                      "body": {"mass_mearth": 9000, "radius_rearth": 11, "envelope_mass_fraction": 0.98}})
    state = generate(spec)[0]
    regular = [m for m in state.system.moons if m.origin == "regular"]
    assert len(regular) >= 3
    roche = 2.44 * state.bulk.radius_m * (state.bulk.density_kg_m3 / regular[0].density_kg_m3) ** (1 / 3)
    assert regular[0].semi_major_axis_m >= 2.99 * roche


def test_densities_mix_ice_and_rock():
    """Ganymede-, Titan- and Io-like water fractions give their densities within 10%."""
    for water, rho in ((0.45, 1936.0), (0.47, 1880.0), (0.0, 3528.0)):
        assert moon_density(water) == pytest.approx(rho, rel=0.1)


def test_listed_moons_win_and_report_round_trip():
    """Listed moons replace drawn ones, an empty list means none; the report and saving keep the drawn system."""
    listed = giant("Jupiter", system={"moons": [{"name": "Solo", "mass_mearth": 0.01, "semi_major_axis_rp": 10}]})
    assert [m.name for m in listed.system.moons] == ["Solo"] and listed.system.source == "user"
    assert giant("Jupiter", system={"moons": []}).system.moons == []
    jupiter = giant("Jupiter", seed=2)
    assert jupiter.system.source == "drawn"
    text = format_report(jupiter)
    assert "Jovian architecture" in text and "2:1 with Jupiter" in text and "ring-born" in text
    again = PlanetState.from_dict(jupiter.to_dict())
    assert again.system.moons == jupiter.system.moons and again.system.moon_regime == "jovian"
    assert all(math.isfinite(m.orbital_period_s) for m in again.system.moons)
