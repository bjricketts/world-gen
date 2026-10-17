"""Habitats of colonists, landing sites and space elevators (task 8.8)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.culture.build import build_civilisation
from worldgen.culture.density import landing_weight
from worldgen.culture.development import Development
from worldgen.culture.habitats import Habitat, air_column_g_cm2, elevator_taper, habitat_capacity, habitat_classes
from worldgen.culture.model import SETTLEMENT_FORMS, Culture
from worldgen.culture.settlements import settlement_form
from worldgen.culture.spec import CultureSpec
from worldgen.culture.suitability import covariates
from worldgen.world import World


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def _with_air(world, pressure_pa, oxygen=None):
    """Return the world with a different surface pressure (and oxygen fraction)."""
    state = world.state
    atm = replace(state.atmosphere, surface_pressure_pa=pressure_pa)
    bio = state.biosphere if oxygen is None else replace(state.biosphere, oxygen_fraction=oxygen)
    return World(spec=world.spec, state=replace(state, atmosphere=atm, biosphere=bio), surface=world.surface,
                 timeline=world.timeline)


def test_earth_air_shields_like_hassler_and_thin_air_does_not():
    # Earth's column is ~1030 g/cm²; the 560 g/cm² threshold is Hassler et al.'s 2 m of rock.
    assert air_column_g_cm2(np.array(c.BAR * 1.013), 9.81) == pytest.approx(1033, rel=0.01)
    assert air_column_g_cm2(np.array(600.0), 3.71) < h.RADIATION_SHIELD_G_CM2


def test_habitats_follow_air_and_temperature(earth):
    land = ~earth.surface["ocean"].values
    home = habitat_classes(earth)
    assert np.mean(home[land] == Habitat.OPEN) > 0.6 and np.all(home[~land] == Habitat.NONE)
    thin = habitat_classes(_with_air(earth, 600.0))
    assert np.all(thin[land & (home != Habitat.NONE)] == Habitat.UNDERGROUND)
    no_oxygen = habitat_classes(_with_air(earth, earth.state.atmosphere.surface_pressure_pa, oxygen=0.0))
    assert not np.any(no_oxygen == Habitat.OPEN) and np.any(no_oxygen == Habitat.DOME)
    share = habitat_capacity(earth, thin)
    assert np.all(share[thin == Habitat.UNDERGROUND] == h.HABITAT_CAPACITY["underground"])  # Earth gravity: no lava-tube bonus


def test_elevator_taper_matches_pearson_on_earth(earth):
    # Pearson (1975): taper ratio 10 for h = 2150 km on Earth.
    assert elevator_taper(earth.state) == pytest.approx(10, rel=0.05)
    locked = replace(earth.state, orbit=replace(earth.state.orbit, rotation_period_s=earth.state.orbit.orbital_period_s))
    assert elevator_taper(locked) == float("inf")


def test_colonist_forms_by_habitat_and_size():
    col = Culture(id=0, name=None, inhabitants="colonist", development=Development(5, 5, 5))
    pops = np.array([500.0, 5e4, 5e4, 5e4])
    hab = np.array([Habitat.OPEN, Habitat.OPEN, Habitat.DOME, Habitat.UNDERGROUND])
    names = [SETTLEMENT_FORMS[f] for f in settlement_form(pops, col, hab)]
    assert names == ["outpost", "town", "dome", "underground"]


def test_landing_sites_prefer_low_latitude_plains(earth):
    cov = covariates(earth)
    w = landing_weight(cov)
    low = (np.abs(cov.lat_deg) <= h.LANDING_MAX_LATITUDE_DEG) & ~cov.rough
    assert np.all(w[low] == 1) and np.all(w[~low] < 1)


def test_colonies_on_thin_air_live_underground_and_anchor_an_elevator(earth):
    mars_like = _with_air(earth, 600.0)
    civ = build_civilisation(mars_like, CultureSpec(cultures=[{"inhabitants": "colonist",
                                                               "colony": {"founded_years_ago": 200}}]))
    forms = {SETTLEMENT_FORMS[f] for f in civ.settlements.form}
    assert "underground" in forms and not forms & {"town", "city", "dome"}
    assert "orbital_anchor" in forms
    assert np.all(civ.layers["habitat"].values[civ.settlements.cell] == Habitat.UNDERGROUND)


def test_homelands_end_at_their_tail(earth):
    civ = build_civilisation(earth, CultureSpec(cultures=[{"homeland": {"lat_deg": 0, "lon_deg": 20, "radius_km": 800},
                                                           "population": 1e6}]))
    grid = earth.grid
    far = np.degrees(np.arccos(np.clip(grid.points @ np.array([np.cos(np.radians(20)), np.sin(np.radians(20)), 0]),
                                       -1, 1))) * 111.2 > 800 * (1 + 5 * h.HOMELAND_EDGE)
    assert np.all(civ.layers["population_density"].values[0][far] == 0)


def test_colonists_need_a_survivable_surface(earth):
    s = earth.surface.copy()
    s["air_temperature"] = s["air_temperature"] * 0 + 740.0
    hot = World(spec=earth.spec, state=earth.state, surface=s, timeline=earth.timeline)
    with pytest.raises(ValueError, match="survivable"):
        build_civilisation(hot, CultureSpec(cultures=[{"inhabitants": "colonist"}]))
