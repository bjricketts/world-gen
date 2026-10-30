"""Geological settings, deep time and resource potentials."""

from dataclasses import replace

import numpy as np
import pytest

from worldgen.culture.geology import Setting, deep_time, geology
from worldgen.culture.resources import RESOURCES, resource_potential
from worldgen.surface.fields import Boundary, OrogenyKind
from worldgen.world import World


@pytest.fixture(scope="module")
def earth(world_cache, request):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec
    from pathlib import Path

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def _with_biosphere(world, **changes):
    """Return a copy of the world whose biosphere state has some fields changed."""
    state = replace(world.state, biosphere=replace(world.state.biosphere, **changes), events=[])
    return World(spec=world.spec, state=state, surface=world.surface, timeline=world.timeline)


def test_snapshot_deep_time_follows_the_biosphere_age(earth):
    t = deep_time(earth.state)
    bio = earth.state.biosphere
    assert t.life_gyr == pytest.approx(bio.age_gyr)
    assert t.oxygenated_gyr == pytest.approx(bio.age_gyr - 1.4)
    assert t.anoxic_ocean_gyr == pytest.approx(t.age_gyr - bio.age_gyr + 1.4 - 0.5)
    assert t.land_life_gyr == pytest.approx(max(bio.age_gyr - 3.4, 0.0))


def test_surface_records_orogeny_kind_and_sediment(earth):
    s = earth.surface
    assert {"orogeny_kind", "sediment"} <= set(s.data_vars)
    assert np.all(s["sediment"].values >= 0)
    kinds = set(np.unique(s["orogeny_kind"].values))
    assert kinds <= {int(k) for k in OrogenyKind} and len(kinds) > 1


def test_settings_follow_the_plate_boundaries(earth):
    geo = geology(earth)
    d = geo.distance_km
    cont = geo.continental
    assert np.all(geo.setting[~cont & ~np.isin(geo.setting, [Setting.VOLCANIC])] == Setting.OCEAN)
    near_sub = cont & (d["subduction"] < 100) & (geo.setting != Setting.OROGEN) & (geo.setting != Setting.VOLCANIC)
    assert near_sub.any() and np.all(geo.setting[near_sub] == Setting.SUBDUCTION_MARGIN)
    conv = earth.surface["boundary"].values == Boundary.CONVERGENT
    assert np.all(np.minimum(d["subduction"][conv], d["collision"][conv]) == 0)


def test_potentials_are_bounded_deterministic_and_complete(earth):
    a = resource_potential(earth)
    b = resource_potential(earth)
    assert list(a["resource"].values) == list(RESOURCES) and a.shape == (len(RESOURCES), earth.grid.size)
    assert float(a.min()) >= 0 and float(a.max()) <= 1
    np.testing.assert_array_equal(a.values, b.values)
    geo = geology(earth)
    for name in ("copper", "gold", "timber", "iron"):
        assert float(a.sel(resource=name).values[geo.land].max()) > 0.5, name


def test_lifeless_world_has_no_organic_resources(earth):
    dead = _with_biosphere(earth, life="none")
    r = resource_potential(dead)
    for name in ("hydrocarbons", "coal"):
        assert float(r.sel(resource=name).max()) == 0.0


def test_uranium_and_lead_zinc_need_oxygen_and_iron_needs_an_anoxic_ocean(earth):
    anoxic = _with_biosphere(earth, product="none")          # life, but no oxygen ever
    r = resource_potential(anoxic)
    assert float(r.sel(resource="uranium").max()) == 0.0
    assert float(r.sel(resource="lead_zinc").max()) == 0.0
    assert float(r.sel(resource="iron").max()) > 0.5


def test_placer_gold_lies_on_rivers_below_gold(earth):
    r = resource_potential(earth)
    placer = r.sel(resource="placer_gold").values
    gold = r.sel(resource="gold").values
    river = earth.surface["discharge"].values >= 10.0
    assert np.all(placer[~river] == 0)
    assert placer.max() <= gold.max() + 1e-6


def test_worlds_without_the_new_surface_fields_still_get_resources(earth):
    old = World(spec=earth.spec, state=earth.state, surface=earth.surface.drop_vars(["orogeny_kind", "sediment"]),
                timeline=earth.timeline)
    r = resource_potential(old)
    assert float(r.sel(resource="copper").max()) > 0 and float(r.sel(resource="hydrocarbons").max()) == 0
