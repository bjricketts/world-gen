"""Culture spec, development axes, resolved cultures, the civilisation data model and its storage (task 8.1)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from worldgen import generate, load_world, save_world
from worldgen.culture import (Civilisation, CultureSpec, Development, Polity, RoadTable, SettlementTable,
                              build_civilisation, load_civilisation, load_culture_spec, resolve_cultures,
                              save_civilisation, save_culture_spec)
from worldgen.culture.development import resolve_development
from worldgen.culture.model import ChronicleEntry
from worldgen.spec import load_spec, save_spec
from worldgen.world import World

SPEC = {
    "cultures": [
        {"name": "Vessari", "inhabitants": "native", "development": {"energy": "farming"},
         "population": [1e7, 1e8], "polities": {"count": [3, 8], "type": "kingdom",
                                                  "capitals": [{"lat_deg": 10, "lon_deg": 20}]}},
        {"inhabitants": "colonist", "development": {"energy": "fusion", "transport": "spaceflight"},
         "colony": {"founded_years_ago": [50, 400], "origin": "Sol",
                    "landing_sites": [{"lat_deg": -5, "lon_deg": 100}]}},
    ]
}


# --- development --------------------------------------------------------------

def test_missing_axes_take_the_mean_of_the_given_ones():
    dev = resolve_development({"energy": "fossil", "transport": None, "communication": None}, {})
    assert (dev.energy, dev.transport, dev.communication) == (3, 3, 3)
    dev = resolve_development({"energy": "foraging", "transport": "rail_and_steam", "communication": None}, {})
    assert dev.communication == 2    # mean of 0 and 3 rounds half up


def test_no_axis_given_uses_the_default():
    default = {"energy": "farming", "transport": "foot", "communication": "oral"}
    assert resolve_development({"energy": None}, default) == Development(1, 0, 0)


def test_levels_outside_the_axis_are_rejected():
    with pytest.raises(ValueError):
        Development(6, 0, 0)


def test_development_names_round_trip():
    dev = Development(2, 4, 1)
    assert Development.from_names(dev.to_names()) == dev


# --- spec ---------------------------------------------------------------------

def test_spec_needs_a_culture():
    with pytest.raises(ValidationError):
        CultureSpec.model_validate({"cultures": []})


@pytest.mark.parametrize("bad", [
    {"inhabitants": "native", "colony": {"origin": "Sol"}},
    {"population": 0},
    {"polities": {"count": 1, "capitals": [{"lat_deg": 0, "lon_deg": 0}, {"lat_deg": 1, "lon_deg": 1}]}},
    {"development": {"energy": "steam"}},
    {"homeland": {"lat_deg": 95, "lon_deg": 0, "radius_km": 100}},
    {"unknown_key": 1},
])
def test_spec_rejects_inconsistent_entries(bad):
    with pytest.raises(ValidationError):
        CultureSpec.model_validate({"cultures": [bad]})


def test_culture_spec_loads_as_a_section_or_a_file(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"culture": SPEC}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(SPEC))
    assert load_culture_spec(tmp_path / "a.yaml") == load_culture_spec(tmp_path / "b.yaml")
    save_culture_spec(load_culture_spec(tmp_path / "a.yaml"), tmp_path / "c.yaml")
    assert load_culture_spec(tmp_path / "c.yaml") == CultureSpec.model_validate(SPEC)


def test_planet_spec_carries_the_culture_section(earth_spec, tmp_path):
    spec = earth_spec.model_copy(update={"culture": CultureSpec.model_validate(SPEC)})
    save_spec(spec, tmp_path / "s.yaml")
    assert load_spec(tmp_path / "s.yaml").culture == spec.culture
    assert earth_spec.culture is None


# --- resolution ---------------------------------------------------------------

def test_resolution_draws_ranges_within_bounds_and_is_deterministic():
    spec = CultureSpec.model_validate(SPEC)
    a, b = resolve_cultures(spec, 7), resolve_cultures(spec, 7)
    assert [c.to_plain() for c in a] == [c.to_plain() for c in b]
    native, colony = a
    assert 1e7 <= native.population <= 1e8
    assert 3 <= native.polity_count <= 8
    assert 50 <= colony.founded_years_ago <= 400
    assert colony.development.communication == 5    # mean of 5 and 5
    assert native.population != resolve_cultures(spec, 8)[0].population


def test_native_development_fills_from_the_given_axis():
    native = resolve_cultures(CultureSpec.model_validate(SPEC), 0)[0]
    assert native.development == Development(1, 1, 1)


def test_adding_a_culture_leaves_the_others_unchanged():
    one = CultureSpec.model_validate({"cultures": SPEC["cultures"][:1]})
    two = CultureSpec.model_validate(SPEC)
    assert resolve_cultures(one, 3)[0].to_plain() == resolve_cultures(two, 3)[0].to_plain()


def test_culture_seed_overrides_the_planet_seed():
    spec = CultureSpec.model_validate({**SPEC, "seed": 11})
    assert resolve_cultures(spec, 1)[0].population == resolve_cultures(spec, 2)[0].population


def test_defaults_by_inhabitants():
    spec = CultureSpec.model_validate({"cultures": [{"inhabitants": "native"}, {"inhabitants": "colonist"}]})
    native, colony = resolve_cultures(spec, 0)
    assert native.development.name("energy") == "farming"
    assert colony.development.name("transport") == "spaceflight"
    assert native.population is None and native.polity_count is None


# --- tables and storage --------------------------------------------------------

def test_tables_fill_missing_columns_and_reject_mismatches():
    t = SettlementTable(id=[0, 1], population=[10.0, 20.0])
    assert len(t) == 2 and np.all(t.polity == -1) and np.all(np.isnan(t.founded_years_ago))
    with pytest.raises(ValueError):
        SettlementTable(id=[0, 1], population=[10.0])
    with pytest.raises(ValueError):
        SettlementTable(id=[0], colour=[1])
    with pytest.raises(ValueError):
        RoadTable(id=[0], a=[0], b=[1], path_offsets=[0, 3], path_cells=[5, 6])


def _civilisation():
    spec = CultureSpec.model_validate(SPEC)
    settlements = SettlementTable(id=[0, 1, 2], culture=[0, 0, 1], polity=[0, 1, -1], cell=[5, 9, 40],
                                  lat_deg=[1.0, 2.0, -5.0], lon_deg=[3.0, 4.0, 100.0],
                                  population=[1e5, 2e4, 300.0], form=[4, 3, 7], name=["Aru", "Bel", "Landing"])
    roads = RoadTable(id=[0, 1], a=[0, 1], b=[1, 2], kind=[2, 0], length_km=[120.0, 900.0],
                      path_offsets=[0, 3, 7], path_cells=[5, 6, 9, 9, 20, 30, 40])
    layers = xr.Dataset({"suitability": ("cell", np.linspace(0, 1, 50, dtype=np.float32)),
                         "polity": ("cell", np.full(50, -1, dtype=np.int32))})
    return Civilisation(spec=spec, seed=4, cultures=resolve_cultures(spec, 4),
                        polities=[Polity(id=0, culture=0, type="kingdom", capital=0),
                                  Polity(id=1, culture=0, type="kingdom", capital=1, parent=0)],
                        settlements=settlements, roads=roads, layers=layers,
                        chronicle=[ChronicleEntry(years_ago=120.0, kind="landing", text="first landing",
                                                  refs={"settlement": 2})])


def test_civilisation_round_trip(tmp_path):
    civ = _civilisation()
    loaded = load_civilisation(save_civilisation(civ, tmp_path / "c"))
    assert [c.to_plain() for c in loaded.cultures] == [c.to_plain() for c in civ.cultures]
    assert loaded.polities == civ.polities and loaded.chronicle == civ.chronicle
    assert loaded.spec == civ.spec and loaded.seed == 4
    for name in SettlementTable.COLUMNS:
        np.testing.assert_array_equal(getattr(loaded.settlements, name), getattr(civ.settlements, name))
    np.testing.assert_array_equal(loaded.roads.path(1), [9, 20, 30, 40])
    xr.testing.assert_equal(loaded.layers, civ.layers)


def test_empty_civilisation_round_trip(tmp_path):
    spec = CultureSpec.model_validate(SPEC)
    civ = Civilisation(spec=spec, seed=0, cultures=resolve_cultures(spec, 0))
    loaded = load_civilisation(save_civilisation(civ, tmp_path / "c"))
    assert len(loaded.settlements) == 0 and len(loaded.roads) == 0 and not loaded.layers.data_vars


def test_unknown_culture_format_is_rejected(tmp_path):
    folder = save_civilisation(_civilisation(), tmp_path / "c")
    data = yaml.safe_load((folder / "culture.yaml").read_text())
    data["culture_format"] = 99
    (folder / "culture.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_civilisation(folder)


@pytest.fixture(scope="module")
def bare_world():
    """An Earth-spec planet without a built surface (fast)."""
    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    state, timeline = generate(spec)
    return World(spec=spec, state=state, surface=None, timeline=timeline)


def test_world_saves_and_loads_its_civilisation(bare_world, tmp_path):
    world = World(**{**bare_world.__dict__, "culture": _civilisation()})
    folder = save_world(world, tmp_path / "w")
    assert (folder / "culture" / "culture.yaml").exists()
    loaded = load_world(folder)
    assert loaded.culture is not None and len(loaded.culture.settlements) == 3
    # Saving the world again without a civilisation removes the stale one.
    save_world(bare_world, folder)
    assert load_world(folder).culture is None


def test_format_5_worlds_still_load(bare_world, tmp_path):
    folder = save_world(bare_world, tmp_path / "w")
    data = yaml.safe_load((folder / "state.yaml").read_text())
    data["format_version"] = 5
    (folder / "state.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    assert load_world(folder).culture is None


def test_build_civilisation_uses_the_world_culture_section(bare_world):
    spec = CultureSpec.model_validate(SPEC)
    with pytest.raises(ValueError):
        build_civilisation(bare_world)
    world = World(**{**bare_world.__dict__, "spec": bare_world.spec.model_copy(update={"culture": spec})})
    civ = build_civilisation(world)
    assert civ.seed == world.spec.seed and len(civ.cultures) == 2
    assert build_civilisation(bare_world, spec).cultures[0].to_plain() == civ.cultures[0].to_plain()


def test_generate_world_places_the_culture_section(world_cache, earth_spec, tmp_path):
    spec = earth_spec.model_copy(update={
        "culture": CultureSpec.model_validate(SPEC),
        "surface": earth_spec.surface.model_copy(update={"tectonics": "heuristic"})})
    world = world_cache(spec)
    assert world.culture is not None and [c.id for c in world.culture.cultures] == [0, 1]
    loaded = load_world(save_world(world, tmp_path / "w"))
    assert loaded.culture.cultures[1].origin == "Sol"
    layers = loaded.culture.layers
    assert layers["resource_potential"].dims == ("resource", "cell")
    assert "hydrocarbons" in list(layers["resource"].values)
    np.testing.assert_array_equal(layers["tectonic_setting"].values, world.culture.layers["tectonic_setting"].values)
    assert layers["population_density"].dims == ("culture", "cell")
    np.testing.assert_array_equal(layers["population_density"].values,
                                  world.culture.layers["population_density"].values)
    assert loaded.culture.cultures[0].population == world.culture.cultures[0].population
    assert len(loaded.culture.settlements) == len(world.culture.settlements) > 0
    np.testing.assert_array_equal(loaded.culture.settlements.population, world.culture.settlements.population)
    assert len(loaded.culture.roads) == len(world.culture.roads) > 0
    np.testing.assert_array_equal(loaded.culture.roads.path_cells, world.culture.roads.path_cells)
    np.testing.assert_array_equal(loaded.culture.roads.flow, world.culture.roads.flow)
    assert [p.to_plain() for p in loaded.culture.polities] == [p.to_plain() for p in world.culture.polities]
    np.testing.assert_array_equal(loaded.culture.layers["polity"].values, world.culture.layers["polity"].values)
