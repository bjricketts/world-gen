"""Relict features left by the planet's past (milestone 6)."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from worldgen import constants as c                                    # noqa: E402
from worldgen import heuristics as h                                   # noqa: E402
from worldgen.generate import generate_world                           # noqa: E402
from worldgen.grid import build_grid                                   # noqa: E402
from worldgen.report import format_report                              # noqa: E402
from worldgen.spec import PlanetSpec, load_spec                        # noqa: E402
from worldgen.state import Timeline                                    # noqa: E402
from worldgen.surface.relicts import Relict, memory_myr, paint, read_history   # noqa: E402

from .conftest import EXAMPLES                                         # noqa: E402


def series(age_gyr=4.0, span_gyr=2.0, points=40, water=1.0, temperature=288.0, melt=1.0, climate="warm"):
    """Return a sampled history over the last ``span_gyr``, constant unless an array is given."""
    times = np.linspace(age_gyr - span_gyr, age_gyr, points)

    def column(value):
        """Return a constant column or the given one."""
        return list(np.full(points, value)) if np.isscalar(value) else list(value)

    return {"time_gyr": list(times), "surface_water_oceans": column(water),
            "surface_temperature_k": column(temperature), "melt": column(melt),
            "climate": list(np.full(points, climate)) if isinstance(climate, str) else list(climate)}


def fake_state(surface_water="liquid", activity=1.0, regime="mobile_lid", ocean_volume_m3=1.3e18,
               age_gyr=4.0):
    """Return the parts of a planet state the relict rules read."""
    spec = load_spec(EXAMPLES / "earth.yaml").model_copy(update={"seed": 3})
    from worldgen.evolve.snapshot import SnapshotEvolver

    state = SnapshotEvolver().build(spec, 3)
    state.star.age_s = age_gyr * c.SECONDS_PER_GYR
    state.atmosphere.surface_water = surface_water
    state.interior.activity_index = activity
    state.interior.tectonic_regime = regime
    state.water.ocean_volume_m3 = ocean_volume_m3
    return state


def test_a_surface_remembers_for_as_long_as_its_erosion_allows():
    """Rain and plate tectonics erase the record in ~100 Myr; a dry dead lid keeps it for Gyr."""
    earth = memory_myr(liquid_share=1.0, activity=1.0)
    assert earth == pytest.approx(100.0, abs=30.0)
    mars = memory_myr(liquid_share=0.0, activity=0.2)
    assert 1000.0 < mars < 4000.0
    assert memory_myr(0.0, 0.05) > mars                                # the deader the planet, the longer
    assert memory_myr(1.0, 0.2) < memory_myr(0.0, 0.2)                 # water wears a surface faster


def test_nothing_recent_leaves_nothing_behind():
    """A planet that has not changed inside its memory has no relicts, and neither has a snapshot world."""
    state = fake_state()
    past = read_history(state, Timeline(series=series()))
    assert past is not None and past.palaeo_ocean_m3 is None and past.cooling_k is None
    assert past.wet_age_myr is None and past.resurfaced_share is None
    assert read_history(state, None) is None
    assert read_history(state, Timeline()) is None


def test_a_lost_ocean_leaves_a_shoreline():
    """A planet holding less water than it did keeps the coastline of the larger sea."""
    water = np.linspace(2.0, 1.0, 40)                                  # half the ocean lost over the run
    state = fake_state(surface_water="ice", activity=0.2)
    past = read_history(state, Timeline(series=series(water=water)))
    assert past.palaeo_ocean_m3 == pytest.approx(2.0 * state.water.ocean_volume_m3, rel=0.02)
    assert past.shoreline_age_myr == pytest.approx(2000.0, rel=0.05)     # the start of the sampled history

    grid = build_grid(2000)
    elevation = np.linspace(-3000.0, 3000.0, grid.size)
    ocean = elevation < 0.0

    class Fill:
        """A stand-in for the ocean fill: the old sea stood 500 m higher."""

        def level_for_volume(self, volume):
            """Return the level the given volume reaches."""
            return 500.0

    relict, features = paint(grid, state, past, elevation, ocean, Fill(),
                             np.full(grid.size, 250.0), None, 1.0, seed=1)
    band = h.RELICT_SHORELINE_BAND_M
    assert (relict == Relict.SHORELINE).sum() > 0
    marked = elevation[relict == Relict.SHORELINE]
    assert np.all(np.abs(marked - 500.0) <= band + 1e-6)               # a terrace at the old sea level
    assert features["relict_shoreline_m"] == pytest.approx(500.0)
    assert features["relict_memory_myr"] == past.memory_myr


def test_a_lost_climate_leaves_valleys_and_scoured_ground():
    """A dry planet that once had rain keeps its valley networks; a warmer one keeps the ice's marks."""
    modes = np.array(["warm"] * 30 + ["snowball"] * 10)
    temperature = np.array([290.0] * 30 + [250.0] * 10)
    frozen = fake_state(surface_water="ice", activity=0.2)
    past = read_history(frozen, Timeline(series=series(temperature=temperature, climate=modes)))
    assert past.wet_age_myr == pytest.approx(513.0, rel=0.1)           # the last warm epoch
    assert past.cooling_k is None                                      # it is coldest now, nothing to recover

    warm_again = np.array(["snowball"] * 20 + ["warm"] * 20)
    swinging = np.array([250.0] * 20 + [290.0] * 20)
    state = fake_state(activity=0.2)                                   # wet and active: a short memory
    past = read_history(state, Timeline(series=series(span_gyr=0.1, temperature=swinging,
                                                      climate=warm_again)))
    assert past.cooling_k == pytest.approx(40.0, abs=1.0)

    grid = build_grid(2000)
    elevation = np.linspace(-2000.0, 4000.0, grid.size)
    ocean = elevation < 0.0
    temperature_k = 290.0 - 0.0065 * np.maximum(elevation, 0.0)        # cooler with height
    relict, features = paint(grid, state, past, elevation, ocean, None, temperature_k, None, 1.0, seed=1)
    scoured = relict == Relict.GLACIAL
    assert scoured.any()
    assert temperature_k[scoured].max() - past.cooling_k < 273.15      # only ground that froze back then
    assert features["relict_glacial_cooling_k"] == pytest.approx(40.0, abs=1.0)


def test_a_quieter_interior_leaves_resurfaced_plains():
    """Plains buried when the planet melted faster stay visible on a lid that no longer erupts."""
    melt = np.concatenate([np.full(20, 2.0), np.full(20, 1e-4)])
    state = fake_state(surface_water="none", activity=0.1, regime="stagnant_lid")
    past = read_history(state, Timeline(series=series(melt=melt)))
    assert past.melt_ratio > h.RELICT_RESURFACING_RATIO
    assert 0.0 < past.resurfaced_share <= h.RELICT_RESURFACED_MAX

    grid = build_grid(2000)
    elevation = np.zeros(grid.size)
    relict, features = paint(grid, state, past, elevation, np.zeros(grid.size, dtype=bool), None,
                             np.full(grid.size, 250.0), None, 1.0, seed=1)
    assert (relict == Relict.VOLCANIC).mean() == pytest.approx(past.resurfaced_share, abs=0.05)

    plated = fake_state(surface_water="none", activity=1.0, regime="mobile_lid")
    relict, _ = paint(grid, plated, past, elevation, np.zeros(grid.size, dtype=bool), None,
                      np.full(grid.size, 250.0), None, 1.0, seed=1)
    assert not (relict == Relict.VOLCANIC).any()                       # plates rework their own plains


@pytest.mark.slow
def test_earth_keeps_no_relicts(world_cache):
    """Earth reworks its surface too fast: nothing from its past survives to be drawn."""
    world = world_cache(load_spec(EXAMPLES / "earth_history.yaml"), "preview")
    features = world.state.surface.features
    assert not any(k.startswith("relict_") for k in features)
    assert "relict" in world.surface                                   # the field is there, and empty
    assert (world.surface["relict"].values == Relict.NONE).all()


@pytest.mark.slow
def test_a_frozen_world_shows_what_it_lost(world_cache):
    """An old ice world keeps the shoreline, the valleys and the scoured ground of its warmer past."""
    spec = PlanetSpec.model_validate({"name": "Relict world", "seed": 5, "mode": "history",
                                      "priors": {"archetype": "ice"}})
    world = world_cache(spec, "preview")
    features = world.state.surface.features
    assert features["relict_memory_myr"] > 1000.0                      # a slow surface remembers
    assert features["relict_river_fraction"] > 0.0
    assert features["relict_glacial_fraction"] > 0.0
    kinds = set(np.unique(world.surface["relict"].values).tolist())
    assert kinds - {int(Relict.NONE)}
    report = format_report(world.state)
    assert "relicts of the past" in report and "surface memory" in report

    from worldgen.render import plot_map

    figure = plot_map(world, "relicts", "mollweide").figure
    assert figure.axes
    with pytest.raises(ValueError):
        plot_map(world_cache(PlanetSpec(name="Plain", seed=2, priors={"archetype": "temperate"}), "preview"),
                 "relicts", "mollweide")
