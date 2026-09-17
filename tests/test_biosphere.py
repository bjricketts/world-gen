from pathlib import Path

import numpy as np
import pytest

from worldgen import PlanetSpec, generate
from worldgen import constants as c
from worldgen.biosphere import Biome, KOPPEN_CODES, classify_biomes, koppen
from worldgen.biosphere.life import default_life, oxygen_level
from worldgen.biosphere.pigment import absorption_peak, pigment_colour
from worldgen.biosphere.vegetation import miami_productivity
from worldgen.render import plot_climate, plot_map
from worldgen.spec import load_spec


def test_default_life_rules():
    assert default_life(4.0, "liquid", 0.3) == "surface"
    assert default_life(4.0, "liquid", 0.001) == "ocean"
    assert default_life(4.0, "ice", 0.3) == "subsurface"
    assert default_life(0.5, "liquid", 0.3) == "none"
    assert default_life(4.0, "none", 0.3) == "none"


def test_oxygen_history():
    assert oxygen_level(0.5) < 0.001
    assert 0.01 < oxygen_level(2.0) < 0.05
    assert oxygen_level(3.87) == pytest.approx(0.21, abs=0.01)


def test_miami_model():
    # Tropical rainforest near the 3000 g/m² ceiling; deserts and cold places far below.
    npp = miami_productivity(np.array([300.0, 300.0, 250.0]), np.array([3.0, 0.1, 1.0]))
    assert npp[0] > 2000.0
    assert npp[1] < 250.0
    assert npp[2] < 400.0
    hot = miami_productivity(np.array([340.0]), np.array([3.0]))
    assert hot[0] < 10.0          # beyond the tolerance


def test_pigment_colours_follow_the_star():
    def hue(hex_colour):
        r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
        return r, g, b
    r, g, b = hue(pigment_colour(absorption_peak(c.T_SUN), c.T_SUN)[0])
    assert g > r and g > b                       # green under the Sun
    r, g, b = hue(pigment_colour(absorption_peak(7200.0), 7200.0)[0])
    assert r > g                                 # reddish under an F star
    colour, brightness = pigment_colour(absorption_peak(3000.0), 3000.0)
    assert brightness < 0.05                     # dark under an M dwarf


def test_koppen_rules():
    months = 12
    north = np.array([True] * 5)
    t = np.zeros((months, 5))
    p = np.zeros((months, 5))
    t[:, 0], p[:, 0] = 300.0, 3.0                          # Af
    t[:, 1], p[:, 1] = 300.0, 0.05                         # BWh
    t[:, 2] = 273.0 + 3 * np.cos(np.linspace(0, 2 * np.pi, months))
    p[:, 2] = 0.5                                          # ET (warmest month below 10 °C)
    t[:, 3] = 283.0 - 10 * np.cos(2 * np.pi * (np.arange(months) + 0.5) / months)
    p[:, 3] = 1.0                                          # Cfb
    t[:, 4] = 263.0 - 25 * np.cos(2 * np.pi * (np.arange(months) + 0.5) / months)
    p[:, 4] = 0.6                                          # Dfa/Dfb
    codes = [KOPPEN_CODES[k] for k in koppen(t, p, north, np.zeros(5, dtype=bool))]
    assert codes[0] == "Af"
    assert codes[1] == "BWh"
    assert codes[2] == "ET"
    assert codes[3] == "Cfb"
    assert codes[4].startswith("Df")


def test_biomes_with_and_without_life():
    temp = np.array([300.0, 300.0, 285.0, 268.0, 285.0])
    rain = np.array([3.0, 0.1, 1.5, 0.6, 1.0])
    monthly = np.tile(temp + 20.0, (12, 1))
    ocean = np.array([False, False, False, False, True])
    ice = np.zeros(5, dtype=bool)
    state, _ = generate(PlanetSpec(seed=0, priors={"archetype": "temperate"}))
    living = classify_biomes(temp, rain, monthly, ocean, ice, state.biosphere)
    assert living.tolist() == [Biome.TROPICAL_RAINFOREST, Biome.HOT_DESERT, Biome.TEMPERATE_FOREST,
                               Biome.BOREAL_FOREST, Biome.OCEAN]
    dead = classify_biomes(temp, rain, monthly, ocean, ice, None)
    assert set(dead[:4].tolist()) == {Biome.BARREN}


def test_earth_state_biosphere(earth_spec):
    state, _ = generate(earth_spec)
    life = state.biosphere
    assert life.life == "surface"
    assert life.oxygen_fraction == pytest.approx(0.21, abs=0.01)
    assert life.productivity == pytest.approx(1.0, abs=0.05)
    assert state.occupiability.factors["biosphere"] == 1.0
    assert state.occupiability.habitable_unaided


def test_life_can_be_switched_off(earth_spec):
    spec = earth_spec.model_copy(deep=True)
    spec.biosphere.life = "none"
    state, _ = generate(spec)
    assert state.biosphere is None
    assert state.occupiability.factors["biosphere"] < 1.0
    assert not state.occupiability.habitable_unaided


@pytest.mark.slow
def test_methanogens_warm_the_planet():
    base = dict(seed=3, priors={"archetype": "temperate"}, interior={"tectonic_regime": "stagnant_lid"},
                atmosphere={"composition": "n2_co2"})
    plain, _ = generate(PlanetSpec(**base, biosphere={"life": "surface", "biochemistry": "oxygenic"}))
    methane, _ = generate(PlanetSpec(**base, biosphere={"life": "surface", "biochemistry": "methanogenic"}))
    assert methane.biosphere.methane_fraction > 0.0
    assert methane.atmosphere.surface_temperature_k > plain.atmosphere.surface_temperature_k


@pytest.fixture(scope="module")
def living_world(world_cache):
    spec = load_spec(Path(__file__).parent.parent / "examples" / "earth.yaml")
    spec.surface.tectonics = "heuristic"
    return world_cache(spec)


def test_surface_vegetation_ice_and_classes(living_world):
    ds = living_world.surface
    ocean = ds.ocean.values
    assert {"biome", "koppen", "ice_thickness", "vegetation_cover", "productivity"} <= set(ds.data_vars)
    assert (ds.biome.values[ocean] == Biome.OCEAN).all()
    assert (ds.biome.values[~ocean] != Biome.OCEAN).all()
    assert (ds.koppen.values[~ocean] > 0).all()
    # Glaciated land is classified as ice.
    glacier = ds.ice_thickness.values > 0
    assert (ds.biome.values[glacier] == Biome.ICE_SHEET).all()
    # Vegetation darkens land.
    veg = ds.vegetation_cover.values
    albedo = ds.land_albedo.values
    land = ~ocean & ~glacier
    assert albedo[land & (veg > 0.8)].mean() < albedo[land & (veg < 0.1)].mean()
    f = living_world.state.surface.features
    assert f["climate_coupling_passes"] >= 2
    assert 0.0 < f["productive_fraction_of_land"] <= 1.0


def test_climate_figures(living_world):
    fig = plot_climate(living_world)
    assert len(fig.axes) >= 9
    for field in ("biomes", "koppen", "ice", "temperature"):
        plot_map(living_world, field, "equirectangular", width=300)
