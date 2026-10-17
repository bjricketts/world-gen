"""Suitability components of the organic history (task 9.2)."""

from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation
from worldgen.culture.build import geology_layers
from worldgen.culture.history.suitability import (ARABLE_LIMITS, COMPONENTS, build_suitability, gentle_share,
                                                  ramp, ruggedness_index, subgrid_slope)
from worldgen.culture.suitability import covariates
from worldgen.surface.fields import Terrain


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


@pytest.fixture(scope="module")
def suit(earth):
    """The suitability components of the preview Earth world, with its covariates."""
    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    return build_suitability(earth, cov, geo["resource_potential"]), cov


def test_ramp_and_gentle_share():
    assert list(ramp([0, 1, 2, 3], 1, 3)) == [0.0, 0.0, 0.5, 1.0]
    shares = gentle_share(np.array([0.01, 0.06, 0.15, 0.4]))
    assert shares[0] == pytest.approx(1.0) and np.all(np.diff(shares) < 0)


def test_ruggedness_index_spans_nunn_and_puga():
    tri = ruggedness_index(np.array([h.SUBGRID_SLOPE["PLAIN"], h.SUBGRID_SLOPE["MOUNTAIN"]]))
    assert tri[0] == pytest.approx(0.26, abs=0.01) and tri[1] == pytest.approx(3.93, abs=0.01)
    assert 0.04 < tri[0] < tri[1] < 6.2          # between the Netherlands and Lesotho


def test_components_are_bounded_and_zero_off_land(suit):
    s, cov = suit
    comps = s.as_array()
    assert comps.shape == (len(COMPONENTS), cov.land.size)
    assert comps.min() >= 0 and comps.max() <= 1
    off = ~cov.land | cov.ice
    assert np.all(comps[:, off] == 0)
    assert s.to_dataarray().dims == ("component", "cell")


def test_arable_share_matches_zabel(suit):
    s, cov = suit
    free = cov.land & ~cov.ice
    # Zabel et al. (2014): 0.61 of ice-free land; the calibration (0.63 on a standard simulated world) is
    # checked in the validation script, and this preview world on heuristic plates only needs to be near it
    assert 0.45 < (s.arable[free] > 0).mean() < 0.85


def test_arable_limits_name_the_weakest_factor(earth, suit):
    s, cov = suit
    free = cov.land & ~cov.ice
    t = earth.surface["air_temperature"].values - 273.15
    frozen = free & (t <= h.ARABLE_PERMAFROST_C)
    assert np.all(s.arable[frozen] == 0)
    assert set(np.unique(s.arable_limit[free])) <= set(range(len(ARABLE_LIMITS)))
    assert np.all(s.arable_limit[free & (s.arable >= 1)] == 0)
    assert np.all(s.arable_limit[free & (s.arable < 1)] > 0)


def test_mountains_and_coasts_are_easier_to_defend(earth, suit):
    s, cov = suit
    free = cov.land & ~cov.ice
    terrain = earth.surface["terrain"].values
    mountain, plain = free & (terrain == Terrain.MOUNTAIN), free & (terrain == Terrain.PLAIN)
    assert s.defensibility[mountain].mean() > s.defensibility[plain].mean()
    inland = plain & (cov.coast_km > 500)
    coastal = plain & (cov.coast_km < cov.spacing_km)
    assert s.defensibility[coastal].mean() > s.defensibility[inland].mean()
    assert np.all(s.ruggedness_index[mountain] > s.ruggedness_index[plain].max())


def test_fresh_water_and_access_follow_rivers_and_coast(earth, suit):
    s, cov = suit
    free = cov.land & ~cov.ice
    big = free & (cov.discharge >= 1000)
    assert np.all(s.fresh_water[big] == 1) and np.all(s.access[big] == 1)
    from worldgen.culture.history.suitability import sea_edge_share
    shore = free & (sea_edge_share(earth, cov.land) > 0)
    assert shore.any() and np.all(s.access[shore] == 1)


def test_subgrid_slope_follows_landforms(earth):
    slope = subgrid_slope(earth)
    terrain = earth.surface["terrain"].values
    assert np.all(slope[terrain == Terrain.PLAIN] == h.SUBGRID_SLOPE["PLAIN"])


def test_epochs_carry_the_suitability_layer(earth):
    spec = CultureSpec.model_validate({"history": {"start_years_ago": 100}, "cultures": [{"mode": "organic"}]})
    civ = build_civilisation(earth, spec)
    layer = civ.layers["suitability_component"]
    assert list(layer["component"].values) == list(COMPONENTS)
