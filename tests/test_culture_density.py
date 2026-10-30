"""Suitability, population density and default populations of cultures."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from worldgen import heuristics as h
from worldgen.culture.build import build_civilisation
from worldgen.culture.density import cluster, colony_age
from worldgen.culture.development import Development
from worldgen.culture.model import Culture
from worldgen.culture.spec import CultureSpec
from worldgen.culture.suitability import Covariates, density_terms, log_capacity
from worldgen.world import World


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def _spec(*cultures):
    return CultureSpec(cultures=list(cultures))


def _culture(energy, inhabitants="native"):
    return Culture(id=0, name=None, inhabitants=inhabitants, development=Development(energy, energy, energy))


def _cov(n, **fields):
    base = dict(land=np.ones(n, bool), ice=np.zeros(n, bool), coast_km=np.full(n, 3000.0),
                discharge=np.zeros(n), elevation=np.zeros(n), temperature_c=np.full(n, 10.0),
                productivity=np.full(n, 500.0), resources=np.zeros(n), cell_area_km2=1.0, spacing_km=1.0,
                radius_km=6371.0, land_life=True)
    base.update(fields)
    return Covariates(**base)


def test_terms_run_from_foragers_to_the_modern_set():
    assert density_terms(_culture(0)) == pytest.approx(h.DENSITY_TERMS_FORAGING)
    assert density_terms(_culture(4)) == pytest.approx(h.DENSITY_TERMS_MODERN)
    assert density_terms(_culture(5)) == pytest.approx(h.DENSITY_TERMS_MODERN)
    mid = density_terms(_culture(2))
    assert mid["clustering"] == pytest.approx(0.5 * h.DENSITY_TERMS_MODERN["clustering"])
    assert density_terms(_culture(5, "colonist"))["log_npp"] == 0.0


def test_forager_capacity_follows_productivity_up_to_the_plateau():
    # Tallavaara et al. (2018): ln density rises 0.003 per g m-2 yr-1 up to ~1360, then levels off.
    cov = _cov(4, productivity=np.array([400.0, 1000.0, 1360.0, 2500.0]))
    ln = log_capacity(cov, density_terms(_culture(0)))
    assert ln[1] - ln[0] == pytest.approx(0.003 * 600)
    assert ln[3] == pytest.approx(ln[2])


def test_heat_elevation_ice_and_sea_lower_or_forbid_capacity():
    cov = _cov(5, temperature_c=np.array([10.0, 28.0, 10, 10, 10]), elevation=np.array([0, 0, 2000.0, 0, 0]),
               ice=np.array([0, 0, 0, 1, 0], bool), land=np.array([1, 1, 1, 1, 0], bool))
    ln = log_capacity(cov, density_terms(_culture(4)))
    assert ln[0] > ln[1] and ln[0] > ln[2] and np.isneginf(ln[3]) and np.isneginf(ln[4])
    colonist = log_capacity(cov, density_terms(_culture(5, "colonist")))
    assert np.isfinite(colonist[3]) and colonist[3] < colonist[0] and np.isneginf(colonist[4])


def test_clustering_keeps_the_total_and_concentrates_people():
    rng = np.random.default_rng(0)
    k = rng.uniform(0.5, 1.5, 200)
    kernel = sparse.diags([0.5, 1.0, 0.5], [-1, 0, 1], shape=(200, 200)).tocsr()
    np.testing.assert_allclose(cluster(k, kernel, 0.0), k)
    d = cluster(k, kernel, 0.4)
    assert d.sum() == pytest.approx(k.sum())
    assert np.std(d / d.mean()) > np.std(k / k.mean())


def test_given_population_is_placed_deterministically(earth):
    spec = _spec({"development": {"energy": "electric"}, "population": 1e9})
    a, b = build_civilisation(earth, spec), build_civilisation(earth, spec)
    radius_km = float(earth.surface.attrs["radius_m"]) / 1e3
    area = 4 * np.pi * radius_km**2 / earth.grid.size
    d = a.layers["population_density"].values[0]
    assert d.sum() * area == pytest.approx(1e9)
    np.testing.assert_array_equal(d, b.layers["population_density"].values[0])
    assert np.all(d[earth.surface["ocean"].values] == 0)
    s = a.layers["suitability"].values[0]
    assert s.max() == pytest.approx(1.0) and s.min() >= 0


def test_default_populations_follow_earth_at_each_energy_level(earth):
    got = {}
    for level in ("foraging", "farming", "electric"):
        civ = build_civilisation(earth, _spec({"development": {"energy": level}}))
        got[level] = civ.cultures[0].population
        assert got[level] == pytest.approx(h.POPULATION_ANCHOR[level], rel=0.5), level
    assert got["foraging"] < got["farming"] < got["electric"]


def test_homelands_confine_a_culture_and_homeless_cultures_divide_the_rest(earth):
    grid = earth.grid
    spec = _spec({"homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 1500}, "population": 1e7},
                 {"population": 1e7}, {"population": 1e7})
    civ = build_civilisation(earth, spec)
    d = civ.layers["population_density"].values
    centre = np.array([np.cos(np.radians(45)) * np.cos(np.radians(10)), np.cos(np.radians(45)) * np.sin(np.radians(10)),
                       np.sin(np.radians(45))])
    far = 6371 * np.arccos(np.clip(grid.points @ centre, -1, 1)) > 1500 * (1 + 4 * h.HOMELAND_EDGE)
    assert d[0][far].sum() < 0.01 * d[0].sum()
    assert d[1].sum() > 0 and d[2].sum() > 0
    assert not np.any((d[1] > 0) & (d[2] > 0))


def test_natives_need_land_life_but_colonists_do_not(earth):
    dead = World(spec=earth.spec, state=replace(earth.state, biosphere=replace(earth.state.biosphere, life="none")),
                 surface=earth.surface, timeline=earth.timeline)
    with pytest.raises(ValueError, match="land life"):
        build_civilisation(dead, _spec({}))
    civ = build_civilisation(dead, _spec({"inhabitants": "colonist", "colony": {"founded_years_ago": 100}}))
    c = civ.cultures[0]
    assert c.population == pytest.approx(h.COLONY_FOUNDERS * np.exp(h.COLONY_GROWTH_PER_YEAR * 100))
    assert len(c.landing_sites) == 1 and colony_age(c) == 100


def test_old_colonies_stop_at_the_land_capacity(earth):
    civ = build_civilisation(earth, _spec({"inhabitants": "colonist", "colony": {
        "founded_years_ago": 5000, "landing_sites": [{"lat_deg": 0, "lon_deg": 20}]}}))
    c = civ.cultures[0]
    assert c.population < h.COLONY_FOUNDERS * np.exp(h.COLONY_GROWTH_PER_YEAR * 5000)
    assert c.landing_sites == [{"lat_deg": 0, "lon_deg": 20}]
    assert c.population <= h.POPULATION_ANCHOR["fusion"] * 1.0
