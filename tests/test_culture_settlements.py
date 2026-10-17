"""Settlement regions, rank-size systems and placement (task 8.4)."""

from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture.build import build_civilisation
from worldgen.culture.development import Development
from worldgen.culture.model import SETTLEMENT_FORMS, Culture
from worldgen.culture.settlements import (_place_region, largest_settlement, listing_cut, settlement_form,
                                          zipf_sizes, zipf_total)
from worldgen.culture.spec import CultureSpec
from worldgen.grid import build_grid


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def _spec(**culture):
    return CultureSpec(cultures=[culture])


@pytest.mark.parametrize("largest, zeta", [(1e4, 1.0), (3e6, 1.0), (3e6, 1.3), (5e5, 0.8)])
def test_zipf_total_matches_the_exact_sum(largest, zeta):
    sizes = zipf_sizes(largest, zeta, h.SETTLEMENT_SMALLEST)
    assert zipf_total(largest, zeta) == pytest.approx(sizes.sum(), rel=1e-3)
    assert largest_settlement(zipf_total(largest, zeta), zeta) == pytest.approx(largest, rel=1e-3)


def test_listing_cut_lists_about_the_target_count():
    largests = np.array([5e6, 1e6, 2e5])
    cut = listing_cut(largests, 1.0, 500)
    assert sum(len(zipf_sizes(L, 1.0, cut)) for L in largests) == pytest.approx(500, abs=3)
    assert listing_cut(largests, 1.0, 1e9) == h.SETTLEMENT_SMALLEST


def test_forms_follow_population_and_foragers_camp():
    farmers = Culture(id=0, name=None, inhabitants="native", development=Development(1, 1, 1))
    pops = np.array([60, 500, 20_000, 300_000, 3e6])
    names = [SETTLEMENT_FORMS[f] for f in settlement_form(pops, farmers)]
    assert names == ["hamlet", "village", "town", "city", "metropolis"]
    foragers = Culture(id=0, name=None, inhabitants="native", development=Development(0, 0, 0))
    assert {SETTLEMENT_FORMS[f] for f in settlement_form(pops, foragers)} == {"camp"}


def test_larger_settlements_keep_their_central_place_distance():
    grid = build_grid(10_000)
    cells = np.flatnonzero(grid.points[:, 2] > 0.5)               # a polar cap
    remaining = np.full(cells.size, 1e6)
    sizes = zipf_sizes(1e6, 1.0, 1e5)
    area = cells.size * 4 * np.pi * 6371.0**2 / grid.size
    where = _place_region(grid.points, cells, remaining, sizes, area, 6371.0, np.random.default_rng(0))
    p = grid.points[cells[where]]
    for r in range(1, len(where)):
        gap = 6371.0 * np.arccos(np.clip(p[:r] @ p[r], -1, 1)).min()
        assert gap >= h.SETTLEMENT_EXCLUSION * np.sqrt(area / (r + 1)) - 1e-6
    assert remaining.sum() == pytest.approx(cells.size * 1e6 - sizes.sum())


def test_listed_and_dispersed_people_add_up_and_regions_are_zipf(earth):
    civ = build_civilisation(earth, _spec(development={"energy": "fossil"}))
    radius_km = float(earth.surface.attrs["radius_m"]) / 1e3
    area = 4 * np.pi * radius_km**2 / earth.grid.size
    st = civ.settlements
    dispersed = civ.layers["dispersed_density"].values[0] * area
    assert np.all(dispersed >= -1e-6)
    assert st.population.sum() + dispersed.sum() == pytest.approx(civ.cultures[0].population, rel=1e-6)
    for k in np.unique(st.region)[:5]:
        pop = st.population[st.region == k]
        rank = st.rank[st.region == k]
        np.testing.assert_allclose(pop, pop[rank == 1][0] / rank)
    regions = civ.layers["region"].values[0]
    occupied = civ.layers["population_density"].values[0] > 0
    assert np.all((regions >= 0) == occupied)
    assert set(np.unique(st.region)) <= set(np.unique(regions[regions >= 0]))
    assert np.all(occupied[st.cell])


def test_placement_is_deterministic_and_uses_the_spec(earth):
    spec = _spec(development={"energy": "fossil"}, settlements={"zipf_exponent": 1.2, "largest_population": 2e6})
    a, b = build_civilisation(earth, spec), build_civilisation(earth, spec)
    np.testing.assert_array_equal(a.settlements.cell, b.settlements.cell)
    np.testing.assert_array_equal(a.settlements.lat_deg, b.settlements.lat_deg)
    assert a.settlements.population.max() == pytest.approx(2e6, rel=0.35)
    top = a.settlements.population[a.settlements.rank == 1][0]
    second = a.settlements.population[(a.settlements.rank == 2) & (a.settlements.region == a.settlements.region[0])][0]
    assert second == pytest.approx(top * 2 ** (-1 / 1.2))
    assert a.cultures[0].zipf_exponent == 1.2


def test_foragers_have_no_permanent_settlements(earth):
    civ = build_civilisation(earth, _spec(development={"energy": "foraging"}))
    assert len(civ.settlements) == 0
    np.testing.assert_allclose(civ.layers["dispersed_density"].values[0], civ.layers["population_density"].values[0])
