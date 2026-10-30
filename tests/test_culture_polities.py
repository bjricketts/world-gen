"""Capitals, XTENT territories, polity counts and types."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from worldgen import heuristics as h
from worldgen.culture.build import build_civilisation
from worldgen.culture.development import Development
from worldgen.culture.model import Culture
from worldgen.culture.polities import _inside, choose_capitals, effective_number, influence_owner, polity_type
from worldgen.culture.spec import CultureSpec
from worldgen.grid import build_grid


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_effective_number_counts_equal_polities_and_discounts_small_ones():
    assert effective_number(np.full(7, 3.0)) == pytest.approx(7)
    assert effective_number(np.array([10.0, 0.1, 0.1])) == pytest.approx(10.2**2 / 100.02)


def test_capitals_are_the_largest_settlements_kept_apart():
    grid = build_grid(2000)
    cells = np.arange(200)
    pop = np.linspace(200, 1, 200)
    caps = choose_capitals(cells, pop, grid.points, 5, 4 * np.pi * 6371.0**2 * 0.1, 6371.0, fixed=(150,))
    assert caps[0] == 150 and 0 in caps and len(caps) == 5
    p = grid.points[cells[caps]]
    gap = 0.5 * np.sqrt(4 * np.pi * 6371.0**2 * 0.1 / 5)
    d = 6371.0 * np.arccos(np.clip(p @ p.T, -1, 1)) + np.eye(5) * 1e9
    assert d[1:, 1:].min() >= gap - 1e-6


def test_influence_goes_to_the_strongest_reachable_capital():
    # a line of 10 cells, 1 h apart; capitals at 0 and 9, the one at 9 with 3.5 h more influence (no cell is equidistant)
    n = 10
    rows = np.r_[np.arange(n - 1), np.arange(1, n)]
    cols = np.r_[np.arange(1, n), np.arange(n - 1)]
    m = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    owner = influence_owner(m, np.array([0, 9]), np.array([0.0, 3.5]))
    assert list(owner) == [0, 0, 0] + [1] * 7
    cut = sparse.csr_matrix((np.ones(2), ([0, 1], [1, 0])), shape=(n, n))
    assert list(influence_owner(cut, np.array([0]), np.array([0.0]))) == [0, 0] + [-1] * 8


def test_polygon_test_and_types():
    lat, lon = np.array([10.0, 30.0]), np.array([10.0, 10.0])
    assert list(_inside(lat, lon, [(0, 0), (20, 0), (20, 20), (0, 20)])) == [True, False]
    farm = Culture(id=0, name=None, inhabitants="native", development=Development(1, 1, 1))
    assert polity_type(farm, 2e6) == "empire" and polity_type(farm, 1e5) == "kingdom"
    modern = Culture(id=0, name=None, inhabitants="native", development=Development(4, 4, 4))
    assert polity_type(modern, 2e6) == "federation"
    colony = Culture(id=0, name=None, inhabitants="colonist", development=Development(5, 5, 5))
    assert polity_type(colony, 1.0) == "colony"


def test_polities_cover_the_culture_and_match_the_effective_number(earth):
    civ = build_civilisation(earth, CultureSpec(cultures=[{"development": {"energy": "fossil"}}]))
    owner = civ.layers["polity"].values[0]
    occupied = civ.layers["population_density"].values[0] > 0
    assert np.all((owner >= 0) == occupied)
    areas = np.array([p.area_km2 for p in civ.polities])
    target = h.POLITY_EFFECTIVE_NUMBER[3] * areas.sum() / h.EARTH_DRY_LAND_KM2
    assert effective_number(areas) == pytest.approx(max(target, 1), rel=0.35)
    st = civ.settlements
    np.testing.assert_array_equal(st.polity, owner[st.cell])
    for p in civ.polities:
        assert st.polity[p.capital] == p.id


def test_spec_counts_capitals_and_territories_are_kept(earth):
    spec = CultureSpec(cultures=[{"development": {"energy": "fossil"}, "polities": {
        "count": 6, "type": "kingdom", "capitals": [{"lat_deg": 45, "lon_deg": 5}],
        "territories": [[(40, 0), (50, 0), (50, 10), (40, 10)]]}}])
    civ = build_civilisation(earth, spec)
    assert len(civ.polities) == 6 and {p.type for p in civ.polities} == {"kingdom"}
    grid = earth.grid
    owner = civ.layers["polity"].values[0]
    box = (grid.lat > 41) & (grid.lat < 49) & (grid.lon > 1) & (grid.lon < 9) & (owner >= 0)
    if box.any():
        assert np.all(owner[box] == 0)


def test_foragers_have_no_polities(earth):
    civ = build_civilisation(earth, CultureSpec(cultures=[{"development": {"energy": "foraging"}}]))
    assert civ.polities == [] and np.all(civ.layers["polity"].values == -1)
