"""Travel costs, gravity links, routing and road classes (task 8.5)."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from worldgen import heuristics as h
from worldgen.culture.build import build_civilisation
from worldgen.culture.model import ROAD_KINDS
from worldgen.culture.roads import TravelGraph, _astar, classify, gravity_pairs, tobler_kmh
from worldgen.culture.spec import CultureSpec
from worldgen.grid import build_grid


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_tobler_peaks_at_6_kmh_slightly_downhill():
    # Goodchild (2020), eq. 1: 6 km/h at a slope of -0.05.
    assert tobler_kmh(np.array(-0.05)) == pytest.approx(6.0)
    assert tobler_kmh(np.array(0.2)) < tobler_kmh(np.array(0.0)) < 6.0


def test_boats_open_the_sea_only_from_pack_and_boat(earth):
    foot, boats = TravelGraph(earth, 0), TravelGraph(earth, 1)
    assert not foot.usable[foot.water_edge].any() and foot.usable[~foot.water_edge].all()
    assert boats.usable.all()
    sea = boats.water_edge & boats.water_cell[boats.row] & boats.water_cell[boats.col]
    np.testing.assert_allclose(boats.base_hours[sea], boats.km[sea] / h.WATER_SPEED_KMH[1])
    ids = boats.undirected
    assert np.all(ids[ids] == ids)                       # each undirected id points at itself


def test_astar_finds_the_least_time_path():
    grid = build_grid(4000)
    m = grid.neighbours.copy().tocsr()
    rng = np.random.default_rng(1)
    m.data = m.data * 6371.0 / rng.uniform(1, 6, m.data.size)          # hours at 1–6 km/h
    m.sort_indices()
    n = grid.size
    best, came, stamp = np.zeros(n), np.zeros(n, np.int64), np.zeros(n, np.int64)
    for k, (a, b) in enumerate([(0, 1500), (10, 3900), (2000, 2100)]):
        path = _astar(m.indptr, m.indices, m.data, grid.points, 6371.0, 6.0, a, b, 1e9, best, came, stamp, k + 1)
        cost = sum(m[path[i], path[i + 1]] for i in range(path.size - 1))
        assert path[0] == a and path[-1] == b
        assert cost == pytest.approx(dijkstra(m, indices=a)[b])
    none = _astar(m.indptr, m.indices, m.data, grid.points, 6371.0, 6.0, 0, 3900, 1.0, best, came, stamp, 99)
    assert none.size == 0


def test_gravity_pairs_prefer_big_near_partners_within_reach():
    grid = build_grid(2000)
    pts = grid.points[:200]
    weight = np.ones(200)
    weight[0] = 1e6
    pairs, g = gravity_pairs(pts, weight, 6371.0, 100.0)
    assert np.all(np.diff(g) <= 0)
    assert np.any(pairs == 0) and np.all(pairs[:, 0] < pairs[:, 1])
    d = 6371.0 * np.arccos(np.clip(np.einsum("ij,ij->i", pts[pairs[:, 0]], pts[pairs[:, 1]]), -1, 1))
    assert d.max() <= h.ROAD_MAX_LINK_KM


def test_classes_fill_from_the_busiest_road_with_doubling_lengths():
    rng = np.random.default_rng(0)
    flow, length = rng.uniform(0, 1, 3000), np.full(3000, 10.0)
    sea = np.zeros(3000, dtype=bool)
    sea[:100] = True
    kinds = classify(flow, length, sea, 4)
    assert np.all(kinds[:100] == ROAD_KINDS.index("sea_route"))
    km = [length[kinds == ROAD_KINDS.index(k)].sum() for k in ("highway", "trunk", "road", "track")]
    assert km[1] / km[0] == pytest.approx(2, rel=0.1) and km[3] / km[2] == pytest.approx(2, rel=0.1)
    top = kinds == ROAD_KINDS.index("highway")
    assert flow[top].min() >= flow[~top & ~sea].max()
    assert set(np.unique(classify(flow, length, sea, 0))) <= {ROAD_KINDS.index("track"), ROAD_KINDS.index("sea_route")}


def test_road_network_is_contiguous_and_joins_settlements(earth):
    civ = build_civilisation(earth, CultureSpec(cultures=[{"development": {"energy": "fossil"}}]))
    roads, st = civ.roads, civ.settlements
    assert len(roads) > 0
    adj = earth.grid.neighbours
    for i in range(0, len(roads), max(1, len(roads) // 200)):
        p = roads.path(i)
        assert p.size >= 2 and np.all(np.asarray(adj[p[:-1], p[1:]]).ravel() > 0)
        assert roads.length_km[i] > 0
    ends = np.r_[roads.a, roads.b]
    linked = np.isin(st.id, ends[ends >= 0])
    assert linked.mean() > 0.3
    assert np.all(np.isin(ends[ends >= 0], st.id))
    again = build_civilisation(earth, CultureSpec(cultures=[{"development": {"energy": "fossil"}}]))
    np.testing.assert_array_equal(roads.path_cells, again.roads.path_cells)


def test_foragers_build_no_roads(earth):
    civ = build_civilisation(earth, CultureSpec(cultures=[{"development": {"energy": "foraging"}}]))
    assert len(civ.roads) == 0


def test_roads_reach_populated_cells_without_settlements(earth):
    civ = build_civilisation(earth, CultureSpec(cultures=[{"development": {"energy": "electric"}}]))
    area = 4 * np.pi * (float(earth.surface.attrs["radius_m"]) / 1e3) ** 2 / earth.grid.size
    people = civ.layers["population_density"].values[0] * area
    rural = people >= h.ROAD_RURAL_MIN_PEOPLE
    rural[civ.settlements.cell] = False
    assert rural.sum() > 100
    touched = np.zeros(earth.grid.size, dtype=bool)
    touched[civ.roads.path_cells] = True
    assert touched[rural].mean() > 0.8
