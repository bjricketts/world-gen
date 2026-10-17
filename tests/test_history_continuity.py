"""Settlements, roads and trade carried through the kept epochs of an organic history (task 9.7)."""

from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation, load_civilisation, save_civilisation
from worldgen.culture.history.continuity import CellRecord, in_cell_order
from worldgen.culture.model import SettlementTable
from worldgen.culture.roads import build_network, gravity_pairs


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_in_cell_order_ranks_settlements_by_size_within_culture_and_cell():
    st = SettlementTable(id=np.arange(5), culture=[0, 0, 0, 1, 0], cell=[3, 3, 4, 3, 3],
                         population=[100.0, 500.0, 50.0, 10.0, 200.0])
    assert list(in_cell_order(st)) == [2, 0, 0, 0, 1]


class _State:
    def __init__(self, people, energy):
        self.active = np.array([True])
        self.population = np.array([people], dtype=float)
        self.development = np.array([[energy, energy, energy]], dtype=float)


def test_the_cell_record_dates_first_farming_and_people():
    rec = CellRecord(3)
    rec.update(_State([10, 0, 5], [0.0, 0.0, 1.0]), 1000.0)
    rec.update(_State([20, 30, 5], [1.0, 1.0, 1.0]), 950.0)        # under the record interval: dates only
    rec.update(_State([40, 60, 5], [1.0, 1.0, 1.0]), 900.0)
    rec.update(_State([80, 90, 5], [1.0, 1.0, 1.0]), 0.0)
    assert list(rec.settled_since) == [950.0, 950.0, 1000.0]
    assert rec.times == [1000.0, 900.0, 0.0]
    assert list(rec.people_at(500.0)) == [40, 60, 5] and list(rec.people_at(2000.0)) == [0, 0, 0]
    assert rec.passed(1, 50.0, 1000.0, 0.0) == 900.0
    assert rec.passed(1, 1000.0, 1000.0, 10.0) == 10.0


def test_gravity_falls_with_the_given_distance_exponent():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(30, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    w = rng.uniform(1, 10, 30)
    p2, g2 = gravity_pairs(pts, w, 6371.0, 1.0, 2.0)
    p1, g1 = gravity_pairs(pts, w, 6371.0, 1.0, 0.91)
    d = 6371.0 * np.arccos(np.clip((pts[p1[:, 0]] * pts[p1[:, 1]]).sum(1), -1, 1))
    assert np.allclose(g1, w[p1[:, 0]] * w[p1[:, 1]] / np.maximum(d, 1.0) ** 0.91)
    assert h.ROAD_DISTANCE_EXPONENT[2] == 1.9 and h.ROAD_DISTANCE_EXPONENT[3] == 0.91


@pytest.fixture(scope="module")
def organic(earth):
    """An organic Earth history with three earlier kept epochs."""
    spec = CultureSpec.model_validate({"history": {"epochs": [3000, 1500, 500]}, "cultures": [{"mode": "organic"}]})
    return build_civilisation(earth, spec)


def test_settlements_keep_ids_names_and_founding_dates(organic):
    civs = organic.epochs + [organic]
    start = organic.spec.history_or_default().start_years_ago
    known = {}
    carried = 0
    for civ in civs:
        st = civ.settlements
        assert len(np.unique(st.id)) == len(st)
        assert np.all(st.founded_years_ago >= civ.epoch_years_ago) and np.all(st.founded_years_ago <= start)
        for k in range(len(st)):
            entry = (int(st.cell[k]), str(st.name[k]), float(st.founded_years_ago[k]))
            if int(st.id[k]) in known:
                assert known[int(st.id[k])] == entry
                carried += 1
            known[int(st.id[k])] = entry
        for culture in set(st.culture.tolist()):
            names = st.name[st.culture == culture]
            assert len(set(names)) == len(names)
        ids = set(st.id.tolist()) | {-1}
        assert set(civ.roads.a.tolist()) <= ids and set(civ.trade.a.tolist()) <= ids
        assert all(p.capital in ids for p in civ.polities)
    assert carried > 0


def test_roads_are_dated_and_persist(organic):
    civs = organic.epochs + [organic]
    for civ in civs:
        r = civ.roads
        assert np.all(r.founded_years_ago >= civ.epoch_years_ago)
        assert np.all(r.founded_years_ago <= civs[0].epoch_years_ago)
    old = [r for r in civs[-1].roads.founded_years_ago if r > 0]
    assert old                                                # some present roads date from earlier epochs


def test_old_roads_are_reused(organic, earth):
    civ = organic
    people = civ.layers["population_density"].values * (4 * np.pi * 6371.0**2 / earth.grid.size)
    fresh, _ = build_network(earth, civ.settlements, civ.cultures, people)
    kept, _ = build_network(earth, civ.settlements, civ.cultures, people, organic.epochs[-1].roads)
    before = set(organic.epochs[-1].roads.path_cells.tolist())
    share = lambda r: np.isin(r.path_cells, list(before)).mean()
    assert share(kept) >= share(fresh)


def test_trade_links_are_routed_and_saved(organic, tmp_path):
    t = organic.trade
    assert len(t) > 0 and np.all(t.km > 0) and np.all(t.hours > 0) and np.all(t.flow > 0)
    back = load_civilisation(save_civilisation(organic, tmp_path / "civ"))
    assert len(back.trade) == len(t) and np.allclose(back.trade.flow, t.flow)
    assert np.array_equal(back.epochs[0].settlements.id, organic.epochs[0].settlements.id)
