"""Farming origins, growth of knowledge and spread of levels in the organic history (task 9.4)."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation
from worldgen.culture.development import ENERGY_LEVELS
from worldgen.culture.history.development import (advance_levels, capacity_gap, companion_levels, energy_rate,
                                                  knowledge_log)
from worldgen.culture.history.population import cell_capacity, front_speed_km_yr
from worldgen.culture.history.state import in_force
from worldgen.culture.spec import HistorySpec


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_level_in_force_rounds_half_up():
    assert list(in_force(np.array([0.0, 0.49, 0.5, 1.2, 2.5]))) == [0, 0, 1, 1, 3]


def test_capacity_is_the_anchor_at_integers_and_geometric_between():
    levels = np.array([[10.0, 0.0], [1000.0, 50.0], [4000.0, 80.0]])
    k = cell_capacity(levels, np.array([1.0, 1.0]))
    assert list(k) == [1000.0, 50.0]
    assert cell_capacity(levels, np.array([0.5, 0.5]))[0] == pytest.approx(100.0)
    assert cell_capacity(levels, np.array([0.5, 0.5]))[1] == 0.0          # no capacity below: lower anchor holds


def test_companion_levels_follow_earths_pairing():
    t, c = companion_levels(np.array([0.0, 1.0, 3.0, 4.0]))
    assert list(t) == [0.0, 2.0, 3.0, 4.0] and list(c) == [0.0, 1.0, 3.0, 4.0]


def test_knowledge_log_and_gaps_follow_the_anchors():
    anchors = np.log([h.POPULATION_ANCHOR[n] for n in ENERGY_LEVELS])
    assert knowledge_log(np.array([0.0, 2.0]))[1] == pytest.approx(anchors[2] - anchors[0])
    assert capacity_gap(np.array([1.3]))[0] == pytest.approx(anchors[2] - anchors[1])


def test_energy_rate_needs_farming_fuel_for_industry_and_scales_with_pace():
    energy = np.array([0.2, 1.0, 2.8, 2.8, h.HISTORY_MAX_ENERGY])
    sphere = np.full(5, 2e6)
    fuel = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    r = energy_rate(energy, sphere, fuel)
    assert r[0] == 0 and r[1] > 0 and r[2] == 0 and r[3] > 0 and r[4] == 0
    assert energy_rate(energy, sphere, fuel, 2.0)[1] == pytest.approx(2 * r[1])
    assert energy_rate(energy, sphere * 2, fuel)[1] == pytest.approx(2 * r[1])       # λ = 1


def test_pace_must_be_positive():
    with pytest.raises(ValidationError):
        HistorySpec(pace=0)
    assert HistorySpec().pace == 1.0


def test_levels_spread_along_a_chain_at_the_front_speed():
    n, km, years = 30, 100.0, 25.0

    class Links:
        rows = np.r_[np.arange(n - 1), np.arange(1, n)]
        cols = np.r_[np.arange(1, n), np.arange(n - 1)]
        flat_km = np.full(rows.size, km)
    level = np.zeros(n); level[0] = 1.2
    progress, credit = np.zeros(n), np.zeros(n)
    arrived = {}
    for step in range(400):
        level, progress, credit, new = advance_levels(level, progress, credit, Links, years)
        for c in new:
            arrived.setdefault(int(c), (step + 1) * years)
    assert np.all(level[1:] == 0.5)                                          # arrivals start at the bottom of the level
    t, _ = companion_levels(np.array([0.5]))
    factor = h.LAND_SPEED_FACTOR[in_force(t)[0]] / h.LAND_SPEED_FACTOR[0]
    cells = sorted(arrived)
    speed = (cells[-1] - cells[1]) * km / (arrived[cells[-1]] - arrived[cells[1]])
    assert speed == pytest.approx(front_speed_km_yr() * factor, rel=0.15)


def test_farming_begins_spreads_and_is_recorded(earth):
    spec = CultureSpec.model_validate({"history": {"start_years_ago": 6000, "epochs": [5000]}, "cultures": [
        {"name": "Oru", "mode": "organic"}]})
    civ = build_civilisation(earth, spec)
    origins = [e for e in civ.chronicle if e.kind == "farming_origin"]
    assert origins and all("cell" in e.refs for e in origins)
    level = civ.layers["development_level"].values[0, 0]
    early = civ.epochs[0].layers["development_level"].values[0, 0]
    assert (level >= h.FARMING_START_ENERGY).sum() > (early >= h.FARMING_START_ENERGY).sum()
    assert civ.cultures[0].development.energy >= 1                          # most people farm by now
    again = build_civilisation(earth, spec)
    assert np.array_equal(again.layers["development_level"].values, civ.layers["development_level"].values)
    assert [e.text for e in again.chronicle] == [e.text for e in civ.chronicle]
