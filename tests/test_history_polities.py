"""States in the organic history: formation, warfare, break-up and their epochs (task 9.5)."""

from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation, resolve_cultures
from worldgen.culture.build import geology_layers
from worldgen.culture.history.polities import (Polities, effective_number_with_subpolities, subpolity_counts)
from worldgen.culture.history.rulers import province_count
from worldgen.culture.history.run import HistoryContext, initial_state
from worldgen.culture.history.suitability import build_suitability
from worldgen.culture.spec import HistorySpec
from worldgen.culture.suitability import covariates


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_subpolity_counts_follow_bands_and_chiefdoms():
    people = np.array([10.0, 10.0, 10.0, 0.0])
    owner = np.array([-1, -1, 3, -1])
    energy = np.array([0.0, 1.0, 1.0, 1.0])
    counts = subpolity_counts(people, owner, energy, 2e4)
    assert list(counts) == [pytest.approx(2e4 / h.SUBPOLITY_FORAGING_KM2), 2e4 / h.SUBPOLITY_FARMING_KM2, 0, 0]


def test_effective_number_counts_states_and_subpolities():
    owner = np.array([0, 0, 1, -1])
    people = np.ones(4)
    energy = np.ones(4)
    area = 1e4                                        # one chiefdom in the free cell
    areas = np.array([2e4, 1e4, 1e4])
    expected = areas.sum() ** 2 / (areas**2).sum()
    assert effective_number_with_subpolities(owner, people, energy, area) == pytest.approx(expected)


def _setup(earth, communication=2.0):
    """Return a context and a state with dense literate people on every land cell."""
    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    suit = build_suitability(earth, cov, geo["resource_potential"])
    cultures = resolve_cultures(CultureSpec.model_validate({"cultures": [{"mode": "organic"}]}), 0)
    ctx = HistoryContext(world=earth, cultures=cultures, history=HistorySpec(start_years_ago=5000), seed=3, cov=cov,
                         suit=suit)
    state = initial_state(ctx)
    state.population[0] = np.where(cov.land, 50.0 * cov.cell_area_km2, 0.0)
    state.development[0] = np.array([1.0, 2.0, communication])[:, None]
    return ctx, state


def _run(ctx, state, steps, years=25.0):
    polities = Polities(ctx)
    for _ in range(steps):
        polities.step(ctx, state, years)
        state.years_ago -= years
        state.step += 1


def test_states_form_only_with_writing(earth):
    ctx, state = _setup(earth, communication=0.0)
    _run(ctx, state, 20)
    assert not state.polities and np.all(state.owner < 0)
    ctx, state = _setup(earth)
    _run(ctx, state, 20)
    assert state.polities
    owned = state.owner >= 0
    assert np.all(state.population[0][owned] > 0)
    alive = [p for p in state.polities if p.alive]
    assert all(state.owner[p.capital] == p.id for p in alive)


def test_warfare_without_break_up_grows_states(earth, monkeypatch):
    monkeypatch.setattr(h, "POLITY_BREAKUP_BASE", 0.0)
    monkeypatch.setattr(h, "POLITY_BREAKUP_SIZE", 0.0)
    monkeypatch.setattr(h, "POLITY_FORM_RATE", 0.002)
    ctx, state = _setup(earth)
    _run(ctx, state, 1)
    founded = len(state.polities)
    assert 0 < founded <= int((state.owner >= 0).sum())
    monkeypatch.setattr(h, "POLITY_FORM_RATE", 0.0)
    _run(ctx, state, 40)
    sizes = np.bincount(state.owner[state.owner >= 0], minlength=founded)
    assert sizes.sum() > 5 * founded and sizes.max() > 10
    for p in state.polities:                                     # conquered states end, survivors hold cells
        assert p.alive == bool(sizes[p.id] > 0) if p.id < sizes.size else not p.alive
    assert all(p.peak_km2 > 0 for p in state.polities if p.alive)


def test_a_break_up_at_a_succession_splits_a_state_into_its_provinces(earth, monkeypatch):
    monkeypatch.setattr(h, "POLITY_FORM_RATE", 0.002)
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 0.0)
    ctx, state = _setup(earth)
    monkeypatch.setattr(h, "POLITY_BREAKUP_BASE", 0.0)
    monkeypatch.setattr(h, "POLITY_BREAKUP_SIZE", 0.0)
    _run(ctx, state, 1)
    monkeypatch.setattr(h, "POLITY_FORM_RATE", 0.0)
    _run(ctx, state, 30)
    before = {p.id: np.flatnonzero(state.owner == p.id) for p in state.polities if p.alive}
    capitals = {p.id: p.capital for p in state.polities if p.alive}
    monkeypatch.setattr(h, "POLITY_BREAKUP_BASE", 0.999)
    monkeypatch.setattr(h, "RULER_REIGN_YEARS", 0.5)
    monkeypatch.setattr(h, "RULER_REIGN_PRIMOGENITURE_YEARS", 0.5)
    monkeypatch.setattr(h, "PRIMOGENITURE_BREAKUP_CUT", 0.0)
    monkeypatch.setattr(h, "POLITY_BREAKUP_COHESION", 0.0)
    monkeypatch.setattr(h, "POLITY_ATTACK_RATE", 0.0)
    count = len(state.polities)
    _run(ctx, state, 1)
    heirs = state.polities[count:]
    broken = [i for i in before if not state.polities[i].alive]
    assert len(broken) >= 0.9 * len(before)                      # 0.999 per 25 years
    for i in broken:
        cells = before[i]
        parts = [p for p in heirs if p.parent == i]
        assert len(parts) == province_count(cells.size, ctx.cov.cell_area_km2)
        assert set(np.unique(state.owner[cells])) == {p.id for p in parts}
        assert parts[0].capital == capitals[i] and state.owner[capitals[i]] == parts[0].id


def test_polity_steps_are_deterministic(earth):
    runs = []
    for _ in range(2):
        ctx, state = _setup(earth)
        _run(ctx, state, 30)
        runs.append((state.owner.copy(), [(p.capital, p.ended_years_ago) for p in state.polities]))
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]


@pytest.fixture(scope="module")
def organic(earth):
    """An organic Earth history from the start of the Holocene, with two earlier epochs."""
    spec = CultureSpec.model_validate({"history": {"epochs": [2000, 500]}, "cultures": [{"mode": "organic"}]})
    return build_civilisation(earth, spec)


def test_epochs_keep_the_historys_states(organic):
    civs = organic.epochs + [organic]
    assert any(civ.polities for civ in civs)
    seen = {}
    for civ in civs:
        owner = civ.layers["polity"].values.max(axis=0)             # one state per cell, over all cultures
        ids = {p.id for p in civ.polities}
        assert set(np.unique(owner[owner >= 0])) == ids
        on = civ.settlements.polity
        assert np.array_equal(on, owner[civ.settlements.cell])
        counts = civ.layers["subpolity_count"].values[0]
        assert np.all(counts[owner >= 0] == 0)
        for p in civ.polities:
            assert p.founded_years_ago >= civ.epoch_years_ago
            if p.id in seen:
                assert seen[p.id] == (p.founded_years_ago, p.parent)
            seen[p.id] = (p.founded_years_ago, p.parent)
    kinds = {e.kind for e in organic.chronicle}
    assert "first_state" in kinds
    assert sum(e.kind == "first_state" and e.refs.get("culture") == 0 for e in organic.chronicle) == 1
