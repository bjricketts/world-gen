"""Rulers, provinces, wars and alliances of states in the organic history."""

from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation, resolve_cultures
from worldgen.culture.build import geology_layers
from worldgen.culture.history.polities import Polities
from worldgen.culture.history.rulers import province_count, provinces, reign_years, succession_times
from worldgen.culture.history.run import HistoryContext, initial_state
from worldgen.culture.history.state import HistoryPolity
from worldgen.culture.history.suitability import build_suitability
from worldgen.culture.history.wars import key_pair, pair_key
from worldgen.culture.spec import HistorySpec
from worldgen.culture.suitability import covariates


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_reigns_follow_kokkonen_and_sundell():
    assert reign_years(False) == 11.4 and reign_years(True) == 20.9
    rng = np.random.default_rng(0)
    gaps = np.diff(np.r_[0.0, succession_times(rng, 11.4, 200000.0)])
    assert gaps.mean() == pytest.approx(11.4, rel=0.03)
    assert succession_times(rng, 1e9, 25.0).size == 0


def test_provinces_split_a_state_around_its_capital():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(400, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    cells = np.arange(400)
    prov = provinces(pts, cells, 17, 5)
    assert prov[17] == 0 and set(np.unique(prov)) == set(range(5))
    assert province_count(10, 1e5) == round(10 * 1e5 / h.PROVINCE_KM2) and province_count(1, 1e3) == 1
    assert province_count(3, 1e7) == 3


def test_pair_keys_are_unordered():
    assert int(pair_key(5, 3)) == int(pair_key(3, 5))
    assert key_pair(int(pair_key(7, 2))) == (2, 7)


def _two_states(earth, big=40, small=3, cultures=1):
    """Return a context and a state with two neighbouring states on a dense literate land: a big one and a small one."""
    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    suit = build_suitability(earth, cov, geo["resource_potential"])
    entries = [{"mode": "organic"} for _ in range(cultures)]
    ctx = HistoryContext(world=earth, cultures=resolve_cultures(CultureSpec.model_validate({"cultures": entries}), 0),
                         history=HistorySpec(start_years_ago=5000), seed=3, cov=cov, suit=suit)
    state = initial_state(ctx)
    for row in range(cultures):
        state.population[row] = np.where(cov.land, 50.0 * cov.cell_area_km2, 0.0)
        state.development[row] = np.array([1.0, 2.0, 2.0])[:, None]
    land = np.flatnonzero(cov.land)
    centre = land[np.argmax(earth.grid.points[land] @ earth.grid.points[land].mean(axis=0))]
    near = land[np.argsort(-(earth.grid.points[land] @ earth.grid.points[centre]))]
    a, b = near[:big], near[big:big + small]
    state.owner[a], state.owner[b] = 0, 1
    state.polities = [HistoryPolity(id=0, culture=ctx.cultures[0].id, capital=int(a[0]), founded_years_ago=5000),
                      HistoryPolity(id=1, culture=ctx.cultures[-1].id, capital=int(b[0]), founded_years_ago=5000)]
    return ctx, state


def _quiet(monkeypatch):
    """Switch off formation, expansion and break-up so only wars act."""
    for name, value in (("POLITY_FORM_RATE", 0.0), ("POLITY_ATTACK_RATE", 0.0), ("POLITY_BREAKUP_BASE", 0.0),
                        ("POLITY_BREAKUP_SIZE", 0.0)):
        monkeypatch.setattr(h, name, value)


def _run(ctx, state, steps, years=25.0):
    polities = Polities(ctx)
    for _ in range(steps):
        polities.step(ctx, state, years)
        state.years_ago -= years
        state.step += 1
    return polities


def test_a_stronger_state_makes_war_and_takes_land(earth, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 1.0)
    ctx, state = _two_states(earth)
    polities = _run(ctx, state, 4)
    wars = polities.wars.records
    assert wars and wars[0].states == 2 and sum(w.cells for w in wars) > 0
    assert (state.owner == 1).sum() < 3 or not state.polities[1].alive
    assert all(w.severity > 0 for w in wars)


def test_equal_states_rarely_attack(earth, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 1.0)
    ctx, state = _two_states(earth, big=10, small=10)
    polities = _run(ctx, state, 2)
    assert not polities.wars.records and not polities.wars.active     # below Cederman's 3:1 threshold


def test_without_onsets_there_is_peace(earth, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 0.0)
    ctx, state = _two_states(earth)
    before = state.owner.copy()
    polities = _run(ctx, state, 3)
    assert np.array_equal(state.owner, before) and not polities.wars.records


def test_wars_cross_cultures(earth, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 1.0)
    ctx, state = _two_states(earth, cultures=2)
    assert state.polities[0].culture != state.polities[1].culture
    _run(ctx, state, 4)
    assert (state.owner == 0).sum() > 40                      # the first culture's state rules the second's land


def test_threatened_neighbours_ally_and_join_wars(earth, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 0.0)
    monkeypatch.setattr(h, "ALLIANCE_RATE", 1e3)
    monkeypatch.setattr(h, "ALLIANCE_BREAK_AT_SUCCESSION", 0.0)
    ctx, state = _two_states(earth, big=60, small=2)
    nbrs = ctx.world.grid.neighbours
    free = np.flatnonzero((state.owner < 0) & ctx.cov.land)
    both = [c for c in free if (state.owner[nbrs[c].indices] == 0).any() and (state.owner[nbrs[c].indices] == 1).any()]
    if not both:
        pytest.skip("no free cell borders both states on this grid")
    state.owner[both[0]] = 2
    state.polities.append(HistoryPolity(id=2, culture=ctx.cultures[0].id, capital=int(both[0]), founded_years_ago=5000))
    polities = _run(ctx, state, 1)
    assert int(pair_key(1, 2)) in polities.wars.allies          # both border a state over twice as strong
    monkeypatch.setattr(h, "WAR_ONSET_RATE", 1.0)
    for _ in range(3):
        polities.step(ctx, state, 25.0)
        state.years_ago -= 25.0
        state.step += 1
    assert any(w.states >= 3 for w in polities.wars.records)


def test_epoch_states_carry_rulers(earth):
    spec = CultureSpec.model_validate({"history": {"epochs": [1000]}, "cultures": [{"mode": "organic"}]})
    civ = build_civilisation(earth, spec)
    states = [p for c in civ.epochs + [civ] for p in c.polities]
    assert states and all(p.rulers >= 1 and p.primogeniture is not None for p in states)
    assert {"war", "break_up"} & {e.kind for e in civ.chronicle}
