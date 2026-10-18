"""Livestock, crowd-disease carriers, colonies passed to heirs and colonists farming on habitable worlds (task 9.14)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, resolve_cultures
from worldgen.culture.history.colonists import heir_state
from worldgen.culture.history.livestock import Livestock
from worldgen.culture.history.population import in_habitats
from worldgen.culture.history.run import HistoryContext, initial_state
from worldgen.culture.history.state import HistoryPolity
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


def _context(earth, colony=None):
    from worldgen.culture.build import geology_layers

    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    entries = [{"mode": "organic"}]
    if colony is not None:
        entries.append({"mode": "organic", "inhabitants": "colonist", "colony": colony})
    cultures = resolve_cultures(CultureSpec.model_validate({"cultures": entries}), 0)
    ctx = HistoryContext(world=earth, cultures=cultures, history=HistorySpec(start_years_ago=5000), seed=1, cov=cov,
                         suit=build_suitability(earth, cov, geo["resource_potential"]))
    return ctx, initial_state(ctx)


def test_origins_draw_herds_more_often_on_open_land(earth, monkeypatch):
    ctx, state = _context(earth)
    herds = Livestock(ctx)
    land = np.flatnonzero(ctx.cov.land)
    shares = np.array([herds.open_share(int(c)) for c in land[::50]])
    assert np.all((shares >= 0) & (shares <= 1)) and shares.max() > shares.min()
    monkeypatch.setattr(h, "LIVESTOCK_P_CLOSED", 0.0)
    monkeypatch.setattr(h, "LIVESTOCK_P_OPEN", 1.0)
    rng = np.random.default_rng(0)
    cells = land[::50]
    open_, closed = cells[shares > 0.6], cells[shares < 0.1]
    if not open_.size or not closed.size:
        pytest.skip("no contrast in open land on this world")
    assert np.mean([herds.originate(state, int(c), rng) for c in open_]) > np.mean([herds.originate(state, int(c), rng)
                                                                           for c in closed])


def test_herds_spread_through_farmland_but_not_where_crops_fail(earth):
    ctx, state = _context(earth)
    herds = Livestock(ctx)
    land = np.flatnonzero(ctx.cov.land)
    state.population[0, land] = 100.0
    state.development[0, 0] = 1.0                       # farming everywhere, with wheel and sail
    state.development[0, 1] = 2.0
    start = int(land[np.argmax(herds.arable[land])])
    state.livestock[start] = 1
    for step in range(200):
        state.step = step
        herds.step(ctx, state, 25.0)
    assert (state.livestock > 0).sum() > 10
    barren = land[herds.arable[land] <= 0]
    assert not (state.livestock[barren] > 0).any()


def test_who_carries_crowd_diseases(earth):
    ctx, state = _context(earth, {"founded_years_ago": 100})
    herds = Livestock(ctx)
    land = np.flatnonzero(ctx.cov.land)
    state.population[0, land] = 10.0
    assert herds.carrier(ctx, state, ctx.cultures[1].id) == 0.0     # from off the planet, by default no
    ctx.cultures[1] = replace(ctx.cultures[1], crowd_diseases=True)
    assert herds.carrier(ctx, state, ctx.cultures[1].id) == 1.0
    assert herds.carrier(ctx, state, ctx.cultures[0].id) == 0.0     # natives without herds
    state.livestock[land] = 1
    assert herds.carrier(ctx, state, ctx.cultures[0].id) == 1.0
    state.livestock[land[:5]] = 2                               # brought by settlers
    assert not herds.immune(state)[land[:5]].any() and herds.immune(state)[land[5:]].all()
    assert herds.carrier(ctx, state, ctx.cultures[0].id) == 1.0


def test_a_colony_passes_to_the_heir_of_a_broken_home_state():
    polities = [HistoryPolity(id=0, culture=0, capital=5, founded_years_ago=500.0, ended_years_ago=200.0),
                HistoryPolity(id=1, culture=0, capital=5, founded_years_ago=200.0, parent=0, ended_years_ago=100.0),
                HistoryPolity(id=2, culture=0, capital=9, founded_years_ago=200.0, parent=0),
                HistoryPolity(id=3, culture=0, capital=5, founded_years_ago=100.0, parent=1),
                HistoryPolity(id=4, culture=0, capital=7, founded_years_ago=300.0, ended_years_ago=250.0)]
    state = SimpleNamespace(polities=polities)
    assert heir_state(state, 0) == 3                    # heir of the heir, holding the old capital
    assert heir_state(state, 4) == -1                   # conquered: no heir
    assert heir_state(state, 2) == 2


def test_colonists_farm_on_worlds_habitable_unaided(earth):
    colonist = SimpleNamespace(inhabitants="colonist")
    assert earth.state.occupiability.habitable_unaided and not in_habitats(colonist, earth)
    assert in_habitats(colonist, None) and not in_habitats(SimpleNamespace(inhabitants="native"), earth)
    hostile = SimpleNamespace(state=SimpleNamespace(occupiability=SimpleNamespace(habitable_unaided=False)))
    assert in_habitats(colonist, hostile)


def test_epochs_keep_the_livestock(earth):
    from worldgen.culture import build_civilisation

    spec = CultureSpec.model_validate({"history": {"start_years_ago": 8000, "epochs": [4000]},
                                       "cultures": [{"name": "Oru", "mode": "organic"}]})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(h, "LIVESTOCK_P_CLOSED", 1.0)
        mp.setattr(h, "LIVESTOCK_P_OPEN", 1.0)
        civ = build_civilisation(earth, spec)
    now, then = civ.layers["livestock"].values, civ.epochs[0].layers["livestock"].values
    assert now.dtype == np.int8 and (now > 0).sum() >= (then > 0).sum() > 0
    herds = [e for e in civ.chronicle if e.kind == "farming_origin"]
    assert herds and all(now[e.refs["cell"]] > 0 for e in herds)
