"""Colonists in the organic history: competition, faster frontiers, colonial states, immigration and contact epidemics (task 9.8)."""

import math
from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, resolve_cultures
from worldgen.culture.history.population import advance_front, front_speed_km_yr, grow, landmasses
from worldgen.culture.history.run import default_processes, run_history
from worldgen.culture.spec import HistorySpec
from worldgen.culture.suitability import covariates


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_the_culture_with_more_capacity_displaces_the_other():
    P = np.array([[90.0], [10.0]])
    K = np.array([[100.0], [1000.0]])
    for _ in range(40):
        P = grow(P, K, 0.03, 25.0)
    assert P[0, 0] < 1.0 and P[1, 0] == pytest.approx(1000.0, rel=0.01)
    alone = grow(np.array([[10.0]]), np.array([[100.0]]), 0.03, 50.0)[0, 0]
    e = math.exp(1.5)
    assert alone == pytest.approx(100 * 10 * e / (100 + 10 * (e - 1)))


def test_a_faster_front_crosses_more_links():
    class Links:
        rows = np.r_[np.arange(59), np.arange(1, 60)]
        cols = np.r_[np.arange(1, 60), np.arange(59)]
        flat_km = np.full(118, 100.0)
    K = np.full(60, 1000.0)
    reached = []
    for factor in (1.0, 5.0):
        P = np.zeros(60); P[0] = 1000.0
        progress, credit = np.zeros(60), np.zeros(60)
        for _ in range(40):
            P, progress, credit, new, _ = advance_front(P, K, progress, credit, Links, 25.0, factor)
            P = grow(P[None], K[None], h.SPREAD_GROWTH_RATE, 25.0)[0]
        reached.append(int((P > 0).sum()))
    assert reached[1] > 3 * reached[0]
    assert reached[0] == pytest.approx(1 + 40 * 25 * front_speed_km_yr() / 100, abs=2)


def _history(earth, herds: float = 0.0, **colony):
    """Run natives and a colony from off the planet that carries crowd diseases (unless told otherwise); natives'
    farming origins keep herd animals with probability ``herds``."""
    colony = {"crowd_diseases": True, **colony}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(h, "LIVESTOCK_P_CLOSED", herds)
        mp.setattr(h, "LIVESTOCK_P_OPEN", herds)
        return _run(earth, colony)


def _run(earth, colony):
    spec = CultureSpec.model_validate({"history": {"start_years_ago": 3000, "epochs": [300, 200, 150]}, "cultures": [
        {"name": "Natives", "mode": "organic", "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}},
        {"name": "Settlers", "mode": "organic", "inhabitants": "colonist",
         "development": {"energy": "fossil", "transport": "rail_and_steam", "communication": "telegraph"},
         "colony": {"founded_years_ago": 300, "landing_sites": [{"lat_deg": 45, "lon_deg": 10}], **colony}}]})
    cultures = resolve_cultures(spec, 0)
    return run_history(earth, cultures, spec.history, 0, covariates(earth), default_processes())


@pytest.fixture(scope="module")
def colony(earth):
    """A colony landing among natives 300 years ago."""
    return _history(earth)


def test_a_colony_founds_its_state_at_landing(colony):
    landed = [e for e in colony.chronicle.entries
              if e.kind == "colony_state" and e.refs["culture"] == colony.cultures[1].id]
    assert len(landed) == 1
    snap = colony.snapshots[-1]
    states = [p for p in snap.polities if p.culture == colony.cultures[1].id]
    assert states and states[0].founded_years_ago == 300.0


def test_contact_epidemics_strike_the_natives_of_the_landmass(colony, earth, monkeypatch):
    assert any(e.kind == "epidemic" for e in colony.chronicle.entries)
    monkeypatch.setattr(h, "EPIDEMIC_DEATH_SHARE", 0.0)
    healthy = _history(earth)
    mass = landmasses(earth)
    home = mass[np.argmax(colony.snapshots[-1].population[1])]
    sick = colony.snapshots[2].population[0][mass == home].sum()            # 150 years ago, 150 after landing
    well = healthy.snapshots[2].population[0][mass == home].sum()
    assert sick < 0.7 * well


def test_no_great_dying_without_herds_or_against_herders(colony, earth):
    from worldgen.culture.history.livestock import Livestock

    settlers = colony.cultures[1].id
    assert any(e.kind == "epidemic" and e.refs.get("culture") == settlers for e in colony.chronicle.entries)
    harmless, herders = _history(earth, crowd_diseases=False), _history(earth, herds=1.0)
    assert not any(e.kind == "epidemic" and e.refs.get("culture") == settlers for e in harmless.chronicle.entries)
    origins = [e for e in herders.chronicle.entries if e.kind == "farming_origin"]
    assert origins and all(e.refs["livestock"] == 1 for e in origins)
    mass = landmasses(earth)
    home = mass[np.argmax(colony.snapshots[-1].population[1])]
    farmed = herders.snapshots[2].population[0][mass == home].sum()
    struck = colony.snapshots[2].population[0][mass == home].sum()
    assert farmed > struck                                                  # herders' descendants withstand them


def test_immigrants_arrive_for_the_given_years(earth):
    base = _history(earth).snapshots
    more = _history(earth, immigrants_per_year=1e4, immigration_years=100).snapshots
    assert more[2].population[1].sum() > base[2].population[1].sum() + 5e5


def test_immigration_is_for_organic_colonies_only():
    with pytest.warns(UserWarning, match="apply to organic colonies only"):
        CultureSpec.model_validate({"cultures": [{"inhabitants": "colonist",
                                                  "colony": {"founded_years_ago": 100, "immigrants_per_year": 10}}]})


def test_a_region_where_another_culture_is_the_majority_breaks_away(earth):
    from worldgen.culture.build import geology_layers
    from worldgen.culture.history.polities import Polities
    from worldgen.culture.history.run import HistoryContext, initial_state
    from worldgen.culture.history.state import HistoryPolity
    from worldgen.culture.history.suitability import build_suitability

    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    spec = CultureSpec.model_validate({"cultures": [{"mode": "organic"}, {"mode": "organic"}]})
    ctx = HistoryContext(world=earth, cultures=resolve_cultures(spec, 0), history=HistorySpec(start_years_ago=500),
                         seed=1, cov=cov, suit=build_suitability(earth, cov, geo["resource_potential"]))
    state = initial_state(ctx)
    land = np.flatnonzero(cov.land)
    centre = land[0]
    near = land[np.argsort(-(earth.grid.points[land] @ earth.grid.points[centre]))][:40]
    state.population[:] = 0.0
    state.population[0, near] = 1e5
    state.development[:] = 2.0
    minority = near[30:]                                      # the far rim of the state
    state.population[1, minority] = 1e6
    state.owner[near] = 0
    state.polities = [HistoryPolity(id=0, culture=ctx.cultures[0].id, capital=int(near[0]), founded_years_ago=500)]
    Polities(ctx)._secessions(ctx, state)
    broke = [p for p in state.polities[1:] if p.culture == ctx.cultures[1].id]
    assert broke and all(p.parent == 0 for p in broke)
    ids = {p.id for p in broke}
    assert np.isin(state.owner[minority], list(ids)).mean() >= 0.5         # rim cells in groups of 3 or more
    assert np.all(state.owner[near[:30]] == 0)


def test_natives_do_not_grow_while_the_epidemic_runs(colony, earth):
    mass = landmasses(earth)
    at, after = colony.snapshots[0], colony.snapshots[1]                     # landing, 100 years later
    assert at.years_ago == 300.0 and after.years_ago == 200.0
    home = mass[np.argmax(after.population[1])]
    near = (mass == home) & (np.arccos(np.clip(earth.grid.points @ earth.grid.points[np.argmax(after.population[1])],
                                               -1, 1)) * 6371.0 < 1000.0)
    assert after.population[0][near].sum() <= 0.15 * at.population[0][near].sum()     # 90% in a century


def _pair(earth, levels=(1.0, 3.0)):
    """A context and state with two cultures sharing a patch of land."""
    from worldgen.culture.build import geology_layers
    from worldgen.culture.history.run import HistoryContext, initial_state
    from worldgen.culture.history.suitability import build_suitability

    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    spec = CultureSpec.model_validate({"cultures": [{"mode": "organic"}, {"mode": "organic"}]})
    ctx = HistoryContext(world=earth, cultures=resolve_cultures(spec, 0), history=HistorySpec(start_years_ago=500),
                         seed=1, cov=cov, suit=build_suitability(earth, cov, geo["resource_potential"]))
    state = initial_state(ctx)
    land = np.flatnonzero(cov.land)[:20]
    state.population[:] = 0.0
    state.population[0, land] = 8e5
    state.population[1, land] = 2e5
    state.development[0] = levels[0]
    state.development[1] = levels[1]
    return ctx, state, land


def test_language_shift_follows_abrams_and_strogatz(earth):
    from worldgen.culture.history.contact import Contact

    ctx, state, land = _pair(earth)
    before = state.population[:, land].sum(axis=0)
    Contact(ctx).shift(ctx, state, 25.0)
    assert np.allclose(state.population[:, land].sum(axis=0), before)
    x, a, c, s = 0.2, h.LANGUAGE_SHIFT_EXPONENT, h.LANGUAGE_SHIFT_RATE, h.LANGUAGE_STATUS   # culture 1 leads
    for _ in range(25):
        x += c * ((1 - x) * x**a * s - x * (1 - x)**a * (1 - s))
    assert state.population[1, land[0]] / before[0] == pytest.approx(x, rel=1e-6)
    ctx, state, land = _pair(earth, (2.0, 2.0))
    Contact(ctx).shift(ctx, state, 25.0)
    assert state.population[0, land[0]] > 8e5                    # equals: the majority gains


def test_technology_passes_only_under_rule(earth):
    from worldgen.culture.history.contact import Contact
    from worldgen.culture.history.state import HistoryPolity

    ctx, state, land = _pair(earth)
    ruled, free = land[:10], land[10:]
    state.owner[ruled] = 0
    state.polities = [HistoryPolity(id=0, culture=ctx.cultures[1].id, capital=int(ruled[0]), founded_years_ago=500)]
    Contact(ctx).transfer(ctx, state, np.random.default_rng(0), 5000.0)
    assert np.all(state.development[0, 0, ruled] == 2.5) and np.all(state.development[0, 0, free] == 1.0)


def test_a_colony_is_tied_to_its_home_power_until_independence(earth):
    colony = _history(earth, independence_years=50)
    kinds = [(e.kind, e.years_ago) for e in colony.chronicle.entries]
    assert ("independence", 250.0) in kinds
    grown = _history(earth)
    freed = [e for e in grown.chronicle.entries if e.kind == "independence"]
    for e in freed:
        assert "strong" in e.text


def _ruled_pair(earth, supremacist):
    """Two cultures sharing land ruled by a state of the leading culture, with supremacy set."""
    from worldgen.culture.history.contact import Contact
    from worldgen.culture.history.state import HistoryPolity

    ctx, state, land = _pair(earth)
    state.owner[land] = 0
    state.polities = [HistoryPolity(id=0, culture=ctx.cultures[1].id, capital=int(land[0]), founded_years_ago=500)]
    contact = Contact(ctx)
    contact.supremacist[(ctx.cultures[1].id, ctx.cultures[0].id)] = supremacist
    return ctx, state, land, contact


def test_rulers_without_supremacy_make_bilinguals_who_become_a_creole(earth):
    ctx, state, land, contact = _ruled_pair(earth, False)
    before = state.population[:, land].sum()
    contact.shift(ctx, state, 25.0)
    creole = [c for c in ctx.cultures if c.creole_of]
    assert len(creole) == 1 and creole[0].creole_of == [0, 1] and creole[0].inhabitants == "native"
    assert state.population[:, land].sum() == pytest.approx(before)
    x, y, a, c, k, s = 0.8, 0.2, h.LANGUAGE_SHIFT_EXPONENT, h.LANGUAGE_SHIFT_RATE, h.BILINGUAL_EASE, h.LANGUAGE_STATUS
    for _ in range(25):                                         # Mira & Paredes (2005) eqs. 1-5; culture 1 leads
        b = 1 - x - y
        pxy, pxb = c * (1 - k) * s * (1 - x) ** a, c * k * s * (1 - x) ** a
        pyx, pyb = c * (1 - k) * (1 - s) * (1 - y) ** a, c * k * (1 - s) * (1 - y) ** a
        x, y = x + y * pyx + b * pyx - x * (pxy + pxb), y + x * pxy + b * pxy - y * (pyx + pyb)
    assert state.population[2, land[0]] / 1e6 == pytest.approx(1 - x - y, rel=1e-6)
    assert np.all(state.development[2][:, land] == 3.0)         # the higher level of its parents
    assert any(e.kind == "creole" for e in ctx.chronicle.entries)


def test_supremacist_rulers_make_no_bilinguals(earth):
    ctx, state, land, contact = _ruled_pair(earth, True)
    contact.shift(ctx, state, 25.0)
    assert len(ctx.cultures) == 2 and not contact.pools
    assert state.population[:, land].sum() == pytest.approx(20 * 1e6)


def test_settler_mortality_follows_malaria_suitability():
    from types import SimpleNamespace

    from worldgen.culture.history.colonists import settler_mortality

    wet = SimpleNamespace(values=np.array([2.0, 2.0, 2.0, 0.1]))
    world = SimpleNamespace(surface={"precipitation": wet, "potential_evaporation": SimpleNamespace(values=np.ones(4))})
    cov = SimpleNamespace(temperature_c=np.array([5.0, 25.0, 40.0, 25.0]))
    m = settler_mortality(world, cov)
    assert m == pytest.approx([h.SETTLER_MORTALITY_MIN, h.SETTLER_MORTALITY_MAX, h.SETTLER_MORTALITY_MIN,
                               h.SETTLER_MORTALITY_MIN])


def test_independence_follows_strang(earth):
    from dataclasses import replace

    from worldgen.culture.history.colonists import independence_rate

    ctx, state, land = _pair(earth)
    home = ctx.cultures[0]
    ctx.cultures += [replace(home, id=2, inhabitants="colonist", home_culture=0),
                     replace(home, id=3, inhabitants="colonist", home_culture=2),
                     replace(home, id=4, inhabitants="colonist", home_culture=1)]
    assert independence_rate(ctx, ctx.cultures[2], 460.0) == pytest.approx(math.exp(-11.61 + 0.012 * 460))
    ctx.cache["independent"] = {3, 4}                           # one in its empire (via culture 2), one outside
    assert independence_rate(ctx, ctx.cultures[2], 460.0) == pytest.approx(
        math.exp(-11.61 + 0.012 * 460 + 0.030 + 0.014))


def test_a_seafaring_state_founds_a_colony_overseas(earth):
    from worldgen.culture.history.overseas import Overseas
    from worldgen.culture.history.state import HistoryPolity

    ctx, state, land = _pair(earth, (1.0, 3.0))
    over = Overseas(ctx)
    ctx.cache["overseas"] = over
    capital = int(np.flatnonzero(over.coast & over.large)[0])
    state.population[1, capital] = 1e6
    state.owner[capital] = 0
    state.polities = [HistoryPolity(id=0, culture=ctx.cultures[1].id, capital=capital, founded_years_ago=500)]
    over.step(ctx, state, 1e4)
    new = ctx.cultures[-1]
    assert len(ctx.cultures) == 3 and new.inhabitants == "colonist" and new.home_culture == ctx.cultures[1].id
    where = np.flatnonzero(state.population[2] > 0)
    assert where.size == 1 and over.mass[where[0]] != over.mass[capital]
    assert state.population[2, where[0]] == pytest.approx(min(h.COLONY_FOUNDERS, 5e5))
    assert ctx.cache["homes"][new.id] == 0 and len(state.polities) == 2


def test_bilinguals_left_outside_rule_return_to_their_cultures(earth):
    ctx, state, land, contact = _ruled_pair(earth, False)
    key = (ctx.cultures[0].id, ctx.cultures[1].id)
    contact.pools[key] = np.zeros(state.population.shape[1])
    free = np.flatnonzero(ctx.cov.land)[-1]
    contact.pools[key][free] = 100.0
    before = state.population[:, free].copy()
    contact.shift(ctx, state, 1.0)
    assert state.population[:2, free].sum() == pytest.approx(before[:2].sum() + 100.0)
    assert state.population[2:, free].sum() == 0.0 and all(pool[free] == 0.0 for pool in contact.pools.values())


def test_a_culture_joins_at_most_one_creole(earth, monkeypatch):
    monkeypatch.setattr(h, "CREOLE_CONTACT", "once")
    ctx, state, land, contact = _ruled_pair(earth, False)
    assert contact._may_mix(ctx, state, {}, 0, 1)
    contact.creoles[(0, 7)] = 8                                    # culture 0 already has a creole with culture 7
    assert not contact._may_mix(ctx, state, {}, 0, 1)
    monkeypatch.setattr(h, "CREOLE_CONTACT", "colonial")
    assert not contact._may_mix(ctx, state, {}, 0, 1)             # two natives


def test_a_homes_own_colonies_do_not_spare_its_natives_from_foreign_diseases(earth):
    from dataclasses import replace

    from worldgen.culture.history.colonists import Epidemics

    ctx, state, land = _pair(earth)
    native = ctx.cultures[0]
    ctx.cultures[1] = replace(ctx.cultures[1], inhabitants="colonist", home_culture=native.id)   # its own colony
    ep = Epidemics(ctx)
    ep.contact = {0: ep.contact.get(0, np.full(state.population.shape[1], np.nan))}
    state.population[1] = 0.0
    state.population[1, land[0]] = 100.0
    ep.step(ctx, state, 25.0)
    assert np.all(np.isnan(ep.contact[0][land]))                 # the home's own pool: no epidemic
    from worldgen.culture.history.run import add_culture
    foreign = replace(ctx.cultures[1], id=7, home_culture=None, name="Others", crowd_diseases=True)
    row = add_culture(ctx, state, foreign, state.development[1].copy())
    state.population[row, land[5]] = 100.0
    ep.step(ctx, state, 25.0)
    same = land[ep.mass[land] == ep.mass[land[5]]]
    assert np.all(np.isfinite(ep.contact[0][same]))              # a foreign pool brings its epidemic
