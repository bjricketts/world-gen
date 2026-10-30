"""Planet events in the organic history: eruptions, droughts, pandemics, impacts, sea level, stress and knowledge
loss."""

import math
from pathlib import Path

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, resolve_cultures
from worldgen.culture.history import events as ev
from worldgen.culture.spec import HistorySpec
from worldgen.culture.suitability import covariates


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def _context(earth, history=None, people=1e5, energy=2.0):
    """A context and state with one farming culture on all land."""
    from worldgen.culture.build import geology_layers
    from worldgen.culture.history.run import HistoryContext, initial_state
    from worldgen.culture.history.suitability import build_suitability

    geo = geology_layers(earth)
    cov = covariates(earth, geo["resource_potential"])
    spec = CultureSpec.model_validate({"cultures": [{"mode": "organic"}]})
    ctx = HistoryContext(world=earth, cultures=resolve_cultures(spec, 0), history=history or HistorySpec(),
                         seed=1, cov=cov, suit=build_suitability(earth, cov, geo["resource_potential"]))
    state = initial_state(ctx)
    state.population[:] = np.where(cov.land, people, 0.0)
    state.development[:] = energy
    state.active[:] = True
    return ctx, state


def test_eruption_sizes_follow_the_fitted_pareto_law():
    rng = np.random.default_rng(0)
    x = ev.pareto_draw(rng, h.VOLCANIC_SULFUR_MIN_TG, h.VOLCANIC_SULFUR_SLOPE, 1e9, 400_000)
    rate = h.VOLCANIC_EVENTS_PER_YEAR * 100
    assert rate * np.mean(x >= 28.1) == pytest.approx(0.25, rel=0.05)          # 6 per 2400 years at Tambora or more
    assert rate * np.mean(x >= 59.0) == pytest.approx(0.083, rel=0.1)         # 2 per 2400 years at 1257 or more
    assert ev.eruption_cooling_k(28.1) == pytest.approx(1.1)


def test_pandemic_intensities_follow_marani():
    mu, sd, xi = h.PANDEMIC_GPD
    rng = np.random.default_rng(1)
    draws = np.array([ev.gpd_draw_between(rng, 1e-3, np.inf, mu, sd, xi) for _ in range(20000)])
    assert draws.min() >= 1e-3
    assert max(ev.gpd_draw_between(rng, 1e-3, 3e-2, mu, sd, xi) for _ in range(2000)) <= 3e-2
    share = ev.gpd_exceedance(5.7e-3, mu, sd, xi) / ev.gpd_exceedance(1e-3, mu, sd, xi)
    assert np.mean(draws > 5.7e-3) == pytest.approx(share, rel=0.1)
    # Spanish-flu intensity (5.7 ‰/yr) once per ~250 years at the 1600–1945 mean rate of events above μ
    assert 1.0 / (h.PANDEMIC_RATE * ev.gpd_exceedance(5.7e-3, mu, sd, xi)) == pytest.approx(248, rel=0.05)


def test_an_eruption_cools_its_hemisphere_or_both_from_the_tropics(earth):
    ctx, state = _context(earth)
    e = ev.Events(ctx)
    north = int(np.argmax(e.lat))
    tropic = int(np.argmin(np.abs(e.lat)))
    e.eruption(ctx, 100.0, north, 59.0, False)
    e.eruption(ctx, 100.0, tropic, 10.0, False)
    big, small = e.shocks
    assert np.all(e.lat[big.cells] > 0) and big.cells.size < small.cells.size == ctx.cov.land.sum()
    assert big.start - big.end == h.VOLCANIC_LONG_YEARS and small.start - small.end == h.VOLCANIC_YEARS
    assert big.sigma[0] == pytest.approx(ev.eruption_cooling_k(59.0) / h.SUMMER_TEMPERATURE_SD_K)
    assert any(c.kind == "eruption" for c in ctx.chronicle.entries)            # above Tambora


def test_stress_counts_the_share_of_the_step_a_shock_lasts(earth):
    ctx, state = _context(earth)
    e = ev.Events(ctx)
    e.eruption(ctx, 110.0, int(np.argmin(np.abs(e.lat))), 10.0, False)       # five years from 110 years ago
    s = e.stress(125.0, 100.0)
    land = np.flatnonzero(ctx.cov.land)
    sigma = ev.eruption_cooling_k(10.0) / h.SUMMER_TEMPERATURE_SD_K
    assert s.sigma[land[0]] == pytest.approx(0.2 * sigma)
    assert s.factor[land[0]] == pytest.approx(1 - 0.2 * h.CAPACITY_LOSS_PER_SIGMA * sigma)
    assert e.stress(100.0, 75.0).sigma.max() == 0.0


def test_a_drought_costs_water_limited_land_most(earth):
    ctx, state = _context(earth)
    e = ev.Events(ctx)
    e.drought(ctx, 50.0, int(np.flatnonzero(ctx.cov.land)[0]), 20000.0, 100.0, 0.4, False)
    shock = e.shocks[0]
    assert np.allclose(shock.loss, 0.4 * e.water_limit[shock.cells])
    assert np.allclose(shock.sigma, 0.4 / h.RAIN_CV)


def test_a_pandemic_kills_its_intensity_among_the_people_it_reaches(earth, monkeypatch):
    monkeypatch.setattr(h, "PANDEMIC_SPREAD_SD", 0.0)
    ctx, state = _context(earth)
    e = ev.Events(ctx)
    before = state.population.sum()
    e.pandemic(ctx, state, np.random.default_rng(0), int(np.flatnonzero(ctx.cov.land)[100]), 1e-3, 3.0, True)
    killed = before - state.population.sum()
    assert killed == pytest.approx(1e-3 * 3.0 * before, rel=1e-6)
    ctx, state = _context(earth)
    e.pandemic(ctx, state, np.random.default_rng(0), int(np.flatnonzero(ctx.cov.land)[100]), 1.0, 3.0, True)
    assert state.population[0][ctx.cov.land].min() >= (1 - h.PANDEMIC_MAX_DEATH_SHARE) * 1e5 - 1e-6


def test_an_impact_devastates_the_land_it_hits(earth):
    ctx, state = _context(earth)
    e = ev.Events(ctx)
    cell = int(np.flatnonzero(ctx.cov.land)[10])
    e.impact(ctx, state, cell, 10.0, True)
    assert state.population[0, cell] == pytest.approx(1e5 * (1 - h.IMPACT_AREA_KM2 / e.cell_km2))
    assert ev.impact_rate(10.0, 1e12) == pytest.approx(1 / 1000, rel=0.3)      # Tunguska every ~1000 years


def test_a_sea_level_rise_floods_low_land_for_good(earth):
    ctx, state = _context(earth)
    e = ev.Events(ctx)
    e.sea_level(ctx, state, 100.0, 50.0)
    low = ctx.cov.land & (e.elevation < 50.0)
    assert low.any() and np.all(e.stress(100.0, 75.0).factor[low] == 0.0)


def test_stress_raises_war_onset_and_break_up(earth):
    from worldgen.culture.history.polities import Polities
    from worldgen.culture.history.state import HistoryPolity

    ctx, state = _context(earth)
    p = Polities(ctx)
    land = np.flatnonzero(ctx.cov.land)
    state.owner[land[:50]] = 0
    state.polities = [HistoryPolity(id=0, culture=0, capital=int(land[0]), founded_years_ago=500)]
    assert p._shocked(state)[0] == 1.0
    sigma = np.zeros(ctx.cov.land.size)
    sigma[land[:50]] = 2.0
    p.stress = ev.Stress(sigma, np.ones_like(sigma))
    assert p._shocked(state)[0] == pytest.approx(1 + 2 * h.CONFLICT_PER_SIGMA)
    keys = np.array([5, 5, 9])
    factor = p.wars._conflict(np.array([land[0], land[60], land[70]]), np.array([land[1], land[61], land[71]]),
                              keys, np.array([5, 9]))
    assert factor == pytest.approx([1 + 2 * h.CONFLICT_PER_SIGMA, 1.0])


def test_a_small_isolated_people_loses_knowledge(earth):
    from worldgen.culture.history.development import Development

    ctx, state = _context(earth, people=0.0, energy=3.0)
    cell = int(np.flatnonzero(ctx.cov.land)[0])
    state.population[0, cell] = 1000.0
    d = Development(ctx)
    sphere = d.sphere_people(state, 0)
    d._lose(ctx, state, 0, sphere, 50.0)
    deficit = math.log(h.KNOWLEDGE_LOSS_PEOPLE[3] / sphere[cell])
    assert state.development[0, 0, cell] == pytest.approx(3.0 - h.KNOWLEDGE_LOSS_RATE * deficit * 50.0)
    d._lose(ctx, state, 0, sphere, 1e6)
    assert state.development[0, 0, cell] == h.FARMING_START_ENERGY
    assert any(e.kind == "knowledge_loss" for e in ctx.chronicle.entries)


def test_scripted_events_happen_when_the_spec_says(earth):
    from worldgen.culture.history.run import default_processes, run_history

    spec = CultureSpec.model_validate({"history": {"start_years_ago": 600, "planet_events": False, "events": [
        {"kind": "eruption", "years_ago": 300, "size": 30},
        {"kind": "drought", "years_ago": 200, "lat_deg": 45, "lon_deg": 10, "size": 0.4, "duration_years": 50},
        {"kind": "sea_level", "years_ago": 100, "size": 5}]},
        "cultures": [{"mode": "organic", "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}}]})
    result = run_history(earth, resolve_cultures(spec, 0), spec.history, 0, covariates(earth), default_processes())
    got = [(e.kind, e.years_ago) for e in result.chronicle.entries if e.kind in ("eruption", "drought", "sea_level",
                                                                                "impact", "pandemic")]
    assert got == [("eruption", 300.0), ("drought", 200.0), ("sea_level", 100.0)]


def test_event_specs_are_checked():
    with pytest.raises(ValueError):
        HistorySpec(events=[{"kind": "drought", "years_ago": 10}])
    with pytest.raises(ValueError):
        HistorySpec(start_years_ago=100, events=[{"kind": "impact", "years_ago": 500}])
    with pytest.raises(ValueError):
        HistorySpec(events=[{"kind": "sea_level", "years_ago": 10}])
