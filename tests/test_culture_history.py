"""Organic mode (milestone 9): history spec, step schedule, state, the history run, kept epochs and their storage."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from worldgen import heuristics as h
from worldgen.culture import CultureSpec, build_civilisation, load_civilisation, resolve_cultures, save_civilisation
from worldgen.culture.history import (HistoryState, next_date, run_history, scoped_at, step_years, stops)
from worldgen.culture.history.state import Chronicle
from worldgen.culture.io import epoch_folder_name
from worldgen.culture.spec import HistorySpec
from worldgen.culture.suitability import covariates

ORGANIC = {
    "history": {"start_years_ago": 3000, "epochs": [2000, 500]},
    "cultures": [
        {"name": "Oru", "mode": "organic"},
        {"name": "Settlers", "mode": "organic", "inhabitants": "colonist",
         "colony": {"founded_years_ago": 300, "landing_sites": [{"lat_deg": 45, "lon_deg": 10}]}},
        {"name": "Moderns", "development": {"energy": "fossil"}},
    ],
}


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world on the fast heuristic plate layout."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


@pytest.fixture(scope="module")
def organic_civ(earth):
    """An organic civilisation with two kept epochs, a colony and a scoped culture."""
    return build_civilisation(earth, CultureSpec.model_validate(ORGANIC))


# --- spec ---------------------------------------------------------------------

def test_mode_defaults_to_scoped_and_history_to_the_holocene():
    spec = CultureSpec.model_validate({"cultures": [{"name": "A"}]})
    assert spec.cultures[0].mode == "scoped" and not spec.organic
    assert spec.history_or_default().start_years_ago == h.HISTORY_START_YEARS_AGO


def test_epochs_are_sorted_oldest_first_and_kept_inside_the_span():
    assert HistorySpec(start_years_ago=5000, epochs=[100, 3000, 100]).epochs == [3000.0, 100.0]
    with pytest.raises(ValidationError):
        HistorySpec(start_years_ago=5000, epochs=[6000])


def test_organic_colony_cannot_land_before_the_history_starts():
    with pytest.raises(ValidationError):
        CultureSpec.model_validate({"history": {"start_years_ago": 1000}, "cultures": [
            {"mode": "organic", "inhabitants": "colonist", "colony": {"founded_years_ago": 2000}}]})


def test_organic_culture_warns_that_given_outcomes_are_ignored():
    with pytest.warns(UserWarning, match="population"):
        CultureSpec.model_validate({"cultures": [{"mode": "organic", "population": 1e6}]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CultureSpec.model_validate({"cultures": [{"mode": "organic"}]})
        CultureSpec.model_validate({"cultures": [{"mode": "organic", "inhabitants": "colonist",
                                                  "development": {"energy": "fusion"}}]})


def test_organic_natives_start_as_foragers_and_drop_given_outcomes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = CultureSpec.model_validate({"cultures": [
            {"mode": "organic", "development": {"energy": "electric"}, "population": 5e6,
             "polities": {"count": 4, "type": "empire"}}]})
    c = resolve_cultures(spec, 0)[0]
    assert c.mode == "organic" and c.development.to_names() == h.HISTORY_START_DEVELOPMENT
    assert c.population is None and c.polity_count is None and c.polity_type is None


# --- clock --------------------------------------------------------------------

def test_step_is_a_generation_then_shorter_from_fossil_energy():
    history = HistorySpec()
    assert step_years(history, 1) == h.HISTORY_STEP_YEARS
    assert step_years(history, h.HISTORY_LATE_FROM_ENERGY) == h.HISTORY_LATE_STEP_YEARS
    assert step_years(HistorySpec(step_years=10), 0) == 10


def test_steps_land_exactly_on_epochs_and_the_present():
    history = HistorySpec(start_years_ago=100, epochs=[37])
    dates = stops(history)
    assert dates == [37.0, 0.0]
    now, seen = 100.0, []
    while now > 0:
        now = next_date(now, 25.0, dates)
        seen.append(now)
    assert seen == [75.0, 50.0, 37.0, 12.0, 0.0]


# --- state and chronicle ------------------------------------------------------

def test_culture_development_is_the_level_most_people_reached():
    dev = np.zeros((1, 3, 4))
    dev[0, 0] = [0.5, 1.2, 1.9, 2.4]
    state = HistoryState(years_ago=0, step=0, culture_ids=[7], population=np.array([[10.0, 30.0, 20.0, 5.0]]),
                         development=dev, active=np.array([True]))
    assert state.culture_development(0).energy == 1      # 55 of 65 people have farming or better
    assert state.top_energy() == 2
    copy = state.copy()
    copy.population[0, 0] = 0
    assert state.population[0, 0] == 10


def test_chronicle_until_keeps_the_events_up_to_a_date():
    ch = Chronicle()
    ch.add(500, "a", "first", culture=1)
    ch.add(100, "b", "second")
    assert [e.kind for e in ch.until(200)] == ["a"] and ch.entries[0].refs == {"culture": 1}


# --- the history run and epochs ------------------------------------------------

def test_history_keeps_the_epochs_and_lands_colonists(earth):
    spec = CultureSpec.model_validate(ORGANIC)
    cultures = [c for c in resolve_cultures(spec, 0) if c.mode == "organic"]
    result = run_history(earth, cultures, spec.history, 0, covariates(earth))
    assert [s.years_ago for s in result.snapshots] == [2000.0, 500.0, 0.0]
    # generation steps until the fusion-age colony lands 300 years ago, then the late step
    assert result.steps == int(2700 / h.HISTORY_STEP_YEARS) + int(300 / h.HISTORY_LATE_STEP_YEARS)
    before, after = result.snapshots[1], result.snapshots[2]
    colony = before.index(1)
    assert not before.active[colony] and after.active[colony]
    assert after.population[colony].sum() == pytest.approx(h.COLONY_FOUNDERS, rel=1e-6)
    assert [e.kind for e in result.chronicle.entries] == ["origin", "landing", "colony_state"]


def test_history_is_deterministic(earth):
    spec = CultureSpec.model_validate(ORGANIC)
    runs = []
    for _ in range(2):
        cultures = [c for c in resolve_cultures(spec, 0) if c.mode == "organic"]
        runs.append(run_history(earth, cultures, spec.history, 0, covariates(earth)))
    for a, b in zip(*(r.snapshots for r in runs)):
        assert np.array_equal(a.population, b.population) and np.array_equal(a.development, b.development)


def test_scoped_cultures_appear_from_their_founding():
    from worldgen.culture.model import Culture
    from worldgen.culture.development import Development
    colony = Culture(id=0, name=None, inhabitants="colonist", development=Development(5, 5, 5), founded_years_ago=300)
    assert scoped_at(colony, 500) is None and scoped_at(colony, 100).founded_years_ago == 200
    native = Culture(id=1, name=None, inhabitants="native", development=Development(1, 1, 1))
    assert scoped_at(native, 100) is None and scoped_at(native, 0) is native


def test_organic_civilisation_carries_its_epochs(organic_civ):
    civ = organic_civ
    assert civ.epoch_years_ago == 0 and [e.epoch_years_ago for e in civ.epochs] == [2000.0, 500.0]
    assert [c.name for c in civ.epochs[0].cultures] == ["Oru"]
    assert {"Oru", "Settlers", "Moderns"} <= {c.name for c in civ.cultures}          # and overseas colonies
    assert civ.at_epoch(500) is civ.epochs[1]
    assert "development_level" in civ.layers and civ.layers["development_level"].dims == ("organic", "axis", "cell")
    oru = next(c for c in civ.cultures if c.name == "Oru")
    assert oru.mode == "organic" and oru.population > 0 and oru.phonology is not None
    assert oru.phonology == civ.epochs[0].cultures[0].phonology            # names follow one sound system throughout
    kinds = [e.kind for e in civ.chronicle]
    assert kinds[0] == "origin" and "landing" in kinds
    assert all(e.years_ago >= 2000 for e in civ.epochs[0].chronicle)
    assert len(civ.epochs[0].chronicle) <= len(civ.chronicle)


def test_epochs_round_trip(organic_civ, tmp_path):
    save_civilisation(organic_civ, tmp_path / "culture")
    assert (tmp_path / "culture" / "epochs" / epoch_folder_name(2000)).is_dir()
    back = load_civilisation(tmp_path / "culture")
    assert [e.epoch_years_ago for e in back.epochs] == [2000.0, 500.0]
    assert len(back.settlements) == len(organic_civ.settlements)
    assert [c.mode for c in back.cultures] == [c.mode for c in organic_civ.cultures]
    save_civilisation(back.epochs[1], tmp_path / "culture")                 # resaving drops stale epochs
    assert not (tmp_path / "culture" / "epochs").exists()


def test_epoch_folder_names_sort_by_date():
    names = [epoch_folder_name(y) for y in (12000, 500, 0.5)]
    assert names[0] == "000012000" and names == sorted(names, reverse=True)
