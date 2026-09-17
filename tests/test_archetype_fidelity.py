"""Milestone 1.1: drawn weathering target, habitable-zone-relative instellation, outcome constraints."""

from functools import lru_cache

import pytest

from worldgen import PlanetSpec, generate
from worldgen import constants as c
from worldgen import heuristics as h
from worldgen import orbit
from worldgen.priors import ARCHETYPES, resolve
from worldgen.priors.constraints import attempt_seed, failed_constraints
from worldgen.priors.distributions import HZFraction

SEEDS = range(40)


def _planets(archetype, **spec):
    """Return planets drawn from an archetype for each seed; plain draws are shared between tests."""
    if not spec:
        return _preset_planets(archetype)
    return [generate(PlanetSpec(seed=s, priors={"archetype": archetype}, **spec))[0] for s in SEEDS]


@lru_cache(maxsize=None)
def _preset_planets(archetype):
    return tuple(generate(PlanetSpec(seed=s, priors={"archetype": archetype}))[0] for s in SEEDS)


# Fix 1: weathering target -------------------------------------------------

@pytest.mark.slow
def test_weathering_target_is_drawn_within_range():
    lo, hi = h.WEATHERING_TARGET_RANGE_K
    temps = [p.atmosphere.surface_temperature_k for p in _planets("temperate")]
    assert all(lo - 1 <= t <= hi + 1 for t in temps)
    assert max(temps) - min(temps) > 20


def test_user_weathering_target_is_used():
    for p in _planets("temperate", atmosphere={"weathering_target_k": 300.0})[:10]:
        assert p.atmosphere.surface_temperature_k == pytest.approx(300.0, abs=0.5)
        assert p.provenance["atmosphere.weathering_target_k"] == "user"


# Tidal Q ------------------------------------------------------------------

@pytest.mark.slow
def test_tidal_q_drawn_for_rocky_planets():
    lo, hi = h.TIDAL_Q_ROCKY_RANGE
    qs = [p.orbit.tidal_q for p in _planets("small_world")]
    assert all(lo <= q <= hi for q in qs)
    assert len(set(qs)) > 1


def test_giants_use_giant_tidal_q():
    p, _ = generate(PlanetSpec(seed=1, priors={"archetype": "giant"}, body={"mass_mearth": 300.0}))
    assert p.orbit.tidal_q == h.TIDAL_Q_GIANT
    assert p.provenance["body.tidal_q"] == "default"


def test_lower_q_locks_sooner(earth_body, sun):
    fast = orbit.tidal_lock_time(0.5 * c.AU, sun.mass_kg, earth_body, 10.0)
    slow = orbit.tidal_lock_time(0.5 * c.AU, sun.mass_kg, earth_body, 1000.0)
    assert slow == pytest.approx(100 * fast)


# Fix 2: habitable-zone-relative instellation ------------------------------

def test_hz_fraction_round_trip(sun):
    for f in (-0.2, 0.0, 0.5, 1.0, 2.0):
        s = orbit.instellation_at_hz_fraction(sun.luminosity_w, sun.effective_temperature_k, 1.0, f)
        hz, _ = orbit.habitable_zone(sun.luminosity_w, sun.effective_temperature_k, 1.0, s)
        assert hz.conservative_fraction == pytest.approx(f, abs=1e-9)


def test_earth_position_in_hz(earth_spec):
    state, _ = generate(earth_spec)
    assert 0.0 < state.orbit.habitable_zone.conservative_fraction < 0.15


@pytest.mark.slow
@pytest.mark.parametrize("name", ["temperate", "ocean", "arid", "tidally_locked", "super_earth", "ice"])
def test_archetype_positions_follow_prior(name):
    dist = ARCHETYPES[name].priors["orbit.instellation_earth"]
    assert isinstance(dist, HZFraction)
    for p in _planets(name):
        assert dist.low - 1e-6 <= p.orbit.habitable_zone.conservative_fraction <= dist.high + 1e-6


def test_user_instellation_overrides_hz_prior():
    p, _ = generate(PlanetSpec(seed=2, priors={"archetype": "temperate", "enforce_constraints": False},
                               orbit={"instellation_earth": 1.0}))
    assert p.orbit.instellation_earth == pytest.approx(1.0)
    assert p.provenance["orbit.instellation_earth"] == "user"


# Fix 3: outcome constraints -----------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, a in ARCHETYPES.items() if a.constraints])
def test_constrained_archetypes_meet_constraints(name):
    for p in _planets(name):
        assert p.constraints_met, (name, p.seed, [c.description for c in failed_constraints(p)])
        assert not failed_constraints(p)


@pytest.mark.slow
def test_earth_like_archetypes_are_not_locked():
    for name in ("temperate", "ocean", "arid", "super_earth"):
        assert all(p.orbit.spin_state.startswith("free") for p in _planets(name))


def test_redraws_are_deterministic():
    a, _ = generate(PlanetSpec(seed=11, priors={"archetype": "ocean"}))
    b, _ = generate(PlanetSpec(seed=11, priors={"archetype": "ocean"}))
    assert a.to_dict() == b.to_dict()


def test_first_attempt_keeps_spec_seed():
    assert attempt_seed(42, 0) == 42
    assert attempt_seed(42, 1) != 42
    assert attempt_seed(42, 1) == attempt_seed(42, 1)


@pytest.mark.slow
def test_redrawn_planet_reports_draw_seed():
    redrawn = [p for p in _planets("ocean") if p.attempts > 1]
    assert redrawn
    p = redrawn[0]
    assert p.draw_seed == attempt_seed(p.seed, p.attempts - 1)
    assert any("redrawn" in i.message for i in p.issues)
    assert resolve(PlanetSpec(seed=p.seed, priors={"archetype": "ocean"}), p.draw_seed).values == p.inputs


def test_impossible_constraints_warn():
    spec = PlanetSpec(seed=0, priors={"archetype": "temperate", "max_attempts": 5},
                      star={"mass_msun": 0.1}, orbit={"instellation_earth": 1.0})
    p, _ = generate(spec)
    assert not p.constraints_met
    assert p.attempts == 5
    assert any(i.level == "warning" and "archetype" in i.message for i in p.issues)


def test_constraints_can_be_disabled():
    spec = PlanetSpec(seed=0, priors={"archetype": "temperate", "enforce_constraints": False},
                      star={"mass_msun": 0.1}, orbit={"instellation_earth": 1.0})
    p, _ = generate(spec)
    assert p.attempts == 1
    assert not p.constraints_met
    assert not any(i.level == "warning" and "archetype" in i.message for i in p.issues)


# Random streams -----------------------------------------------------------

def test_random_streams_are_independent_across_seeds_and_fields():
    import numpy as np

    from worldgen.util import named_rng

    names = ["a", "b", "atmosphere.weathering_target_k", "atmosphere.surface_pressure_bar"]
    x = np.array([[named_rng(s, n).random() for n in names] for s in range(1000)])
    assert np.abs(np.corrcoef(x.T) - np.eye(len(names))).max() < 0.1
    for j in range(len(names)):
        assert abs(np.corrcoef(x[:-1, j], x[1:, j])[0, 1]) < 0.1
