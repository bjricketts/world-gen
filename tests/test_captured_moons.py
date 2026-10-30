"""Captured moons and impact systems of giants."""

import math

import numpy as np
import pytest

from worldgen import constants as c
from worldgen import generate, generate_world, heuristics as h
from worldgen.giants import moon_spec
from worldgen.giants.moons import (capture_mass_limit, capture_survival, draw_moons, survival_among_moons,
                                   tidal_decay, tidal_reach)
from worldgen.priors import resolve
from worldgen.report import format_report
from worldgen.spec import PlanetSpec
from worldgen.state import PlanetState

from tests.test_moon_systems import giant


def draws(name, n=200, a=None):
    """Return the drawn moons and regimes of ``n`` seeds for a giant analogue."""
    state = giant(name, a=a)
    sy = state.system
    return state, [draw_moons(name, state.star, state.orbit, state.bulk, sy.limits, sy.ring_regime, 0.3, s)
                   for s in range(n)]


def test_capture_limit_matches_williams_and_zugger():
    """An Earth at 1 AU splitting a binary with a 0.005 M⊕ companion at 10 R⊕ and 1.5 km/s keeps just under a Moon."""
    gm = c.G * c.M_EARTH
    q = 10 * c.R_EARTH
    apo = 0.5 * c.AU * (c.M_EARTH / (3 * c.M_SUN)) ** (1 / 3)
    v_enc = math.sqrt(1500.0**2 + 2 * gm / q)
    v_peri = math.sqrt(gm * (2 / q - 2 / (q + apo)))
    limit = capture_mass_limit(c.M_EARTH, q, v_enc, v_peri, 0.005 * c.M_EARTH) / c.M_EARTH
    assert 0.007 < limit < 0.0123
    assert capture_mass_limit(c.M_EARTH, q, v_enc, v_peri, 0.05 * c.M_EARTH) > 10 * limit * c.M_EARTH


def test_survival_curves_pass_through_their_papers():
    """Porter & Grundy's shares by star and Rufu & Canup's by mass ratio, clipped to [0, 1] beyond."""
    for star, share in h.CAPTURE_SURVIVAL_BY_STAR:
        assert capture_survival(star * c.M_SUN) == pytest.approx(share)
    for ratio, share in h.CAPTURE_AMONG_MOONS:
        assert survival_among_moons(ratio) == pytest.approx(share)
    assert survival_among_moons(0.1) == 0.0 and survival_among_moons(100.0) == 1.0


def test_retrograde_decay_undoes_prograde_recession():
    """Tides carry a prograde moon out and a retrograde one in along the same a^(13/2) law."""
    bulk = giant("Neptune").bulk
    a0 = 15 * bulk.radius_m
    out = tidal_reach(2e-4, bulk, 1e4, 0.3, 4.5e9 * c.SECONDS_PER_YEAR, a0)
    assert tidal_decay(2e-4, bulk, 1e4, 0.3, 4.5e9 * c.SECONDS_PER_YEAR, out) == pytest.approx(a0, rel=1e-6)
    assert tidal_decay(2e-4, bulk, 1e4, 0.3, 4.5e9 * c.SECONDS_PER_YEAR, 1.5 * bulk.radius_m) == 0.0


def test_ice_giants_capture_tritons_and_gas_giants_keep_their_moons():
    """A quarter of ice giants keep a captured Triton-class moon; Jupiter and Saturn analogues keep none."""
    for name in ("Uranus", "Neptune"):
        _, out = draws(name)
        captured = [m for moons, _, _ in out for m in moons if m.origin == "captured"]
        assert len(captured) / len(out) == pytest.approx(0.25, abs=0.08)
        assert all(1e-4 * 0.99 < m.mass_kg / c.M_EARTH < 1e-2 for m in captured)
        assert 0.3 < np.mean([m.retrograde for m in captured]) < 0.8
        retro = [m for m in captured if m.retrograde]
        assert all(m.inclination_rad > math.pi / 2 for m in retro)
    for name in ("Jupiter", "Saturn"):
        _, out = draws(name, 100)
        assert not any(m.origin == "captured" for moons, _, _ in out for m in moons)
        assert all("capture:" in " ".join(i.message for i in issues) for _, _, issues in out)


def test_half_of_ice_giants_have_a_uranian_impact_system():
    """Uranian systems hold ~1e-4 of the planet's mass, heavier outward, with no ring-born chain beside them."""
    state, out = draws("Uranus")
    uranian = [moons for moons, regime, _ in out if regime == "uranian"]
    assert len(uranian) / len(out) == pytest.approx(h.URANIAN_IMPACT_PROBABILITY, abs=0.08)
    for moons in uranian:
        assert not any(m.origin == "ring" for m in moons)
        impact = [m for m in moons if m.origin == "impact"]
        if len(impact) == 5:
            total = sum(m.mass_kg for m in impact) / state.bulk.mass_kg
            assert 3e-5 < total < 3e-4
            assert impact[0].mass_kg < min(impact[3].mass_kg, impact[4].mass_kg)


def test_a_surviving_capture_clears_the_regular_moons():
    """A capture around a Jupiter at 1.5 AU removes the regular moons, keeps the ring-born ones, and is reported."""
    state, out = draws("Jupiter", 400, a=1.5)
    hit = [(moons, issues) for moons, _, issues in out if any(m.origin == "captured" for m in moons)]
    assert hit
    for moons, issues in hit:
        assert not any(m.origin == "regular" for m in moons)
        text = " ".join(i.message for i in issues)
        assert "MJ/kg" in text and "regular moon(s)" in text
    kept = [moons for moons, _, _ in out if not any(m.origin == "captured" for m in moons)]
    assert all(any(m.origin == "regular" for m in moons) for moons in kept)


def test_captured_moons_report_and_become_worlds():
    """A drawn captured moon is reported as captured and its world's parent records the origin."""
    world = generate_world(PlanetSpec.model_validate({
        "name": "Uranus", "seed": 4, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
        "orbit": {"semi_major_axis_au": 19.2, "rotation_period_h": 17.24},
        "body": {"mass_mearth": 14.5, "radius_rearth": 4.01, "envelope_mass_fraction": 0.15}}), "preview")
    moons = world.state.system.moons
    k = next(i for i, m in enumerate(moons) if m.origin == "captured")
    assert "captured" in format_report(world.state)
    again = PlanetState.from_dict(world.state.to_dict())
    assert again.system.moons[k].retrograde == moons[k].retrograde
    moon, _ = generate(moon_spec(world, k))
    assert moon.parent.moon_origin == "captured"


def test_archetype_captured_moons_and_impact_refusal():
    """Captured habitable moons are half retrograde, with the retrograde stable limit; impact origins are refused."""
    picked = [resolve(PlanetSpec.model_validate({"name": "M", "seed": s, "priors": {"archetype": "habitable_moon"},
                                                 "parent": {"moon_origin": "captured"}})) for s in range(200)]
    tilts = np.array([r.get("parent.inclination_deg") for r in picked])
    assert np.mean(tilts > 90) == pytest.approx(h.CAPTURE_RETROGRADE_SHARE, abs=0.08)
    assert all(t <= h.CAPTURED_TILT_DEG[1] or t >= 180 - h.CAPTURED_TILT_DEG[1] for t in tilts)
    base = {"name": "M", "seed": 1, "star": {"mass_msun": 1.0}, "orbit": {"semi_major_axis_au": 1.0},
            "body": {"mass_mearth": 0.5}, "parent": {"mass_mearth": 300.0, "semi_major_axis_rp": 20.0}}
    pro, _ = generate(PlanetSpec.model_validate(base))
    retro, _ = generate(PlanetSpec.model_validate({**base, "parent": {**base["parent"], "inclination_deg": 160.0}}))
    assert retro.parent.stable_limit_m / pro.parent.stable_limit_m == pytest.approx(
        h.MOON_STABLE_FRACTION_RETROGRADE / h.MOON_STABLE_FRACTION_PROGRADE)
    with pytest.raises(ValueError, match="Woo et al. 2021"):
        resolve(PlanetSpec.model_validate({"name": "M", "seed": 1, "priors": {"archetype": "habitable_moon"},
                                           "parent": {"moon_origin": "impact"}}))
