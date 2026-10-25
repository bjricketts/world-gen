"""The habitable_moon archetype: priors, the parent it brings, the archetype mix and the parent's glow (11.7)."""

import math

import pytest

from worldgen import constants as c
from worldgen import generate, generate_world, heuristics as h
from worldgen.priors import resolve
from worldgen.priors.archetypes import ARCHETYPES
from worldgen.priors.sampling import compatible_archetypes
from worldgen.report import format_report
from worldgen.spec import PlanetSpec


def moon(seed):
    """Return a habitable-moon state for a seed."""
    return generate(PlanetSpec.model_validate({"name": "Moon", "seed": seed,
                                               "priors": {"archetype": "habitable_moon"}}))[0]


def test_draws_meet_the_archetype():
    """Draws are temperate moons from their origin's priors, within the constraints."""
    for seed in range(1, 9):
        state = moon(seed)
        p = state.parent
        assert state.constraints_met and state.archetype == "habitable_moon"
        assert p.moon_origin in h.HABITABLE_MOON_ORIGINS
        if p.moon_origin == "regular":
            ratio = state.bulk.mass_kg / p.mass_kg
            assert h.HABITABLE_MOON_MASS_RATIO[0] * 0.999 <= ratio <= h.HABITABLE_MOON_MASS_RATIO[1] * 1.001
        else:
            lo, hi = h.HABITABLE_MOON_CAPTURED_PARENT_MEARTH
            assert lo * 0.999 <= p.mass_kg / c.M_EARTH <= hi * 1.001
            lo, hi = h.HABITABLE_MOON_CAPTURED_MEARTH
            assert lo * 0.999 <= state.bulk.mass_kg / c.M_EARTH <= hi * 1.001
        lo, hi = h.HABITABLE_MOON_ORBIT_RP if p.moon_origin == "regular" else h.HABITABLE_MOON_CAPTURED_ORBIT_RP
        assert lo <= p.semi_major_axis_m / p.radius_m <= hi
        assert p.semi_major_axis_m <= p.stable_limit_m
        assert state.atmosphere.surface_water == "liquid" and not state.atmosphere.runaway_greenhouse
        assert state.orbit.spin_state == "synchronous with parent" and state.system is None
        assert h.HABITABLE_MOON_STAR_MSUN[0] <= state.star.mass_kg / c.M_SUN <= h.HABITABLE_MOON_STAR_MSUN[1]


def test_origins_follow_their_shares():
    """Regular, captured and impact moons come out in about their heuristic shares, each with its own priors."""
    origins = [resolve(PlanetSpec.model_validate({"name": "M", "seed": s, "priors": {"archetype": "habitable_moon"}}))
               for s in range(300)]
    for origin, share in h.HABITABLE_MOON_ORIGINS.items():
        picked = [r for r in origins if r.get("parent.moon_origin") == origin]
        assert len(picked) / len(origins) == pytest.approx(share, abs=0.07)
        parents = [r.get("parent.mass_mearth") for r in picked]
        lo, hi = (h.HABITABLE_MOON_REGULAR_PARENT_MEARTH if origin == "regular"
                  else h.HABITABLE_MOON_CAPTURED_PARENT_MEARTH)
        assert all(lo <= m <= hi for m in parents)
    fixed = resolve(PlanetSpec.model_validate({"name": "M", "seed": 1, "priors": {"archetype": "habitable_moon"},
                                               "parent": {"moon_origin": "captured"}}))
    assert fixed.get("parent.mass_mearth") <= h.HABITABLE_MOON_CAPTURED_PARENT_MEARTH[1]
    assert "moon's origin" in format_report(moon(2))


def test_parent_glow_includes_internal_heat():
    """A giant parent glows above its equilibrium temperature: T⁴ = T_eq⁴ + T_int⁴."""
    state = moon(3)
    p = state.parent
    t_eq = (state.star.luminosity_w * (1 - p.bond_albedo) / (16 * math.pi * c.SIGMA_SB *
                                                            (c.AU * math.sqrt(state.star.luminosity_w / c.L_SUN
                                                                              / p.star_instellation_earth)) ** 2)) ** 0.25
    assert p.effective_temperature_k > t_eq + 5
    assert state.provenance["parent.effective_temperature_k"] == "derived"


def test_specs_with_a_parent_pick_moon_archetypes():
    """A spec with a parent section and no archetype becomes a habitable moon; a fixed Earth mass rules it out."""
    spec = PlanetSpec.model_validate({"name": "M", "seed": 2, "parent": {"mass_mearth": 2000, "semi_major_axis_rp": 15}})
    assert resolve(spec).archetype == "habitable_moon"
    assert compatible_archetypes({"parent.mass_mearth": 2000}) == ["habitable_moon"]
    assert "habitable_moon" in compatible_archetypes({"parent.mass_mearth": 300, "body.mass_mearth": 1.0})
    assert "habitable_moon" not in compatible_archetypes({"parent.mass_mearth": 300, "body.mass_mearth": 3.0})
    assert "habitable_moon" not in compatible_archetypes({"body.mass_mearth": 0.5})   # a described planet stays one
    assert "habitable_moon" in compatible_archetypes({"star.mass_msun": 1.0})


def test_the_random_mix_sometimes_makes_a_moon():
    """With no archetype asked for, about 3% of specs become habitable moons, with their parent drawn."""
    picks = [resolve(PlanetSpec.model_validate({"name": "X", "seed": s})).archetype for s in range(400)]
    share = picks.count("habitable_moon") / len(picks)
    total = sum(a.mix_weight for a in ARCHETYPES.values())
    assert share == pytest.approx(ARCHETYPES["habitable_moon"].mix_weight / total, abs=0.02)
    seed = picks.index("habitable_moon")
    inputs = resolve(PlanetSpec.model_validate({"name": "X", "seed": seed}))
    assert inputs.get("parent.moon_origin") in h.HABITABLE_MOON_ORIGINS
    if inputs.get("parent.moon_origin") == "regular":
        ratio = inputs.get("body.mass_mearth") / inputs.get("parent.mass_mearth")
        assert h.HABITABLE_MOON_MASS_RATIO[0] <= ratio <= h.HABITABLE_MOON_MASS_RATIO[1]
    assert inputs.get("parent.semi_major_axis_rp") is not None


@pytest.mark.slow
def test_a_habitable_moon_world_builds():
    """A habitable moon gets a full surface at preview resolution."""
    world = generate_world(PlanetSpec.model_validate({"name": "Moon", "seed": 4,
                                                      "priors": {"archetype": "habitable_moon"}}), "preview")
    assert world.state.parent is not None and world.state.surface is not None
    assert world.surface["elevation"].values.size > 0
