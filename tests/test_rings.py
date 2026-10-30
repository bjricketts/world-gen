"""Rings: which planets have which, their layout from the Roche limits, colours, meshes and saving."""

import numpy as np
import pytest

from worldgen import constants as c
from worldgen import generate, heuristics as h
from worldgen.giants.limits import roche_limit_rubble
from worldgen.giants.rings import pr_lifetime_gyr, ring_window, zero_albedo_temperature
from worldgen.render.rings import ring_mesh_arrays
from worldgen.report import format_report
from worldgen.spec import PlanetSpec
from worldgen.state import PlanetState

GIANTS = {
    "Jupiter": (5.2, 317.8, 11.21, 0.9, 9.93),
    "Saturn": (9.58, 95.2, 9.45, 0.85, 10.56),
    "Uranus": (19.2, 14.5, 4.01, 0.15, 17.24),
    "Neptune": (30.1, 17.1, 3.88, 0.15, 16.11),
}


def giant(name, rings="auto", moons=None, seed=1, a=None):
    """Return the state of a Solar System giant analogue with the given ring setting."""
    a_au, mass, radius, envelope, rotation = GIANTS[name]
    system = {"rings": rings}
    if moons is not None:
        system["moons"] = moons
    spec = PlanetSpec.model_validate({
        "name": name, "seed": seed, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
        "orbit": {"semi_major_axis_au": a if a is not None else a_au, "rotation_period_h": rotation},
        "body": {"mass_mearth": mass, "radius_rearth": radius, "envelope_mass_fraction": envelope},
        "system": system})
    return generate(spec)[0]


@pytest.mark.parametrize("name, dense, needed", [("Jupiter", False, None), ("Saturn", True, 0.66),
                                                 ("Uranus", True, 0.02), ("Neptune", False, None)])
def test_seventy_kelvin_window(name, dense, needed):
    """Saturn and Uranus can hold dense rings near 70 K (bright and dark), Jupiter and Neptune cannot."""
    state = giant(name, "none")
    t0 = zero_albedo_temperature(state.star, state.orbit)
    ok, albedo = ring_window(t0)
    assert ok == dense
    if needed is not None:
        assert albedo == pytest.approx(needed, abs=0.03)


def test_regimes_of_the_solar_system_giants():
    """Most seeds give each giant its own kind of rings; hot giants and rocky planets get none."""
    expected = {"Jupiter": "dusty warm", "Saturn": "broad", "Uranus": "narrow", "Neptune": "dusty cold"}
    for name, regime in expected.items():
        regimes = [giant(name, seed=s).system.ring_regime for s in range(1, 9)]
        assert regimes.count(regime) >= 5, (name, regimes)
    assert giant("Jupiter", a=0.05).system.ring_regime == "none"
    earth, _ = generate(PlanetSpec.model_validate({"seed": 1, "star": {"mass_msun": 1.0},
                                                   "orbit": {"semi_major_axis_au": 1.0},
                                                   "body": {"mass_mearth": 1.0, "radius_rearth": 1.0}}))
    assert earth.system.ring_regime == "none" and earth.system.rings == []


def test_pr_lifetime_formula():
    """A 1 m rock particle at 1 AU from a Sun-like star lasts 4.19 Gyr (Hyodo & Ida 2025, eq. 4)."""
    state = giant("Jupiter", "none", a=1.0)
    lum = state.star.luminosity_w / c.L_SUN
    assert pr_lifetime_gyr(state.star, state.orbit, 1.0, 3000.0) * lum == pytest.approx(4.19, rel=1e-6)


def test_broad_rings_match_saturn():
    """Saturn's rings come out with the C ring inner edge, A ring outer edge and mass of the real rings."""
    rings = giant("Saturn", "broad").system.rings
    names = [r.name for r in rings]
    assert names == ["C", "B inner", "B", "Cassini Division", "A"]
    assert rings[0].inner_radius_m / 1e3 == pytest.approx(74_490, rel=0.02)
    assert rings[-1].outer_radius_m / 1e3 == pytest.approx(136_770, rel=0.02)
    assert sum(r.mass_kg for r in rings) == pytest.approx(1.5e19, rel=0.3)
    b = rings[2]
    assert b.mass_kg > 0.5 * sum(r.mass_kg for r in rings)
    assert b.bond_albedo > rings[0].bond_albedo
    red, _, blue = (int(b.colour[i:i + 2], 16) for i in (1, 3, 5))
    assert red > blue


def test_cassini_division_follows_a_moon():
    """A Mimas-like moon puts the Cassini Division's inner edge at its 2:1 resonance."""
    mimas = {"name": "Mimas", "mass_mearth": 6.3e-6, "radius_rearth": 0.031, "semi_major_axis_rp": 185_520 / 60_268}
    state = giant("Saturn", "broad", moons=[mimas])
    division = next(r for r in state.system.rings if r.name == "Cassini Division")
    expected = state.system.moons[0].semi_major_axis_m * 2 ** (-2 / 3)
    assert division.inner_radius_m == pytest.approx(expected, rel=1e-6)
    assert division.inner_radius_m / 1e3 == pytest.approx(117_570, rel=0.01)


def test_narrow_rings_match_uranus():
    """Uranus-like ringlets are narrow, dark and end at the Roche limit for 1.2 g/cm³ (the ε ring)."""
    state = giant("Uranus", "narrow")
    rings = state.system.rings
    assert h.NARROW_RING_COUNT[0] <= len(rings) <= h.NARROW_RING_COUNT[1]
    a_r = roche_limit_rubble(state.bulk.mass_kg, h.NARROW_RING_DENSITY)
    assert rings[-1].outer_radius_m / 1e3 == pytest.approx(51_150, rel=0.03)
    assert all(r.inner_radius_m >= 0.8 * a_r for r in rings)
    assert all((r.outer_radius_m - r.inner_radius_m) / 1e3 <= 100 for r in rings)
    assert all(r.bond_albedo < 0.05 for r in rings)


def test_dusty_rings_match_jupiter_and_neptune():
    """Jupiter's main ring and Neptune's Adams ring with its arcs come out where they are."""
    jupiter = giant("Jupiter", "dusty warm")
    main = next(r for r in jupiter.system.rings if r.name == "main")
    assert main.inner_radius_m / 1e3 == pytest.approx(122_500, rel=0.02)
    assert main.optical_depth == pytest.approx(8e-6)
    neptune = giant("Neptune", "dusty cold")
    adams = next(r for r in neptune.system.rings if r.name == "Adams")
    assert adams.inner_radius_m / 1e3 == pytest.approx(62_900, rel=0.02)
    assert h.RING_ARC_COUNT[0] <= len(adams.arcs) <= h.RING_ARC_COUNT[1]


def test_user_rings_and_settings():
    """Listed rings are used as given; 'none' removes rings; a forced regime applies; bad edges are refused."""
    state = giant("Jupiter", [{"name": "Broad", "inner_radius_rp": 1.5, "outer_radius_rp": 2.0,
                               "optical_depth": 1.0, "bond_albedo": 0.6}])
    assert state.system.ring_regime == "user" and state.system.rings[0].name == "Broad"
    assert giant("Saturn", "none").system.rings == []
    assert giant("Jupiter", "broad").system.ring_regime == "broad"
    with pytest.raises(ValueError):
        PlanetSpec.model_validate({"system": {"rings": [{"inner_radius_rp": 2.0, "outer_radius_rp": 1.5}]}})


def test_ring_mesh_arrays_and_arcs():
    """Each ring is one annulus; arcs are more opaque than the rest of their ring."""
    rings = giant("Neptune", "dusty cold").system.rings
    vertices, faces, rgba = ring_mesh_arrays(rings, 360)
    assert len(vertices) == 2 * 360 * len(rings) and len(faces) == 2 * 360 * len(rings)
    assert np.allclose(vertices[:, 2], 0.0)
    adams = rgba[-720:-360, 3]
    assert adams.max() > 10 * max(adams.min(), 1)


def test_report_and_round_trip():
    """The report lists the rings, and they survive saving to plain data and back."""
    state = giant("Saturn", "broad")
    text = format_report(state)
    assert "broad icy rings" in text and "Cassini Division" in text
    neptune = giant("Neptune", "dusty cold")
    again = PlanetState.from_dict(neptune.to_dict())
    assert again.system.rings == neptune.system.rings and again.system.ring_regime == "dusty cold"
