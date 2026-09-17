import math

import pytest

from worldgen import constants as c
from worldgen import orbit, star
from worldgen.orbit import build_orbit, equilibrium_temperature, habitable_zone


def _orbit(star_state, body, a_au, rotation_h=24.0, e=0.0):
    state, issues, prov = build_orbit(
        star=star_state, bulk=body, semi_major_axis_m=a_au * c.AU, instellation_earth=None,
        eccentricity=e, obliquity_rad=math.radians(23.0), rotation_period_s=rotation_h * 3600,
        rotation_is_user=False, obliquity_is_user=False)
    return state, issues, prov


def test_earth_orbit(sun, earth_body):
    o, _, _ = _orbit(sun, earth_body, 1.0)
    assert o.orbital_period_s / c.SECONDS_PER_DAY == pytest.approx(365.25, rel=1e-3)
    assert o.instellation_earth == pytest.approx(1.0)
    assert o.spin_state == "free"
    assert o.habitable_zone.position == "conservative habitable zone"


def test_equilibrium_temperature_earth():
    assert equilibrium_temperature(1.0, 0.3) == pytest.approx(255, abs=1)
    assert equilibrium_temperature(1.0, 0.0) == pytest.approx(278.3, abs=0.5)


def test_hz_limits_for_sun():
    hz, _ = habitable_zone(c.L_SUN, 5780.0, 1.0, 1.0)
    assert hz.runaway_greenhouse_s == pytest.approx(1.107)
    assert hz.maximum_greenhouse_s == pytest.approx(0.356)
    assert hz.runaway_greenhouse_m / c.AU == pytest.approx(0.950, abs=0.005)
    assert hz.maximum_greenhouse_m / c.AU == pytest.approx(1.676, abs=0.005)


def test_hz_widens_with_planet_mass():
    small = orbit.runaway_greenhouse_limit(5780.0, 0.1)
    big = orbit.runaway_greenhouse_limit(5780.0, 5.0)
    assert small == pytest.approx(0.99)
    assert big == pytest.approx(1.188)


def test_close_red_dwarf_planet_is_locked(earth_body):
    m_dwarf, _ = star.build_star(0.2, 5.0, 0.0)
    o, _, prov = _orbit(m_dwarf, earth_body, 0.05)
    assert o.spin_state == "synchronous"
    assert o.rotation_period_s == pytest.approx(o.orbital_period_s)
    assert o.obliquity_rad == 0.0
    assert prov["orbit.rotation_period_h"] == "derived"


def test_eccentric_locked_planet_resonance(earth_body):
    m_dwarf, _ = star.build_star(0.2, 5.0, 0.0)
    o, _, _ = _orbit(m_dwarf, earth_body, 0.05, e=0.2)
    assert o.spin_state == "3:2 resonance"
    assert o.rotation_period_s == pytest.approx(o.orbital_period_s * 2 / 3)


def test_instellation_round_trip(sun):
    a = orbit.semi_major_axis_for_instellation(sun.luminosity_w, 0.5, 0.1)
    assert orbit.instellation(sun.luminosity_w, a, 0.1) == pytest.approx(0.5)
