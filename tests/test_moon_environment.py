"""Moons through the rocky pipeline: eclipses, planetshine, tides, magnetic shelter, ice shells, small bodies (11.6)."""

import math

import pytest

from worldgen import constants as c
from worldgen import generate, generate_world, heuristics as h
from worldgen.bulk import rocky_cmf, rocky_mass, rocky_radius
from worldgen.giants import moon_spec
from worldgen.giants.moon_energy import (dipole_moment, eclipse_fraction, magnetosphere_fraction, parent_heat_flux,
                                         reflected_fraction, standoff_distance, tidal_heating)
from worldgen.report import format_report
from worldgen.spec import PlanetSpec
from worldgen.state import PlanetState
from worldgen.water import melting_depth

from tests.test_system import jupiter_spec

R_J = 7.149e7


def test_eclipses_follow_heller():
    """A coplanar orbit at 5 planet radii loses 1/(5π) = 6.4% of its starlight; tilted orbits lose less."""
    assert eclipse_fraction(1.0, 5.0, 0.0) == pytest.approx(1 - 1 / (5 * math.pi))
    assert eclipse_fraction(1.0, 32.0, 0.0) > 0.99
    assert eclipse_fraction(1.0, 5.0, math.radians(30)) > eclipse_fraction(1.0, 5.0, 0.0)
    assert 1 - eclipse_fraction(1.0, 20.0, math.radians(98)) < 1e-3


def test_planetshine_terms_follow_heller_and_barnes():
    """Reflected light is π R² α / 2a² of the starlight; the glow is R² σ T⁴ / a² × (1 − α_IR) / 4."""
    assert reflected_fraction(1.0, 10.0, 0.3) == pytest.approx(math.pi * 0.3 / 200)
    glow = parent_heat_flux(R_J, 10 * R_J, 124.0)
    assert glow == pytest.approx(c.SIGMA_SB * 124.0**4 / 100 * 0.95 / 4)


def test_io_tidal_heat():
    """Io's measured k₂/Q of 0.015 gives ~10¹⁴ W, Io's observed output (2 W/m²)."""
    power = tidal_heating(1.898e27, 1.8216e6, 4.217e8, 0.0041, 0.015, 1.0)
    assert power == pytest.approx(1e14, rel=0.15)
    assert power / (4 * math.pi * 1.8216e6**2) == pytest.approx(h.TIDAL_HAZARD_FLUX_W_M2, rel=0.2)


@pytest.mark.parametrize("mass, d_au, radius, expected", [(317.8, 5.2, R_J, 43), (95.2, 9.58, 6.03e7, 20),
                                                          (1.0, 1.0, c.R_EARTH, 10)])
def test_standoff_distances(mass, d_au, radius, expected):
    """Heller & Zuluaga's standoff with the Solar System dipoles: Jupiter ~43, Saturn ~20, Earth ~10 radii."""
    moment = dipole_moment(mass * c.M_EARTH, mass > 10)
    assert standoff_distance(moment, d_au * c.AU) / radius == pytest.approx(expected, rel=0.1)


def test_dipole_table_and_magnetosphere_share():
    """The interpolation passes through Saturn and Jupiter; 20% of an orbit at 2 R_S lies inside."""
    assert dipole_moment(95.2 * c.M_EARTH, True) == pytest.approx(4.6e25, rel=1e-6)
    assert dipole_moment(317.8 * c.M_EARTH, True) == pytest.approx(1.56e27, rel=1e-6)
    assert dipole_moment(1.0 * c.M_EARTH, False, dynamo=False) == 0.0
    assert magnetosphere_fraction(1.0, 2.0) == pytest.approx(0.2, abs=0.02)
    assert magnetosphere_fraction(1.0, 0.9) == 1.0


@pytest.mark.parametrize("power, radius, surface, low, high", [
    (1.5e12 + 8e10, 1.5608e6, 100.0, 5e3, 30e3),       # Europa: 15–25 km
    (2.5e11, 2.634e6, 100.0, 100e3, 250e3),            # Ganymede: ~150 km
    (1.5e10, 2.52e5, 70.0, 10e3, 60e3),                # Enceladus: ~20 km
    (3e11, 2.575e6, 94.0, 50e3, 200e3),                # Titan: 50–100 km
])
def test_ice_shells_within_a_factor_of_two(power, radius, surface, low, high):
    """The conductive shell (Tjoa et al. 2020, corrected conductivity) lands within ~2× of the icy moons' shells."""
    assert low < melting_depth(power, radius, surface) < high


def test_small_body_core_fractions():
    """Radii give the Moon no core, Io ~0.06, Mars 0.24, Mercury ~0.66 and Earth 0.33; inverses agree."""
    for mass, radius, wmf, expected in ((0.0123, 0.2727, 0.0, 0.0), (0.01496, 0.286, 0.0, 0.06),
                                         (0.107, 0.532, 0.0, 0.24), (0.0553, 0.383, 0.0, 0.66), (1.0, 1.0, 0.0, 0.33)):
        assert rocky_cmf(mass, radius, wmf) == pytest.approx(expected, abs=0.05)
    for mass in (0.01, 0.3, 3.0):
        r = rocky_radius(mass, 0.2, 0.1)
        assert rocky_mass(r, 0.2, 0.1) == pytest.approx(mass, rel=1e-6)
        assert rocky_cmf(mass, r, 0.1) == pytest.approx(0.2, abs=1e-6)


@pytest.fixture(scope="module")
def galilean():
    """Return the Galilean moons generated from a Jupiter world."""
    world = generate_world(jupiter_spec(), "preview")
    return [generate(moon_spec(world, i))[0] for i in range(4)]


def test_moons_receive_eclipsed_starlight_planetshine_and_tides(galilean):
    """Each moon's instellation is the starlight after eclipses plus reflection; glow and tides are extra flux."""
    for moon in galilean:
        p = moon.parent
        assert moon.orbit.instellation_earth == pytest.approx(
            p.star_instellation_earth * (p.eclipse_fraction + p.reflected_fraction))
        area = 4 * math.pi * moon.bulk.radius_m**2
        assert moon.orbit.extra_flux_w_m2 == pytest.approx(p.planet_heat_w_m2 + p.tidal_heating_w / area)
        assert moon.interior.tidal_power_w == pytest.approx(p.tidal_heating_w)
        assert p.magnetosphere_fraction == 1.0       # all four lie inside Jupiter's ~43 R_J standoff
    io, europa = galilean[0], galilean[1]
    assert io.parent.eclipse_fraction < europa.parent.eclipse_fraction
    assert io.parent.planet_heat_w_m2 > europa.parent.planet_heat_w_m2


def test_icy_moons_have_shells_and_oceans(galilean):
    """Europa, Ganymede and Callisto are frozen with shells over oceans; Io has no water; none has an envelope."""
    io, europa, ganymede, callisto = galilean
    assert io.ice_shell is None
    assert 3e3 < europa.ice_shell.shell_thickness_m < 30e3 and europa.ice_shell.ocean_depth_m > 50e3
    assert 100e3 < ganymede.ice_shell.shell_thickness_m < 300e3
    assert all(m.bulk.planet_class not in ("sub-Neptune", "ice giant") for m in galilean)
    assert io.interior.radiogenic_power_w > 1e11 and ganymede.interior.radiogenic_power_w > 1e11
    text = format_report(europa)
    assert "subsurface ocean" in text and "eclipses remove" in text and "parent's magnetosphere" in text
    again = PlanetState.from_dict(europa.to_dict())
    assert again.ice_shell == europa.ice_shell and again.parent == europa.parent


def test_extra_flux_warms_and_tides_can_trigger_a_runaway():
    """Tidal heat given to a planet warms it; an Earth-mass moon close to a Jupiter on an eccentric orbit runs away."""
    base = {"name": "T", "seed": 1, "star": {"mass_msun": 1.0}, "orbit": {"semi_major_axis_au": 1.3},
            "body": {"mass_mearth": 1.0}, "atmosphere": {"surface_pressure_bar": 1.0}}
    cold, _ = generate(PlanetSpec.model_validate(base))
    warm, _ = generate(PlanetSpec.model_validate({**base, "body": {"mass_mearth": 1.0, "tidal_heating_w": 5e15}}))
    assert warm.orbit.extra_flux_w_m2 == pytest.approx(5e15 / (4 * math.pi * warm.bulk.radius_m**2))
    assert warm.atmosphere.equilibrium_temperature_k > cold.atmosphere.equilibrium_temperature_k + 2
    roasted, _ = generate(PlanetSpec.model_validate({
        "name": "Hot moon", "seed": 1, "star": {"mass_msun": 1.0}, "orbit": {"semi_major_axis_au": 1.2},
        "body": {"mass_mearth": 1.0, "water_mass_fraction": 0.001}, "atmosphere": {"surface_pressure_bar": 1.0},
        "parent": {"mass_mearth": 317.8, "semi_major_axis_rp": 6.0, "eccentricity": 0.05}}))
    assert roasted.parent.tidal_heating_w / (4 * math.pi * roasted.bulk.radius_m**2) > 100
    assert roasted.atmosphere.runaway_greenhouse
