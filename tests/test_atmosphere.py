import pytest

from worldgen import atmosphere as atm
from worldgen import constants as c
from worldgen.orbit import equilibrium_temperature

# (instellation, escape velocity km/s) for the cosmic shoreline
BODIES = {
    "Earth": (1.0, 11.19), "Venus": (1.91, 10.36), "Mars": (0.431, 5.03),
    "Titan": (0.011, 2.64), "Moon": (1.0, 2.38), "Mercury": (6.67, 4.25),
    "Callisto": (0.037, 2.44),
}


@pytest.mark.parametrize("name, keeps", [
    ("Earth", True), ("Venus", True), ("Titan", True),
    ("Moon", False), ("Mercury", False), ("Callisto", False),
])
def test_cosmic_shoreline(name, keeps):
    s, v = BODIES[name]
    ratio = atm.shoreline_ratio(s, 1.0, v * 1e3)
    assert (ratio < 1) == keeps


def test_mars_sits_near_shoreline():
    s, v = BODIES["Mars"]
    assert 0.7 < atm.shoreline_ratio(s, 1.0, v * 1e3) < 1.4


def _surface_t(s, albedo, composition, bar):
    t_eq = equilibrium_temperature(s, albedo)
    return atm.surface_temperature(t_eq, atm.greenhouse_optical_depth(composition, bar * c.BAR))


def test_greenhouse_calibration():
    assert _surface_t(1.0, 0.3, "n2_o2", 1.0) == pytest.approx(288, abs=3)
    assert _surface_t(1.91, 0.76, "co2", 92.0) == pytest.approx(737, rel=0.05)
    assert _surface_t(0.431, 0.25, "co2", 0.006) == pytest.approx(215, abs=5)


def test_boiling_point():
    assert atm.boiling_point(c.P_EARTH) == pytest.approx(373.15, abs=0.1)
    assert atm.boiling_point(92 * c.BAR) == pytest.approx(578, rel=0.03)   # steam tables
    assert atm.boiling_point(300 * c.BAR) == pytest.approx(atm.WATER_CRITICAL_K)
    assert atm.boiling_point(100.0) == pytest.approx(atm.WATER_TRIPLE_POINT_K)


@pytest.mark.parametrize("args, expected", [
    ((True, True, c.P_EARTH, 288, False), "liquid"),
    ((True, True, c.P_EARTH, 250, False, 0.0, 0.0), "ice"),
    ((True, True, c.P_EARTH, 265, False, 0.15, 0.3), "liquid"),      # open tropical ocean on a cold world
    ((True, True, c.P_EARTH, 280, False, 0.005, 0.01), "ice"),       # warm mean, but the ocean is frozen
    ((True, False, 0.0, 200, False), "ice"),
    ((True, True, c.P_EARTH, 400, False), "vapour"),
    ((True, True, c.P_EARTH, 288, True), "none"),
    ((False, True, c.P_EARTH, 288, False), "none"),
    ((True, False, 0.0, 300, False), "none"),
])
def test_water_state(args, expected):
    """The phase follows the open-water fractions from the climate model, not only the mean temperature."""
    assert atm.water_state(*args) == expected


def test_temperature_round_trip():
    tau = atm.optical_depth_for_temperature(255.0, 288.0)
    assert atm.surface_temperature(255.0, tau) == pytest.approx(288.0)
