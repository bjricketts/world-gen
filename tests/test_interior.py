import pytest

from worldgen import interior
from worldgen.util import named_rng

from .conftest import rocky_body

# (mass M_E, radius R_E, core mass fraction, surface T, surface water, tidal W)
EARTH = (1.0, 1.0, 0.325, 288, True, 0.0)
VENUS = (0.815, 0.95, 0.30, 737, False, 0.0)
MARS = (0.107, 0.532, 0.24, 215, False, 0.0)
MOON = (0.0123, 0.273, 0.02, 250, False, 0.0)
MERCURY = (0.0553, 0.383, 0.70, 440, False, 0.0)
IO = (0.015, 0.286, 0.20, 110, False, 1e14)


def _interior(body_params, regime=None):
    m, r, cmf, t, water, tidal = body_params
    body = rocky_body(m, r, cmf)
    state, issues, _ = interior.build_interior(
        body, 4.57, 1.0, tidal, t, water, regime, None, named_rng(0, "r"), named_rng(0, "d"))
    return state, issues


def test_earth_heat_budget():
    state, _ = _interior(EARTH)
    assert state.radiogenic_power_w / 1e12 == pytest.approx(20, rel=0.15)
    assert state.surface_heat_flux_w_m2 == pytest.approx(0.087, rel=0.15)
    assert state.activity_index == pytest.approx(1.0, rel=0.05)


def _most_likely(params):
    state, _ = _interior(params)
    return max(state.regime_probabilities, key=state.regime_probabilities.get)


@pytest.mark.parametrize("body, expected", [
    (EARTH, "mobile_lid"), (MARS, "stagnant_lid"), (MOON, "inactive"),
    (MERCURY, "inactive"), (IO, "heat_pipe"),
])
def test_solar_system_regimes(body, expected):
    assert _most_likely(body) == expected


def test_venus_does_not_favour_plates():
    state, _ = _interior(VENUS)
    p = state.regime_probabilities
    assert p["mobile_lid"] < p["stagnant_lid"]
    assert p["mobile_lid"] < p["episodic"]


def test_probabilities_normalised():
    state, _ = _interior(EARTH)
    assert sum(state.regime_probabilities.values()) == pytest.approx(1.0)


def test_unlikely_override_warns():
    _, issues = _interior(MOON, regime="mobile_lid")
    assert any(i.kind == "conflict" for i in issues)


def test_radiogenic_heat_decays():
    assert interior.radiogenic_heat_per_kg(0.0) > 2 * interior.radiogenic_heat_per_kg(4.57)
