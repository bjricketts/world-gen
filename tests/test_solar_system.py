"""Solar System validation: icy air, high-pressure ice, melt feedback and the validation script (task 11.8)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from worldgen import constants as c
from worldgen import generate, heuristics as h
from worldgen.atmosphere import n2_vapour_pressure_bar
from worldgen.giants import moon_spec
from worldgen.giants.moon_energy import tidal_heating
from worldgen.spec import PlanetSpec
from worldgen.water import (high_pressure_ice_onset_mpa, ice_ih_melting_pressure_mpa, melting_temperature_k)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_solar_system.py"


def load_script():
    """Return the validation script as a module."""
    spec = importlib.util.spec_from_file_location("validate_solar_system", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_solar_system"] = module     # its dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


def test_n2_vapour_pressure_matches_grundy():
    """Grundy et al.'s α fit gives their 0.76 nbar at 27 K; the β fit gives ~19 µbar near Triton's 38 K."""
    assert n2_vapour_pressure_bar(27.0) == pytest.approx(0.76e-9, rel=0.1)
    assert n2_vapour_pressure_bar(38.0) == pytest.approx(19e-6, rel=0.1)
    assert n2_vapour_pressure_bar(45.0) > 10 * n2_vapour_pressure_bar(38.0)


def test_iapws_melting_curves():
    """The melting curves meet at the IAPWS triple points; water freezes to ice V above ~570 MPa at 270 K."""
    assert ice_ih_melting_pressure_mpa(251.165) == pytest.approx(208.566, rel=1e-4)
    assert melting_temperature_k(0.1) == pytest.approx(273.15, abs=0.02)
    for p, t in ((350.1, 256.164), (632.4, 273.31), (2216.0, 355.0)):
        assert melting_temperature_k(p) == pytest.approx(t, rel=1e-3)
    assert high_pressure_ice_onset_mpa(270.0) == pytest.approx(567.6, rel=0.01)
    assert high_pressure_ice_onset_mpa(252.0) < high_pressure_ice_onset_mpa(260.0)


def test_melt_feedback_picks_hot_moons():
    """Io's measured k₂/Q gives its ~10¹⁴ W; a Ganymede-like orbit stays solid; a fixed Q is kept."""
    assert tidal_heating(1.898e27, 1.8216e6, 4.217e8, 0.0041, h.TIDAL_MOLTEN_K2_OVER_Q, 1.0) == pytest.approx(
        1e14, rel=0.15)
    base = {"name": "M", "seed": 3, "star": {"mass_msun": 1.0}, "orbit": {"semi_major_axis_au": 5.2},
            "body": {"mass_mearth": 0.015, "radius_rearth": 0.286},
            "parent": {"mass_mearth": 317.8, "semi_major_axis_rp": 5.9, "eccentricity": 0.0041}}
    io, _ = generate(PlanetSpec.model_validate(base))
    assert io.parent.partially_molten and io.parent.tidal_k2_over_q == h.TIDAL_MOLTEN_K2_OVER_Q
    calm, _ = generate(PlanetSpec.model_validate({**base, "parent": {**base["parent"], "semi_major_axis_rp": 15.0,
                                                                     "eccentricity": 0.0013}}))
    assert not calm.parent.partially_molten
    fixed, _ = generate(PlanetSpec.model_validate({**base, "body": {**base["body"], "tidal_q": 300.0}}))
    assert not fixed.parent.partially_molten


def test_icy_air_titan_and_triton():
    """Titan's air comes out near 1.5 bar and 94 K with a hazy albedo; Triton's sits at the N₂ vapour pressure."""
    vss = load_script()
    for planet, index, check in (("Saturn", 1, "titan"), ("Neptune", 0, "triton")):
        spec = vss.giant_spec(planet, moons=vss.MOONS[planet])
        state = generate(spec)[0]
        moon, _ = generate(moon_spec(types.SimpleNamespace(state=state, spec=spec), index))
        a = moon.atmosphere
        if check == "titan":
            assert 1.0 < a.surface_pressure_pa / c.BAR < 2.5 and abs(a.surface_temperature_k - 93.7) < 6
            assert a.bond_albedo == h.HAZE_ALBEDO and moon.ice_shell.high_pressure_ice_m > 0
        else:
            assert a.surface_pressure_pa / c.BAR == pytest.approx(n2_vapour_pressure_bar(a.surface_temperature_k))
            assert 1e-6 < a.surface_pressure_pa / c.BAR < 1e-3


def test_rocky_planets_keep_their_air():
    """Warm or dry worlds are untouched by the icy inventory: Earth stays near 1 bar."""
    earth, _ = generate(PlanetSpec.model_validate({"name": "E", "seed": 1, "star": {"mass_msun": 1.0},
                                                    "orbit": {"semi_major_axis_au": 1.0},
                                                    "body": {"mass_mearth": 1.0, "radius_rearth": 1.0}}))
    assert not any("nitrogen inventory" in i.message for i in earth.issues)


@pytest.mark.slow
def test_validation_script_passes():
    """Every Solar System check passes (100 draws for the system statistics)."""
    assert load_script().main(["--seeds", "100"]) == 0
