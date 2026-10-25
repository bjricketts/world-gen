"""Moon calibration fixes (2026-10-25): ice-shell dissipation, free-eccentricity damping, porous ice lid."""

import math
import types

import pytest

from tests.test_system import jupiter_spec
from worldgen import constants as c
from worldgen import generate
from worldgen import heuristics as h
from worldgen.giants import moon_spec
from worldgen.giants.moon_energy import (damped_eccentricity, eccentricity_damping_time, ice_shell_dissipation,
                                         ice_shell_world)
from worldgen.spec import PlanetSpec


@pytest.fixture(scope="module")
def jupiter_moons():
    spec = jupiter_spec()
    state = generate(spec)[0]
    world = types.SimpleNamespace(state=state, spec=spec)
    return {m.name: generate(moon_spec(world, k))[0] for k, m in enumerate(state.system.moons)}


def test_ice_shell_dissipation_anchors():
    assert ice_shell_dissipation(4.80e22, 1.5608e6) == pytest.approx(1.0e-3, rel=0.02)      # Europa
    assert ice_shell_dissipation(1.08e20, 2.52e5) == pytest.approx(0.015, rel=0.02)         # Enceladus
    assert ice_shell_dissipation(1e18, 5e4) == 0.015                                        # small moons clip
    assert h.ICE_SHELL_DISSIPATION_MIN <= ice_shell_dissipation(5e24, 7e6) < 1e-3
    assert ice_shell_world(0.3, 0.037) and not ice_shell_world(0.3, 1.0) and not ice_shell_world(0.01, 0.037)


def test_damping_time_and_decay():
    # Io-like moon: (21/2)(M/m)(R/a)^5 (k2/Q) n, Tokadjian & Piro (2023) eq. 20
    t = eccentricity_damping_time(1.898e27, 8.93e22, 1.8216e6, 4.217e8, 0.015)
    n = math.sqrt(c.G * (1.898e27 + 8.93e22) / 4.217e8**3)
    assert t == pytest.approx(1 / (10.5 * (1.898e27 / 8.93e22) * (1.8216e6 / 4.217e8) ** 5 * 0.015 * n))
    assert damped_eccentricity(0.01, t, t) == pytest.approx(0.01 / math.e)
    assert damped_eccentricity(0.01, 1e20, t) == 0.0
    assert damped_eccentricity(0.01, 1e20, math.inf) == 0.01


def test_galilean_moons(jupiter_moons):
    io, europa, ganymede, callisto = (jupiter_moons[n] for n in ("Io", "Europa", "Ganymede", "Callisto"))
    assert io.parent.partially_molten and not io.parent.ice_shell_dissipation
    area = 4 * math.pi * europa.bulk.radius_m**2
    flux = (europa.parent.tidal_heating_w + europa.interior.radiogenic_power_w) / area
    assert 0.015 <= flux <= 0.035                                        # Bagheri et al. 2026
    assert europa.parent.ice_shell_dissipation
    assert 19e3 <= europa.ice_shell.shell_thickness_m <= 39e3            # Juno MWR, via Bagheri et al. 2026
    assert ganymede.ice_shell.ocean_depth_m > 100e3 and ganymede.ice_shell.shell_thickness_m < 200e3
    assert callisto.ice_shell.ocean_depth_m > 0
    # listed eccentricities are real (forced) and are not damped
    assert europa.parent.eccentricity == europa.parent.initial_eccentricity
    assert not math.isfinite(europa.parent.eccentricity_damping_time_s)


def test_porous_lid():
    from worldgen.water import build_ice_shell

    europa = generate(PlanetSpec.model_validate({
        "name": "e", "seed": 1, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
        "orbit": {"semi_major_axis_au": 5.2},
        "body": {"mass_mearth": 0.008, "radius_rearth": 0.245, "water_mass_fraction": 0.3}}))[0]
    if europa.ice_shell is None:
        pytest.skip("no ice shell for this draw")
    a = build_ice_shell(europa.bulk, europa.water, 90.0, 2e11)
    b = build_ice_shell(europa.bulk, europa.water, h.ICE_LID_EFFECTIVE_K, 2e11)
    assert a.shell_thickness_m == pytest.approx(b.shell_thickness_m)


def test_habitable_moons_free_eccentricity_damped():
    damped = 0
    for seed in range(1, 21):
        st = generate(PlanetSpec.model_validate({"name": "m", "seed": seed,
                                                 "priors": {"archetype": "habitable_moon"}}))[0]
        p = st.parent
        assert p.eccentricity <= p.initial_eccentricity
        damped += p.eccentricity < 0.1 * p.initial_eccentricity
        assert p.tidal_heating_w / (4 * math.pi * st.bulk.radius_m**2) < h.TIDAL_HAZARD_FLUX_W_M2
    assert damped >= 12


def test_given_eccentricity_is_kept():
    spec = {"name": "m", "seed": 3, "priors": {"archetype": "habitable_moon"},
            "parent": {"eccentricity": 0.01}}
    st = generate(PlanetSpec.model_validate(spec))[0]
    assert st.parent.eccentricity == pytest.approx(0.01)
    assert st.parent.tidal_heating_w > 0


def test_drawn_resonant_moons_keep_eccentricity():
    from worldgen.giants.moons import DrawnMoon, _damp_free_eccentricities
    from types import SimpleNamespace

    inner = DrawnMoon(a_m=4.2e8, mass_kg=8.9e22, water=0.0, origin="regular", density=3500.0, eccentricity=0.004)
    outer = DrawnMoon(a_m=6.7e8, mass_kg=4.8e22, water=0.08, origin="regular", density=3000.0, eccentricity=0.009,
                      resonance_with=inner, resonance="2:1")
    free = DrawnMoon(a_m=1.9e9, mass_kg=1.1e23, water=0.5, origin="regular", density=1850.0, eccentricity=0.007)
    close = DrawnMoon(a_m=2.0e8, mass_kg=1e22, water=0.0, origin="regular", density=3000.0, eccentricity=0.01)
    bulk = SimpleNamespace(mass_kg=1.898e27)
    orbit = SimpleNamespace(instellation_earth=0.037)
    _damp_free_eccentricities([close, inner, outer, free], bulk, orbit, 4.6 * c.SECONDS_PER_GYR)
    assert inner.eccentricity == 0.004 and outer.eccentricity == 0.009
    assert close.eccentricity < 1e-6                     # a close free moon circularises
    assert free.eccentricity > 0.004                      # a distant one keeps most of it (Callisto-like)
