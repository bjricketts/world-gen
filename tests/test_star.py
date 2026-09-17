import pytest

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen import star


def test_sun_matches_reference(sun):
    assert sun.luminosity_w / 3.828e26 == pytest.approx(1.0, rel=1e-6)
    assert sun.radius_m / 6.957e8 == pytest.approx(1.0, rel=1e-6)
    assert sun.effective_temperature_k == pytest.approx(5772, abs=1)
    assert sun.spectral_type == "G2V"
    assert sun.xuv_fraction_relative_sun == pytest.approx(1.0)


def test_young_sun_is_fainter():
    young, _ = star.build_star(1.0, 0.0, 0.0)
    assert young.luminosity_w / 3.828e26 == pytest.approx(0.71, abs=0.02)


def test_red_dwarf_is_faint_long_lived_and_active():
    m_dwarf, _ = star.build_star(0.2, 5.0, 0.0)
    assert m_dwarf.luminosity_w / 3.828e26 < 0.01
    assert m_dwarf.main_sequence_lifetime_s > 100 * 3.156e16
    assert m_dwarf.spectral_type.startswith("M")
    assert m_dwarf.xuv_fraction_relative_sun > 10


def test_validity_errors():
    _, issues = star.build_star(0.05, 1.0, 0.0)
    assert any(i.level == "error" for i in issues)
    _, issues = star.build_star(1.5, 5.0, 0.0)   # lifetime ~2 Gyr
    assert any("lifetime" in i.message for i in issues)


def test_pre_main_sequence_brightening():
    """A contracting star is brighter than its main-sequence value; the Sun is already settled at 30 Myr."""
    assert star.zams_time_gyr(1.0) == pytest.approx(0.03)
    assert star.zams_time_gyr(0.1) > 0.5
    assert star.pre_main_sequence_factor(1.0, 0.03) == 1.0
    early = star.pre_main_sequence_factor(0.1, 0.05)
    assert 2.0 < early < h.PRE_MAIN_SEQUENCE_MAX
    assert star.pre_main_sequence_factor(0.1, 1.0) == 1.0
    lum_early = star.luminosity_track(0.1, 0.05)[0]
    lum_late = star.luminosity_track(0.1, 1.0)[0]
    assert lum_early > lum_late
    # The Sun's track is unchanged where it matters.
    assert star.luminosity_track(1.0, c.AGE_SUN_GYR)[0] == pytest.approx(1.0, rel=1e-3)


def test_xuv_history_percentiles():
    """Fast rotators stay saturated longer and emit more XUV; all tracks join the snapshot relation."""
    slow = star.saturation_time_gyr(1.0, 0.1)
    fast = star.saturation_time_gyr(1.0, 0.9)
    assert slow == pytest.approx(0.0057)
    assert fast == pytest.approx(0.226)
    assert star.saturation_time_gyr(0.3, 0.5) > star.saturation_time_gyr(1.0, 0.5)
    ages = [0.01, 0.05, 0.1, 0.5]
    assert all(star.xuv_fraction_history(1.0, t, 0.9) >= star.xuv_fraction_history(1.0, t, 0.1) for t in ages)
    integrated = [sum(star.xuv_fraction_history(1.0, 0.002 * k, p) for k in range(1, 500)) for p in (0.1, 0.9)]
    assert 2.0 < integrated[1] / integrated[0] < 20.0
    for p in (0.1, 0.5, 0.9):
        assert star.xuv_fraction_history(1.0, 4.57, p) == pytest.approx(star.xuv_fraction(1.0, 4.57))
        assert star.xuv_fraction_history(1.0, 0.001, p) == pytest.approx(h.XUV_SATURATED)


def test_history_star_matches_build_star():
    """The star built at an epoch matches the luminosity track used inside the integration."""
    state, _ = star.build_star(0.4, 0.2, 0.0, activity_percentile=0.5)
    lum, radius, _ = star.luminosity_track(0.4, 0.2)
    assert state.luminosity_w / c.L_SUN == pytest.approx(lum)
    assert state.radius_m / c.R_SUN == pytest.approx(radius)
    assert state.xuv_fraction == pytest.approx(star.xuv_fraction_history(0.4, 0.2, 0.5))
