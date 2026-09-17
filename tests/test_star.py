import pytest

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
