import pytest

from worldgen import bulk
from worldgen import constants as c


def build(mass=None, radius=None, cmf=None, wmf=0.0, envelope=None, wmf_is_user=True):
    return bulk.build_bulk(mass, radius, cmf, wmf, envelope, wmf_is_user)


def test_earth_radius_from_mass():
    state, _, prov = build(mass=1.0, cmf=0.325)
    assert state.radius_m / c.R_EARTH == pytest.approx(1.0, abs=0.01)
    assert state.surface_gravity_m_s2 == pytest.approx(9.8, abs=0.1)
    assert state.escape_velocity_m_s == pytest.approx(11_186, rel=0.01)
    assert prov["body.radius_rearth"] == "derived"
    assert state.planet_class == "rocky"


def test_mass_radius_cmf_inversions_agree():
    r = bulk.rocky_radius(4.0, 0.2)
    assert bulk.rocky_mass(r, 0.2) == pytest.approx(4.0)
    assert bulk.rocky_cmf(4.0, r) == pytest.approx(0.2)


def test_cmf_derived_from_mass_and_radius():
    state, _, prov = build(mass=3.0, radius=bulk.rocky_radius(3.0, 0.1))
    assert state.core_mass_fraction == pytest.approx(0.1)
    assert prov["body.core_mass_fraction"] == "derived"


def test_low_density_rocky_planet_gets_water():
    radius = bulk.rocky_radius(2.0, 0.0, 0.2)
    state, _, prov = build(mass=2.0, radius=radius, wmf=0.0, wmf_is_user=False)
    assert state.water_mass_fraction == pytest.approx(0.2, abs=1e-6)
    assert state.planet_class == "water world"
    assert prov["body.water_mass_fraction"] == "derived"


def test_overdetermined_bulk_warns():
    _, issues, _ = build(mass=1.0, radius=1.2, cmf=0.3)
    assert any(i.kind == "conflict" for i in issues)


def test_chen_kipping_continuity():
    # The published coefficients are rounded, leaving steps of up to ~2%.
    for m in (2.04, 132.0, 26600.0):
        below = bulk.chen_kipping_radius(m * 0.9999)
        above = bulk.chen_kipping_radius(m * 1.0001)
        assert above == pytest.approx(below, rel=0.02)


@pytest.mark.parametrize("mass, expected", [
    (0.05, "dwarf rocky world"), (1.0, "rocky"), (5.0, "super-Earth"),
    (17.0, "ice giant"), (318.0, "gas giant"), (5000.0, "brown dwarf"),
])
def test_classes(mass, expected):
    state, _, _ = build(mass=mass, cmf=0.3)
    assert state.planet_class == expected


def test_jupiter_radius_near_chen_kipping():
    state, _, _ = build(mass=317.8)
    assert 10 < state.radius_m / c.R_EARTH < 15


def test_neptune_mass_from_radius():
    state, _, prov = build(radius=3.86, envelope=0.15)
    assert 10 < state.mass_kg / c.M_EARTH < 30
    assert prov["body.mass_mearth"] == "derived"
