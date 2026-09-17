import pytest

from worldgen import PlanetSpec, generate
from worldgen import heuristics as h
from worldgen.water import land_fraction_estimate, mantle_water, total_water_for_surface, volume_for_land_fraction

EARTH_TOTAL = h.SURFACE_WATER_EARTH + h.MANTLE_MASS_FRACTION * h.MANTLE_WATER_EARTH


def test_earth_partition_is_steady_state():
    mantle = mantle_water(EARTH_TOTAL, 1.0, plate_cycling=True)
    assert mantle == pytest.approx(h.MANTLE_MASS_FRACTION * h.MANTLE_WATER_EARTH, rel=1e-3)


def test_higher_gravity_stores_more_water_in_the_mantle():
    shares = [mantle_water(1e-3, g, True) / 1e-3 for g in (0.5, 1.0, 1.5, 2.5)]
    assert shares == sorted(shares)
    assert shares[-1] > 0.9


def test_mantle_capacity_limits_storage():
    assert mantle_water(0.05, 3.0, True) == pytest.approx(h.MANTLE_WATER_MAX * h.MANTLE_MASS_FRACTION)


def test_partition_inverts():
    for g in (0.6, 1.0, 1.8):
        for plates in (True, False):
            for total in (5e-5, 6e-4, 3e-3):
                surface = total - mantle_water(total, g, plates)
                assert total_water_for_surface(surface, g, plates) == pytest.approx(total, rel=1e-6)


def test_without_plates_the_mantle_keeps_a_fixed_share():
    assert mantle_water(1e-4, 1.0, False) == pytest.approx(h.NONPLATE_MANTLE_SHARE * 1e-4)
    assert mantle_water(1.0, 1.0, False) == pytest.approx(h.MANTLE_WATER_EARTH * h.MANTLE_MASS_FRACTION)


def test_land_estimate_decreases_with_volume_and_relief():
    r = 6.371e6
    earth = land_fraction_estimate(1.34e18, r, 1.0)
    assert 0.2 < earth < 0.4
    assert land_fraction_estimate(3e18, r, 1.0) < earth < land_fraction_estimate(0.5e18, r, 1.0)
    assert land_fraction_estimate(1.34e18, r, 0.6) < earth       # shallower basins overflow
    assert land_fraction_estimate(0.0, r, 1.0) == 1.0
    assert land_fraction_estimate(1e21, r, 1.0) == 0.0


def test_earth_state_has_earths_ocean(earth_spec):
    state, _ = generate(earth_spec)
    w = state.water
    assert w.surface_mass_kg == pytest.approx(1.4e21, rel=0.05)
    assert w.seafloor_pressure_ratio == pytest.approx(1.0, abs=0.05)
    assert w.plate_cycling


def test_waterworld_has_no_weathering_regulation():
    state, _ = generate(PlanetSpec(seed=3, priors={"archetype": "ocean", "enforce_constraints": False},
                                   body={"water_mass_fraction": 0.02}))
    assert state.water.land_fraction_estimate < h.WATERWORLD_LAND_FRACTION
    assert not state.atmosphere.weathering_regulated


def test_arid_worlds_are_mostly_land():
    for seed in range(5):
        state, _ = generate(PlanetSpec(seed=seed, priors={"archetype": "arid"}))
        assert state.water.land_fraction_estimate >= 0.5


def test_land_estimate_depends_on_regime():
    r = 6.371e6
    plates = land_fraction_estimate(1e18, r, 1.0, "mobile_lid")
    heat_pipe = land_fraction_estimate(1e18, r, 1.0, "heat_pipe")
    assert heat_pipe < plates
    for regime in ("mobile_lid", "stagnant_lid", "episodic", "heat_pipe", "inactive"):
        volume = volume_for_land_fraction(0.4, r, 1.3, regime)
        assert land_fraction_estimate(volume, r, 1.3, regime) == pytest.approx(0.4, abs=1e-6)
