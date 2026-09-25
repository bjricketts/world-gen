import numpy as np
import pytest

from worldgen import PlanetSpec, generate
from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.climate import COLD_START_K, climate_setting, planet_climate, run_climate, solve_zonal
from worldgen.climate.ebm import SurfaceState, TemperatureControl, solve_thermal
from worldgen.climate.ice import build_ice_sheets, buzzsaw, equilibrium_line, positive_degree_days
from worldgen.climate.insolation import harmonics, seasonal_insolation, substellar_insolation, synthesise
from worldgen.climate.mesh import climate_grid, cotangent_weights, laplacian, regridder
from worldgen.climate.moisture import solve_moisture
from worldgen.grid import build_grid

R = 6.371e6


def earth_setting(instellation=1.0, synchronous=False, rotation_h=23.93, albedo=0.3, fixed=True, water=True):
    return climate_setting(instellation, 0.0167, np.radians(23.44), np.radians(283.0), synchronous, c.P_EARTH, 9.81,
                           "n2_o2", 0.85, rotation_h * 3600.0, 3.156e7, albedo, fixed, water, 288.0)


# Insolation -----------------------------------------------------------------------

def test_insolation_global_mean_and_seasons():
    x = -1.0 + (np.arange(400) + 0.5) / 200.0
    q = seasonal_insolation(x, c.S_EARTH, np.radians(23.44), 0.0, 0.0, 48)
    assert q.mean() == pytest.approx(c.S_EARTH / 4, rel=0.01)
    # Northern summer (sample 24 of 48) is bright at the north pole and dark at the south pole.
    assert q[24, -1] > q[24, 200] > 0.0 == q[24, 0]
    assert substellar_insolation(np.array([1.0, 0.0, -0.5]), 1000.0).tolist() == [1000.0, 0.0, 0.0]


def test_harmonics_round_trip():
    t = 2 * np.pi * np.arange(24) / 24
    values = (3.0 + 2.0 * np.cos(t - 0.4) - 0.5 * np.sin(2 * t))[:, None]
    np.testing.assert_allclose(synthesise(harmonics(values, 3), 24), values, atol=1e-12)


# Grid operators -------------------------------------------------------------------

def test_cotangent_laplacian_eigenvalue():
    z = climate_grid().points[:, 2]
    lap = laplacian(cotangent_weights(climate_grid().size))
    assert np.median((lap @ z)[np.abs(z) > 0.2] / z[np.abs(z) > 0.2]) == pytest.approx(-2.0, rel=0.05)


def test_regridding_keeps_constant_fields():
    rg = regridder(20_000)
    np.testing.assert_allclose(rg.to_surface(np.full(climate_grid().size, 5.0)), 5.0)
    np.testing.assert_allclose(rg.to_climate(np.full(20_000, 5.0)), 5.0)


# Tier 0 ---------------------------------------------------------------------------

def test_zonal_earth():
    z = solve_zonal(earth_setting(), 0.3)
    lat = np.degrees(np.arcsin(z.coordinate))
    annual = (0.7 * z.ocean_k + 0.3 * z.land_k).mean(axis=0)
    assert z.converged
    assert z.albedo == pytest.approx(0.3, abs=0.005)
    assert 280.0 < z.mean_k < 295.0
    assert annual[np.abs(lat) < 10].mean() - annual[np.abs(lat) > 70].mean() > 25.0
    assert 45.0 < z.ice_line_deg < 80.0
    assert 0.6 < z.open_ocean_fraction < 0.97
    # Land has a larger seasonal cycle than ocean at high latitude.
    band = (lat > 50) & (lat < 70)
    assert np.ptp(z.land_k[:, band].mean(axis=1)) > 3 * np.ptp(z.ocean_k[:, band].mean(axis=1))


def test_zonal_land_per_band():
    setting = earth_setting()
    uniform = solve_zonal(setting, 0.3)
    banded = solve_zonal(setting, np.full(36, 0.3))
    assert banded.mean_k == pytest.approx(uniform.mean_k, abs=1e-6)
    # With the land at the poles, no ocean is left to freeze there; tropical land brightens the planet.
    polar = np.where(np.abs(banded.coordinate) > 0.7, 1.0, 0.0)
    tropical = np.where(np.abs(banded.coordinate) < 0.3, 1.0, 0.0)
    a, b = solve_zonal(earth_setting(fixed=False, albedo=0.27), polar), \
        solve_zonal(earth_setting(fixed=False, albedo=0.27), tropical)
    assert a.open_ocean_fraction > b.open_ocean_fraction
    assert a.albedo < b.albedo


def test_ocean_heat_transport():
    z = solve_zonal(earth_setting(), 0.3, target_mean_k=288.0)
    lat = np.degrees(np.arcsin(-1.0 + 2.0 / 36 * np.arange(1, 36)))
    ocean, atmosphere = z.ocean_transport, z.atmosphere_transport
    assert (ocean[lat > 1] >= 0).all() and (ocean[lat < -1] <= 0).all()        # poleward in both hemispheres
    assert 10.0 < lat[np.argmax(ocean)] < 30.0                                  # largest in the subtropics
    at35 = np.argmin(np.abs(lat - 35.0))
    assert 0.15 < ocean[at35] / (ocean[at35] + atmosphere[at35]) < 0.4          # Earth: 22% (Trenberth & Caron 2001)
    near_equator = np.argmin(np.abs(lat - 4.0))
    assert ocean[near_equator] > atmosphere[near_equator]                        # Earth: ocean larger to 17° N
    assert not solve_zonal(earth_setting(water=False), 1.0).ocean_transport.any()


def test_zonal_target_mean_is_held():
    z = solve_zonal(earth_setting(), 0.3, target_mean_k=275.0)
    assert z.mean_k == pytest.approx(275.0, abs=0.2)


def test_snowball_bistability():
    setting = earth_setting(instellation=0.87, albedo=0.27, fixed=False)
    warm = solve_zonal(setting, 0.3)
    cold = solve_zonal(setting, 0.3, start_k=COLD_START_K)
    assert warm.open_water_fraction > 0.3
    assert cold.open_water_fraction < 0.02
    assert warm.mean_k > cold.mean_k + 15


def test_synchronous_day_night_contrast():
    z = solve_zonal(earth_setting(instellation=0.9, synchronous=True, rotation_h=480.0), 0.3)
    assert z.dayside_k > z.nightside_k + 30
    assert 0.0 < z.open_ocean_fraction < 1.0
    # Thick air carries more heat to the night side.
    thick = climate_setting(0.9, 0.0, 0.0, 0.0, True, 5 * c.P_EARTH, 9.81, "n2_o2", 0.85 * 5**0.72,
                            480 * 3600.0, 480 * 3600.0, 0.3, True, True, 300.0)
    z5 = solve_zonal(thick, 0.3)
    assert z5.dayside_k - z5.nightside_k < z.dayside_k - z.nightside_k


def test_dry_planet_has_no_open_water():
    z = solve_zonal(earth_setting(water=False), 1.0)
    assert z.open_water_fraction == 0.0


# Tier 1 ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aquaplanet():
    setting = earth_setting()
    zonal = solve_zonal(setting, 0.0)
    n = climate_grid().size
    surface = SurfaceState(np.zeros(n), np.zeros(n), np.full(n, 0.2), np.zeros(n))
    thermal = solve_thermal(setting, surface, zonal)
    return setting, zonal, thermal, solve_moisture(setting, thermal, np.zeros(n))


def test_tier1_agrees_with_tier0(aquaplanet):
    setting, zonal, thermal, _ = aquaplanet
    assert thermal.converged
    assert thermal.mean_k == pytest.approx(zonal.mean_k, abs=2.0)
    # The ocean heats the mid-latitudes and cools the tropics without adding energy.
    lat = climate_grid().lat
    assert abs(float(np.mean(thermal.ocean_warming_k))) < 0.5
    assert np.mean(thermal.ocean_warming_k[np.abs(lat) < 10]) < 0 < np.mean(thermal.ocean_warming_k[(np.abs(lat) > 30)
                                                                                                      & (np.abs(lat) < 60)])
    annual = solve_thermal(setting, SurfaceState(*(np.zeros(climate_grid().size),) * 2,
                                                 np.full(climate_grid().size, 0.2), np.zeros(climate_grid().size)),
                           zonal, seasonal=False)
    assert annual.samples == 1
    assert annual.mean_k == pytest.approx(thermal.mean_k, abs=2.0)


def test_tier1_holds_a_target_within_limits(aquaplanet):
    setting, zonal, thermal, _ = aquaplanet
    n = climate_grid().size
    surface = SurfaceState(np.zeros(n), np.zeros(n), np.full(n, 0.2), np.zeros(n))
    held = solve_thermal(setting, surface, zonal, control=TemperatureControl(thermal.mean_k + 5.0, 0.0, 50.0))
    assert held.mean_k == pytest.approx(thermal.mean_k + 5.0, abs=0.3)
    assert held.optical_depth > setting.optical_depth
    capped = solve_thermal(setting, surface, zonal,
                           control=TemperatureControl(thermal.mean_k + 5.0, 0.0, setting.optical_depth))
    assert capped.mean_k == pytest.approx(thermal.mean_k, abs=0.3)
    assert thermal.optical_depth == pytest.approx(setting.optical_depth)


def test_moisture_budget_and_belts(aquaplanet):
    _, _, thermal, moist = aquaplanet
    lat = climate_grid().lat
    # Moisture transport only moves water: E − P sums to zero.
    assert abs(moist.net_export_m.mean()) < 1e-3
    p = moist.precipitation_m.mean(axis=0)
    tropics, subtropics, midlat = p[np.abs(lat) < 8].mean(), p[(np.abs(lat) > 18) & (np.abs(lat) < 30)].mean(), \
        p[(np.abs(lat) > 40) & (np.abs(lat) < 55)].mean()
    assert tropics > 2 * subtropics
    assert midlat > subtropics
    # The rising branch follows the sun: south in month 1, north half a year later.
    assert moist.itcz_sin[0] < 0.0 < moist.itcz_sin[12]


def test_run_climate_on_earth_like_surface(world_cache, earth_spec):
    earth_spec.surface.tectonics = "heuristic"
    w = world_cache(earth_spec)
    ds = w.surface
    ocean = ds.ocean.values
    lat = ds.lat.values
    assert ds.monthly_temperature.shape == (12, ds.sizes["cell"])
    assert 1.0 < ds.precipitation.values[ocean].mean() < 1.5
    assert 0.5 < ds.precipitation.values[~ocean].mean() < 1.1
    assert ds.air_temperature.values[np.abs(lat) < 10].mean() > ds.air_temperature.values[np.abs(lat) > 70].mean()
    assert 0.02 < ds.sea_ice.values[ocean].mean() < 0.3
    f = w.state.surface.features
    assert f["climate_mean_temperature_k"] == pytest.approx(288.0, abs=2.0)


# Ice sheets -----------------------------------------------------------------------

def test_positive_degree_days():
    assert positive_degree_days(np.full((12, 1), 263.15))[0] < 10.0
    assert positive_degree_days(np.full((12, 1), 283.15))[0] == pytest.approx(3652.5, rel=0.01)


def test_ice_sheet_on_cold_wet_land():
    grid = build_grid(10_000)
    ocean = grid.points[:, 2] < 0.5          # a polar continent north of 30°
    ground = np.where(ocean, 0.0, 500.0)
    cold = np.where(grid.lat > 60, 250.0, 290.0)
    months = np.tile(cold, (12, 1))
    rain = np.full((12, grid.size), 0.5)
    ice = build_ice_sheets(grid, R, ground, ocean, months, rain, 9.81)
    assert ice.glaciated[grid.lat > 65].all()
    assert not ice.glaciated[ocean].any()
    assert not ice.glaciated[(grid.lat < 55) & ~ocean].any()
    # Thickest far from the margin, a few km at most.
    assert ice.thickness_m[grid.lat > 85].mean() > ice.thickness_m[(grid.lat > 60) & (grid.lat < 63)].mean()
    assert 1000.0 < ice.thickness_m.max() < 7000.0      # Nye profile 3300 km from the margin
    # Snowline: low where it is cold, high where it is warm.
    ela = equilibrium_line(months, rain, ground, ~ocean)
    assert np.all(ela[grid.lat > 65] < 1.0)
    assert np.all(ela[(grid.lat < 55) & ~ocean] > 2000.0)


def test_snowline_lies_where_the_mass_balance_is_zero():
    """The snowline is interpolated between height steps, not rounded to them."""
    from worldgen.climate.ice import mass_balance

    ground = np.zeros(40)
    warm = np.linspace(274.0, 300.0, 40)
    months = warm[None, :] + 8.0 * np.cos(2 * np.pi * (np.arange(12)[:, None] + 0.5) / 12)
    rain = np.full((12, 40), 1.0)
    ela = equilibrium_line(months, rain, ground)
    found = np.isfinite(ela) & (ela > 0.0)
    assert found.sum() > 30
    assert np.mean(np.mod(ela[found], 500.0) == 0.0) < 0.1          # not stuck on the 500 m levels
    at_line = mass_balance(months[:, found], rain[:, found], h.LAPSE_RATE_K_PER_M * ela[found])
    assert np.abs(at_line).max() < 0.25                             # balance near zero at the snowline (m/yr)
    below = mass_balance(months[:, found], rain[:, found], h.LAPSE_RATE_K_PER_M * (ela[found] - 300.0))
    above = mass_balance(months[:, found], rain[:, found], h.LAPSE_RATE_K_PER_M * (ela[found] + 300.0))
    assert (below < 0.0).all() and (above > 0.0).all()


def test_snow_melts_first_and_more_slowly_than_ice():
    """The year's snow melts at the snow rate; only degree-days left over melt ice, at the higher ice rate."""
    from worldgen.climate.ice import mass_balance, positive_degree_days, snow_balance, snow_share

    months = np.array([[255.0, 275.0]] * 9 + [[270.0, 290.0]] * 3)     # a cold and a warm site
    rain = np.full((12, 2), 1.0)
    snowfall = (rain * snow_share(months)).mean(axis=0)
    degree_days = positive_degree_days(months)
    balance = mass_balance(months, rain)
    cold = snowfall[0] > h.SNOW_DEGREE_DAY_FACTOR_M * degree_days[0]
    assert cold and balance[0] == pytest.approx(snowfall[0] - h.SNOW_DEGREE_DAY_FACTOR_M * degree_days[0])
    left = degree_days[1] - snowfall[1] / h.SNOW_DEGREE_DAY_FACTOR_M
    assert left > 0.0 and balance[1] == pytest.approx(-h.DEGREE_DAY_FACTOR_M * left)
    assert np.array_equal(np.sign(balance), np.sign(snow_balance(months, rain)))
    assert h.SNOW_DEGREE_DAY_FACTOR_M < 0.6 * h.DEGREE_DAY_FACTOR_M


def test_snowline_climate_matches_glaciers():
    """At the snowline, summer temperature and precipitation follow the glacier relation within ~2 K.

    Ohmura et al. (1992): P = 645 + 296 T + 9 T² (mm w.e., June–August °C) at 70 glaciers' ELAs.
    """
    annual = np.linspace(270.0, 300.0, 200)
    previous = -np.inf
    for p_mm in (500.0, 1000.0, 2000.0):
        for annual_range in (15.0, 25.0):
            months = annual[None, :] + 0.5 * annual_range * np.cos(2 * np.pi * (np.arange(12)[:, None] + 0.5) / 12)
            ela = equilibrium_line(months, np.full((12, 200), p_mm / 1000.0), np.zeros(200))
            k = np.flatnonzero(np.isfinite(ela) & (ela > 100.0))[0]
            summer = np.sort(months[:, k])[-3:].mean() - 273.15 - h.LAPSE_RATE_K_PER_M * ela[k]
            glaciers = (-296.0 + np.sqrt(296.0**2 - 4 * 9.0 * (645.0 - p_mm))) / (2 * 9.0)
            assert abs(summer - glaciers) < 2.5
        assert summer > previous                          # wetter glaciers reach down into warmer air
        previous = summer


def test_buzzsaw_wears_peaks_toward_the_snowline():
    height = np.array([500.0, 2000.0, 6000.0, 9000.0, 6000.0])
    ocean = np.array([False, False, False, False, True])
    ela = np.full(5, 1000.0)
    out = buzzsaw(height, ocean, ela, 10.0)
    base = 1000.0 + h.GLACIAL_PEAK_OFFSET_M
    assert out[0] == 500.0 and out[1] == 2000.0 and out[4] == 6000.0
    assert base < out[2] < 6000.0
    assert out[2] < out[3] < 9000.0                 # no fixed ceiling
    assert out[3] - out[2] < 3000.0                 # but high peaks lose more
    # Over long times peaks approach the base.
    assert buzzsaw(height, ocean, ela, 1e5)[3] == pytest.approx(base, abs=100.0)


# Global state ---------------------------------------------------------------------

def test_earth_state_climate(earth_spec):
    state, _ = generate(earth_spec)
    cl = state.climate
    assert state.atmosphere.surface_temperature_k == pytest.approx(288.0, abs=0.5)
    assert state.atmosphere.bond_albedo == pytest.approx(0.3)
    assert 0.7 < cl.open_ocean_fraction < 0.95
    assert 50.0 < cl.ice_line_deg < 80.0
    assert not cl.snowball_stable
    assert state.atmosphere.surface_water == "liquid"


def test_cold_world_with_open_tropics_counts_as_liquid(earth_spec):
    spec = earth_spec.model_copy(update={"atmosphere": earth_spec.atmosphere.model_copy(
        update={"surface_temperature_k": 267.0})})
    state, _ = generate(spec)
    assert state.atmosphere.surface_temperature_k == 267.0
    assert state.climate.open_water_fraction > 0.02
    assert state.atmosphere.surface_water == "liquid"


@pytest.mark.slow
@pytest.mark.slow
def test_surface_climate_keeps_the_weathering_target(world_cache):
    # Arid seed 6 falls 17 K below its target on the real map without the Tier 1 control.
    w = world_cache(PlanetSpec(seed=6, priors={"archetype": "arid"}, surface={"tectonics": "heuristic"}))
    state = w.state
    assert state.atmosphere.weathering_regulated
    assert w.state.surface.features["climate_mean_temperature_k"] == pytest.approx(
        state.atmosphere.weathering_target_k, abs=1.0)
    assert state.climate.mean_temperature_k == pytest.approx(state.atmosphere.weathering_target_k, abs=1.0)


def test_tidally_locked_archetype_has_temperate_day_side():
    for seed in range(4):
        state, _ = generate(PlanetSpec(seed=seed, priors={"archetype": "tidally_locked"}))
        assert state.orbit.spin_state == "synchronous"
        assert 250.0 <= state.climate.dayside_temperature_k <= 330.0
        assert state.climate.dayside_temperature_k > state.climate.nightside_temperature_k


def test_planet_climate_rebuilds_the_state_climate(earth_spec):
    state, _ = generate(earth_spec)
    setting, zonal = planet_climate(state)
    assert zonal.mean_k == pytest.approx(state.climate.mean_temperature_k, abs=1.5)
    grid = build_grid(10_000)
    z = np.where(np.abs(grid.lon) < 40, 500.0, -3000.0)
    climate = run_climate(grid, R, z, z < 0, setting, zonal, seasonal=False)
    assert climate.rainfall.monthly_temperature_k is None
    assert np.isfinite(climate.ela_m[z > 0]).all()
