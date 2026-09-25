import numpy as np
import pytest

from worldgen import PlanetSpec, generate_world
from worldgen import constants as c
from worldgen.grid import build_grid
from worldgen.hydrology import WaterSetting, build_drainage, erode
from worldgen.hydrology.graph import accumulate, pass_levels, priority_flood, steepest_receivers, upstream_order
from worldgen.climate import climate_setting, run_climate, solve_zonal
from worldgen.hydrology.rainfall import Rainfall, fu_runoff
from worldgen.surface import Terrain
from worldgen.surface.sealevel import ocean_fill

R = 6.371e6


@pytest.fixture(scope="module")
def grid():
    return build_grid(10_000)


def dome(grid, centre=(1.0, 0.0, 0.0), radius=0.9, height=2000.0):
    """A round continent rising from −3 km sea floor, with a central hollow."""
    centre = np.asarray(centre) / np.linalg.norm(centre)
    angle = np.arccos(np.clip(grid.points @ centre, -1, 1))
    z = np.where(angle < radius, height * np.cos(angle / radius * np.pi / 2) + 200.0, -3000.0)
    hollow = angle < 0.15
    z[hollow] -= 1500.0 * np.cos(angle[hollow] / 0.15 * np.pi / 2)
    return z


def earth_climate(synchronous=False, rotation_h=24.0):
    """Return an Earth-like climate setting and its zonal solution (land fraction 0.3)."""
    setting = climate_setting(1.0, 0.0167, np.radians(23.44), np.radians(283.0), synchronous, c.P_EARTH, 9.81,
                              "n2_o2", 0.85, rotation_h * 3600.0, 3.156e7, 0.3, True, True, 288.0)
    return setting, solve_zonal(setting, 0.3)


def climate_rain(grid, z, ocean, synchronous=False, rotation_h=24.0, seasonal=True):
    """Return the Tier 1 water balance of a surface on an Earth-like planet."""
    setting, zonal = earth_climate(synchronous, rotation_h)
    return run_climate(grid, R, z, ocean, setting, zonal, seasonal=seasonal).rainfall


def uniform_rain(grid, rain, evaporation):
    ones = np.ones(grid.size)
    runoff = fu_runoff(rain * ones, evaporation * ones)
    return Rainfall(precipitation_m=rain * ones, evaporation_m=evaporation * ones, runoff_m=runoff,
                    temperature_k=288.0 * ones)


# Graph kernels ------------------------------------------------------------------

def test_priority_flood_fills_hollows_and_drains_everything(grid):
    z = dome(grid)
    ocean = z < 0
    adj = grid.neighbours
    filled = priority_flood(z, adj.indptr, adj.indices, ocean, 1e-3)
    assert (filled >= z).all()
    assert (filled - z).max() > 1000.0
    receiver, _ = steepest_receivers(filled, adj.indptr, adj.indices, adj.data, ocean)
    order = upstream_order(receiver)
    roots = receiver.copy()
    for i in order:
        roots[i] = i if receiver[i] == i else roots[receiver[i]]
    assert ocean[roots].all()


def test_accumulation_conserves_water(grid):
    z = dome(grid)
    ocean = z < 0
    adj = grid.neighbours
    filled = priority_flood(z, adj.indptr, adj.indices, ocean, 1e-3)
    receiver, _ = steepest_receivers(filled, adj.indptr, adj.indices, adj.data, ocean)
    order = upstream_order(receiver)
    flow = accumulate(order, receiver, np.where(ocean, 0.0, 1.0))
    outlets = receiver == np.arange(grid.size)
    assert flow[outlets].sum() == pytest.approx((~ocean).sum())


def test_loops_are_rejected():
    with pytest.raises(ValueError):
        upstream_order(np.array([1, 0, 2]))


def test_pass_levels_and_ocean_volume(grid):
    z = dome(grid)
    adj = grid.neighbours
    level = pass_levels(z, adj.indptr, adj.indices, int(np.argmin(z)))
    assert (level >= z).all()
    fill = ocean_fill(grid, z, R)
    area = 4 * np.pi * R**2 / grid.size
    assert fill.volume(0.0) == pytest.approx((-z[z < 0]).sum() * area)
    for volume in (1e17, fill.volume(0.0), 3e18):
        assert fill.volume(fill.level_for_volume(volume)) == pytest.approx(volume, rel=1e-6)
    # The hollow on the continent is below the rim but not part of the ocean.
    assert fill.level_for_land_fraction(0.3) < 1000.0


# Rainfall ----------------------------------------------------------------------

def test_fu_runoff_limits():
    assert fu_runoff(np.array([3.0]), np.array([0.2]))[0] == pytest.approx(2.8, abs=0.05)
    assert fu_runoff(np.array([0.1]), np.array([2.0]))[0] < 0.005
    assert fu_runoff(np.array([1.0]), np.array([0.0]))[0] == pytest.approx(1.0)


@pytest.mark.slow
def test_mountains_cast_a_rain_shadow(grid):
    # A tropical continent with a north–south range near its east coast; trade winds blow west.
    lon = grid.lon
    lat = grid.lat
    land = (np.abs(lat) < 25) & (lon > -40) & (lon < 40)
    z = np.where(land, 300.0, -3000.0)
    ridge = land & (np.abs(lon - 25) < 5)
    z[ridge] = 4500.0
    rain = climate_rain(grid, z, ~land)
    windward = land & (lon > 32)
    leeward = land & (lon < 15) & (lon > 0)
    assert rain.precipitation_m[windward].mean() > 2 * rain.precipitation_m[leeward].mean()
    no_ridge = climate_rain(grid, np.where(land, 300.0, -3000.0), ~land)
    assert rain.precipitation_m[leeward].mean() < no_ridge.precipitation_m[leeward].mean()


def test_synchronous_rain_centres_on_the_substellar_point(grid):
    ocean = np.ones(grid.size, dtype=bool)
    rain = climate_rain(grid, np.full(grid.size, -3000.0), ocean, synchronous=True, rotation_h=500.0)
    day = grid.points[:, 0] > 0.9
    night = grid.points[:, 0] < -0.5
    assert rain.precipitation_m[day].mean() > 5 * rain.precipitation_m[night].mean()
    assert rain.temperature_k[day].mean() > rain.temperature_k[night].mean() + 30
    assert rain.monthly_temperature_k is not None


# Drainage and lakes ------------------------------------------------------------

def test_wet_hollow_overflows_and_dry_hollow_closes(grid):
    z = dome(grid)
    ocean = z < 0
    wet = build_drainage(grid, R, z, ocean, uniform_rain(grid, 2.0, 0.5))
    dry = build_drainage(grid, R, z, ocean, uniform_rain(grid, 0.4, 2.0))
    centre = np.argmax(grid.points[:, 0])
    assert wet.lake[centre] and dry.lake[centre]
    assert wet.lake.sum() > dry.lake.sum()
    assert not wet.endorheic[centre] and dry.endorheic[centre]
    assert np.nanmax(wet.lake_level_m) > np.nanmax(dry.lake_level_m)


def test_water_balance_closes(grid):
    z = dome(grid)
    ocean = z < 0
    rain = uniform_rain(grid, 0.6, 1.2)
    d = build_drainage(grid, R, z, ocean, rain)
    area = 4 * np.pi * R**2 / grid.size
    land = ~ocean
    mouths = land & ocean[d.receiver]
    to_sea = d.discharge_m3_s[mouths].sum() * c.SECONDS_PER_YEAR
    in_lake = d.lake
    supplied = (rain.runoff_m[land & ~in_lake].sum()
                + np.maximum(rain.precipitation_m - rain.evaporation_m, 0)[in_lake].sum()) * area
    lost = np.maximum(rain.evaporation_m - rain.precipitation_m, 0)[in_lake].sum() * area
    assert to_sea <= supplied
    assert to_sea == pytest.approx(max(supplied - lost, 0.0), rel=0.35, abs=1e9)


def test_rivers_have_increasing_order_downstream(grid):
    z = dome(grid)
    ocean = z < 0
    d = build_drainage(grid, R, z, ocean, uniform_rain(grid, 2.0, 0.5))
    river = np.flatnonzero(d.river_order > 0)
    assert river.size > 0
    down = d.receiver[river]
    both = d.river_order[down] > 0
    assert (d.river_order[down][both] >= d.river_order[river][both]).all()
    assert d.basin[ocean].max() == 0 and d.basin[~ocean].min() >= 1


# Erosion -----------------------------------------------------------------------

def test_erosion_lowers_mountains_and_conserves_sediment(grid):
    z = dome(grid, height=4000.0)
    ocean = z < 0
    runoff = np.where(ocean, 0.0, 0.5)
    result = erode(grid, R, z, ocean, runoff, duration_myr=20.0, relief=1.0)
    land = ~ocean
    assert result.height_m[land].max() < z[land].max()
    assert result.eroded_m.sum() == pytest.approx(result.deposited_m.sum(), rel=1e-6)
    assert result.deposited_m[ocean].sum() > 0
    # The hollow collects sediment.
    centre = np.argmax(grid.points[:, 0])
    assert result.deposited_m[centre] > 0
    assert (result.height_m[ocean] <= 0.0).all()


def test_no_erosion_without_runoff(grid):
    z = dome(grid)
    ocean = z < 0
    result = erode(grid, R, z, ocean, np.zeros(grid.size), duration_myr=5.0, relief=1.0)
    assert result.eroded_m.max() == 0.0


# Worlds ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wet_world():
    return generate_world(PlanetSpec(seed=4, priors={"archetype": "temperate"},
                                     surface={"tectonics_duration_myr": 100}), "preview")


def test_wet_world_has_rivers_and_lakes(wet_world):
    ds = wet_world.surface
    for name in ("precipitation", "runoff", "discharge", "flow_to", "drainage_basin", "lake", "river_order"):
        assert name in ds
    assert (ds["river_order"].values > 0).any()
    lake = ds["lake"].values
    assert (ds["terrain"].values[lake] == Terrain.LAKE).all()
    assert not (lake & ds["ocean"].values).any()
    features = wet_world.state.surface.features
    assert features["river_outflow_km3_yr"] > 0


def test_land_below_sea_level_is_enclosed(wet_world):
    ds = wet_world.surface
    low = (ds["elevation"].values < 0) & ~ds["ocean"].values
    grid = wet_world.grid
    ocean = ds["ocean"].values
    for cell in np.flatnonzero(low):
        assert not ocean[grid.neighbour_indices(cell)].all()
    # Such cells hold lakes or drain inland.
    if low.any():
        assert (ds["lake"].values[low] | ds["endorheic"].values[low]).mean() > 0.5


def test_frozen_and_dry_worlds_have_no_rivers(world_cache):
    frozen = world_cache(PlanetSpec(seed=1, priors={"archetype": "ice"}, surface={"tectonics": "heuristic"}))
    dry = world_cache(PlanetSpec(seed=2, priors={"archetype": "hot_volcanic"}))
    for w in (frozen, dry):
        assert "river_order" not in w.surface
        assert "river_outflow_km3_yr" not in w.state.surface.features


def test_water_setting_uses_the_annual_climate(grid):
    climate, zonal = earth_climate()
    setting = WaterSetting(radius_m=R, ocean_volume_m3=1e18, climate=climate, zonal=zonal, relief=1.0)
    z = dome(grid)
    a = setting.rainfall(grid, z, z < 0)
    b = climate_rain(grid, z, z < 0, seasonal=False)
    np.testing.assert_array_equal(a.precipitation_m, b.precipitation_m)
    assert a.monthly_precipitation_m is None


def test_priority_breach_drains_shallow_hollows_and_keeps_deep_basins():
    """A hollow behind a low sill is cut through; one behind a high sill is left to fill as a lake."""
    from worldgen.hydrology.graph import priority_breach, priority_flood
    from worldgen.zoom import region_grid

    grid = region_grid(4, 5, 10, 10, 10, 10, nodes_per_tile=30)
    rows, cols = grid.shape
    r, c = np.divmod(np.arange(grid.size), cols)
    z = 5.0 * c.astype(float)                                   # a slope down to the left edge
    outlet = c == 0
    hollow = (np.abs(r - rows // 2) < 4) & (np.abs(c - 15) < 4)
    z[hollow] -= 30.0                                           # a 30 m hollow behind a ~15 m cut
    adj = grid.neighbours

    def pooled(surface):
        return (priority_flood(surface, adj.indptr, adj.indices, outlet, 1e-3) - surface > 1.0).any()

    assert pooled(z)
    assert not pooled(priority_breach(z, adj.indptr, adj.indices, outlet, 100.0, 1e-3))   # breached
    assert pooled(priority_breach(z, adj.indptr, adj.indices, outlet, 5.0, 1e-3))         # too deep: a lake
    breached = priority_breach(z, adj.indptr, adj.indices, outlet, 100.0, 1e-3)
    assert (breached <= z + 1e-9).all()                         # breaching only ever lowers ground


def test_open_edge_exports_sediment_instead_of_raising_the_edge():
    """On a local grid, sediment reaching the open edge leaves; it is not piled on the edge cells."""
    from worldgen.hydrology import erode
    from worldgen.zoom import region_grid

    grid = region_grid(4, 6, 20, 20, 20, 20, nodes_per_tile=24)
    rows, cols = grid.shape
    r, c = np.divmod(np.arange(grid.size), cols)
    height = 400.0 + 10.0 * np.sin(r / 3.0) * np.cos(c / 4.0)          # rolling ground
    height += 3.0 * np.minimum.reduce([r, rows - 1 - r, c, cols - 1 - c])   # a dome draining to the frame
    edge = np.zeros(grid.size, bool)
    edge[(r == 0) | (r == rows - 1) | (c == 0) | (c == cols - 1)] = True
    area = grid.cell_area_m2(6.371e6)
    result = erode(grid, 6.371e6, height, np.zeros(grid.size, bool), np.full(grid.size, 0.5), 5.0, 1.0, 1.0,
                   cell_area_m2=area, creep_m2_per_myr=1e4, edge=edge)
    assert np.abs(result.height_m[edge] - height[edge]).max() < 1e-9          # the edge is untouched
    eroded = result.eroded_m.sum() * area
    assert eroded > 0.0 and result.exported_m3 > 0.0
    assert abs(eroded - result.deposited_m.sum() * area - result.exported_m3) < 1e-6 * eroded
