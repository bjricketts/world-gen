import numpy as np
import pytest

from worldgen import PlanetSpec, generate_world
from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.grid import build_grid
from worldgen.surface import Boundary, Crust, Terrain, relief_factor
from worldgen.surface.distance import distance_from, smooth
from worldgen.surface.fields import SurfaceFields
from worldgen.surface.landforms import add_crater, add_volcano, crater_depth, sample_crater_diameters
from worldgen.surface.plates import Plates, classify_boundaries
from worldgen.surface.sealevel import ocean_fill
from worldgen.util import named_rng

RES = "preview"


def world(seed=0, archetype="temperate", resolution=RES, tectonics="heuristic", **spec):
    """Generate a world; heuristic plates unless the test concerns the simulation."""
    spec["surface"] = {"tectonics": tectonics, **spec.get("surface", {})}
    return generate_world(PlanetSpec(seed=seed, priors={"archetype": archetype}, **spec), resolution)


@pytest.fixture(scope="module")
def temperate():
    return world(seed=3, resolution=20_000, tectonics="simulated")


REGIME_CASES = {
    "mobile_lid": dict(archetype="temperate"),
    "stagnant_lid": dict(archetype="small_world", interior={"tectonic_regime": "stagnant_lid"}),
    "episodic": dict(archetype="arid", interior={"tectonic_regime": "episodic"}),
    "heat_pipe": dict(archetype="hot_volcanic", interior={"tectonic_regime": "heat_pipe"}),
    "inactive": dict(archetype="small_world", interior={"tectonic_regime": "inactive"}),
}


@pytest.fixture(scope="module")
def regime_worlds():
    out = {}
    for regime, kw in REGIME_CASES.items():
        kw = dict(kw)
        priors = {"archetype": kw.pop("archetype"), "enforce_constraints": False}
        out[regime] = generate_world(PlanetSpec(seed=4, priors=priors, **kw), RES)
    return out


# Dataset structure ----------------------------------------------------------

def test_dataset_structure(temperate):
    ds = temperate.surface
    for name in ("elevation", "ocean", "terrain", "crust", "plate", "boundary", "crust_age"):
        assert ds[name].dims == ("cell",)
        assert ds[name].size == 20_000
    assert ds.attrs["regime"] == "mobile_lid"
    assert ds["elevation"].attrs["units"] == "m"
    assert temperate.grid.size == 20_000


@pytest.mark.parametrize("regime", list(REGIME_CASES))
def test_every_regime_builds(regime_worlds, regime):
    w = regime_worlds[regime]
    assert w.state.interior.tectonic_regime == regime
    e = w.surface["elevation"].values
    assert np.isfinite(e).all()
    assert e.max() - e.min() > 1000
    assert w.state.surface.max_elevation_m == pytest.approx(e.max(), rel=1e-5)


@pytest.mark.slow
def test_land_fraction_target_is_met():
    for target in (0.05, 0.3, 0.7):
        w = world(seed=1, surface={"land_fraction": target})
        # Large enclosed basins hold seas at the ocean's level; only flat sediment plains flood at once.
        assert w.state.surface.land_fraction == pytest.approx(target, abs=0.02)
        assert w.state.provenance["surface.land_fraction"] == "user"
        ocean = w.surface["ocean"].values
        assert (w.surface["elevation"].values[ocean] < 0).all()
        water = w.state.water
        assert water.ocean_volume_m3 > 0
        assert any("sets the total water mass fraction" in i.message for i in w.state.issues)


def test_enclosed_basins_hold_seas():
    grid = build_grid(5000)
    z = grid.points[:, 2]
    elevation = np.where(z > 0.5, -4000.0 * (z - 0.5), 500.0)   # a basin enclosed by land in the north
    elevation[z < -0.8] = -3000.0                          # the ocean around the south pole
    fill = ocean_fill(grid, elevation, 6.4e6)
    level = fill.level_for_land_fraction(0.7)
    assert np.mean(fill.pass_level < level) == pytest.approx(0.3, abs=0.005)
    assert 0.5 < np.mean(fill.pass_level[z > 0.5] < level) < 1.0      # the sea fills part of the basin
    # A small closed hollow stays dry.
    small = np.where(z > 0.995, -1000.0, 500.0)     # 0.25% of the surface
    small[z < -0.8] = -3000.0
    assert (ocean_fill(grid, small, 6.4e6).pass_level[z > 0.995] >= 500.0).all()


@pytest.mark.slow
def test_sea_level_follows_water_volume():
    dry = world(seed=1, body={"water_mass_fraction": 3e-4})
    wet = world(seed=1, body={"water_mass_fraction": 1.5e-3})
    assert dry.state.surface.land_fraction > wet.state.surface.land_fraction
    for w in (dry, wet):
        ocean = w.surface["ocean"].values
        area = 4 * np.pi * w.state.bulk.radius_m**2 / ocean.size
        volume = -(w.surface["elevation"].values[ocean].astype(float)).sum() * area
        assert volume == pytest.approx(w.state.water.ocean_volume_m3, rel=0.02)
        # Ice sheets hold the rest of the surface water.
        ice = w.surface["ice_thickness"].values.astype(float).sum() * area * 917.0 / h.SEAWATER_DENSITY
        surface = w.state.water.surface_mass_kg / h.SEAWATER_DENSITY
        assert volume + ice == pytest.approx(surface, rel=0.02)


@pytest.mark.slow
def test_user_water_and_land_fraction_conflict_is_reported():
    w = world(seed=1, body={"water_mass_fraction": 3e-4}, surface={"land_fraction": 0.1})
    assert w.state.surface.land_fraction == pytest.approx(0.1, abs=0.01)
    assert any(i.kind == "conflict" and "land fraction 0.10" in i.message for i in w.state.issues)


@pytest.mark.slow
def test_dry_planet_has_no_ocean_and_warns_on_land_fraction():
    w = generate_world(PlanetSpec(seed=2, priors={"archetype": "hot_volcanic"}, surface={"land_fraction": 0.4}), RES)
    assert w.state.atmosphere.surface_water == "none"
    assert not w.surface["ocean"].values.any()
    assert w.state.surface.land_fraction == 1.0
    assert w.surface["elevation"].values.mean() == pytest.approx(0.0, abs=1.0)
    assert any(i.kind == "conflict" and i.subsystem == "surface" for i in w.state.issues)


@pytest.mark.slow
def test_frozen_ocean_flag(world_cache):
    w = world_cache(PlanetSpec(seed=1, priors={"archetype": "ice"}, surface={"tectonics": "heuristic"}))
    assert w.state.atmosphere.surface_water == "ice"
    assert w.surface.attrs["frozen_ocean"] == 1
    assert w.state.surface.frozen_ocean


@pytest.mark.slow
def test_giant_has_no_surface():
    w = generate_world(PlanetSpec(seed=1, priors={"archetype": "giant"}, body={"mass_mearth": 300.0}), RES)
    assert w.surface is None and w.grid is None
    assert w.state.surface is None


@pytest.mark.slow
def test_surface_is_deterministic():
    a, b = world(seed=5), world(seed=5)
    assert np.array_equal(a.surface["elevation"].values, b.surface["elevation"].values)
    assert not np.array_equal(a.surface["elevation"].values, world(seed=6).surface["elevation"].values)


# Plate terrain ---------------------------------------------------------------

def test_earth_like_hypsometry(temperate):
    e = temperate.surface["elevation"].values
    crust = temperate.surface["crust"].values
    assert np.median(e[crust == Crust.OCEANIC]) < -2500
    assert np.median(e[crust == Crust.CONTINENTAL]) - np.median(e[crust == Crust.OCEANIC]) > 3000
    assert np.median(e[crust == Crust.CONTINENTAL]) < 1500
    assert e.min() < -5000 and 2000 < e.max() < h.MAX_ELEVATION_EARTH_M * 1.5


def test_plate_features_present(temperate):
    ds = temperate.surface
    terrain = set(np.unique(ds["terrain"].values).tolist())
    assert {Terrain.MOUNTAIN, Terrain.TRENCH, Terrain.RIDGE, Terrain.ABYSSAL_PLAIN} <= terrain
    assert set(np.unique(ds["boundary"].values).tolist()) == {int(b) for b in Boundary}
    assert ds["plate"].values.min() == 0
    assert temperate.state.surface.features["plates"] >= h.PLATES_RANGE[0]


def test_oceanic_age_increases_away_from_ridges(temperate):
    ds = temperate.surface
    age = ds["crust_age"].values
    ridge = ds["terrain"].values == Terrain.RIDGE
    oceanic = np.isfinite(age)
    assert np.median(age[ridge]) < np.median(age[oceanic & ~ridge])
    assert np.isnan(age[ds["crust"].values == Crust.CONTINENTAL]).all()


def test_continents_avoid_spreading_ridges(temperate):
    ds = temperate.surface
    grid = temperate.grid
    ridge_cells = np.flatnonzero(ds["terrain"].values == Terrain.RIDGE)
    dist, _ = distance_from(grid, ridge_cells, temperate.state.bulk.radius_m / 1e3)
    continental = ds["crust"].values == Crust.CONTINENTAL
    assert np.median(dist[continental]) > np.median(dist[~continental])


def test_boundary_classification_on_two_plates():
    grid = build_grid(5000)
    fields = SurfaceFields(grid=grid, radius_m=c.R_EARTH)
    east = grid.points[:, 0] > 0
    # Plate 0 (x < 0) spins about +z, plate 1 about -z: they collide on one meridian and separate on the other.
    plates = Plates(count=2, seeds=np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
                    euler_poles=np.array([[0, 0, 1.0], [0, 0, -1.0]]),
                    angular_speed=np.array([1e-8, 1e-8]), cell_plate=east.astype(np.int16))
    omega = plates.euler_poles[plates.cell_plate] * plates.angular_speed[plates.cell_plate, None]
    velocity = np.cross(omega, grid.points) * fields.radius_m
    i, j, kind, _ = classify_boundaries(fields, plates, velocity)
    y = grid.points[i, 1]
    conv = kind == Boundary.CONVERGENT
    div = kind == Boundary.DIVERGENT
    # Plate 0 moves toward +y at x<0, plate 1 toward -y... collisions on one side, spreading on the other.
    assert conv.any() and div.any()
    assert np.sign(np.median(y[conv])) != np.sign(np.median(y[div]))


# Relief and landforms --------------------------------------------------------

def test_relief_factor_scales_inversely_with_gravity():
    assert relief_factor(c.G_EARTH) == pytest.approx(1.0)
    assert relief_factor(c.G_EARTH / 2) == pytest.approx(2.0)
    assert relief_factor(c.G_EARTH * 100) == h.RELIEF_FACTOR_RANGE[0]


@pytest.mark.slow
def test_low_gravity_worlds_have_taller_relief():
    small = generate_world(PlanetSpec(seed=7, priors={"archetype": "small_world", "enforce_constraints": False},
                                      body={"mass_mearth": 0.1}, interior={"tectonic_regime": "stagnant_lid"}), RES)
    big = generate_world(PlanetSpec(seed=7, priors={"archetype": "small_world", "enforce_constraints": False},
                                    body={"mass_mearth": 2.0}, interior={"tectonic_regime": "stagnant_lid"}), RES)
    assert small.state.surface.relief_factor > big.state.surface.relief_factor
    assert np.ptp(small.surface["elevation"].values) > np.ptp(big.surface["elevation"].values)


def test_crater_population_scales_with_age():
    rng = named_rng(0, "t")
    young = sample_crater_diameters(rng, 4e7, 0.5, 100, 2000)
    old = sample_crater_diameters(rng, 4e7, 4.5, 100, 2000)
    assert len(old) > 5 * len(young)
    assert old.min() >= 100 and old.max() <= 2000
    assert list(old) == sorted(old, reverse=True)
    assert sample_crater_diameters(rng, 4e7, 0.0, 100, 2000).size == 0


def test_crater_depth_regimes():
    assert crater_depth(1.0, 1.0) == pytest.approx(200.0)
    assert crater_depth(500.0, 1.0) < 0.2 * 500e3
    assert crater_depth(500.0, 2.0) > crater_depth(500.0, 1.0)


def test_single_landforms():
    grid = build_grid(5000)
    fields = SurfaceFields(grid=grid, radius_m=c.R_EARTH)
    centre = grid.points[0]
    add_volcano(fields, centre, 5000.0)
    assert 3000 < fields.elevation.max() <= 5000
    assert (fields.terrain == Terrain.VOLCANO).any()
    fields2 = SurfaceFields(grid=grid, radius_m=c.R_EARTH)
    add_crater(fields2, grid.points[100], 2000.0, 1.0)
    assert fields2.elevation.min() < -1000
    assert (fields2.terrain == Terrain.CRATER).any()


def test_inactive_worlds_are_cratered_and_heat_pipe_worlds_are_not(regime_worlds):
    assert regime_worlds["inactive"].state.surface.features["craters"] > 0
    assert regime_worlds["heat_pipe"].state.surface.features["craters"] == 0
    terrain = regime_worlds["heat_pipe"].surface["terrain"].values
    assert (terrain == Terrain.VOLCANO).any()


# Distances ---------------------------------------------------------------------

def test_graph_distance_approximates_great_circle():
    grid = build_grid(10_000)
    src = 0
    dist, nearest = distance_from(grid, [src], 1.0)
    true = np.arccos(np.clip(grid.points @ grid.points[src], -1, 1))
    far = true > 0.3
    ratio = dist[far] / true[far]
    assert 1.0 <= ratio.min() + 1e-9 and np.median(ratio) < 1.12
    assert (nearest == src).all()


def test_distance_limit_and_smoothing():
    grid = build_grid(5000)
    dist, nearest = distance_from(grid, [0], 1.0, limit_km=0.2)
    assert np.isinf(dist).any() and (nearest[np.isinf(dist)] == -1).all()
    spike = np.zeros(grid.size)
    spike[0] = 1.0
    out = smooth(grid, spike, 3)
    assert out.sum() > 0 and out.max() < 1.0
