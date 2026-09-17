import numpy as np
import pytest

from worldgen import PlanetSpec, generate_world
from worldgen import heuristics as h
from worldgen.grid import build_grid
from worldgen.surface import Crust as CrustType
from worldgen.surface import Terrain
from worldgen.surface.plates import seafloor_depth
from worldgen.surface.sealevel import ocean_fill
from worldgen.tectonics import run
from worldgen.tectonics.crust import FRONT_ANDEAN, FRONT_ARC, Crust, Plates
from worldgen.tectonics.initial import continent_mask
from worldgen.tectonics.kernels import find_contacts, rotate_points
from worldgen.tectonics.processes import StepContext, count_feedback
from worldgen.tectonics.remap import balance_continents, remap, tidy_plates
from worldgen.world import load_world, save_world

R = 6.371e6
N = 10_000


@pytest.fixture(scope="module")
def grid():
    return build_grid(N)


@pytest.fixture(scope="module")
def result(grid):
    return run(grid, R, 1.0, 0.4, 1.0, "supercontinent", 100.0, seed=5)


def ctx(grid, speed=5e4):
    return StepContext(radius_m=R, edge_rad=grid.mean_spacing(), relief=1.0, typical_speed=speed, dt=2.0)


def test_rotation_preserves_length_and_follows_plate():
    pts = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotvec = np.array([[0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]])
    out = rotate_points(pts, np.array([0, 1], dtype=np.int32), rotvec)
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)


def _pair(continental, ages, omega):
    """Two points 0.01 rad apart on the equator, on plates 0 and 1."""
    pts = np.array([[1.0, 0.0, 0.0], [np.cos(0.01), np.sin(0.01), 0.0]])
    nbr = np.array([[1, 2], [0, 2]])
    return find_contacts(pts, nbr, np.array([0, 1], dtype=np.int32), np.array(continental),
                         np.array(ages, dtype=float), np.ones(2, dtype=bool), np.array(omega), R, 2)


# Plate 0 moves toward +y (toward point 1), plate 1 is fixed: the points converge.
CONVERGING = [[0.0, 0.0, 8e-3], [0.0, 0.0, 0.0]]


def test_ocean_subducts_beneath_continent():
    consumed, front, speed, collision, _ = _pair([False, True], [50, 0], CONVERGING)
    assert consumed.tolist() == [True, False]
    assert front[1] == FRONT_ANDEAN and speed[1] > 0 and not collision.any()


def test_older_ocean_subducts():
    consumed, front, *_ = _pair([False, False], [20, 90], CONVERGING)
    assert consumed.tolist() == [False, True]
    assert front[0] == FRONT_ARC


def test_continents_collide_without_subduction():
    consumed, _, _, collision, pairs = _pair([True, True], [0, 0], CONVERGING)
    assert not consumed.any()
    assert (collision > 0).all() and pairs[0, 1] == pairs[1, 0] > 0


def test_diverging_points_do_not_interact():
    consumed, front, _, collision, _ = _pair([False, True], [50, 0], [[0.0, 0.0, -8e-3], [0.0, 0.0, 0.0]])
    assert not consumed.any() and not front.any() and not collision.any()


def test_remap_fills_gaps_with_young_ocean(grid):
    crust = Crust.empty(grid.points)
    crust.continental[:] = True
    crust.ocean_age[:] = 80.0
    lon = np.degrees(np.arctan2(grid.points[:, 1], grid.points[:, 0]))
    crust.alive[np.abs(lon) < 10] = False           # a 20° wide strip has no crust
    plates = Plates(omega=np.array([[0.0, 0.0, 8e-3]]), active=np.array([True]))   # ~5 cm/yr at the equator
    new = remap(crust, plates, grid, elapsed_myr=20.0, ctx=ctx(grid))
    middle = np.abs(lon) < 3
    assert (~new.continental[middle]).all()
    assert (new.ocean_age[middle] <= 20.0).all()
    assert new.continental[np.abs(lon) > 30].all()
    edge = (~new.continental) & (np.abs(lon) > 7)
    assert new.ocean_age[edge].mean() > new.ocean_age[middle].mean()


def test_tidy_plates_removes_stray_cells(grid):
    crust = Crust.empty(grid.points)
    crust.plate[:] = (grid.points[:, 2] > 0).astype(np.int32)
    stray = np.flatnonzero(grid.points[:, 2] > 0.5)[:5]
    crust.plate[stray] = 0
    tidy_plates(crust, Plates(omega=np.zeros((2, 3)), active=np.ones(2, dtype=bool)), grid)
    assert (crust.plate[stray] == 1).all()


@pytest.mark.parametrize("shift", [-0.05, 0.05])
def test_continental_area_is_restored(grid, shift):
    crust = Crust.empty(grid.points)
    crust.continental[:] = grid.points[:, 2] > 0.2 + shift * 2
    crust.cont_elev[crust.continental] = h.CONTINENT_BASE_M
    crust.ocean_age[:] = 50.0
    target = float((grid.points[:, 2] > 0.2).mean())
    balance_continents(crust, grid, target)
    assert crust.continental.mean() == pytest.approx(target, abs=0.003)


@pytest.mark.parametrize("start", ["supercontinent", "cratons"])
def test_initial_continents_cover_target_area(grid, start):
    mask = continent_mask(grid, start, 0.35, seed=2)
    assert mask.mean() == pytest.approx(0.35, abs=0.01)


def test_unknown_start_raises(grid):
    with pytest.raises(ValueError):
        continent_mask(grid, "pangaea", 0.3, seed=1)


def test_simulation_keeps_plates_and_continents(result):
    crust = result.crust
    assert crust.size == N and crust.alive.all()
    assert crust.continental.mean() == pytest.approx(0.4, abs=0.005)
    assert h.PLATES_RANGE[0] <= result.plates.count <= h.PLATES_RANGE[1]
    assert set(np.unique(crust.plate)) <= set(np.flatnonzero(result.plates.active))
    assert result.stats.subducted > 0 and result.stats.collisions >= 0


def test_simulation_creates_young_sea_floor(result):
    ocean_age = result.crust.ocean_age[~result.crust.continental]
    assert ocean_age.min() < 5.0
    assert np.median(ocean_age) < 150.0


def test_simulation_builds_mountains(result):
    crust = result.crust
    assert (crust.cont_elev[crust.continental] > h.CONTINENT_BASE_M + 1000).any()
    assert np.isfinite(crust.orogeny_age).any()


@pytest.mark.slow
def test_simulation_is_deterministic(grid):
    a = run(grid, R, 1.0, 0.4, 1.0, "cratons", 40.0, seed=9)
    b = run(grid, R, 1.0, 0.4, 1.0, "cratons", 40.0, seed=9)
    np.testing.assert_array_equal(a.crust.plate, b.crust.plate)
    np.testing.assert_array_equal(a.crust.cont_elev, b.crust.cont_elev)


@pytest.mark.slow
def test_snapshot_interval_is_rounded_to_remap_interval(grid):
    r = run(grid, R, 1.0, 0.4, 1.0, "supercontinent", 60.0, seed=1, snapshot_interval_myr=30.0)
    assert [s.time_myr for s in r.snapshots] == [-60.0, -20.0, 0.0]


@pytest.mark.slow
def test_snapshots_span_the_simulation(grid):
    r = run(grid, R, 1.0, 0.4, 1.0, "supercontinent", 60.0, seed=1, snapshot_interval_myr=20.0)
    assert [s.time_myr for s in r.snapshots] == [-60.0, -40.0, -20.0, 0.0]


def test_count_feedback_is_bounded():
    lo, hi = h.PLATE_COUNT_FEEDBACK
    assert count_feedback(100, 10) == lo
    assert count_feedback(1, 50) == hi
    assert count_feedback(10, 10) == 1.0


def test_seafloor_deepens_with_age_and_levels_off():
    depth = seafloor_depth(np.array([0.0, 10.0, 20.0, 60.0, 200.0, 1000.0]))
    assert (np.diff(depth) < 0).all()
    assert depth[0] == h.OCEAN_RIDGE_DEPTH_M
    assert depth[-1] == pytest.approx(h.OCEAN_MAX_DEPTH_M, abs=1.0)
    young = h.OCEAN_RIDGE_DEPTH_M + h.OCEAN_SUBSIDENCE_M_PER_SQRT_MYR * np.sqrt(h.OCEAN_PLATE_TRANSITION_MYR)
    assert seafloor_depth(np.array([h.OCEAN_PLATE_TRANSITION_MYR]))[0] == pytest.approx(young, abs=5.0)


def spec(**surface):
    return PlanetSpec(seed=2, priors={"archetype": "temperate"}, interior={"tectonic_regime": "mobile_lid"},
                      surface={"tectonics_duration_myr": 60, **surface})


@pytest.fixture(scope="module")
def simulated_world():
    return generate_world(spec(tectonics_start="cratons"), "preview", snapshot_interval_myr=30)


def test_simulated_world_surface(simulated_world):
    ds = simulated_world.surface
    features = simulated_world.state.surface.features
    assert ds.attrs["tectonics"] == "simulated"
    assert features["simulated_myr"] == 60
    assert features["plates"] == len(np.unique(ds["plate"].values))
    assert ds["plate"].values.min() == 0
    oceanic = ds["crust"].values == CrustType.OCEANIC
    assert np.isfinite(ds["crust_age"].values[oceanic]).all()
    assert np.isnan(ds["crust_age"].values[~oceanic]).all()
    present = set(np.unique(ds["terrain"].values).tolist())
    assert {Terrain.RIDGE, Terrain.MOUNTAIN} <= present
    assert any("simulated for 60 Myr from a cratons start" in i.message for i in simulated_world.state.issues)


def test_snapshots_saved_and_loaded(simulated_world, tmp_path):
    ds = simulated_world.surface
    assert ds["snapshot_elevation"].dims == ("time", "cell")
    assert ds["time"].values.tolist() == [-60.0, -20.0, 0.0]
    # Every snapshot holds the same ocean volume as the present surface.
    for row in ds["snapshot_elevation"].values.astype(float):
        fill = ocean_fill(build_grid(ds.sizes["cell"]), row, simulated_world.state.bulk.radius_m)
        assert fill.volume(0.0) == pytest.approx(simulated_world.state.water.ocean_volume_m3, rel=0.01)
    loaded = load_world(save_world(simulated_world, tmp_path / "w"))
    np.testing.assert_array_equal(loaded.surface["snapshot_plate"].values, ds["snapshot_plate"].values)


@pytest.mark.slow
def test_duration_is_limited_by_planet_age():
    w = generate_world(PlanetSpec(seed=1, priors={"archetype": "temperate"}, star={"age_gyr": 0.03},
                                  interior={"tectonic_regime": "mobile_lid"}), "preview")
    assert w.state.surface.features["simulated_myr"] == pytest.approx(30.0)


@pytest.mark.slow
def test_heuristic_mode_still_available():
    w = generate_world(spec(tectonics="heuristic"), "preview")
    assert w.surface.attrs["tectonics"] == "heuristic"
    assert "simulated_myr" not in w.state.surface.features
    assert "time" not in w.surface.dims


def test_start_is_drawn_per_seed():
    from worldgen.priors.sampling import resolve

    starts = {resolve(PlanetSpec(seed=s)).values["surface.tectonics_start"] for s in range(12)}
    assert starts == {"supercontinent", "cratons"}


def test_invalid_tectonics_option_is_rejected():
    with pytest.raises(ValueError):
        PlanetSpec(surface={"tectonics": "magic"})
