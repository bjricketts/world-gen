"""Structural fabric: the tangent gradient and the fold/ridge grain it produces."""

import numpy as np
import pytest

from worldgen import PlanetSpec, load_world, save_world
from worldgen.grid import build_grid
from worldgen.surface.fabric import structural_fabric, tangent_gradient
from worldgen.surface.fields import Crust


def _unit(v):
    """Return the rows of ``v`` normalised, leaving zero rows zero."""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def test_tangent_gradient_matches_an_analytic_field():
    """The gradient of the height field z points toward the pole, along the tangent plane."""
    grid = build_grid(6000)
    p = grid.points
    gradient = tangent_gradient(grid, p[:, 2])
    zhat = np.array([0.0, 0.0, 1.0])
    expected = zhat - p[:, 2:3] * p            # z-hat projected into each tangent plane
    mid = np.abs(p[:, 2]) < 0.9
    cosine = np.einsum("ij,ij->i", _unit(gradient[mid]), _unit(expected[mid]))
    assert np.median(cosine) > 0.99
    # Its magnitude is the slope of z along the surface, cos(latitude).
    magnitude = np.linalg.norm(gradient[mid], axis=1)
    assert np.median(magnitude / np.linalg.norm(expected[mid], axis=1)) == pytest.approx(1.0, abs=0.1)


def test_gradient_and_fabric_lie_in_the_tangent_plane():
    grid = build_grid(3000)
    values = grid.points[:, 0] ** 2 + grid.points[:, 1]
    gradient = tangent_gradient(grid, values)
    assert np.abs(np.einsum("ij,ij->i", gradient, grid.points)).max() < 1e-9
    fabric = structural_fabric(grid, values * 3000.0, np.full(grid.size, Crust.CONTINENTAL, np.int8),
                               np.full(grid.size, np.nan), np.zeros(grid.size), 1.0)
    assert np.abs(np.einsum("ij,ij->i", fabric, grid.points)).max() < 1e-6
    assert np.linalg.norm(fabric, axis=1).max() <= 1.0 + 1e-9


def test_ocean_fabric_runs_along_the_ridge_and_fades_with_age():
    """A ridge at x = 0 (age growing with |x|) gives a grain along the ridge, strong only when young."""
    grid = build_grid(6000)
    p = grid.points
    age = np.abs(p[:, 0]) * 300.0                       # 0 at the ridge, older away from it
    fabric = structural_fabric(grid, np.zeros(grid.size), np.full(grid.size, Crust.OCEANIC, np.int8),
                               age, np.full(grid.size, np.nan), 1.0)
    ridge_axis = _unit(np.cross(np.array([1.0, 0.0, 0.0]), p))   # tangent to the constant-x contours
    young = (age < 60.0) & (np.abs(p[:, 0]) > 0.05)
    alignment = np.abs(np.einsum("ij,ij->i", _unit(fabric[young]), ridge_axis[young]))
    assert np.median(alignment) > 0.9
    assert np.linalg.norm(fabric[young], axis=1).mean() > 0.4
    assert np.linalg.norm(fabric[age > 200.0], axis=1).max() < 0.1   # old sea floor has lost its grain


def test_continental_fabric_runs_along_the_belt_not_across_it():
    """A mountain belt along x = 0 gives a grain along the belt, and none on the flat plains beside it."""
    grid = build_grid(6000)
    p = grid.points
    from worldgen import heuristics as h
    elevation = h.CONTINENT_BASE_M + 3500.0 * np.exp(-(p[:, 0] / 0.12) ** 2)
    fabric = structural_fabric(grid, elevation, np.full(grid.size, Crust.CONTINENTAL, np.int8),
                               np.full(grid.size, np.nan), np.zeros(grid.size), 1.0)
    belt_axis = _unit(np.cross(np.array([1.0, 0.0, 0.0]), p))
    on_belt = np.abs(p[:, 0]) < 0.08
    alignment = np.abs(np.einsum("ij,ij->i", _unit(fabric[on_belt]), belt_axis[on_belt]))
    assert np.median(alignment) > 0.9
    assert np.linalg.norm(fabric[on_belt], axis=1).mean() > 0.4
    plains = np.abs(p[:, 0]) > 0.5
    assert np.linalg.norm(fabric[plains], axis=1).max() < 0.1   # flat ground has no grain


def test_old_orogeny_and_single_plate_crust_have_no_grain():
    grid = build_grid(3000)
    p = grid.points
    from worldgen import heuristics as h
    elevation = h.CONTINENT_BASE_M + 3500.0 * np.exp(-(p[:, 0] / 0.12) ** 2)
    worn = structural_fabric(grid, elevation, np.full(grid.size, Crust.CONTINENTAL, np.int8),
                             np.full(grid.size, np.nan), np.full(grid.size, 5000.0), 1.0)
    assert np.linalg.norm(worn, axis=1).max() < 0.1               # a long-worn orogen has subdued grain
    primary = structural_fabric(grid, elevation, np.full(grid.size, Crust.PRIMARY, np.int8),
                                np.full(grid.size, np.nan), np.zeros(grid.size), 1.0)
    assert not primary.any()                                      # non-plate crust carries no grain


def test_fabric_is_deterministic():
    grid = build_grid(2000)
    p = grid.points
    args = (grid, p[:, 2] * 4000.0, np.full(grid.size, Crust.CONTINENTAL, np.int8),
            np.full(grid.size, np.nan), np.zeros(grid.size), 1.0)
    assert np.array_equal(structural_fabric(*args), structural_fabric(*args))


def test_built_world_stores_a_fabric_that_survives_saving(world_cache, tmp_path):
    """A generated surface carries a tangent-vector fabric that round-trips through save/load."""
    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"},
                                   surface={"tectonics": "heuristic"}))
    ds = world.surface
    assert "fabric" in ds
    fabric = ds["fabric"].values                    # (vec, cell)
    assert fabric.shape == (3, ds.sizes["cell"])
    assert np.isfinite(fabric).all()
    magnitude = np.linalg.norm(fabric, axis=0)
    assert magnitude.max() <= 1.0 + 1e-4
    assert magnitude.max() > 0.1                     # an Earth-like world has real grain somewhere
    points = build_grid(int(ds.attrs["grid_size"])).points
    tangent = np.einsum("ic,ci->c", fabric, points)
    assert np.abs(tangent).max() < 1e-4             # the grain lies in the tangent plane

    folder = save_world(world, tmp_path / "world")
    reloaded = load_world(folder)
    assert np.allclose(reloaded.surface["fabric"].values, fabric, atol=1e-5)
