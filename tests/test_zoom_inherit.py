"""Inheriting the global surface into a region, and downscaling its climate."""

import numpy as np
import pytest

from worldgen import PlanetSpec, heuristics as h
from worldgen.constants import R_EARTH
from worldgen.grid import build_grid
from worldgen.zoom import (GlobalSampler, downscale_climate, inherit_region, lapse_temperature,
                           orographic_factor, prevailing_wind, region_for, region_grid)


def test_sampler_reproduces_node_values_exactly():
    """A direction on a grid point interpolates back to that point's value: no drift."""
    grid = build_grid(4000)
    sampler = GlobalSampler(grid)
    picked = grid.points[::37]
    field = np.random.default_rng(0).normal(size=grid.size)
    verts, weights = sampler.barycentric(picked)
    assert np.allclose(sampler.interpolate(field, verts, weights), field[::37], atol=1e-9)


def test_sampler_weights_form_a_partition_of_unity():
    grid = build_grid(3000)
    sampler = GlobalSampler(grid)
    q = grid.points[1::5] * 0.7 + grid.points[2::5][:len(grid.points[1::5])] * 0.3
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    _, weights = sampler.barycentric(q)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert (weights >= -1e-9).all()


def test_sampler_interpolates_a_constant_and_a_smooth_field():
    grid = build_grid(6000)
    sampler = GlobalSampler(grid)
    q = build_grid(2000).points                                   # a different set of directions
    verts, weights = sampler.barycentric(q)
    assert np.allclose(sampler.interpolate(np.full(grid.size, 3.5), verts, weights), 3.5)
    height = grid.points[:, 2]
    interpolated = sampler.interpolate(height, verts, weights)
    assert np.abs(interpolated - q[:, 2]).max() < 1e-3            # linear error over a triangle


def test_sampler_nearest_matches_the_kd_tree():
    grid = build_grid(3000)
    sampler = GlobalSampler(grid)
    q = build_grid(1000).points
    assert np.array_equal(sampler.nearest(q), grid.tree.query(q, k=1)[1])


def test_lapse_temperature_cools_with_height():
    t = np.array([288.0, 300.0])
    assert np.allclose(lapse_temperature(t, np.array([0.0, 1000.0])), [288.0, 300.0 - 1000.0 * h.LAPSE_RATE_K_PER_M])


def test_orographic_factor_wets_windward_and_dries_lee_slopes():
    grid = region_grid(4, 5, 10, 10, 10, 10, nodes_per_tile=32)
    from worldgen.surface.fabric import tangent_gradient
    delta = grid.points[:, 2] * 1.0e4                             # relief rising toward +z
    uphill = tangent_gradient(grid, delta)
    uphill /= np.maximum(np.linalg.norm(uphill, axis=1, keepdims=True), 1e-30)
    windward = orographic_factor(grid, R_EARTH, uphill, delta)
    lee = orographic_factor(grid, R_EARTH, -uphill, delta)
    assert np.median(windward) > 1.0 and np.median(lee) < 1.0
    flat = orographic_factor(grid, R_EARTH, uphill, np.zeros(grid.size))
    assert np.allclose(flat, 1.0)


@pytest.fixture
def zoomed(world_cache):
    """A temperate world and a region zoomed on its highest land."""
    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"},
                                   surface={"tectonics": "heuristic"}))
    peak = int(np.argmax(world.surface["elevation"].values))
    centre = world.grid.points[peak]
    block = region_for(centre, level=7, radius_m=float(world.surface.attrs["radius_m"]), tiles_each_side=1)
    return world, region_grid(*block, nodes_per_tile=24)


def test_inherited_region_matches_the_global_surface(zoomed):
    world, region = zoomed
    inherited = inherit_region(world, region)
    assert inherited.elevation.shape == (region.size,)
    assert np.isfinite(inherited.elevation).all()
    lo, hi = world.surface["elevation"].values.min(), world.surface["elevation"].values.max()
    assert lo - 1.0 <= inherited.elevation.min() and inherited.elevation.max() <= hi + 1.0
    assert np.array_equal(inherited.ocean, inherited.elevation < 0.0)
    assert set(inherited.categorical) >= {"crust", "terrain", "biome"}
    assert all(v.shape == (region.size,) for v in inherited.categorical.values())
    assert inherited.temperature_k.shape == (region.size,) and np.isfinite(inherited.temperature_k).all()
    assert inherited.precipitation_m is not None                 # a temperate world has rain


def test_inherited_fabric_lies_in_the_tangent_plane(zoomed):
    world, region = zoomed
    inherited = inherit_region(world, region)
    assert np.abs(np.einsum("ij,ij->i", inherited.fabric, region.points)).max() < 1e-6
    assert np.linalg.norm(inherited.fabric, axis=1).max() <= 1.0 + 1e-6


def test_prevailing_wind_is_a_unit_field(zoomed):
    world, region = zoomed
    wind = prevailing_wind(region.points, world.state)
    assert wind.shape == (region.size, 3)
    assert np.allclose(np.linalg.norm(wind, axis=1), 1.0, atol=1e-6)


def test_downscale_is_a_no_op_without_added_relief(zoomed):
    world, region = zoomed
    inherited = inherit_region(world, region)
    temperature, precipitation, _ = downscale_climate(inherited, inherited.elevation)
    assert np.allclose(temperature, inherited.temperature_k)
    assert np.allclose(precipitation, inherited.precipitation_m)
