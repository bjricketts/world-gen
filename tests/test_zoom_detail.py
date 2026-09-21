"""Sub-grid detail synthesis: octave scheduling, grain-aligned ridges, seamless regeneration."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from worldgen import PlanetSpec, heuristics as h
from worldgen.zoom import inherit_region, region_for, region_grid, synthesize_region
from worldgen.zoom.detail import _detail_octaves, _elongate, synthesize_detail


def test_detail_adds_more_octaves_the_deeper_the_zoom():
    coarse = _detail_octaves(0.02, 0.02)
    fine = _detail_octaves(0.02, 0.02 / 16)
    assert fine > coarse
    assert _detail_octaves(0.02, 1e-9) == h.ZOOM_DETAIL_MAX_OCTAVES


def test_elongate_stretches_features_along_the_grain():
    """After grain-aligned smoothing a field varies less along the grain than across it."""
    from worldgen.surface.fabric import tangent_gradient

    grid = region_grid(4, 5, 10, 10, 10, 10, nodes_per_tile=48)
    up = np.array([0.0, 1.0, 0.0])
    grain = up - (grid.points @ up)[:, None] * grid.points
    grain /= np.linalg.norm(grain, axis=1, keepdims=True)        # unit grain, full strength
    field = np.random.default_rng(0).random(grid.size)
    gradient = tangent_gradient(grid, _elongate(grid, field, grain, np.ones(grid.size)))
    across = np.cross(grid.points, grain)                        # the tangent perpendicular to the grain
    along_grain = np.median(np.abs(np.einsum("ij,ij->i", gradient, grain)))
    across_grain = np.median(np.abs(np.einsum("ij,ij->i", gradient, across)))
    assert along_grain < 0.85 * across_grain                     # relief runs along the grain, not across it


@pytest.fixture
def detailed(world_cache):
    """A temperate world zoomed on its highest land, with detail synthesised."""
    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"},
                                   surface={"tectonics": "heuristic"}))
    peak = int(np.argmax(world.surface["elevation"].values))
    radius = float(world.surface.attrs["radius_m"])
    block = region_for(world.grid.points[peak], level=7, radius_m=radius, tiles_each_side=1)
    region = region_grid(*block, nodes_per_tile=24)
    return world, region, synthesize_region(world, region)


def test_detail_is_bounded_and_refines_the_surface(detailed):
    world, region, out = detailed
    assert np.isfinite(out.elevation).all()
    assert np.abs(out.detail).max() < 10_000.0                   # detail stays within a sensible range
    assert out.detail.std() > 1.0                                # it actually adds relief
    inherited = inherit_region(world, region)
    assert np.allclose(out.elevation, inherited.elevation + out.detail)
    assert np.array_equal(out.ocean, out.elevation < 0.0)
    changed = out.detail != 0.0
    assert not np.allclose(out.temperature_k[changed], inherited.temperature_k[changed])  # lapse applied


def test_detail_is_deterministic(detailed):
    world, region, out = detailed
    again = synthesize_region(world, region)
    assert np.array_equal(out.elevation, again.elevation)


def test_adjacent_regions_agree_at_shared_interior_nodes(world_cache):
    """Two overlapping regions produce the same detailed elevation on the nodes they share."""
    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"},
                                   surface={"tectonics": "heuristic"}))
    face, level, x0, y0, _, _ = region_for(world.grid.points[0], level=7,
                                           radius_m=float(world.surface.attrs["radius_m"]))
    left = synthesize_region(world, region_grid(face, level, x0, y0, x0 + 1, y0, nodes_per_tile=24))
    right = synthesize_region(world, region_grid(face, level, x0 + 1, y0, x0 + 2, y0, nodes_per_tile=24))

    dist, match = cKDTree(left.grid.points).query(right.grid.points, distance_upper_bound=1e-9)
    shared_right = np.flatnonzero(np.isfinite(dist))
    margin = h.ZOOM_DETAIL_SMOOTH_STEPS + 1
    lr, lc = left.grid.shape
    rr, rc = right.grid.shape

    def interior(flat, rows, cols):
        r, c = flat // cols, flat % cols
        return (r >= margin) & (r < rows - margin) & (c >= margin) & (c < cols - margin)

    keep = shared_right[interior(match[shared_right], lr, lc) & interior(shared_right, rr, rc)]
    assert keep.size > 0
    assert np.abs(left.elevation[match[keep]] - right.elevation[keep]).max() < 1e-6
