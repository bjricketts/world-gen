"""Local erosion and drainage on a zoomed region: conservation, seams and continued rivers."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from worldgen import PlanetSpec, heuristics as h
from worldgen.zoom import erode_region, region_for, region_grid, synthesize_region


@pytest.fixture(scope="module")
def wet_world(world_cache):
    return world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"}, surface={"tectonics": "heuristic"}))


def _erode(world, block, nodes_per_tile=20):
    region = region_grid(*block, nodes_per_tile=nodes_per_tile)
    return erode_region(world, synthesize_region(world, region))


def test_region_erosion_conserves_sediment(wet_world):
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = int(np.argmax(wet_world.surface["elevation"].values))
    out = _erode(wet_world, region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1))
    eroded, deposited = out.eroded_m.sum(), out.deposited_m.sum()
    assert eroded > 0.0
    assert abs(eroded - deposited) < 0.02 * eroded            # river erosion moves material, it does not lose it


def test_region_erosion_is_deterministic(wet_world):
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = int(np.argmax(wet_world.surface["elevation"].values))
    block = region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1)
    a, b = _erode(wet_world, block), _erode(wet_world, block)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.discharge_m3_s, b.discharge_m3_s)


def test_region_has_a_drainage_network(wet_world):
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = int(np.argmax(wet_world.surface["elevation"].values))
    out = _erode(wet_world, region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1))
    assert np.isfinite(out.discharge_m3_s).all() and (out.discharge_m3_s >= 0.0).all()
    assert out.river_order.max() >= 1                         # some cells carry rivers
    assert (out.receiver >= 0).all() and (out.receiver < out.grid.size).all()


def test_adjacent_regions_agree_on_their_pinned_edge(wet_world):
    """Two neighbouring regions share the eroded elevation of the edge between them."""
    radius = float(wet_world.surface.attrs["radius_m"])
    face, level, x0, y0, _, _ = region_for(wet_world.grid.points[0], 7, radius)
    left = _erode(wet_world, (face, level, x0, y0, x0, y0))
    right = _erode(wet_world, (face, level, x0 + 1, y0, x0 + 1, y0))
    dist, match = cKDTree(left.grid.points).query(right.grid.points, distance_upper_bound=1e-9)
    shared = np.flatnonzero(np.isfinite(dist))
    assert shared.size > 0
    # The pinned edge is the inherited surface plus detail; the grain-smoothing in detail leaves a
    # sub-centimetre discrepancy at the very edge (below float32 elevation precision at this relief).
    assert np.abs(left.elevation[match[shared]] - right.elevation[shared]).max() < 0.02


def test_global_rivers_continue_into_the_region(wet_world):
    """A region a major river crosses receives that river's discharge at its edge."""
    discharge = wet_world.surface["discharge"].values
    land = ~wet_world.surface["ocean"].values
    mouth = int(np.argmax(np.where(land, discharge, 0.0)))
    radius = float(wet_world.surface.attrs["radius_m"])
    out = _erode(wet_world, region_for(wet_world.grid.points[mouth], 8, radius, tiles_each_side=1), nodes_per_tile=16)
    assert out.inflow_m3_s.sum() > 0.0                        # a global river enters across the edge
    assert out.discharge_m3_s.max() >= out.inflow_m3_s.max()  # its water is carried through the region
    assert out.river_order.max() >= 1
