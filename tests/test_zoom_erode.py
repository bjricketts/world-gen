"""Local erosion and drainage on a zoomed region: conservation, seams and continued rivers."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from worldgen import PlanetSpec, heuristics as h
from worldgen.zoom import erode_region, region_for, region_grid, synthesize_region


@pytest.fixture(scope="module")
def wet_world(world_cache):
    return world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"}, surface={"tectonics": "heuristic"}))


def _ice_free_peak(world):
    """Return the highest land cell with no ice and well below the snowline: mountains that rivers shape."""
    s = world.surface
    elevation = s["elevation"].values
    clear = (~s["ocean"].values) & (s["ice_thickness"].values == 0.0) & (elevation < s["snowline"].values - 1000.0)
    cells = np.flatnonzero(clear)
    return int(cells[np.argmax(elevation[cells])])


def _erode(world, block, nodes_per_tile=20):
    region = region_grid(*block, nodes_per_tile=nodes_per_tile)
    return erode_region(world, synthesize_region(world, region))


def test_region_erosion_conserves_sediment(wet_world):
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = int(np.argmax(wet_world.surface["elevation"].values))
    out = _erode(wet_world, region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1))
    area = out.grid.cell_area_m2(radius)
    eroded, deposited = out.eroded_m.sum() * area, out.deposited_m.sum() * area
    assert eroded > 0.0
    # Every eroded cubic metre is either deposited in the region or carried out across its edge.
    assert abs(eroded - deposited - out.exported_m3) < 0.02 * eroded


def test_region_erosion_is_deterministic(wet_world):
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = int(np.argmax(wet_world.surface["elevation"].values))
    block = region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1)
    a, b = _erode(wet_world, block), _erode(wet_world, block)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.discharge_m3_s, b.discharge_m3_s)


def test_region_has_a_drainage_network(wet_world):
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = _ice_free_peak(wet_world)
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
    # The pinned edge is the inherited surface plus detail, computed with a halo, so both agree exactly.
    assert np.abs(left.elevation[match[shared]] - right.elevation[shared]).max() < 1e-6


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


def test_dry_channels_carry_less_than_the_perennial_threshold(wet_world):
    """Channels are keyed to catchment area; only those with enough mean flow are perennial."""
    radius = float(wet_world.surface.attrs["radius_m"])
    peak = _ice_free_peak(wet_world)
    out = _erode(wet_world, region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1))
    channel = out.river_order > 0
    assert out.perennial is not None and not (out.perennial & ~channel).any()
    assert (out.discharge_m3_s[out.perennial] >= h.ZOOM_STREAM_MIN_DISCHARGE_M3_S).all()
    assert (out.discharge_m3_s[channel & ~out.perennial] < h.ZOOM_STREAM_MIN_DISCHARGE_M3_S).all()


def test_a_drier_climate_has_fewer_perennial_streams(wet_world):
    """The same terrain under less rain keeps its channels but fewer of them flow all year."""
    from dataclasses import replace

    from worldgen.zoom import erode_region

    radius = float(wet_world.surface.attrs["radius_m"])
    peak = _ice_free_peak(wet_world)
    region = region_grid(*region_for(wet_world.grid.points[peak], 7, radius, tiles_each_side=1), nodes_per_tile=20)
    detailed = synthesize_region(wet_world, region)
    wet = erode_region(wet_world, detailed)
    dry = erode_region(wet_world, replace(detailed, precipitation_m=0.2 * detailed.precipitation_m))
    assert dry.perennial.sum() < wet.perennial.sum()


def test_channels_start_sooner_on_steeper_ground():
    """Channel heads follow A·S² = const: a slope twice as steep needs a quarter of the catchment."""
    from worldgen.zoom.erode import channel_heads_km2

    region = region_grid(4, 8, 80, 80, 80, 80, nodes_per_tile=48)      # ~0.8 km lattice
    rows, cols = region.shape
    _, c = np.divmod(np.arange(region.size), cols)
    step = region.mean_spacing(6.371e6)
    gentle = channel_heads_km2(region, 0.005 * c * step, 6.371e6)
    steep = channel_heads_km2(region, 0.01 * c * step, 6.371e6)
    middle = (rows // 2) * cols + cols // 2
    assert gentle[middle] == pytest.approx(4.0 * steep[middle], rel=0.05)
    flat = channel_heads_km2(region, np.zeros(region.size), 6.371e6)
    assert (flat == h.ZOOM_CHANNEL_HEAD_MAX_KM2).all()


def test_channels_continue_downstream(wet_world):
    """Once a channel starts it runs on to the region's edge, sea or a lake, even across flatter ground."""
    radius = float(wet_world.surface.attrs["radius_m"])
    out = _erode(wet_world, region_for(wet_world.grid.points[_ice_free_peak(wet_world)], 7, radius,
                                       tiles_each_side=1))
    channel = np.flatnonzero(out.river_order > 0)
    down = out.receiver[channel]
    ends = (down == channel) | out.ocean[down] | out.lake[down] | out.grid.boundary()[down] \
        | (out.ice_thickness_m[down] > 0.0)
    assert ((out.river_order[down] > 0) | ends).all()
