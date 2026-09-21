"""Cube-sphere quadtree tiles, addressing, seeding and the region lattice."""

import numpy as np
import pytest

from worldgen.constants import R_EARTH
from worldgen.hydrology.graph import priority_flood, steepest_receivers
from worldgen.zoom import tiles
from worldgen.zoom.tiles import (RegionGrid, TileId, face_uv_to_unit, level_for_resolution, region_for,
                                 region_grid, tile_at, unit_to_face_uv)


def _random_unit_vectors(n, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_face_uv_round_trips_with_unit_vectors():
    """Every direction maps to a face coordinate and back to itself."""
    worst = 0.0
    for v in _random_unit_vectors(400):
        face, s, t = unit_to_face_uv(v)
        assert 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0
        back = face_uv_to_unit(face, s, t)
        assert np.linalg.norm(back) == pytest.approx(1.0, abs=1e-9)
        worst = max(worst, np.linalg.norm(back - v))
    assert worst < 1e-9


def test_face_coordinates_produce_unit_vectors():
    s, t = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
    for face in range(6):
        p = face_uv_to_unit(face, s.ravel(), t.ravel())
        assert np.allclose(np.linalg.norm(p, axis=1), 1.0)


def test_tile_seed_is_deterministic_and_distinct():
    a = TileId(2, 5, 10, 12)
    assert a.seed(7) == TileId(2, 5, 10, 12).seed(7)          # same tile, same seed
    assert a.seed(7) != a.seed(8)                             # a different planet
    assert a.seed(7) != TileId(2, 5, 10, 13).seed(7)          # a neighbouring tile
    assert a.seed(7) != TileId(3, 5, 10, 12).seed(7)          # a different face


def test_parent_and_children_are_consistent():
    tile = TileId(4, 6, 21, 34)
    assert tile.parent() == TileId(4, 5, 10, 17)
    children = tile.children()
    assert len(children) == 4
    assert all(c.parent() == tile for c in children)
    with pytest.raises(ValueError):
        TileId(0, 0, 0, 0).parent()


def test_tile_neighbours_stay_on_the_face():
    assert len(TileId(0, 3, 3, 3).neighbours()) == 4
    assert len(TileId(0, 3, 0, 0).neighbours()) == 2          # a corner tile has two edge neighbours
    assert len(TileId(0, 3, 0, 0).neighbours(diagonal=True)) == 3


def test_adjacent_tiles_share_their_edge_nodes_exactly():
    left = region_grid(4, 3, 2, 2, 2, 2, nodes_per_tile=16)
    right = region_grid(4, 3, 3, 2, 3, 2, nodes_per_tile=16)
    rows, cols = left.shape
    left_edge = left.points.reshape(rows, cols, 3)[:, -1]     # right column of the left tile
    right_edge = right.points.reshape(rows, cols, 3)[:, 0]    # left column of the right tile
    assert np.array_equal(left_edge, right_edge)


def test_region_lattice_is_regular_and_nearly_uniform():
    grid = region_grid(4, 4, 5, 5, 5, 5, nodes_per_tile=16)
    rows, cols = grid.shape
    p = grid.points.reshape(rows, cols, 3)
    steps = np.linalg.norm(np.diff(p, axis=1), axis=2)
    assert steps.max() / steps.min() < 1.05                  # a small equiangular tile is almost even


def test_region_grid_matches_the_global_grid_interface():
    grid = region_grid(4, 4, 5, 5, 6, 6, nodes_per_tile=16)
    rows, cols = grid.shape
    assert grid.size == rows * cols == len(grid.points)
    assert (grid.neighbours != grid.neighbours.T).nnz == 0   # symmetric graph
    degree = np.diff(grid.neighbours.indptr)
    interior = np.ones((rows, cols), bool)
    interior[[0, -1], :] = interior[:, [0, -1]] = False
    assert (degree.reshape(rows, cols)[interior] == 8).all()  # eight-connected interior
    i, j = grid.edges()
    assert (i < j).all() and i.size == grid.neighbours.nnz // 2


def test_hydrology_kernels_run_on_a_region_grid():
    """A region grid drives the drainage kernels: every node reaches a border outlet."""
    grid = region_grid(4, 4, 5, 5, 6, 6, nodes_per_tile=16)
    rows, cols = grid.shape
    z = (grid.points[:, 2] - grid.points[:, 2].min()) * 1e4   # a smooth slope
    outlet = np.zeros(grid.size, bool)
    border = np.zeros((rows, cols), bool)
    border[[0, -1], :] = border[:, [0, -1]] = True
    outlet[border.ravel()] = True
    adj = grid.neighbours
    filled = priority_flood(z, adj.indptr, adj.indices, outlet, 1e-3)
    receiver, distance = steepest_receivers(filled, adj.indptr, adj.indices, adj.data, outlet)
    assert filled.shape == z.shape
    assert (receiver[~outlet] != np.arange(grid.size)[~outlet]).all()   # interior cells all drain


def test_region_for_centres_and_clips_to_the_face():
    centre = face_uv_to_unit(4, 0.5, 0.5)
    face, level, x0, y0, x1, y1 = region_for(centre, 5, R_EARTH, tiles_each_side=1)
    assert face == 4 and (x1 - x0, y1 - y0) == (2, 2)
    count = 1 << level
    assert x0 <= count // 2 <= x1
    corner = face_uv_to_unit(4, 0.01, 0.01)
    _, _, x0, y0, x1, y1 = region_for(corner, 5, R_EARTH, tiles_each_side=2)
    assert x0 == 0 and y0 == 0                                # clipped at the face edge

    wide = region_for(centre, 6, R_EARTH, half_width_km=500.0)
    narrow = region_for(centre, 6, R_EARTH, half_width_km=100.0)
    assert (wide[4] - wide[2]) >= (narrow[4] - narrow[2])     # a wider region spans more tiles


def test_level_for_resolution_gets_finer_with_smaller_spacing():
    coarse = level_for_resolution(R_EARTH, 1000.0)
    fine = level_for_resolution(R_EARTH, 100.0)
    assert fine > coarse
    grid = region_grid(4, fine, 0, 0, 0, 0, nodes_per_tile=tiles.NODES_PER_TILE)
    assert grid.mean_spacing(R_EARTH) < 400.0                 # ~100 m target, within a small factor


def test_cell_area_is_positive_and_scales_with_radius():
    grid = region_grid(4, 4, 5, 5, 5, 5, nodes_per_tile=16)
    area = grid.cell_area_m2(R_EARTH)
    assert area > 0.0
    assert grid.cell_area_m2(2 * R_EARTH) == pytest.approx(4 * area, rel=1e-6)
