"""Tile pyramid (task 10.3): global levels, re-levelling, decimation, chains with blend bands, tile files."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from worldgen import heuristics as h
from worldgen.export import add_chain, build_global, export_pyramid
from worldgen.export.pyramid import blend_weight
from worldgen.export.sampling import point_sampler
from worldgen.zoom import region_grid
from worldgen.zoom.tiles import face_uv_to_unit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PLACE = (40.0, 20.0)


@pytest.fixture(scope="module")
def world(world_cache):
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


@pytest.fixture(scope="module")
def pyramid(world):
    pyr = build_global(world, top_level=1)
    add_chain(pyr, world, *PLACE, 4)
    return pyr


def _points(block):
    x0, y0, x1, y1 = block.tiles
    return region_grid(block.face, block.level, x0, y0, x1, y1, nodes_per_tile=h.PYRAMID_NODES_PER_TILE).points


def test_top_level_faces_meet_exactly(pyramid):
    blocks = pyramid.blocks(1)
    points = [_points(b) for b in blocks]
    heights = [b.fields["elevation"].ravel() for b in blocks]
    worst = 0.0
    for i in range(6):
        tree = cKDTree(points[i])
        for j in range(i + 1, 6):
            dist, idx = tree.query(points[j], distance_upper_bound=1e-7)
            shared = np.isfinite(dist)
            if shared.any():
                worst = max(worst, np.abs(heights[j][shared] - heights[i][idx[shared]]).max())
    assert worst < 1e-3


def test_top_level_keeps_the_global_surface_at_large_scale(world, pyramid):
    surface = world.surface["elevation"].values
    offsets = []
    for b in pyramid.blocks(1):
        sampler = point_sampler(world.grid, _points(b))
        under = sampler.sample(surface).ravel()
        land = under > 0
        offsets.append((b.fields["elevation"].ravel() - under)[land])
    assert abs(np.concatenate(offsets).mean()) < 10.0


def test_coarser_levels_are_every_other_node(pyramid):
    for fine, coarse in zip(pyramid.blocks(1), pyramid.blocks(0)):
        assert np.array_equal(coarse.fields["elevation"], fine.fields["elevation"][::2, ::2])
        assert coarse.fields["elevation"].shape == (h.PYRAMID_NODES_PER_TILE,) * 2


def test_chains_meet_their_parent_on_the_coverage_edge(pyramid):
    per = h.PYRAMID_NODES_PER_TILE - 1
    for level in (2, 3, 4):
        (child,) = pyramid.blocks(level)
        x0, y0, _, _ = child.tiles
        parent = pyramid.block_at(child.face, level - 1, x0 // 2, y0 // 2)
        px0, py0, _, _ = parent.tiles
        pz = parent.fields["elevation"]
        r0, c0 = (y0 // 2 - py0) * per, (x0 // 2 - px0) * per        # the parent tile under the child block
        under = pz[r0:r0 + per + 1, c0:c0 + per + 1]
        z = child.fields["elevation"]
        # parent nodes on the edge are reproduced, and edge midpoints are the mean of their two parent neighbours
        assert np.allclose(z[0, ::2], under[0]) and np.allclose(z[-1, ::2], under[-1])
        assert np.allclose(z[::2, 0], under[:, 0]) and np.allclose(z[::2, -1], under[:, -1])
        assert np.allclose(z[0, 1::2], 0.5 * (under[0, :-1] + under[0, 1:]))
        # inside the band the level keeps its own relief
        inner = z[2 * per // 4:2 * per * 3 // 4:2, 2 * per // 4:2 * per * 3 // 4:2]
        assert np.std(inner - under[per // 4:per * 3 // 4, per // 4:per * 3 // 4]) > 1.0


def test_blend_weight_ramps_over_a_tenth_of_a_tile():
    w = blend_weight((257, 257))
    band = round(128 * h.PYRAMID_BLEND_FRACTION)
    assert w[0, 100] == 1.0 and w[band, 100] == 0.0 and 0.0 < w[band // 2, 100] < 1.0
    assert w[128, 128] == 0.0


def test_tiles_and_index_place_every_node(pyramid, world, tmp_path):
    folder = export_pyramid(pyramid, world, tmp_path / "pyr", fields=("elevation", "ocean"))
    index = json.loads((folder / "tileset.json").read_text())
    counts = {int(k): len(v["tiles"]) for k, v in index["levels"].items()}
    assert counts == {0: 6, 1: 24, 2: 4, 3: 4, 4: 4}
    tile = index["levels"]["2"]["tiles"][0]
    from PIL import Image

    codes = np.asarray(Image.open(folder / tile["files"]["elevation"]))
    enc = index["elevation_encoding"]
    heights = enc["offset"] + enc["scale"] * codes.astype(float)
    block = pyramid.blocks(2)[0]
    assert np.abs(heights - block.tile(tile["x"], tile["y"])[::-1]).max() <= enc["scale"]
    # the index's node formula gives the lattice positions
    i, j = 5, 17
    s = (tile["x"] * 128 + j) / (4 * 128)
    t = (tile["y"] * 128 + i) / (4 * 128)
    x0, y0, _, _ = block.tiles
    grid = region_grid(block.face, 2, x0, y0, x0 + 1, y0 + 1, nodes_per_tile=129)
    row = (tile["y"] - y0) * 128 + i
    col = (tile["x"] - x0) * 128 + j
    assert np.allclose(face_uv_to_unit(block.face, s, t), grid.points[row * grid.shape[1] + col])
