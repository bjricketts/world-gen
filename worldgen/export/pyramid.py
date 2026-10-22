"""A tile pyramid of a world's heights: the globe → continent → region → local chain with consistent seams.

Every level is a cube-sphere quadtree level on a lattice of 129 nodes per
tile edge, so a parent's nodes are a subset of its children's and tiles of one
block meet exactly.

- The global part is built at the top level (one zoom block per cube face,
  detail and erosion), re-levelled onto the global surface by removing its
  offset averaged over each global cell (the same at shared face edges),
  stitched where faces meet (detail is smoothed along the grain on each face's
  own lattice, so faces differ slightly there), and the coarser levels keep
  every other node of the level below, so they agree exactly with no blending.
- Chains on request run from the top level down to a finest level around a
  place: each level is the 2 × 2-tile block over its parent tile, re-levelled
  onto the parent (the large-scale part of their difference removed) and
  ramped to the parent's surface over a band where its coverage ends, so it
  meets the parent tiles around it without cracks.
- Re-levelling and blending leave shallow hollows; each level's heights are
  breached again (the zoom's hybrid breaching) so they drain, with the sea,
  lakes and the block's edge kept as they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .. import __version__
from .. import heuristics as h
from ..zoom import zoom_block
from ..zoom.assemble import unit_vector
from ..zoom.tiles import FACES, unit_to_face_uv
from .regions import world_height_limits
from .sampling import point_sampler
from .writers import write_raster

KEPT = ("elevation", "ocean", "lake", "rivers", "ice")
_SOURCES = {"ocean": "ocean", "lake": "lake", "rivers": "river_order", "ice": "ice_thickness_m"}
_DTYPES = {"elevation": np.float32, "ocean": np.bool_, "lake": np.bool_, "rivers": np.int8, "ice": np.float32}


@dataclass
class Block:
    """One built block of a level: a rectangle of tiles on one face with its fields per lattice node (rows along t)."""

    face: int
    level: int
    tiles: tuple[int, int, int, int]          # x0, y0, x1, y1
    fields: dict
    coverage: str                             # 'global' or 'chain'
    blended: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, cols) of the block's lattice."""
        return self.fields["elevation"].shape

    def tile(self, x: int, y: int, name: str = "elevation") -> np.ndarray:
        """Return one tile's nodes (129 × 129, rows along t) of a field."""
        per = h.PYRAMID_NODES_PER_TILE - 1
        x0, y0, _, _ = self.tiles
        r, c = (y - y0) * per, (x - x0) * per
        return self.fields[name][r:r + per + 1, c:c + per + 1]

    def tile_ids(self) -> list[tuple[int, int]]:
        """Return the (x, y) of the block's tiles."""
        x0, y0, x1, y1 = self.tiles
        return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


@dataclass
class Pyramid:
    """The levels of a world's tile pyramid: blocks per level, the global top level and the chains below it."""

    planet: str
    radius_m: float
    top_level: int
    levels: dict = field(default_factory=dict)      # level → list of Block

    def blocks(self, level: int) -> list[Block]:
        """Return the blocks built at a level."""
        return self.levels.get(level, [])

    def block_at(self, face: int, level: int, x: int, y: int) -> Optional[Block]:
        """Return the block holding a tile, or None."""
        for b in self.blocks(level):
            x0, y0, x1, y1 = b.tiles
            if b.face == face and x0 <= x <= x1 and y0 <= y <= y1:
                return b
        return None


def _fields_of(region) -> dict:
    """Return the kept fields of an eroded region as lattice arrays."""
    rows, cols = region.grid.shape
    out = {"elevation": np.asarray(region.elevation, dtype=float).reshape(rows, cols)}
    for name, attr in _SOURCES.items():
        values = getattr(region, attr)
        if values is not None:
            out[name] = np.asarray(values).reshape(rows, cols).astype(_DTYPES[name])
    return out


def _relevel_global(world, blocks: list[Block], grids: list) -> None:
    """Remove each block's offset from the global surface averaged over every global cell (in place).

    Averaging over cells pools all faces, and the correction is interpolated
    back barycentrically, so nodes shared by two faces get the same value.
    """
    surface = world.surface["elevation"].values
    size = world.grid.size
    total, count = np.zeros(size), np.zeros(size)
    samplers = []
    for b, g in zip(blocks, grids):
        sampler = point_sampler(world.grid, g.points)
        samplers.append(sampler)
        diff = b.fields["elevation"].ravel() - sampler.sample(surface).ravel()
        total += np.bincount(sampler.nearest, weights=diff, minlength=size)
        count += np.bincount(sampler.nearest, minlength=size)
    offset = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    for b, sampler in zip(blocks, samplers):
        b.fields["elevation"] = b.fields["elevation"] - sampler.sample(offset).reshape(b.shape)


def _shared_targets(blocks: list[Block], grids: list) -> list[dict]:
    """Return, per face block, its edge nodes shared with other faces and the mean height all faces give them."""
    from scipy.spatial import cKDTree

    edges = []
    for g in grids:
        idx = np.flatnonzero(g.boundary())
        edges.append((idx, g.points[idx]))
    sums = [np.array(b.fields["elevation"].ravel()[e[0]], dtype=float) for b, e in zip(blocks, edges)]
    counts = [np.ones(len(e[0])) for e in edges]
    for i in range(len(blocks)):
        tree = cKDTree(edges[i][1])
        for j in range(len(blocks)):
            if i == j:
                continue
            dist, k = tree.query(edges[j][1], distance_upper_bound=1e-7)
            shared = np.isfinite(dist)
            zj = blocks[j].fields["elevation"].ravel()[edges[j][0]]
            np.add.at(sums[i], k[shared], zj[shared])
            np.add.at(counts[i], k[shared], 1.0)
    return [dict(zip(e[0].tolist(), (s / c).tolist())) for e, s, c in zip(edges, sums, counts)]


def _stitch_faces(blocks: list[Block], grids: list) -> float:
    """Make face blocks agree on their shared edges (the mean of the faces there), ramping each face's change
    inward over the blend band; return the largest change at an edge.

    Detail is smoothed along the grain on each face's own lattice, so two faces
    differ slightly where they meet; one side at a time, each face takes the
    remaining difference at its edge and fades it out over the band.
    """
    targets = _shared_targets(blocks, grids)
    band = max(round((h.PYRAMID_NODES_PER_TILE - 1) * h.PYRAMID_BLEND_FRACTION), 1)
    ramp = np.clip(1.0 - np.arange(band + 1) / band, 0.0, 1.0)
    worst = 0.0
    for b, target in zip(blocks, targets):
        z = b.fields["elevation"]
        rows, cols = z.shape
        goal = np.full(rows * cols, np.nan)
        goal[list(target)] = list(target.values())
        goal = goal.reshape(rows, cols)
        for side in range(4):
            view = np.rot90(z, side)          # the side being stitched is row 0 of the view
            want = np.rot90(goal, side)[0]
            change = np.where(np.isfinite(want), want - view[0], 0.0)
            worst = max(worst, float(np.abs(change).max(initial=0.0)))
            view[:band + 1] += ramp[:, None] * change[None, :]
    return worst


def _drain(block: Block, grid) -> None:
    """Cut through the shallow hollows re-levelling and blending leave, so the heights drain again (in place).

    The sea, lakes and the block's edge are outlets and stay as they are.
    """
    from ..hydrology.graph import priority_breach

    z = block.fields["elevation"]
    outlet = grid.boundary().reshape(z.shape)
    for name in ("ocean", "lake"):
        if name in block.fields:
            outlet = outlet | block.fields[name].astype(bool)
    adj = grid.neighbours
    block.fields["elevation"] = priority_breach(z.ravel().astype(float), adj.indptr, adj.indices, outlet.ravel(),
                                                h.ZOOM_BREACH_MAX_M, 1e-3).reshape(z.shape)


def _decimate(block: Block) -> Block:
    """Return the block one level coarser, keeping every other node."""
    x0, y0, x1, y1 = block.tiles
    fields = {k: v[::2, ::2].copy() for k, v in block.fields.items()}
    return Block(block.face, block.level - 1, (x0 // 2, y0 // 2, x1 // 2, y1 // 2), fields, block.coverage)


def build_global(world, top_level: int = h.PYRAMID_TOP_LEVEL, ruggedness: str = "earth",
                 cache_dir=None) -> Pyramid:
    """Return a pyramid with its global part: levels 0 to ``top_level`` over the whole planet.

    ``top_level`` is built as one block per cube face (8 × 8 tiles at level 3,
    ~9 km on an Earth-sized planet, ~1 min per face); coarser levels keep every
    other node.
    """
    count = 1 << top_level
    if (count * (h.PYRAMID_NODES_PER_TILE - 1) + 1) ** 2 > h.ZOOM_MAX_NODES:
        raise ValueError(f"a face at level {top_level} exceeds the zoom's node limit; use level 3 or below")
    blocks, grids = [], []
    for face in range(6):
        block = (face, top_level, 0, 0, count - 1, count - 1)
        region = zoom_block(world, block, nodes_per_tile=h.PYRAMID_NODES_PER_TILE, ruggedness=ruggedness,
                            cache_dir=cache_dir)
        blocks.append(Block(face, top_level, block[2:], _fields_of(region), "global"))
        grids.append(region.grid)
    _relevel_global(world, blocks, grids)
    _stitch_faces(blocks, grids)
    for b, g in zip(blocks, grids):
        _drain(b, g)
    pyramid = Pyramid(world.state.name, float(world.surface.attrs["radius_m"]), top_level, {top_level: blocks})
    for level in range(top_level - 1, -1, -1):
        pyramid.levels[level] = [_decimate(b) for b in pyramid.levels[level + 1]]
    return pyramid


def _parent_values(parent: Block, child_tiles, shape) -> np.ndarray:
    """Return the parent's elevation bilinearly interpolated at a child block's nodes."""
    per = h.PYRAMID_NODES_PER_TILE - 1
    px0, py0, _, _ = parent.tiles
    cx0, cy0, _, _ = child_tiles
    rows, cols = shape
    # child node (r, c) sits at face index (cy0·per + r) on the child level, half that on the parent's
    rr = (cy0 * per + np.arange(rows)) / 2.0 - py0 * per
    cc = (cx0 * per + np.arange(cols)) / 2.0 - px0 * per
    grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
    return map_coordinates(parent.fields["elevation"], [grid_r, grid_c], order=1, mode="nearest")


def blend_weight(shape: tuple[int, int]) -> np.ndarray:
    """Return the weight of the parent's surface: 1 on the coverage edge, falling linearly to 0 over the band."""
    rows, cols = shape
    band = max(round((h.PYRAMID_NODES_PER_TILE - 1) * h.PYRAMID_BLEND_FRACTION), 1)
    r = np.minimum(np.arange(rows), rows - 1 - np.arange(rows))[:, None]
    c = np.minimum(np.arange(cols), cols - 1 - np.arange(cols))[None, :]
    return np.clip(1.0 - np.minimum(r, c) / band, 0.0, 1.0)


def add_chain(pyramid: Pyramid, world, lat_deg: float, lon_deg: float, finest_level: int,
              ruggedness: str = "earth", cache_dir=None) -> list[Block]:
    """Add the blocks from the level below the top down to ``finest_level`` around a place; return them.

    Each block is the 2 × 2 tiles over its parent tile, re-levelled onto the
    parent and ramped to the parent's surface where its coverage ends. Levels
    already holding a block over the place are reused.
    """
    if finest_level <= pyramid.top_level:
        raise ValueError(f"chains start below the top level ({pyramid.top_level})")
    face, s, t = unit_to_face_uv(unit_vector(lat_deg, lon_deg))
    added = []
    for level in range(pyramid.top_level + 1, finest_level + 1):
        n = 1 << (level - 1)
        px, py = min(int(s * n), n - 1), min(int(t * n), n - 1)
        parent = pyramid.block_at(face, level - 1, px, py)
        if pyramid.block_at(face, level, 2 * px, 2 * py) is not None:
            continue
        tiles = (2 * px, 2 * py, 2 * px + 1, 2 * py + 1)
        region = zoom_block(world, (face, level) + tiles, nodes_per_tile=h.PYRAMID_NODES_PER_TILE,
                            ruggedness=ruggedness, cache_dir=cache_dir)
        fields = _fields_of(region)
        z = fields["elevation"]
        under = _parent_values(parent, tiles, z.shape)
        sigma = h.PYRAMID_RELEVEL_SIGMA * 2.0          # the parent's spacing is two child steps
        z = z - gaussian_filter(z - under, sigma, mode="nearest")
        alpha = blend_weight(z.shape)
        fields["elevation"] = (1.0 - alpha) * z + alpha * under
        block = Block(face, level, tiles, fields, "chain", blended=True)
        _drain(block, region.grid)
        pyramid.levels.setdefault(level, []).append(block)
        added.append(block)
    return added


def build_pyramid(world, places=(), top_level: int = h.PYRAMID_TOP_LEVEL, ruggedness: str = "earth",
                  cache_dir=None) -> Pyramid:
    """Return a world's pyramid: the global levels and a chain for each (lat, lon, finest level) in ``places``."""
    pyramid = build_global(world, top_level, ruggedness, cache_dir)
    for lat, lon, finest in places:
        add_chain(pyramid, world, lat, lon, finest, ruggedness, cache_dir)
    return pyramid


def tile_corners(face: int, level: int, x: int, y: int) -> dict:
    """Return the longitude and latitude (degrees) of a tile's corners, image orientation (high t on top)."""
    from ..zoom.tiles import face_uv_to_unit

    count = 1 << level
    s = np.array([x, x + 1, x + 1, x]) / count
    t = np.array([y + 1, y + 1, y, y]) / count
    p = face_uv_to_unit(face, s, t)
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1)))
    return {k: [round(float(a), 7), round(float(b), 7)]
            for k, a, b in zip(("top_left", "top_right", "bottom_right", "bottom_left"), lon, lat)}


def export_pyramid(pyramid: Pyramid, world, folder: str | Path, fields=("elevation",), suffix: str = ".png",
                   limits: Optional[tuple[float, float]] = None) -> Path:
    """Write every tile of a pyramid as ``L<level>/<face>/<x>_<y><suffix>`` with a ``tileset.json`` index.

    Heights in 16-bit files share the world's height scale. Tiles hold 129 ×
    129 nodes with the tile's high-t edge on top; neighbouring tiles repeat
    their shared edge. The index describes how to place every node.
    """
    folder = Path(folder)
    limits = limits or world_height_limits(world)
    index = {"worldgen_version": __version__, "planet": pyramid.planet, "radius_m": pyramid.radius_m,
             "nodes_per_tile": h.PYRAMID_NODES_PER_TILE, "top_level": pyramid.top_level, "fields": list(fields),
             "height_limits_m": list(limits),
             "node_position": ("node (row i, column j) of tile (face, level, x, y), i = 0 on the image's bottom row: "
                               "s = (x·128 + j) / (2^level·128), t = (y·128 + i) / (2^level·128); "
                               "direction ∝ normal + tan((2s − 1)π/4)·right + tan((2t − 1)π/4)·up"),
             "faces": [{"normal": f[0].tolist(), "right": f[1].tolist(), "up": f[2].tolist()} for f in FACES],
             "levels": {}}
    encoding = None
    for level in sorted(pyramid.levels):
        entries = []
        for block in pyramid.blocks(level):
            for x, y in block.tile_ids():
                for name in fields:
                    if name not in block.fields:
                        continue
                    data = block.tile(x, y, name)[::-1]
                    kind = "values" if name in ("elevation", "ice") else "codes"
                    path = folder / f"L{level}" / str(block.face) / f"{x}_{y}_{name}{suffix}"
                    meta = {"field": name}
                    write_raster(path, data, meta, kind, limits if name == "elevation" else None, sidecar=False)
                    if name == "elevation":
                        encoding = meta["encoding"]
                entries.append({"face": block.face, "x": x, "y": y, "coverage": block.coverage,
                                "blended": block.blended, "corners_lonlat": tile_corners(block.face, level, x, y),
                                "files": {n: f"L{level}/{block.face}/{x}_{y}_{n}{suffix}" for n in fields
                                          if n in block.fields}})
        nominal = (np.pi / 2) / (1 << level) / (h.PYRAMID_NODES_PER_TILE - 1) * pyramid.radius_m
        index["levels"][str(level)] = {"nominal_step_m": float(nominal), "tiles": entries}
    index["elevation_encoding"] = encoding
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tileset.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return folder
