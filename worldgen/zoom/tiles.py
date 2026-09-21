"""Cube-sphere quadtree tiles and the local grid a zoomed region is built on (milestone 7).

A region of the planet is addressed on a cube-sphere: six faces, each an
equiangular square that a quadtree subdivides into tiles. A tile is named by
``TileId(face, level, x, y)`` and its seed is a hash of the master seed and
that id, so any tile regenerates identically and neighbouring tiles share
their edge nodes exactly.

The equiangular mapping (``tan`` of the face angle) keeps cell sizes nearly
even across a face, and within one tile the lattice is regular. A region is
sampled on that lattice as a ``RegionGrid`` that exposes the same points,
neighbour graph and edges as the global grid, so the hydrology and rendering
code runs on it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from ..util import stable_hash

QUARTER = np.pi / 4.0
NODES_PER_TILE = 128         # samples along each side of a tile, edges included
# Each face is (outward normal, +s axis, +t axis); the axes are right-handed with the normal.
FACES = (
    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # +x
    (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),  # -x
    (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),   # +y
    (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # -y
    (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),   # +z
    (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # -z
)
_NORMALS = np.array([f[0] for f in FACES])
_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def face_uv_to_unit(face: int, s, t) -> np.ndarray:
    """Return the unit vector(s) at face coordinates ``s``, ``t`` in [0, 1]."""
    normal, right, up = FACES[face]
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    x = np.tan((2.0 * s - 1.0) * QUARTER)
    y = np.tan((2.0 * t - 1.0) * QUARTER)
    cube = normal + x[..., None] * right + y[..., None] * up
    return cube / np.linalg.norm(cube, axis=-1, keepdims=True)


def unit_to_face_uv(point: np.ndarray) -> tuple[int, float, float]:
    """Return the face and the ``s``, ``t`` in [0, 1] of a single unit vector."""
    point = np.asarray(point, dtype=float)
    face = int(np.argmax(_NORMALS @ point))
    normal, right, up = FACES[face]
    x = (point @ right) / (point @ normal)
    y = (point @ up) / (point @ normal)
    s = np.arctan(x) / QUARTER * 0.5 + 0.5
    t = np.arctan(y) / QUARTER * 0.5 + 0.5
    return face, float(s), float(t)


def _angular(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the angle (radians) between paired unit vectors, accurate at small angles."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("ij,ij->i", a, b))


def level_for_resolution(radius_m: float, spacing_m: float, nodes_per_tile: int = NODES_PER_TILE) -> int:
    """Return the quadtree level whose tiles sample the surface near ``spacing_m``."""
    face_arc_m = QUARTER * 2.0 * radius_m       # a face spans 90° of arc
    ratio = face_arc_m / max(spacing_m * (nodes_per_tile - 1), 1e-9)
    return max(int(np.ceil(np.log2(max(ratio, 1.0)))), 0)


@dataclass(frozen=True)
class TileId:
    """One tile of the cube-sphere quadtree: face, subdivision level and position on the face."""

    face: int
    level: int
    x: int
    y: int

    @property
    def count(self) -> int:
        """Return the number of tiles along one side of the face at this level."""
        return 1 << self.level

    def bounds(self) -> tuple[float, float, float, float]:
        """Return the tile's face coordinates (s0, s1, t0, t1)."""
        step = 1.0 / self.count
        return self.x * step, (self.x + 1) * step, self.y * step, (self.y + 1) * step

    def center_unit(self) -> np.ndarray:
        """Return the unit vector at the tile centre."""
        return face_uv_to_unit(self.face, (self.x + 0.5) / self.count, (self.y + 0.5) / self.count)

    def seed(self, master_seed: int) -> int:
        """Return the deterministic seed of this tile under a planet's master seed."""
        return stable_hash("worldgen.zoom.tile", master_seed, self.face, self.level, self.x, self.y)

    def parent(self) -> "TileId":
        """Return the tile one level coarser that contains this one."""
        if self.level == 0:
            raise ValueError("a level-0 tile has no parent")
        return TileId(self.face, self.level - 1, self.x // 2, self.y // 2)

    def children(self) -> list["TileId"]:
        """Return the four tiles one level finer that subdivide this one."""
        return [TileId(self.face, self.level + 1, 2 * self.x + dx, 2 * self.y + dy)
                for dy in (0, 1) for dx in (0, 1)]

    def neighbours(self, diagonal: bool = False) -> list["TileId"]:
        """Return the same-level tiles adjacent to this one on the same face."""
        offsets = _NEIGHBOUR_OFFSETS if diagonal else ((-1, 0), (1, 0), (0, -1), (0, 1))
        out = []
        for dx, dy in offsets:
            x, y = self.x + dx, self.y + dy
            if 0 <= x < self.count and 0 <= y < self.count:
                out.append(TileId(self.face, self.level, x, y))
        return out


@dataclass(frozen=True)
class RegionGrid:
    """A regular cube-sphere lattice over a rectangular block of tiles.

    ``points`` are unit vectors and ``neighbours`` is the eight-connected
    lattice graph with angular edge lengths, matching the global grid's
    interface so the drainage, erosion and distance code runs unchanged.
    ``shape`` is (rows, cols); row-major order matches ``points``.
    """

    points: np.ndarray
    neighbours: csr_matrix
    shape: tuple[int, int]
    face: int
    level: int
    origin: tuple[int, int]     # (x0, y0): the tile at the lower-left of the block
    nodes_per_tile: int

    @property
    def size(self) -> int:
        """Return the number of nodes."""
        return len(self.points)

    @property
    def lat(self) -> np.ndarray:
        """Return latitudes in degrees."""
        return np.degrees(np.arcsin(np.clip(self.points[:, 2], -1, 1)))

    @property
    def lon(self) -> np.ndarray:
        """Return longitudes in degrees, −180 to 180."""
        return np.degrees(np.arctan2(self.points[:, 1], self.points[:, 0]))

    def mean_spacing(self, radius_m: float = 1.0) -> float:
        """Return the mean distance between neighbouring nodes, in units of ``radius_m``."""
        return float(self.neighbours.data.mean()) * radius_m

    def cell_area_m2(self, radius_m: float) -> float:
        """Return the mean area a node represents (the mean lattice step squared)."""
        rows, cols = self.shape
        grid = self.points.reshape(rows, cols, 3)
        along_s = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean()
        along_t = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
        return float(along_s * along_t) * radius_m ** 2

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return each undirected edge once, as index arrays (i, j) with i < j."""
        coo = self.neighbours.tocoo()
        keep = coo.row < coo.col
        return coo.row[keep], coo.col[keep]

    def tree(self) -> cKDTree:
        """Return a KD-tree over the nodes for nearest-node queries."""
        return cKDTree(self.points)

    def tile_of(self, row: int, col: int) -> TileId:
        """Return the tile a lattice node falls in (edge nodes belong to the lower tile)."""
        per = self.nodes_per_tile - 1
        return TileId(self.face, self.level, self.origin[0] + col // per, self.origin[1] + row // per)


def _lattice_neighbours(points: np.ndarray, rows: int, cols: int) -> csr_matrix:
    """Return the eight-connected lattice graph with angular edge lengths."""
    index = np.arange(rows * cols).reshape(rows, cols)
    src, dst = [], []
    for dr, dc in _NEIGHBOUR_OFFSETS:
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        src.append(index[r0:r1, c0:c1].ravel())
        dst.append(index[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel())
    i = np.concatenate(src)
    j = np.concatenate(dst)
    data = _angular(points[i], points[j])
    return coo_matrix((data, (i, j)), shape=(rows * cols, rows * cols)).tocsr()


def region_grid(face: int, level: int, x0: int, y0: int, x1: int, y1: int,
                nodes_per_tile: int = NODES_PER_TILE) -> RegionGrid:
    """Return the lattice over the tile block [x0, x1] × [y0, y1] of a face at ``level``."""
    per = nodes_per_tile - 1
    cols = (x1 - x0) * per + nodes_per_tile
    rows = (y1 - y0) * per + nodes_per_tile
    count = 1 << level
    s = np.linspace(x0 / count, (x1 + 1) / count, cols)
    t = np.linspace(y0 / count, (y1 + 1) / count, rows)
    ss, tt = np.meshgrid(s, t)
    points = face_uv_to_unit(face, ss.ravel(), tt.ravel())
    return RegionGrid(points=points, neighbours=_lattice_neighbours(points, rows, cols), shape=(rows, cols),
                      face=face, level=level, origin=(x0, y0), nodes_per_tile=nodes_per_tile)


def tile_at(face: int, level: int, s: float, t: float) -> TileId:
    """Return the tile at a point given by its face coordinates."""
    count = 1 << level
    x = int(np.clip(int(s * count), 0, count - 1))
    y = int(np.clip(int(t * count), 0, count - 1))
    return TileId(face, level, x, y)


def region_for(center: np.ndarray, level: int, radius_m: float,
               half_width_km: float | None = None, tiles_each_side: int = 0
               ) -> tuple[int, int, int, int, int, int]:
    """Return the tile block (face, level, x0, y0, x1, y1) around a centre point.

    The block spans ``tiles_each_side`` tiles around the centre tile, or enough
    tiles to cover ``half_width_km`` of surface. It is clipped to the face, so a
    region does not cross a cube edge.
    """
    face, s, t = unit_to_face_uv(center)
    count = 1 << level
    cx = int(np.clip(int(s * count), 0, count - 1))
    cy = int(np.clip(int(t * count), 0, count - 1))
    if half_width_km is not None:
        tile_arc = QUARTER * 2.0 / count                     # a tile spans this much arc (radians)
        tiles_each_side = int(np.ceil(half_width_km * 1e3 / radius_m / tile_arc))
    x0 = max(cx - tiles_each_side, 0)
    y0 = max(cy - tiles_each_side, 0)
    x1 = min(cx + tiles_each_side, count - 1)
    y1 = min(cy + tiles_each_side, count - 1)
    return face, level, x0, y0, x1, y1
