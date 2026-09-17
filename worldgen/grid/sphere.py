"""Global spherical grid: a Fibonacci point set with its triangulation and neighbours.

Points are unit vectors. Physical distances are obtained by multiplying
angular distances by the planet radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import ConvexHull, cKDTree

RESOLUTIONS = {"preview": 10_000, "standard": 40_000, "high": 160_000}
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_points(n: int) -> np.ndarray:
    """Return ``n`` nearly uniformly spaced unit vectors (shape n×3)."""
    i = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    theta = GOLDEN_ANGLE * i
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


@dataclass(frozen=True)
class SphereGrid:
    """Points on the unit sphere with triangles, neighbour lists and edge lengths."""

    points: np.ndarray        # (n, 3) unit vectors
    triangles: np.ndarray     # (m, 3) point indices
    neighbours: csr_matrix    # symmetric adjacency; data = angular edge length [rad]

    @property
    def size(self) -> int:
        """Return the number of points."""
        return len(self.points)

    @property
    def lat(self) -> np.ndarray:
        """Return latitudes in degrees."""
        return np.degrees(np.arcsin(np.clip(self.points[:, 2], -1, 1)))

    @property
    def lon(self) -> np.ndarray:
        """Return longitudes in degrees, −180 to 180."""
        return np.degrees(np.arctan2(self.points[:, 1], self.points[:, 0]))

    @property
    def cell_solid_angle(self) -> float:
        """Return the solid angle (sr) represented by each point; the point set is uniform."""
        return 4.0 * np.pi / self.size

    def mean_spacing(self, radius_m: float = 1.0) -> float:
        """Return the mean distance between neighbouring points, in units of ``radius_m``."""
        return float(self.neighbours.data.mean()) * radius_m

    def neighbour_indices(self, i: int) -> np.ndarray:
        """Return the indices of the points adjacent to point ``i``."""
        return self.neighbours.indices[self.neighbours.indptr[i]:self.neighbours.indptr[i + 1]]

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return each undirected edge once, as index arrays (i, j) with i < j."""
        coo = self.neighbours.tocoo()
        keep = coo.row < coo.col
        return coo.row[keep], coo.col[keep]

    @property
    def tree(self) -> cKDTree:
        """Return a KD-tree over the points for nearest-point queries."""
        return _tree(self.size)

    def nearest(self, vectors: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return chord distances and indices of the ``k`` grid points nearest to each vector."""
        return self.tree.query(vectors, k=k)


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the angle (rad) between paired unit vectors."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("ij,ij->i", a, b))


@lru_cache(maxsize=4)
def _tree(n: int) -> cKDTree:
    """Return the cached KD-tree for a grid size."""
    return cKDTree(fibonacci_points(n))


@lru_cache(maxsize=4)
def build_grid(n: int) -> SphereGrid:
    """Return the Fibonacci grid with ``n`` points (cached, so repeated calls are free)."""
    if n < 12:
        raise ValueError("a grid needs at least 12 points")
    points = fibonacci_points(n)
    triangles = ConvexHull(points).simplices.astype(np.int64)
    rows = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    cols = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
    adjacency = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    adjacency.sum_duplicates()
    # Edge weights are the angular lengths of the edges.
    row_index = np.repeat(np.arange(n), np.diff(adjacency.indptr))
    adjacency.data = _angular_distance(points[row_index], points[adjacency.indices])
    return SphereGrid(points=points, triangles=triangles, neighbours=adjacency)


def resolve_resolution(resolution: int | str) -> int:
    """Return the point count for a named resolution or an explicit integer."""
    if isinstance(resolution, str):
        try:
            return RESOLUTIONS[resolution]
        except KeyError:
            raise ValueError(f"unknown resolution '{resolution}'; choose from {', '.join(RESOLUTIONS)} "
                             "or give a point count") from None
    return int(resolution)
