"""Structural fabric: the dominant grain of the surface, for oriented zoom detail.

Mountain belts fold along strike and abyssal hills run parallel to the
spreading ridge, so relief is elongated along a preferred direction. Zoom
lays anisotropic detail along that direction (milestone 7), which needs the
grain to be known at generation time and stored on the surface.

The grain is recovered from the tectonic state the surface already carries:
the belt axis on continents runs along the elevation contours (perpendicular
to the topographic gradient), and the ridge axis on sea floor runs along the
crust-age contours (perpendicular to the age gradient). Each cell holds a
tangent vector whose direction is the grain and whose length (0 to 1) is how
pronounced it is; cells with no grain hold the zero vector. The direction is
an undirected line, so its sign is arbitrary and consumers align signs before
interpolating.
"""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from ..grid import SphereGrid
from .distance import smooth
from .fields import Crust


def tangent_gradient(grid: SphereGrid, values: np.ndarray) -> np.ndarray:
    """Return the surface gradient of a per-cell field, as a tangent vector at each cell.

    The gradient is the least-squares fit of a plane to each cell and its
    neighbours, in the tangent plane at the cell. Units are value per radian
    of arc.
    """
    p = grid.points
    n = grid.size
    adj = grid.neighbours
    rows = np.repeat(np.arange(n), np.diff(adj.indptr))
    cols = adj.indices
    diff = p[cols] - p[rows]
    edge_normal = p[rows]
    diff -= np.einsum("ij,ij->i", diff, edge_normal)[:, None] * edge_normal   # into the tangent plane
    dvalue = values[cols] - values[rows]

    a = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    np.add.at(a, rows, diff[:, :, None] * diff[:, None, :])
    np.add.at(b, rows, dvalue[:, None] * diff)
    # The normal equations are singular along each cell's own normal; pin that direction.
    reg = np.maximum(np.trace(a, axis1=1, axis2=2) / 2.0, 1e-12)
    a += reg[:, None, None] * np.einsum("ij,ik->ijk", p, p)
    gradient = np.linalg.solve(a, b[:, :, None])[:, :, 0]
    return gradient - np.einsum("ij,ij->i", gradient, p)[:, None] * p


def _along_contours(grid: SphereGrid, scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit direction along the contours of a field, and the gradient magnitude."""
    gradient = tangent_gradient(grid, smooth(grid, scalar, h.FABRIC_SMOOTH_STEPS))
    magnitude = np.linalg.norm(gradient, axis=1)
    along = np.cross(grid.points, gradient)   # the gradient turned 90° into the tangent plane
    norm = np.linalg.norm(along, axis=1, keepdims=True)
    return along / np.maximum(norm, 1e-30), magnitude


def structural_fabric(grid: SphereGrid, elevation: np.ndarray, crust: np.ndarray, crust_age_myr: np.ndarray,
                      orogeny_age_myr: np.ndarray, relief: float) -> np.ndarray:
    """Return the per-cell structural grain (n×3 tangent vectors, length 0 to 1).

    Continental relief takes the belt axis along its elevation contours,
    weighted by how high and how recent the orogeny is; oceanic crust takes
    the ridge axis along its crust-age contours, weighted toward young sea
    floor whose abyssal-hill fabric is not yet buried. Cells with no grain
    (flat plains, single-plate crust) hold the zero vector.
    """
    fabric = np.zeros((grid.size, 3))
    continental = crust == Crust.CONTINENTAL
    oceanic = crust == Crust.OCEANIC

    if oceanic.any():
        age = np.where(np.isfinite(crust_age_myr), crust_age_myr, h.FABRIC_OCEAN_AGE_FADE_MYR)
        along, _ = _along_contours(grid, age)
        strength = np.clip(1.0 - age / h.FABRIC_OCEAN_AGE_FADE_MYR, 0.0, 1.0)
        fabric[oceanic] = along[oceanic] * strength[oceanic, None]

    if continental.any():
        along, magnitude = _along_contours(grid, elevation)
        uplift = np.maximum(elevation - h.CONTINENT_BASE_M, 0.0) / (h.FABRIC_MOUNTAIN_REF_M * relief)
        recency = np.exp(-np.nan_to_num(orogeny_age_myr, nan=np.inf) / h.FABRIC_OROGENY_FADE_MYR)
        # A grain needs a defined slope: flat plateaus have relief but no direction.
        typical = np.median(magnitude[continental & (uplift > 0.0)]) if (continental & (uplift > 0.0)).any() else 0.0
        confidence = magnitude / (magnitude + h.FABRIC_CONFIDENCE_FRACTION * typical + 1e-30)
        strength = np.clip(uplift, 0.0, 1.0) * recency * confidence
        fabric[continental] = along[continental] * strength[continental, None]

    weak = np.linalg.norm(fabric, axis=1) < h.FABRIC_MIN_STRENGTH
    fabric[weak] = 0.0
    return fabric
