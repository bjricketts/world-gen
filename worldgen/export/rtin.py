"""Simplified triangulations of 2ⁿ + 1 height grids: right-triangulated irregular networks.

The grid is split recursively into right triangles (Evans, Kirkpatrick &
Townsend 2001, in the formulation of Mapbox's Martini). A triangle's error is
the largest vertical distance between its plane and the heights of the nodes it
covers, plus, on a sphere, how far its hypotenuse chord sags below the surface;
errors propagate up to parents, so cutting at a tolerance leaves a conforming
mesh within that tolerance of every node. (Martini measures only the hypotenuse
midpoint, which lets the mesh depart from the heights by up to twice the
tolerance.)
With the border locked every border node is kept, so neighbouring tiles share
their border vertices and meet without cracks.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _triangle_coords(size):
    """Return the hypotenuse ends (ax, ay, bx, by) of every triangle of the implicit binary tree."""
    tile = size - 1
    count = tile * tile * 2 - 2
    out = np.zeros((count, 4), np.int32)
    for i in range(count):
        node = i + 2
        ax = ay = bx = by = cx = cy = 0
        if node & 1:
            bx = by = cx = tile
        else:
            ax = ay = cy = tile
        node >>= 1
        while node > 1:
            mx = (ax + bx) >> 1
            my = (ay + by) >> 1
            if node & 1:
                bx, by, ax, ay = ax, ay, cx, cy
            else:
                ax, ay, bx, by = bx, by, cx, cy
            cx, cy = mx, my
            node >>= 1
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = ax, ay, bx, by
    return out


@njit(cache=True)
def _plane_error(heights, ax, ay, bx, by, cx, cy):
    """Return the largest vertical distance between a triangle's plane and the heights of the nodes it covers."""
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    za, zb, zc = heights[ay, ax], heights[by, bx], heights[cy, cx]
    e = 0.0
    for y in range(min(ay, by, cy), max(ay, by, cy) + 1):
        for x in range(min(ax, bx, cx), max(ax, bx, cx) + 1):
            l1 = (by - cy) * (x - cx) + (cx - bx) * (y - cy)
            l2 = (cy - ay) * (x - cx) + (ax - cx) * (y - cy)
            if det > 0:
                if l1 < 0 or l2 < 0 or l1 + l2 > det:
                    continue
            elif l1 > 0 or l2 > 0 or l1 + l2 < det:
                continue
            w1 = l1 / det
            w2 = l2 / det
            d = abs(w1 * za + w2 * zb + (1.0 - w1 - w2) * zc - heights[y, x])
            if d > e:
                e = d
    return e


@njit(cache=True)
def rtin_errors(heights, points, radius, lock_border):
    """Return the error map of a (2ⁿ + 1)² height grid at unit-vector positions ``points`` on a sphere of
    ``radius`` (0 for a flat grid); with ``lock_border`` border nodes are always kept."""
    size = heights.shape[0]
    tile = size - 1
    coords = _triangle_coords(size)
    count = coords.shape[0]
    parents = count - tile * tile
    err = np.zeros((size, size))
    for i in range(count - 1, -1, -1):
        ax, ay, bx, by = coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3]
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        cx = mx + my - ay
        cy = my + ax - mx
        e = _plane_error(heights, ax, ay, bx, by, cx, cy)
        if radius > 0.0:
            pa = points[ay, ax]
            pb = points[by, bx]
            dot = min(1.0, pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2])
            e += radius * (1.0 - np.cos(0.5 * np.arccos(dot)))
        if lock_border and (mx == 0 or my == 0 or mx == tile or my == tile):
            e = np.inf
        if e > err[my, mx]:
            err[my, mx] = e
        if i < parents:
            left = err[(ay + cy) >> 1, (ax + cx) >> 1]
            right = err[(by + cy) >> 1, (bx + cx) >> 1]
            if left > err[my, mx]:
                err[my, mx] = left
            if right > err[my, mx]:
                err[my, mx] = right
    return err


@njit(cache=True)
def rtin_triangles(err, max_error):
    """Return the triangles (flat node indices, counter-clockwise in (column, row)) whose error is within
    ``max_error``."""
    size = err.shape[0]
    tile = size - 1
    out = np.empty((2 * tile * tile, 3), np.int64)
    k = 0
    stack = np.empty((64 * size + 64, 6), np.int64)
    stack[0] = (0, 0, tile, tile, tile, 0)
    stack[1] = (tile, tile, 0, 0, 0, tile)
    top = 2
    while top > 0:
        top -= 1
        ax, ay, bx, by, cx, cy = stack[top]
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        if abs(ax - cx) + abs(ay - cy) > 1 and err[my, mx] > max_error:
            stack[top] = (cx, cy, ax, ay, mx, my)
            stack[top + 1] = (bx, by, cx, cy, mx, my)
            top += 2
        else:
            out[k, 0] = ay * size + ax
            out[k, 1] = by * size + bx
            out[k, 2] = cy * size + cx
            k += 1
    return out[:k]


def is_rtin_size(n: int) -> bool:
    """Return whether a grid side of ``n`` nodes is 2ⁿ + 1."""
    return n >= 3 and ((n - 1) & (n - 2)) == 0
