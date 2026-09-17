"""River lines for maps: channels traced along the flow network and drawn as smoothed curves.

The river data stays on the grid; only the drawing is smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection

from ..world import World

SMOOTHING_PASSES = 3
RIVER_COLOUR = "#1d5fa8"
LAKE_COLOUR = "#4f8fd0"


@dataclass
class RiverLine:
    """One stretch of river with a constant stream order."""

    points: np.ndarray    # (k, 3) unit vectors from upstream to downstream
    order: int            # Strahler order
    discharge_m3_s: float  # flow at the downstream end


def has_rivers(world: World) -> bool:
    """Return whether the world carries river data."""
    return world.surface is not None and "river_order" in world.surface


def trace_rivers(world: World) -> list[RiverLine]:
    """Return the river network as stretches of constant order, each ending where it joins the next."""
    ds = world.surface
    order = ds["river_order"].values.astype(int)
    flow_to = ds["flow_to"].values.astype(np.int64)
    discharge = ds["discharge"].values
    points = world.grid.points
    river = order > 0
    lake = ds["lake"].values.astype(bool)

    # A stretch starts at a source or where the stream order changes.
    donors_same = np.zeros(order.size, dtype=bool)
    ups = np.flatnonzero(river & (flow_to != np.arange(order.size)))
    same = river[flow_to[ups]] & (order[flow_to[ups]] == order[ups])
    donors_same[flow_to[ups][same]] = True
    starts = np.flatnonzero(river & ~donors_same)

    lines = []
    for start in starts:
        cells = [start]
        cell = start
        while True:
            nxt = flow_to[cell]
            if nxt == cell:
                break
            if lake[nxt]:
                break
            cells.append(nxt)
            if not river[nxt] or order[nxt] != order[start]:
                break
            cell = nxt
        if len(cells) < 2:
            continue
        lines.append(RiverLine(points=points[cells], order=int(order[start]),
                               discharge_m3_s=float(discharge[cells[-1] if river[cells[-1]] else cells[-2]])))
    return lines


def smooth_line(points: np.ndarray, passes: int = SMOOTHING_PASSES) -> np.ndarray:
    """Return a polyline on the sphere smoothed by corner cutting, keeping its end points."""
    line = points
    for _ in range(passes):
        if len(line) < 3:
            break
        a, b = line[:-1], line[1:]
        cut = np.empty((2 * len(a), 3))
        cut[0::2] = 0.75 * a + 0.25 * b
        cut[1::2] = 0.25 * a + 0.75 * b
        line = np.vstack([line[:1], cut[1:-1], line[-1:]])
    return line / np.linalg.norm(line, axis=1, keepdims=True)


def _to_lonlat(points: np.ndarray) -> np.ndarray:
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1, 1)))
    return np.column_stack([lon, lat])


def river_collection(world: World, proj, min_order: int = 1, width_scale: float = 1.0) -> LineCollection:
    """Return the rivers of a world as a line collection in projection coordinates."""
    import cartopy.crs as ccrs

    x0, x1 = proj.x_limits
    y0, y1 = proj.y_limits
    span = max(x1 - x0, y1 - y0)
    segments, widths = [], []
    top = max(int(world.surface["river_order"].values.max()), 1)
    for line in trace_rivers(world):
        if line.order < min_order:
            continue
        lonlat = _to_lonlat(smooth_line(line.points))
        with np.errstate(invalid="ignore"):
            xy = proj.transform_points(ccrs.Geodetic(), lonlat[:, 0], lonlat[:, 1])[:, :2]
        ok = np.isfinite(xy).all(axis=1)
        # Split where the line leaves the visible globe or wraps around the map edge.
        jump = np.r_[False, np.hypot(*np.diff(xy, axis=0).T) > 0.05 * span]
        breaks = ~ok | jump
        piece = []
        for k in range(len(xy)):
            if breaks[k] and len(piece) > 1:
                segments.append(np.array(piece))
                widths.append(line.order)
                piece = []
            elif breaks[k]:
                piece = []
            if ok[k]:
                piece.append(xy[k])
        if len(piece) > 1:
            segments.append(np.array(piece))
            widths.append(line.order)
    lw = width_scale * (0.25 + 1.1 * (np.asarray(widths, dtype=float) / top) ** 1.5)
    return LineCollection(segments, linewidths=lw, colors=RIVER_COLOUR, capstyle="round", joinstyle="round",
                          zorder=3)


def draw_rivers(ax, world: World, proj, min_order: int = 1, width_scale: float = 1.0) -> None:
    """Add the world's rivers to a map axes drawn in ``proj``."""
    if has_rivers(world):
        ax.add_collection(river_collection(world, proj, min_order, width_scale))
