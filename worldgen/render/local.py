"""Stylised 2D map of a zoomed region (milestone 7).

A region is a regular lattice, so its map is a flat raster rather than a
projected globe: hypsometric or biome tints, hillshaded relief, a crenellated
coastline, lakes, and the local stream network drawn at a width that grows with
each river's discharge. Channels too dry to flow all year are drawn as dashed
washes, and glaciers and ice sheets in white with their margin outlined. The
palette follows the global maps so a zoomed map reads as the same world close
up. How much of the channel network is drawn follows the map's scale, and the
thresholds can be overridden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from .. import heuristics as h
from ..constants import SECONDS_PER_YEAR
from ..biosphere import Biome
from .maps import BIOME_COLOURS, elevation_colours, save_figure
from .rivers import LAKE_COLOUR, RIVER_COLOUR

COAST_COLOUR = "#22333b"
# The map shows as much of the network as its scale allows (Töpfer & Pillewizer 1966): drawn channels
# stay about STREAM_SPACING_MM apart on the page, so a channel is drawn once it drains a square of that
# side on the ground, or once it carries what such a catchment yields at REFERENCE_RUNOFF_M. Wet ground
# therefore shows more flowing streams than dry ground, and zooming in reveals smaller ones.
STREAM_SPACING_MM = 8.0
REFERENCE_RUNOFF_M = 0.3         # yearly runoff that sets the discharge threshold
STREAM_BASE_WIDTH = 0.35         # line width (pt) of the smallest drawn stream
STREAM_WIDTH_PER_DECADE = 0.6    # extra width per tenfold rise in discharge
STREAM_MAX_WIDTH = 2.4
DRY_CHANNEL_COLOUR = "#5a4630"   # dry washes: dashed and thin, so they read as channels without water
DRY_CHANNEL_WIDTH = 0.7
ICE_COLOUR = "#f4f8fb"           # glaciers and ice sheets, as on the global maps
ICE_EDGE_COLOUR = "#7d98ad"


def _base_rgb(world, region, biome: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the land/ocean colour image (rows, cols, 3) and the elevation and ocean rasters.

    Heights use the world's colour scale, so a lowland plain keeps its lowland
    colour however little relief the region spans.
    """
    rows, cols = region.grid.shape
    elevation = region.elevation.reshape(rows, cols)
    ocean = region.ocean.reshape(rows, cols)
    sea_ice = None if region.sea_ice is None else region.sea_ice.reshape(rows, cols)
    rgb, _, _ = elevation_colours(world, elevation, ocean, sea_ice=sea_ice)
    if biome and region.categorical and "biome" in region.categorical:
        table = np.array([to_rgb(BIOME_COLOURS.get(Biome(k), "#8a8a8a")) for k in range(len(Biome))])
        rgb = np.where(ocean[..., None], rgb, table[region.categorical["biome"].reshape(rows, cols)])
    return rgb, elevation, ocean


def _axial_step_m(region) -> float:
    """Return the mean distance between lattice neighbours along a row (m)."""
    rows, cols = region.grid.shape
    p = region.grid.points.reshape(rows, cols, 3)
    return float(np.linalg.norm(np.diff(p, axis=1), axis=2).mean()) * region.radius_m


def _shade(elevation: np.ndarray, step_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 35.0
           ) -> np.ndarray:
    """Return Lambertian shading relative to flat ground (1 = flat, >1 facing the light, <1 away).

    The scale is fixed, unlike a min–max stretched hillshade, so a few steep
    cells cannot wash out the rest of the map.
    """
    d_row, d_col = np.gradient(elevation * h.ZOOM_HILLSHADE_EXAGGERATION, step_m)
    normal = np.stack([-d_col, -d_row, np.ones_like(elevation)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    light = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    return np.clip(normal @ light, 0.0, 1.0) / np.sin(alt)


def _smooth2d(line: np.ndarray, passes: int = 3) -> np.ndarray:
    """Return a 2-D polyline smoothed by corner cutting, keeping its end points."""
    for _ in range(passes):
        if len(line) < 3:
            break
        a, b = line[:-1], line[1:]
        cut = np.empty((2 * len(a), 2))
        cut[0::2] = 0.75 * a + 0.25 * b
        cut[1::2] = 0.25 * a + 0.75 * b
        line = np.vstack([line[:1], cut[1:-1], line[-1:]])
    return line


def drawing_thresholds(km_per_mm: float) -> tuple[float, float]:
    """Return the catchment (km²) and discharge (m³/s) from which channels are drawn at a map scale."""
    area_km2 = (STREAM_SPACING_MM * km_per_mm) ** 2
    discharge = REFERENCE_RUNOFF_M * area_km2 * 1e6 / SECONDS_PER_YEAR
    return area_km2, discharge


def _river_lines(region, min_order: int, min_area_km2: float, min_discharge_m3_s: float = np.inf
                 ) -> tuple[list, list, list]:
    """Return smoothed channel stretches (in lattice coordinates), their line widths and whether each is wet.

    Channels are traced downstream as stretches of constant Strahler order and
    wetness. A stretch is drawn from ``min_order`` upward once its catchment
    reaches ``min_area_km2`` or its discharge ``min_discharge_m3_s``. Widths grow
    with discharge, so trunks are wider than tributaries.
    """
    cols = region.grid.shape[1]
    order, receiver = region.river_order, region.receiver
    discharge = region.discharge_m3_s
    wet = region.perennial if region.perennial is not None else order > 0
    river = (order >= min_order) & ((region.drainage_area_km2 >= min_area_km2) | (discharge >= min_discharge_m3_s))
    index = np.arange(region.grid.size)
    # A stretch starts at a source, where the stream order changes, or where the channel turns wet.
    ups = np.flatnonzero(river & (receiver != index))
    down = receiver[ups]
    same = river[down] & (order[down] == order[ups]) & (wet[down] == wet[ups])
    donor_same = np.zeros(region.grid.size, dtype=bool)
    donor_same[down[same]] = True
    smallest = max(float(discharge[river].min()), 1e-6) if river.any() else 1.0

    segments, widths, wets = [], [], []
    for start in np.flatnonzero(river & ~donor_same):
        cells, cell = [start], start
        while True:
            nxt = receiver[cell]
            if nxt == cell or region.lake[nxt]:
                break
            cells.append(nxt)
            if not river[nxt] or order[nxt] != order[start] or wet[nxt] != wet[start]:
                break
            cell = nxt
        if len(cells) < 2:
            continue
        cells = np.asarray(cells)
        segments.append(_smooth2d(np.column_stack([cells % cols, cells // cols]).astype(float)))
        decades = np.log10(max(float(discharge[cells[-1]]), smallest) / smallest)
        widths.append(min(STREAM_BASE_WIDTH + STREAM_WIDTH_PER_DECADE * decades, STREAM_MAX_WIDTH))
        wets.append(bool(wet[start]))
    return segments, widths, wets


def plot_local_map(world, region, *, mode: str = "relief", hillshade: bool = True, rivers: bool = True,
                   min_stream_order: int = 1, min_stream_area_km2: Optional[float] = None,
                   min_stream_discharge_m3_s: Optional[float] = None, graticule: bool = True,
                   title: Optional[str] = None, figsize: float = 7.0):
    """Return a stylised 2D map figure of an eroded region.

    ``mode`` is 'relief' (hypsometric tints) or 'biome' (biome colours).
    Channels are drawn once their catchment or discharge is large enough for
    the map's scale; ``min_stream_area_km2`` and ``min_stream_discharge_m3_s``
    override those thresholds and ``min_stream_order`` drops the smallest
    orders. Perennial streams are blue; dry channels are dashed.
    """
    import matplotlib.pyplot as plt

    rows, cols = region.grid.shape
    rgb, elevation, ocean = _base_rgb(world, region, biome=(mode == "biome"))

    ice = np.zeros((rows, cols)) if region.ice_thickness_m is None else region.ice_thickness_m.reshape(rows, cols)
    covered = ice > 0.0
    rgb[covered] = 0.25 * rgb[covered] + 0.75 * np.array(to_rgb(ICE_COLOUR))
    if hillshade:
        shade = _shade(elevation + ice, _axial_step_m(region))      # light falls on the ice surface
        strength = np.where(ocean, 0.25, np.where(covered, 0.45, 0.65))[..., None]
        rgb = np.clip(rgb * (1.0 - strength + strength * shade[..., None]), 0.0, 1.0)
    rgb[region.lake.reshape(rows, cols)] = to_rgb(LAKE_COLOUR)     # lake surfaces are flat: no shading

    fig, ax = plt.subplots(figsize=(figsize, figsize * rows / cols))
    ax.imshow(rgb, origin="lower", interpolation="bilinear")
    if ocean.any():
        # Dry ground below sea level (closed basins) is lifted out of the contour so only the sea's edge is drawn.
        coast = np.where(ocean, elevation, np.maximum(elevation, 1.0))
        ax.contour(coast, levels=[0.0], colors=[COAST_COLOUR], linewidths=0.8)
    if covered.any() and not covered.all():
        ax.contour(covered.astype(float), levels=[0.5], colors=[ICE_EDGE_COLOUR], linewidths=0.5)

    if graticule:
        lat = region.grid.lat.reshape(rows, cols)
        lon = region.grid.lon.reshape(rows, cols)
        ax.contour(lon, levels=8, colors="white", alpha=0.25, linewidths=0.4)
        ax.contour(lat, levels=8, colors="white", alpha=0.25, linewidths=0.4)

    _scale_bar(ax, region, cols, rows)
    ax.set_xlim(0, cols - 1)
    ax.set_ylim(0, rows - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if rivers and region.river_order.max() > 0:
        width_mm = fig.get_size_inches()[0] * ax.get_position().width * 25.4
        km_per_mm = cols * _axial_step_m(region) / 1e3 / width_mm
        area, discharge = drawing_thresholds(km_per_mm)
        area = area if min_stream_area_km2 is None else min_stream_area_km2
        discharge = discharge if min_stream_discharge_m3_s is None else min_stream_discharge_m3_s
        segments, widths, wets = _river_lines(region, min_stream_order, area, discharge)
        dry = [k for k, w in enumerate(wets) if not w]
        wet = [k for k, w in enumerate(wets) if w]
        if dry:
            ax.add_collection(LineCollection([segments[k] for k in dry], linewidths=DRY_CHANNEL_WIDTH,
                                             colors=DRY_CHANNEL_COLOUR, linestyles=(0, (2.5, 1.5))))
        if wet:
            ax.add_collection(LineCollection([segments[k] for k in wet], linewidths=[widths[k] for k in wet],
                                             colors=RIVER_COLOUR, capstyle="round", joinstyle="round"))
    return fig


def _scale_bar(ax, region, cols: int, rows: int) -> None:
    """Draw a rounded-length scale bar in the lower-left corner."""
    km_per_cell = region.grid.mean_spacing(region.radius_m) / 1e3
    span_km = km_per_cell * cols
    nice = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
    length_km = nice[np.searchsorted(nice, span_km * 0.25)] if span_km * 0.25 < nice[-1] else nice[-1]
    cells = length_km / km_per_cell
    x0, y0 = 0.04 * cols, 0.04 * rows
    ax.plot([x0, x0 + cells], [y0, y0], color="black", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + cells / 2, y0 + 0.015 * rows, f"{length_km:g} km", ha="center", va="bottom", fontsize=8)


def save_local_map(world, region, path: str | Path, **kwargs) -> Path:
    """Draw a region's local map and write it to an image file."""
    import matplotlib

    matplotlib.use("Agg")
    return save_figure(plot_local_map(world, region, **kwargs), path)
