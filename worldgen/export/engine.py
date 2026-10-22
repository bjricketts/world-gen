"""Heightmaps sized for game engines, built on a lattice whose node count matches the engine's sizes.

Unity terrains take square heightmaps of 2ⁿ + 1 samples (up to 4097), which
blocks of 2ᵏ tiles of 129 nodes give exactly; Unreal landscapes recommend 127,
253, 505, 1009, 2017 and 4033 vertices (k·126 + 1), which blocks of k tiles of
127 nodes give (its largest recommended size, 8129, is 64·127 + 1 and has no
such lattice). Such a region is built on that lattice, so it needs no resampling.
Heights are 16-bit codes on the world's height scale; the sidecar gives the
values the engine's import dialog asks for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..zoom import level_for_resolution, square_block, zoom_block
from .regions import axial_step_m, region_raster, world_height_limits
from .writers import write_raster

ENGINES = {"unity": 129, "unreal": 127}          # nodes per tile edge
ENGINE_TILES = (1, 2, 4, 8, 16, 32)            # tiles per side; above 8 the block passes the zoom's node limit


def engine_sizes(engine: str) -> list[int]:
    """Return the heightmap sizes an engine lattice gives with 1, 2, 4, … tiles per side."""
    per = ENGINES[_check(engine)] - 1
    return [k * per + 1 for k in ENGINE_TILES]


def _check(engine: str) -> str:
    """Return the engine name, or raise for an unknown one."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine '{engine}'; choose from {', '.join(ENGINES)}")
    return engine


def engine_region(world, lat_deg: float, lon_deg: float, engine: str = "unity", size: int = 513,
                  resolution_m: Optional[float] = None, level: Optional[int] = None, ruggedness: str = "earth",
                  cache_dir=None):
    """Return a square region around a place whose lattice is ``size`` × ``size`` nodes for the engine.

    The level comes from ``level`` or the spacing ``resolution_m`` (default
    ``ZOOM_DEFAULT_RESOLUTION_M``), as for ``zoom_region``.
    """
    nodes = ENGINES[_check(engine)]
    sizes = engine_sizes(engine)
    if size not in sizes:
        raise ValueError(f"{engine} sizes are {', '.join(map(str, sizes))}")
    tiles = (size - 1) // (nodes - 1)
    if level is None:
        radius = float(world.surface.attrs["radius_m"])
        level = level_for_resolution(radius, resolution_m or h.ZOOM_DEFAULT_RESOLUTION_M, nodes)
    block = square_block(world, lat_deg, lon_deg, tiles, level)
    return zoom_block(world, block, nodes_per_tile=nodes, ruggedness=ruggedness, cache_dir=cache_dir)


def import_settings(engine: str, size: int, step_m: float, limits: tuple[float, float]) -> dict:
    """Return the values an engine's heightmap import needs for a map of ``size`` samples ``step_m`` apart."""
    lo, hi = limits
    extent = (size - 1) * step_m
    if engine == "unity":
        return {"resolution": size, "depth": "16 bit", "byte_order": "little-endian (Windows)",
                "terrain_width_m": extent, "terrain_length_m": extent, "terrain_height_m": hi - lo,
                "terrain_y_position_m": lo}
    # Unreal: code 32768 sits at the actor's height; Z scale 100 spans 512 m (Landscape Technical Guide).
    return {"resolution": size, "xy_scale": step_m * 100.0, "z_scale": (hi - lo) / 512.0 * 100.0,
            "location_z_cm": (lo + (hi - lo) * 32768 / 65535) * 100.0}


def export_engine_heightmap(world, region, path: str | Path, engine: str = "unity",
                            limits: Optional[tuple[float, float]] = None) -> Path:
    """Write an engine region's heights as a 16-bit .r16/.raw or .png on the world's height scale."""
    nodes = ENGINES[_check(engine)]
    rows, cols = region.grid.shape
    if region.grid.nodes_per_tile != nodes or rows != cols or rows not in engine_sizes(engine):
        raise ValueError(f"build the region with engine_region(..., engine='{engine}') first")
    if Path(path).suffix.lower() == ".tif":
        raise ValueError("engine heightmaps are 16-bit .r16, .raw or .png files")
    limits = limits or world_height_limits(world)
    raster = region_raster(world, region, "elevation")
    if np.nanmax(raster.data) > limits[1] or np.nanmin(raster.data) < limits[0]:
        raise ValueError("the region's heights leave the world's height scale; pass limits=")
    meta = dict(raster.meta, engine=engine, engine_import=import_settings(engine, rows, axial_step_m(region), limits))
    return write_raster(path, raster.data, meta, "values", limits)
