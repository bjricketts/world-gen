"""Zoom: cube-sphere quadtree tiles and local maps of a region (milestone 7)."""

from .detail import DetailedRegion, synthesize_detail, synthesize_region
from .erode import ErodedRegion, erode_region
from .inherit import (GlobalSampler, InheritedRegion, downscale_climate, inherit_region, lapse_temperature,
                      orographic_factor, prevailing_wind)
from .tiles import (NODES_PER_TILE, RegionGrid, TileId, face_uv_to_unit, faces_uv, level_for_resolution,
                    region_for, region_grid, tile_at, unit_to_face_uv)

__all__ = ["DetailedRegion", "ErodedRegion", "GlobalSampler", "InheritedRegion", "NODES_PER_TILE", "RegionGrid",
           "TileId", "downscale_climate", "erode_region", "face_uv_to_unit", "faces_uv", "inherit_region",
           "lapse_temperature", "level_for_resolution", "orographic_factor", "prevailing_wind", "region_for",
           "region_grid", "synthesize_detail", "synthesize_region", "tile_at", "unit_to_face_uv"]
