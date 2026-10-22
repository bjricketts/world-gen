"""Zoom: cube-sphere quadtree tiles and local maps of a region (milestone 7)."""

from .assemble import load_region, region_block, save_region, square_block, zoom_block, zoom_region
from .detail import DetailedRegion, synthesize_detail, synthesize_region
from .erode import ErodedRegion, erode_region
from .glacier import IceCover, glaciate, ice_cover
from .inherit import (GlobalSampler, InheritedRegion, connected_sea, downscale_climate, inherit_region,
                      lapse_temperature, orographic_factor, prevailing_wind)
from .tiles import (NODES_PER_TILE, RegionGrid, TileId, face_uv_to_unit, faces_uv, level_for_resolution,
                    region_for, region_grid, tile_at, unit_to_face_uv)

__all__ = ["connected_sea", "DetailedRegion", "downscale_climate", "erode_region", "ErodedRegion", "face_uv_to_unit",
           "faces_uv", "glaciate", "GlobalSampler", "ice_cover", "IceCover", "inherit_region", "InheritedRegion",
           "lapse_temperature", "level_for_resolution", "load_region", "NODES_PER_TILE", "orographic_factor",
           "prevailing_wind", "region_block", "region_for", "region_grid", "RegionGrid", "save_region", "square_block", "zoom_block",
           "synthesize_detail", "synthesize_region", "tile_at", "TileId", "unit_to_face_uv", "zoom_region"]
