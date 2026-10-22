"""Zoom into a world: pick the tile block around a place, build its local map, and keep it for reuse.

``zoom_region`` runs the whole chain (tiles, inherited surface, sub-grid
detail, downscaled climate, local erosion, glaciers and drainage) for the
region around a latitude and longitude. A region depends only on the world and
its request, so it can always be rebuilt; a cache folder, off by default,
keeps built regions so they load instead of being rebuilt.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from .. import __version__
from .. import heuristics as h
from .erode import ErodedRegion, erode_region
from .detail import RUGGEDNESS, synthesize_region
from .tiles import NODES_PER_TILE, level_for_resolution, region_for, region_grid, unit_to_face_uv

REGION_FORMAT = 1
# Fields of an eroded region stored per node, and those that may be absent.
NODE_FIELDS = ("elevation", "ocean", "receiver", "discharge_m3_s", "river_order", "basin", "lake", "lake_level_m",
               "endorheic", "drainage_area_km2", "inflow_m3_s", "eroded_m", "deposited_m", "temperature_k")
OPTIONAL_FIELDS = ("precipitation_m", "runoff_m", "perennial", "channel_head_km2", "ice_thickness_m", "glacial_m",
                   "sea_ice")


def unit_vector(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the unit vector at a latitude and longitude (degrees)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def region_block(world, lat_deg: float, lon_deg: float, resolution_m: Optional[float] = None,
                 level: Optional[int] = None, half_width_km: float = h.ZOOM_DEFAULT_HALF_WIDTH_KM,
                 nodes_per_tile: int = NODES_PER_TILE) -> tuple[int, int, int, int, int, int]:
    """Return the tile block (face, level, x0, y0, x1, y1) of a zoom request.

    The level comes from ``level``, else from the lattice spacing
    ``resolution_m`` (default ``ZOOM_DEFAULT_RESOLUTION_M``). The block covers
    ``half_width_km`` around the place and stays on one cube face.
    """
    radius = float(world.surface.attrs["radius_m"])
    if level is None:
        level = level_for_resolution(radius, resolution_m or h.ZOOM_DEFAULT_RESOLUTION_M, nodes_per_tile)
    return region_for(unit_vector(lat_deg, lon_deg), level, radius, half_width_km=half_width_km)


def zoom_region(world, lat_deg: float, lon_deg: float, *, resolution_m: Optional[float] = None,
                level: Optional[int] = None, half_width_km: float = h.ZOOM_DEFAULT_HALF_WIDTH_KM,
                nodes_per_tile: int = NODES_PER_TILE, ruggedness: str = "earth",
                cache_dir: Optional[str | Path] = None) -> ErodedRegion:
    """Return the local map of the region around a place on a world.

    ``ruggedness`` is 'earth' (default), 'dramatic' or 'gentle'. With
    ``cache_dir`` a region built before for the same world and request is
    loaded from there, and a new one is stored there.
    """
    if world.surface is None:
        raise ValueError(f"{world.state.name} has no solid surface to zoom into")
    block = region_block(world, lat_deg, lon_deg, resolution_m, level, half_width_km, nodes_per_tile)
    return zoom_block(world, block, nodes_per_tile=nodes_per_tile, ruggedness=ruggedness, cache_dir=cache_dir)


def square_block(world, lat_deg: float, lon_deg: float, tiles: int, level: int) -> tuple[int, int, int, int, int, int]:
    """Return a block of ``tiles`` × ``tiles`` tiles at ``level`` around a place, shifted to stay on one cube face."""
    face, s, t = unit_to_face_uv(unit_vector(lat_deg, lon_deg))
    count = 1 << level
    if tiles > count:
        raise ValueError(f"a face holds only {count} tiles per side at level {level}")
    x0 = int(np.clip(round(s * count - tiles / 2), 0, count - tiles))
    y0 = int(np.clip(round(t * count - tiles / 2), 0, count - tiles))
    return face, level, x0, y0, x0 + tiles - 1, y0 + tiles - 1


def zoom_block(world, block: tuple[int, int, int, int, int, int], *, nodes_per_tile: int = NODES_PER_TILE,
               ruggedness: str = "earth", cache_dir: Optional[str | Path] = None) -> ErodedRegion:
    """Return the local map of a tile block (face, level, x0, y0, x1, y1), built or read from ``cache_dir``."""
    if world.surface is None:
        raise ValueError(f"{world.state.name} has no solid surface to zoom into")
    if ruggedness not in RUGGEDNESS:
        raise ValueError(f"unknown ruggedness '{ruggedness}'; choose from: {', '.join(RUGGEDNESS)}")
    _, _, x0, y0, x1, y1 = block
    nodes = ((x1 - x0 + 1) * (nodes_per_tile - 1) + 1) * ((y1 - y0 + 1) * (nodes_per_tile - 1) + 1)
    if nodes > h.ZOOM_MAX_NODES:
        raise ValueError(f"the region would have {nodes:,} nodes (limit {h.ZOOM_MAX_NODES:,}); "
                         "ask for a coarser resolution or a smaller half-width")

    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / _cache_name(world, block, nodes_per_tile, ruggedness)
        if path.exists():
            return load_region(path)
    region = erode_region(world, synthesize_region(world, region_grid(*block, nodes_per_tile=nodes_per_tile),
                                                   ruggedness))
    if path is not None:
        save_region(region, path)
    return region


def _world_key(world) -> str:
    """Return a short fingerprint of a world's surface, so a cache never serves another world's regions."""
    elevation = np.ascontiguousarray(world.surface["elevation"].values, dtype=np.float32)
    return f"{zlib.crc32(elevation.tobytes()):08x}"


def _cache_name(world, block, nodes_per_tile: int, ruggedness: str) -> str:
    """Return the cache folder name of a region request."""
    face, level, x0, y0, x1, y1 = block
    return (f"f{face}_l{level}_x{x0}-{x1}_y{y0}-{y1}_n{nodes_per_tile}_{ruggedness}_"
            f"{_world_key(world)}_v{__version__}.r{REGION_FORMAT}.zarr")


def region_to_dataset(region: ErodedRegion) -> xr.Dataset:
    """Return an eroded region as a dataset: node fields, the fabric, categorical fields and the lattice block."""
    grid = region.grid
    data = {name: ("node", np.asarray(getattr(region, name))) for name in NODE_FIELDS}
    for name in OPTIONAL_FIELDS:
        value = getattr(region, name)
        if value is not None:
            data[name] = ("node", np.asarray(value))
    data["fabric"] = (("node", "vec"), np.asarray(region.fabric))
    for name, values in (region.categorical or {}).items():
        data[f"categorical_{name}"] = ("node", np.asarray(values))
    x0, y0, x1, y1 = grid.tiles()
    attrs = {"region_format": REGION_FORMAT, "worldgen_version": __version__, "face": grid.face,
             "level": grid.level, "tiles": [x0, y0, x1, y1], "nodes_per_tile": grid.nodes_per_tile,
             "radius_m": region.radius_m, "seed": int(region.seed), "exported_m3": float(region.exported_m3)}
    return xr.Dataset(data_vars=data, attrs=attrs)


def region_from_dataset(ds: xr.Dataset) -> ErodedRegion:
    """Return the eroded region stored in a dataset written by ``region_to_dataset``."""
    if int(ds.attrs.get("region_format", -1)) != REGION_FORMAT:
        raise ValueError(f"unsupported region format {ds.attrs.get('region_format')} (expected {REGION_FORMAT})")
    x0, y0, x1, y1 = (int(v) for v in ds.attrs["tiles"])
    grid = region_grid(int(ds.attrs["face"]), int(ds.attrs["level"]), x0, y0, x1, y1,
                       nodes_per_tile=int(ds.attrs["nodes_per_tile"]))
    fields = {name: ds[name].values for name in NODE_FIELDS}
    fields.update({name: (ds[name].values if name in ds else None) for name in OPTIONAL_FIELDS})
    categorical = {name[len("categorical_"):]: ds[name].values for name in ds.data_vars
                   if name.startswith("categorical_")}
    return ErodedRegion(grid=grid, radius_m=float(ds.attrs["radius_m"]), seed=int(ds.attrs["seed"]),
                        fabric=ds["fabric"].values, categorical=categorical or None,
                        exported_m3=float(ds.attrs["exported_m3"]), **fields)


def save_region(region: ErodedRegion, path: str | Path) -> Path:
    """Write an eroded region to a zarr folder and return its path."""
    import warnings

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        region_to_dataset(region).to_zarr(path, mode="w", consolidated=False)
    return path


def load_region(path: str | Path) -> ErodedRegion:
    """Read an eroded region written by ``save_region``."""
    return region_from_dataset(xr.open_zarr(Path(path), consolidated=False).load())
