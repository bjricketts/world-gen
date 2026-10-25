"""Global rasters of a world on a latitude–longitude grid, as values or as the colours of its maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .crs import equirect_geotransform, planet_crs_wkt
from .fields import ExportField, export_field
from .sampling import world_equirect_sampler
from .writers import write_raster

DEFAULT_WIDTH = 2048


@dataclass(frozen=True)
class Raster:
    """A sampled field with the description and georeferencing a file needs."""

    data: np.ndarray
    field: Optional[ExportField]
    meta: dict
    kind: str                       # 'values', 'codes' or 'colour'


def _world(world, epoch: Optional[float]):
    """Return the world at a kept epoch of its organic history, or as it is."""
    if epoch is None:
        return world
    from ..culture import world_at_epoch

    return world_at_epoch(world, epoch)


def _meta(world, field: str, rows: int, cols: int, epoch: Optional[float]) -> dict:
    """Return the description and georeferencing of a global raster."""
    radius = float(world.surface.attrs["radius_m"])
    meta = {"planet": world.state.name, "radius_m": radius, "field": field, "grid": "equirectangular",
            "bounds_deg": [-180.0, -90.0, 180.0, 90.0], "pixel_deg": [360.0 / cols, 180.0 / rows],
            "crs_wkt": planet_crs_wkt(world.state.name, radius), "geotransform": equirect_geotransform(rows, cols)}
    if epoch is not None:
        meta["epoch_years_ago"] = float(epoch)
    return meta


def global_raster(world, field: str = "elevation", width: int = DEFAULT_WIDTH,
                  epoch: Optional[float] = None) -> Raster:
    """Return a field of the world on a global equirectangular raster ``width`` × ``width/2`` (north up, 180° W first).

    ``epoch`` (years ago) takes a culture field from a kept epoch of an organic history.
    """
    world = _world(world, epoch)
    spec = export_field(world, field)
    sampler = world_equirect_sampler(world, width)
    data = spec.sample(sampler, world.surface)
    rows, cols = sampler.shape
    meta = _meta(world, field, rows, cols, epoch)
    meta.update(units=spec.units, description=spec.description, sampling=spec.sampling)
    if spec.labels is not None:
        meta["labels"] = {str(k): v for k, v in spec.labels.items()}
    return Raster(data, spec, meta, "codes" if spec.categorical else "values")


def global_colours(world, field: str = "elevation", width: int = DEFAULT_WIDTH, hillshade: bool = False,
                   epoch: Optional[float] = None) -> Raster:
    """Return a field's map colours on a global equirectangular raster (RGBA in 0–1)."""
    from ..render.culture import split_field
    from ..render.maps import field_colours

    world = _world(world, epoch)
    name, option = split_field(field)
    month = int(option) if name in ("temperature", "rainfall") and option is not None else None
    sampler = world_equirect_sampler(world, width)
    rgba = field_colours(world, name if month is not None else field, sampler, month=month, hillshade=hillshade)
    rows, cols = sampler.shape
    meta = _meta(world, field, rows, cols, epoch)
    meta.update(description=f"map colours of '{field}'", hillshade=hillshade)
    return Raster(rgba[..., :3], None, meta, "colour")


def export_global(world, path: str | Path, field: str = "elevation", width: int = DEFAULT_WIDTH,
                  colour: bool = False, hillshade: bool = False, epoch: Optional[float] = None,
                  limits: Optional[tuple[float, float]] = None) -> Path:
    """Write a field of the world as a global raster (.tif, .png, .r16 or .raw, chosen by suffix) and return the path.

    ``colour`` writes the map colours instead of values. ``limits`` fixes the
    value range of 16-bit PNG and RAW codes; heights default to the world's
    height scale (``world_height_limits``), shared with region exports.
    """
    raster = (global_colours(world, field, width, hillshade, epoch) if colour
              else global_raster(world, field, width, epoch))
    if limits is None and raster.kind == "values" and field in ("elevation", "surface"):
        from .regions import world_height_limits

        limits = world_height_limits(world)
    return write_raster(path, raster.data, raster.meta, raster.kind, limits)
