"""Raster writers: GeoTIFF, 16-bit PNG and RAW, each with a JSON sidecar that records how to read the values.

Heights and other continuous values go to GeoTIFF as 32-bit floats; PNG and
RAW hold 16-bit codes mapped linearly onto the value range (the sidecar gives
the offset and scale). Category codes are stored as integers with −1 (GeoTIFF)
or the largest code (PNG, RAW) marking empty cells. Colour images are 8-bit
RGB(A). GeoTIFF needs rasterio (the ``export`` extra).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .. import __version__

RASTER_FORMATS = {".tif": "geotiff", ".tiff": "geotiff", ".png": "png", ".r16": "raw16", ".raw": "raw16"}
U16_MAX = 65535


def raster_format(path: Path) -> str:
    """Return the format a file name asks for."""
    fmt = RASTER_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"unknown raster format '{path.suffix}'; use one of {', '.join(RASTER_FORMATS)}")
    return fmt


def sidecar_path(path: Path) -> Path:
    """Return the JSON sidecar of a raster file (the file name plus .json)."""
    return path.with_name(path.name + ".json")


def _encode_u16(data: np.ndarray, limits: Optional[tuple[float, float]]) -> tuple[np.ndarray, dict]:
    """Return 16-bit codes of a continuous field and the encoding that reads them back."""
    finite = np.isfinite(data)
    lo, hi = limits if limits is not None else ((float(data[finite].min()), float(data[finite].max()))
                                                if finite.any() else (0.0, 1.0))
    hi = hi if hi > lo else lo + 1.0
    top = U16_MAX - 1 if not finite.all() else U16_MAX
    scale = (hi - lo) / top
    codes = np.round((np.clip(np.where(finite, data, lo), lo, hi) - lo) / scale).astype(np.uint16)
    encoding = {"dtype": "uint16", "offset": lo, "scale": scale, "value": "offset + scale * code"}
    if not finite.all():
        codes[~finite] = U16_MAX
        encoding["nodata"] = U16_MAX
    return codes, encoding


def _encode_codes(data: np.ndarray, wide: bool) -> tuple[np.ndarray, dict]:
    """Return unsigned category codes, empty cells (−1) mapped to the largest code of the type."""
    data = np.asarray(data, dtype=np.int64)
    small = not wide and data.max(initial=0) < 255
    dtype, top = (np.uint8, 255) if small else (np.uint16, U16_MAX)
    if data.max(initial=0) >= top:
        raise ValueError(f"category codes up to {int(data.max())} do not fit in 16 bits")
    codes = np.where(data < 0, top, data).astype(dtype)
    return codes, {"dtype": np.dtype(dtype).name, "nodata": top}


def _colour_bytes(data: np.ndarray) -> np.ndarray:
    """Return 8-bit RGB(A) from colours in 0–1."""
    return np.round(np.clip(np.nan_to_num(data), 0.0, 1.0) * 255).astype(np.uint8)


def _write_geotiff(path: Path, data: np.ndarray, meta: dict, kind: str) -> dict:
    """Write a GeoTIFF with its georeferencing and tags; return the encoding."""
    try:
        import rasterio
        from rasterio.control import GroundControlPoint
        from rasterio.transform import Affine
    except ImportError as err:      # pragma: no cover - depends on the installation
        raise ImportError("GeoTIFF export needs rasterio: pip install 'worldgen[export]'") from err
    if kind == "colour":
        bands = np.moveaxis(_colour_bytes(data), -1, 0)
        encoding, nodata, dtype = {"dtype": "uint8", "bands": "RGBA"[:bands.shape[0]]}, None, "uint8"
    elif kind == "codes":
        bands = np.asarray(data, dtype=np.int32)[None]
        encoding, nodata, dtype = {"dtype": "int32", "nodata": -1}, -1, "int32"
    else:
        bands = np.asarray(data, dtype=np.float32)[None]
        encoding, nodata, dtype = {"dtype": "float32", "nodata": "nan"}, np.nan, "float32"
    count, rows, cols = bands.shape
    options = dict(driver="GTiff", height=rows, width=cols, count=count, dtype=dtype, nodata=nodata,
                   compress="deflate", predictor=3 if dtype == "float32" else 2)
    if rows >= 256 and cols >= 256:
        options.update(tiled=True, blockxsize=256, blockysize=256)
    if kind == "colour":
        options["photometric"] = "RGB"
    if "geotransform" in meta:
        options.update(crs=meta["crs_wkt"], transform=Affine.from_gdal(*meta["geotransform"]))
    with warnings.catch_warnings():
        # A file referenced by ground control points has no geotransform until they are written.
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        ds = rasterio.open(path, "w", **options)
    with ds:
        ds.write(bands)
        if "gcps" in meta:
            ds.gcps = ([GroundControlPoint(row=r, col=c, x=lon, y=lat) for r, c, lon, lat in meta["gcps"]],
                       rasterio.crs.CRS.from_wkt(meta["crs_wkt"]))
        tags = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in meta.items()
                if k in ("planet", "field", "units", "description", "labels", "epoch_years_ago")}
        ds.update_tags(**tags)
        if kind != "colour":
            ds.set_band_description(1, str(meta.get("field", "")))
    return encoding


def write_raster(path: str | Path, data: np.ndarray, meta: dict, kind: str = "values",
                 limits: Optional[tuple[float, float]] = None, sidecar: bool = True) -> Path:
    """Write a raster in the format its suffix names, plus its JSON sidecar, and return the path.

    ``kind`` is 'values' (continuous), 'codes' (categories, −1 empty) or
    'colour' (rows × cols × 3 or 4 in 0–1). ``meta`` carries the
    georeferencing (``crs_wkt`` with ``geotransform`` or ``gcps``) and the
    description that goes into the sidecar. ``limits`` fixes the value range
    16-bit codes span (default: the data's range). Without ``sidecar`` the
    encoding is returned in ``meta['encoding']`` instead of a file.
    """
    path = Path(path)
    fmt = raster_format(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if fmt == "geotiff":
        encoding = _write_geotiff(path, data, meta, kind)
    else:
        from PIL import Image

        if kind == "colour":
            if fmt == "raw16":
                raise ValueError("colour images cannot be written as RAW heightmaps; use .png or .tif")
            Image.fromarray(_colour_bytes(data)).save(path)
            encoding = {"dtype": "uint8", "bands": "RGBA"[:data.shape[-1]]}
        else:
            codes, encoding = (_encode_codes(data, wide=fmt == "raw16") if kind == "codes"
                               else _encode_u16(np.asarray(data, dtype=float), limits))
            if fmt == "png":
                Image.fromarray(codes).save(path)
            else:
                codes.astype("<u2").tofile(path)
                encoding["byte_order"] = "little-endian"
    if not sidecar:
        meta["encoding"] = encoding
        return path
    rows, cols = data.shape[:2]
    record = {"worldgen_version": __version__, "file": path.name, "format": fmt, "width": cols, "height": rows,
              "kind": kind, **meta, "encoding": encoding}
    sidecar_path(path).write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    return path


def read_sidecar(path: str | Path) -> dict:
    """Return the JSON sidecar of a raster written by ``write_raster``."""
    return json.loads(sidecar_path(Path(path)).read_text(encoding="utf-8"))


def decode(codes: np.ndarray, encoding: dict) -> np.ndarray:
    """Return the values of 16-bit codes read from a PNG or RAW file, NaN where empty."""
    values = encoding["offset"] + encoding["scale"] * codes.astype(float)
    if "nodata" in encoding:
        values[codes == encoding["nodata"]] = np.nan
    return values
