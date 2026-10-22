"""Heightfields and other layers of zoomed regions, on their native cube-face lattice or a north-up local grid.

A region's own lattice is the master: no resampling, and regions at one level
meet exactly. Files store the lattice with its high-t edge at the top (as the
local maps draw it) and are georeferenced by ground control points through the
lattice's nodes, to be warped with a thin-plate spline; an optional north-up
copy is resampled onto a Lambert azimuthal equal-area grid centred on the
region. Heights in 16-bit files share the world's height scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates

from .. import heuristics as h
from ..zoom.tiles import face_uv_to_unit, faces_uv
from .crs import laea_crs_wkt, laea_forward, laea_inverse, planet_crs_wkt
from .fields import CATEGORICAL, LINEAR, NEAREST, ExportField, _labels
from .rasters import Raster
from .writers import write_raster

# name: (region attribute, sampling, units, description)
REGION_NODE_FIELDS = {
    "elevation": ("elevation", LINEAR, "m", "ground height above sea level (below ice)"),
    "surface": (None, LINEAR, "m", "height of the top surface above sea level, ice included"),
    "ocean": ("ocean", CATEGORICAL, "", "1 where the sea lies"),
    "lake": ("lake", CATEGORICAL, "", "1 where a lake lies"),
    "rivers": ("river_order", NEAREST, "Strahler order", "channel order (0: no channel)"),
    "perennial": ("perennial", CATEGORICAL, "", "1 on channels that flow all year"),
    "discharge": ("discharge_m3_s", NEAREST, "m3/s", "mean discharge"),
    "drainage_area": ("drainage_area_km2", NEAREST, "km2", "upstream catchment area"),
    "ice": ("ice_thickness_m", LINEAR, "m", "ice thickness"),
    "glacial": ("glacial_m", LINEAR, "m", "depth cut by glaciers"),
    "eroded": ("eroded_m", LINEAR, "m", "depth removed by erosion"),
    "deposited": ("deposited_m", LINEAR, "m", "sediment deposited"),
    "temperature": ("temperature_k", LINEAR, "K", "annual mean air temperature"),
    "rainfall": ("precipitation_m", LINEAR, "m/yr", "annual precipitation"),
    "runoff": ("runoff_m", LINEAR, "m/yr", "annual runoff"),
    "biomes": ("biome", CATEGORICAL, "biome", "biome (from the nearest global cell)"),
    "koppen": ("koppen", CATEGORICAL, "Koppen class", "Köppen–Geiger class (from the nearest global cell)"),
    "terrain": ("terrain", CATEGORICAL, "terrain", "landform class (from the nearest global cell)"),
}
REGION_CULTURE_FIELDS = {
    "political": ("polity id", "polity holding the node"),
    "cultures": ("culture id", "culture with the most people at the node"),
    "settlements": ("settlement", "settlement whose footprint covers the node (index into settlements.geojson)"),
    "roads": ("road class", "highest class of road through the node"),
    "habitat": ("habitat", "habitat colonists need at the node"),
}
REGION_FIELDS = tuple(REGION_NODE_FIELDS) + tuple(REGION_CULTURE_FIELDS)
CATEGORY_KEYS = {"biomes": "biome", "koppen": "koppen", "terrain": "terrain"}


def world_height_limits(world) -> tuple[float, float]:
    """Return the height range 16-bit files of a world span: its elevation range widened on both sides."""
    elevation = world.surface["elevation"].values
    lo, hi = float(np.nanmin(elevation)), float(np.nanmax(elevation))
    margin = h.EXPORT_HEIGHT_MARGIN_FRACTION * (hi - lo)
    return lo - margin, hi + margin


def axial_step_m(region) -> float:
    """Return the mean distance between neighbouring lattice nodes along a row or column (m)."""
    rows, cols = region.grid.shape
    p = region.grid.points.reshape(rows, cols, 3)
    along = np.linalg.norm(np.diff(p, axis=1), axis=2).mean() + np.linalg.norm(np.diff(p, axis=0), axis=2).mean()
    return float(along / 2.0) * region.radius_m


def region_field(region, field: str, culture=None) -> ExportField:
    """Return a field of an eroded region per lattice node; culture fields need ``culture`` from ``zoom_culture``."""
    if field in REGION_NODE_FIELDS:
        attr, sampling, units, description = REGION_NODE_FIELDS[field]
        if field == "surface":
            ice = region.ice_thickness_m if region.ice_thickness_m is not None else 0.0
            values = region.elevation + ice
        elif field in CATEGORY_KEYS:
            if not region.categorical or CATEGORY_KEYS[field] not in region.categorical:
                raise ValueError(f"this region carries no '{field}' field")
            values = region.categorical[CATEGORY_KEYS[field]]
        else:
            values = getattr(region, attr)
            if values is None:
                raise ValueError(f"this region has no '{field}' field (its planet has no {description})")
        values = np.asarray(values)
        if sampling == CATEGORICAL or field == "rivers":
            values = values.astype(np.int32)
        labels = _labels(field) if field != "perennial" else {0: "no", 1: "yes"}
        return ExportField(field, values, sampling, units, description, labels,
                           -1 if sampling == CATEGORICAL else None)
    if field in REGION_CULTURE_FIELDS:
        if culture is None:
            raise ValueError(f"'{field}' needs the region's culture: pass culture=zoom_culture(world, region)")
        units, description = REGION_CULTURE_FIELDS[field]
        values, labels = _culture_layer(region, culture, field)
        return ExportField(field, values.astype(np.int32), CATEGORICAL, units, description, labels, -1)
    raise ValueError(f"unknown region field '{field}'; choose from {', '.join(REGION_FIELDS)}")


def _culture_layer(region, culture, field: str) -> tuple[np.ndarray, Optional[dict]]:
    """Return a per-node culture layer and its code labels."""
    from ..culture.habitats import Habitat
    from ..culture.model import ROAD_KINDS

    if field == "political":
        return np.asarray(culture.polity_map), None
    if field == "cultures":
        return np.asarray(culture.culture_map), None
    if field == "settlements":
        return np.asarray(culture.footprint), None
    if field == "habitat":
        if culture.habitat is None:
            raise ValueError("this region's culture has no habitat layer")
        return np.asarray(culture.habitat), {int(k): k.name.lower() for k in Habitat}
    codes = np.full(region.grid.size, -1, dtype=np.int32)
    for nodes, kind, _ in culture.roads:
        nodes = np.asarray(nodes)
        codes[nodes] = np.maximum(codes[nodes], int(kind))
    return codes, {k: name.replace("_", " ") for k, name in enumerate(ROAD_KINDS)}


def lattice_to_unit(region, rows_f: np.ndarray, cols_f: np.ndarray) -> np.ndarray:
    """Return unit vectors at fractional lattice positions (row along t, column along s) of a region."""
    grid = region.grid
    per = grid.nodes_per_tile - 1
    step = 1.0 / ((1 << grid.level) * per)
    x0, y0 = grid.origin
    s = (x0 * per - grid.pad + np.asarray(cols_f, dtype=float)) * step
    t = (y0 * per - grid.pad + np.asarray(rows_f, dtype=float)) * step
    return face_uv_to_unit(grid.face, s, t)


def _unit_to_lattice(region, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fractional (row, column) of unit vectors on a region's lattice, and whether they lie on its face."""
    grid = region.grid
    per = grid.nodes_per_tile - 1
    count = (1 << grid.level) * per
    x0, y0 = grid.origin
    faces, s, t = faces_uv(vectors)
    return t * count - y0 * per + grid.pad, s * count - x0 * per + grid.pad, faces == grid.face


def _lonlat(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude and latitude (degrees) of unit vectors."""
    return (np.degrees(np.arctan2(vectors[..., 1], vectors[..., 0])),
            np.degrees(np.arcsin(np.clip(vectors[..., 2], -1.0, 1.0))))


def ground_control_points(region) -> list[tuple[float, float, float, float]]:
    """Return GCPs (image row, column, longitude, latitude) at every tile corner and at least
    ``EXPORT_GCP_MIN_PER_SIDE`` nodes along each side, for an image with the region's high-t edge on top."""
    rows, cols = region.grid.shape
    per = region.grid.nodes_per_tile - 1

    def picks(n):
        nodes = set(range(0, n, per)) | {n - 1}
        nodes |= set(np.linspace(0, n - 1, h.EXPORT_GCP_MIN_PER_SIDE).round().astype(int).tolist())
        return np.array(sorted(nodes))

    r, c = np.meshgrid(picks(rows), picks(cols), indexing="ij")
    lon, lat = _lonlat(region.grid.points.reshape(rows, cols, 3)[r, c])
    image_row = rows - 1 - r
    return [(float(a) + 0.5, float(b) + 0.5, float(x), float(y))
            for a, b, x, y in zip(image_row.ravel(), c.ravel(), lon.ravel(), lat.ravel())]


def north_angle_deg(region) -> float:
    """Return the direction of north at the region's centre, in degrees clockwise from the image's up."""
    rows, cols = region.grid.shape
    centre = region.grid.points[(rows // 2) * cols + cols // 2]
    north = np.array([0.0, 0.0, 1.0]) - centre[2] * centre
    if np.linalg.norm(north) < 1e-12:
        return 0.0
    probe = centre + 1e-6 * north / np.linalg.norm(north)
    r, c, _ = _unit_to_lattice(region, (probe / np.linalg.norm(probe))[None])
    r0, c0, _ = _unit_to_lattice(region, centre[None])
    return float(np.degrees(np.arctan2(c[0] - c0[0], r[0] - r0[0])))


def _region_meta(world, region, field: str) -> dict:
    """Return the description of a region file shared by every format."""
    grid = region.grid
    x0, y0, x1, y1 = grid.tiles()
    rows, cols = grid.shape
    corners = grid.points.reshape(rows, cols, 3)[[rows - 1, rows - 1, 0, 0], [0, cols - 1, cols - 1, 0]]
    lon, lat = _lonlat(corners)
    return {"planet": world.state.name, "radius_m": float(region.radius_m), "field": field,
            "lattice": {"face": grid.face, "level": grid.level, "tiles": [x0, y0, x1, y1],
                        "nodes_per_tile": grid.nodes_per_tile, "step_m": axial_step_m(region)},
            "corners_lonlat": {k: [float(a), float(b)] for k, a, b in zip(("top_left", "top_right", "bottom_right",
                                                                            "bottom_left"), lon, lat)}}


def region_raster(world, region, field: str = "elevation", culture=None, crs: str = "native",
                  colour: bool = False, mode: str = "relief", hillshade: bool = True) -> Raster:
    """Return a region field as a raster: on the native lattice (``crs='native'``, GCP-referenced) or resampled
    onto a north-up LAEA grid centred on the region (``crs='laea'``). ``colour`` gives the local map's colours
    (``mode`` 'relief' or 'biome') instead of values."""
    if crs not in ("native", "laea"):
        raise ValueError("crs is 'native' or 'laea'")
    rows, cols = region.grid.shape
    meta = _region_meta(world, region, field)
    if colour:
        from ..render.local import local_colours

        spec = None
        grid_values = local_colours(world, region, mode, hillshade)
        kind, categorical = "colour", False
        meta.update(description=f"local map colours ({mode})", hillshade=hillshade)
    else:
        spec = region_field(region, field, culture)
        grid_values = spec.values.reshape(rows, cols)
        kind, categorical = ("codes" if spec.categorical else "values"), spec.sampling != LINEAR
        meta.update(units=spec.units, description=spec.description, sampling=spec.sampling)
        if spec.labels is not None:
            meta["labels"] = {str(k): v for k, v in spec.labels.items()}
    if crs == "native":
        meta.update(grid="cube-face lattice", crs_wkt=planet_crs_wkt(world.state.name, region.radius_m),
                    gcps=ground_control_points(region), warp="thin plate spline", north_deg=north_angle_deg(region),
                    image_rows="top row is the lattice's high-t edge")
        return Raster(grid_values[::-1].copy(), spec, meta, kind)
    data, laea_meta = _resample_laea(region, grid_values, categorical, kind)
    meta.update(laea_meta, crs_wkt=laea_crs_wkt(world.state.name, region.radius_m, *laea_meta["centre_latlon"]))
    return Raster(data, spec, meta, kind)


def _resample_laea(region, values: np.ndarray, categorical: bool, kind: str) -> tuple[np.ndarray, dict]:
    """Return values resampled onto the north-up LAEA grid that covers a region at its lattice step."""
    rows, cols = region.grid.shape
    radius = region.radius_m
    centre = region.grid.points[(rows // 2) * cols + cols // 2]
    lat0, lon0 = np.arcsin(centre[2]), np.arctan2(centre[1], centre[0])
    edge = np.r_[np.arange(cols), (rows - 1) * cols + np.arange(cols), np.arange(rows) * cols,
                 np.arange(rows) * cols + cols - 1]
    p = region.grid.points[edge]
    ex, ey = laea_forward(np.arcsin(p[:, 2]), np.arctan2(p[:, 1], p[:, 0]), radius, lat0, lon0)
    step = axial_step_m(region)
    x0, x1 = np.floor(ex.min() / step) * step, np.ceil(ex.max() / step) * step
    y0, y1 = np.floor(ey.min() / step) * step, np.ceil(ey.max() / step) * step
    nx, ny = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    x = x0 + (np.arange(nx) + 0.5) * step
    y = y1 - (np.arange(ny) + 0.5) * step
    xx, yy = np.meshgrid(x, y)
    lat, lon = laea_inverse(xx.ravel(), yy.ravel(), radius, lat0, lon0)
    vec = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    r, c, on_face = _unit_to_lattice(region, vec)
    inside = on_face & (r >= 0) & (r <= rows - 1) & (c >= 0) & (c <= cols - 1)
    if kind == "colour":
        out = np.stack([map_coordinates(values[..., k], [r, c], order=1, mode="nearest") for k in range(3)], -1)
        out[~inside] = 1.0
        out = out.reshape(ny, nx, 3)
    elif categorical:
        out = values[np.clip(np.round(r), 0, rows - 1).astype(int), np.clip(np.round(c), 0, cols - 1).astype(int)]
        out = np.where(inside, out, -1 if kind == "codes" else np.nan).reshape(ny, nx)
    else:
        out = map_coordinates(np.asarray(values, dtype=float), [r, c], order=1, mode="nearest")
        out = np.where(inside, out, np.nan).reshape(ny, nx)
    meta = {"grid": "laea", "centre_latlon": [float(np.degrees(lat0)), float(np.degrees(lon0))],
            "geotransform": (float(x0), float(step), 0.0, float(y1), 0.0, -float(step)),
            "resampled": "nearest node" if categorical else "bilinear", "inside_share": float(inside.mean())}
    return out, meta


def region_lonlat(region) -> tuple[np.ndarray, np.ndarray]:
    """Return each node's longitude and latitude (degrees) as images with the high-t edge on top."""
    rows, cols = region.grid.shape
    lon, lat = _lonlat(region.grid.points.reshape(rows, cols, 3))
    return lon[::-1].copy(), lat[::-1].copy()


def export_region(world, region, path: str | Path, field: str = "elevation", culture=None, crs: str = "native",
                  colour: bool = False, mode: str = "relief", hillshade: bool = True,
                  limits: Optional[tuple[float, float]] = None) -> Path:
    """Write one field of a zoomed region (.tif, .png, .r16 or .raw by suffix) with its JSON sidecar.

    Heights in PNG and RAW span the world's height scale unless ``limits`` is
    given, so regions and global rasters of one world line up in height.
    """
    raster = region_raster(world, region, field, culture, crs, colour, mode, hillshade)
    if limits is None and raster.kind == "values" and field in ("elevation", "surface"):
        limits = world_height_limits(world)
        if np.nanmax(raster.data) > limits[1] or np.nanmin(raster.data) < limits[0]:
            raise ValueError("the region's heights leave the world's height scale; pass limits=")
    return write_raster(path, raster.data, raster.meta, raster.kind, limits)


DEFAULT_BUNDLE = ("elevation", "ocean", "lake", "rivers", "biomes")


def export_region_bundle(world, region, folder: str | Path, fields=DEFAULT_BUNDLE, suffix: str = ".tif",
                         culture=None, crs: str = "native", colour: bool = True, vectors: bool = True) -> Path:
    """Write a region's fields, its map colours, node coordinates and (with ``culture``) culture layers and
    GeoJSON vectors into a folder, with a ``region.json`` index; return the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fields = list(fields) + ([f for f in REGION_CULTURE_FIELDS if f != "habitat" or culture.habitat is not None]
                             if culture is not None else [])
    files = {}
    for field in fields:
        try:
            files[field] = export_region(world, region, folder / f"{field}{suffix}", field, culture, crs).name
        except ValueError as err:
            if "has no" not in str(err):
                raise
    if colour:
        files["colour"] = export_region(world, region, folder / f"colour{'.png' if suffix != '.tif' else suffix}",
                                        colour=True, crs=crs).name
    lon, lat = region_lonlat(region)
    np.savez_compressed(folder / "lonlat.npz", lon=lon, lat=lat)
    files["lonlat"] = "lonlat.npz"
    if vectors and culture is not None:
        from .vectors import write_region_vectors

        files.update(write_region_vectors(world, region, culture, folder))
    index = _region_meta(world, region, "bundle")
    index.update(files=files, crs=crs)
    (folder / "region.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return folder
