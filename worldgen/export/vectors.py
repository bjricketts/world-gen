"""GeoJSON layers of a zoomed region's culture: settlements, roads and polity borders.

Coordinates are longitude and latitude in degrees on the planet's sphere
(GeoJSON assumes Earth's datum; the region's sidecar names the planet and its
radius). Borders follow the lattice cell edges between nodes held by different
polities, joined into lines.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .regions import _lonlat, lattice_to_unit


def _round(values) -> list:
    """Return coordinates rounded to ~1 cm on an Earth-sized planet."""
    return [round(float(v), 7) for v in values]


def _feature(geometry: dict, properties: dict, fid: int) -> dict:
    """Return a GeoJSON feature."""
    return {"type": "Feature", "id": fid, "geometry": geometry, "properties": properties}


def _collection(features: list, world, layer: str) -> dict:
    """Return a GeoJSON feature collection with the planet named in its properties."""
    return {"type": "FeatureCollection", "name": layer,
            "planet": {"name": world.state.name, "radius_m": float(world.surface.attrs["radius_m"])},
            "features": features}


def _names(world) -> tuple[dict, dict]:
    """Return culture and polity names by id."""
    civ = world.culture
    cultures = {c.id: c.name or f"culture {c.id}" for c in civ.cultures}
    polities = {p.id: p.name or f"polity {p.id}" for p in civ.polities}
    return cultures, polities


def settlement_features(world, region, culture) -> list:
    """Return a point feature per settlement: name, population, form, culture, polity, footprint radius."""
    from ..culture.model import SETTLEMENT_FORMS

    cultures, polities = _names(world)
    lon, lat = _lonlat(region.grid.points[culture.node])
    out = []
    for i in range(len(culture)):
        polity = int(culture.polity[i])
        out.append(_feature({"type": "Point", "coordinates": _round((lon[i], lat[i]))},
                            {"name": str(culture.name[i]), "population": float(culture.population[i]),
                             "form": SETTLEMENT_FORMS[int(culture.form[i])], "culture": int(culture.culture[i]),
                             "culture_name": cultures.get(int(culture.culture[i])), "polity": polity,
                             "polity_name": polities.get(polity) if polity >= 0 else None,
                             "global_id": int(culture.listed[i]), "radius_km": float(culture.radius_km[i])}, i))
    return out


def road_features(world, region, culture) -> list:
    """Return a line feature per road segment, with its class and culture."""
    from ..culture.model import ROAD_KINDS

    cultures, _ = _names(world)
    out = []
    for k, (nodes, kind, cid) in enumerate(culture.roads):
        lon, lat = _lonlat(region.grid.points[np.asarray(nodes)])
        out.append(_feature({"type": "LineString", "coordinates": [_round(p) for p in zip(lon, lat)]},
                            {"class": ROAD_KINDS[int(kind)], "culture": int(cid),
                             "culture_name": cultures.get(int(cid))}, k))
    return out


def _chain(segments: list) -> list:
    """Return polylines joining segments that share end points."""
    ends = defaultdict(list)
    for k, (a, b) in enumerate(segments):
        ends[a].append(k)
        ends[b].append(k)
    used = np.zeros(len(segments), dtype=bool)
    lines = []
    for start in range(len(segments)):
        if used[start]:
            continue
        used[start] = True
        line = list(segments[start])
        for forward in (True, False):
            while True:
                tip = line[-1] if forward else line[0]
                nxt = next((k for k in ends[tip] if not used[k]), None)
                if nxt is None:
                    break
                used[nxt] = True
                a, b = segments[nxt]
                point = b if a == tip else a
                if forward:
                    line.append(point)
                else:
                    line.insert(0, point)
        lines.append(line)
    return lines


def border_features(world, region, culture) -> list:
    """Return line features along the lattice cell edges between nodes held by different polities (or none)."""
    _, polities = _names(world)
    rows, cols = region.grid.shape
    owner = np.asarray(culture.polity_map).reshape(rows, cols)
    pairs = defaultdict(list)
    # Corners of a node's cell sit half a step from it; a differing pair of neighbours shares one cell edge.
    r, c = np.nonzero(owner[:, 1:] != owner[:, :-1])
    for i, j in zip(r, c):
        key = tuple(sorted((int(owner[i, j]), int(owner[i, j + 1]))))
        pairs[key].append(((2 * i - 1, 2 * j + 1), (2 * i + 1, 2 * j + 1)))
    r, c = np.nonzero(owner[1:, :] != owner[:-1, :])
    for i, j in zip(r, c):
        key = tuple(sorted((int(owner[i, j]), int(owner[i + 1, j]))))
        pairs[key].append(((2 * i + 1, 2 * j - 1), (2 * i + 1, 2 * j + 1)))
    out = []
    for (a, b), segments in pairs.items():
        if a < 0 and b < 0:
            continue
        for line in _chain(segments):
            half = np.array(line, dtype=float) / 2.0
            rr = np.clip(half[:, 0], 0, rows - 1)
            cc = np.clip(half[:, 1], 0, cols - 1)
            lon, lat = _lonlat(lattice_to_unit(region, rr, cc))
            out.append(_feature({"type": "LineString", "coordinates": [_round(p) for p in zip(lon, lat)]},
                                {"left": a, "right": b, "left_name": polities.get(a), "right_name": polities.get(b)},
                                len(out)))
    return out


def write_region_vectors(world, region, culture, folder: str | Path) -> dict:
    """Write settlements.geojson, roads.geojson and borders.geojson for a region; return the file names by layer."""
    folder = Path(folder)
    layers = {"settlements": settlement_features, "roads": road_features, "borders": border_features}
    files = {}
    for name, build in layers.items():
        path = folder / f"{name}.geojson"
        path.write_text(json.dumps(_collection(build(world, region, culture), world, name)), encoding="utf-8")
        files[f"{name}_vectors"] = path.name
    return files
