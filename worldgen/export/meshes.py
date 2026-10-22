"""Triangle meshes of the globe, of pyramid tiles and of zoomed regions, with per-vertex colours.

A mesh is built per tile of 2ⁿ + 1 nodes and simplified with a right-
triangulated irregular network whose border is locked, so neighbouring tiles
share their border vertices. Heights may be exaggerated. The sea floor stays
in the terrain mesh, and a separate sea-surface mesh at sea level repeats the
terrain triangles that touch the sea; the coast is where the two cross. Each tile's
vertices are in east–north–up metres about its centre and carry a transform to
planet-centred metres (z towards the north pole), which glTF files store as
node transforms; OBJ and PLY files take the vertices in planet-centred or in
one shared local frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..zoom import region_grid
from ..zoom.tiles import face_uv_to_unit
from .rtin import is_rtin_size, rtin_errors, rtin_triangles
from .sampling import point_sampler

SEA_COLOUR = (38, 84, 140, 255)
MESH_FORMATS = (".glb", ".gltf", ".ply", ".obj")


@dataclass
class TileMesh:
    """One mesh in its local east–north–up frame, with the transform that places it in planet-centred metres."""

    name: str
    vertices: np.ndarray            # (n, 3) float, local metres (east, north, up)
    faces: np.ndarray               # (m, 3) int, counter-clockwise seen from above
    colours: Optional[np.ndarray]   # (n, 4) uint8
    transform: np.ndarray           # (4, 4) local → planet-centred
    layer: str = "terrain"          # 'terrain' or 'sea'

    def planet_vertices(self) -> np.ndarray:
        """Return the vertices in planet-centred metres."""
        return self.vertices @ self.transform[:3, :3].T + self.transform[:3, 3]


def enu_frame(centre_unit: np.ndarray, radius_m: float) -> np.ndarray:
    """Return the 4 × 4 transform from east–north–up metres at a point on the sphere to planet-centred metres."""
    up = centre_unit / np.linalg.norm(centre_unit)
    east = np.cross([0.0, 0.0, 1.0], up)
    if np.linalg.norm(east) < 1e-9:                     # at a pole east is taken along +y
        east = np.array([0.0, 1.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = east, north, up, up * radius_m
    return m


def default_error(step_m: float) -> float:
    """Return the default vertical tolerance of a tile mesh: a share of its node spacing."""
    return h.MESH_ERROR_FRACTION * step_m


def tile_mesh(name: str, points: np.ndarray, heights: np.ndarray, radius_m: float, max_error_m: float,
              ocean: Optional[np.ndarray] = None, colours: Optional[np.ndarray] = None,
              exaggeration: float = 1.0, sea: bool = True, simplify: bool = True) -> list[TileMesh]:
    """Return the terrain mesh of a square tile, and its sea-surface mesh where it touches the sea.

    ``points`` are the nodes' unit vectors (n × n × 3), ``heights`` their
    heights (m above sea level), ``colours`` RGBA per node. Without
    ``simplify``, or on a grid that is not 2ⁿ + 1, every node is kept.
    """
    rows, cols = heights.shape
    z = np.asarray(heights, dtype=float) * exaggeration
    pts = np.ascontiguousarray(points, dtype=float)
    centre = pts[rows // 2, cols // 2]
    frame = enu_frame(centre, radius_m)
    inverse = np.linalg.inv(frame)

    def triangulate(values):
        if simplify and rows == cols and is_rtin_size(rows):
            return rtin_triangles(rtin_errors(values, pts, radius_m, True), max_error_m)
        index = np.arange(rows * cols).reshape(rows, cols)
        a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
        return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

    def build(values, faces, layer, rgba):
        used = np.unique(faces)
        remap = np.full(rows * cols, -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        planet = pts.reshape(-1, 3)[used] * (radius_m + values.ravel()[used])[:, None]
        local = planet @ inverse[:3, :3].T + inverse[:3, 3]
        tris = remap[faces]
        # outward (counter-clockwise from above): the triangle normal points away from the planet
        normal = np.cross(planet[tris[:, 1]] - planet[tris[:, 0]], planet[tris[:, 2]] - planet[tris[:, 0]])
        flip = np.einsum("ij,ij->i", normal, planet[tris[:, 0]]) < 0
        tris[flip] = tris[flip][:, ::-1]
        return TileMesh(f"{name}_{layer}", local, tris.astype(np.int32), rgba[used] if rgba is not None else None,
                        frame, layer)

    rgba = None if colours is None else np.asarray(colours).reshape(rows * cols, 4)
    faces = triangulate(z)
    out = [build(z, faces, "terrain", rgba)]
    if sea and ocean is not None and np.asarray(ocean).any():
        # The sea takes the terrain's own triangles that touch the sea, so both share every chord and the
        # coast is where the two planes of each triangle cross.
        wet = np.asarray(ocean).ravel()[faces].any(axis=1)
        if wet.any():
            out.append(build(np.zeros_like(z), faces[wet], "sea",
                             np.tile(np.array(SEA_COLOUR, np.uint8), (rows * cols, 1))))
    return out


def _rgba(rgb: np.ndarray) -> np.ndarray:
    """Return 8-bit RGBA from colours in 0–1."""
    rgb = np.asarray(rgb, dtype=float)
    alpha = np.ones(rgb.shape[:-1] + (1,))
    return np.round(np.clip(np.concatenate([rgb[..., :3], alpha], -1), 0, 1) * 255).astype(np.uint8)


def _node_colours(world, field: str, points: np.ndarray, heights: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Return RGBA colours at nodes: the world's height colours from the tile's own heights, or a map field's."""
    from ..render.maps import elevation_colours, field_colours

    if field == "elevation":
        return _rgba(elevation_colours(world, heights, ocean)[0])
    sampler = point_sampler(world.grid, points.reshape(-1, 3), heights.shape)
    return _rgba(field_colours(world, field, sampler))


def pyramid_meshes(pyramid, world, level: int, field: str = "elevation", max_error_m: Optional[float] = None,
                   exaggeration: float = 1.0, sea: bool = True, epoch: Optional[float] = None) -> list[TileMesh]:
    """Return the meshes of every tile of one pyramid level, coloured by a map field."""
    if epoch is not None:
        from ..culture import world_at_epoch

        world = world_at_epoch(world, epoch)
    per = h.PYRAMID_NODES_PER_TILE - 1
    step = (np.pi / 2) / (1 << level) / per * pyramid.radius_m
    error = default_error(step) if max_error_m is None else max_error_m
    out = []
    for block in pyramid.blocks(level):
        x0, y0, x1, y1 = block.tiles
        points = region_grid(block.face, level, x0, y0, x1, y1, nodes_per_tile=per + 1).points
        points = points.reshape(*block.shape, 3)
        for x, y in block.tile_ids():
            r, c = (y - y0) * per, (x - x0) * per
            pts = points[r:r + per + 1, c:c + per + 1]
            z = np.asarray(block.tile(x, y), dtype=float)
            ocean = block.tile(x, y, "ocean").astype(bool) if "ocean" in block.fields else z < 0
            colours = _node_colours(world, field, pts, z, ocean)
            out += tile_mesh(f"L{level}_f{block.face}_{x}_{y}", pts, z, pyramid.radius_m, error, ocean, colours,
                             exaggeration, sea)
    return out


def globe_meshes(world, level: int = 2, field: str = "elevation", max_error_m: Optional[float] = None,
                 exaggeration: float = 1.0, sea: bool = True, pyramid=None, epoch: Optional[float] = None
                 ) -> list[TileMesh]:
    """Return the whole globe as tile meshes at a quadtree level.

    With a ``pyramid`` the tiles are its level; without one the global grid is
    resampled onto the cube-sphere lattice (fast, with the grid's ~124 km relief).
    """
    if pyramid is not None:
        return pyramid_meshes(pyramid, world, level, field, max_error_m, exaggeration, sea, epoch)
    if world.surface is None:
        raise ValueError(f"{world.state.name} has no solid surface")
    if epoch is not None:
        from ..culture import world_at_epoch

        world = world_at_epoch(world, epoch)
    radius = float(world.surface.attrs["radius_m"])
    per = h.PYRAMID_NODES_PER_TILE - 1
    count = 1 << level
    step = (np.pi / 2) / count / per * radius
    error = default_error(step) if max_error_m is None else max_error_m
    surface = world.surface
    out = []
    span = (np.arange(per + 1)) / (count * per)
    for face in range(6):
        for y in range(count):
            for x in range(count):
                s, t = np.meshgrid(x / count + span, y / count + span)
                pts = face_uv_to_unit(face, s, t)
                sampler = point_sampler(world.grid, pts.reshape(-1, 3), s.shape)
                z = sampler.sample(surface["elevation"].values)
                near_sea = sampler.sample(surface["ocean"].values, categorical=True).astype(bool)
                ocean = (z < 0) & near_sea
                colours = _node_colours(world, field, pts, z, ocean)
                out += tile_mesh(f"L{level}_f{face}_{x}_{y}", pts, z, radius, error, ocean, colours, exaggeration,
                                 sea)
    return out


def region_meshes(world, region, mode: str = "relief", max_error_m: Optional[float] = None,
                  exaggeration: float = 1.0, sea: bool = True, hillshade: bool = False) -> list[TileMesh]:
    """Return a zoomed region as one mesh coloured like its local map (``mode`` 'relief' or 'biome').

    Engine and pyramid lattices (2ⁿ + 1 nodes square) are simplified; other
    regions keep every node.
    """
    from ..render.local import local_colours
    from .regions import axial_step_m

    rows, cols = region.grid.shape
    pts = region.grid.points.reshape(rows, cols, 3)
    z = np.asarray(region.elevation, dtype=float).reshape(rows, cols)
    if region.ice_thickness_m is not None:
        z = z + region.ice_thickness_m.reshape(rows, cols)
    ocean = np.asarray(region.ocean).reshape(rows, cols)
    colours = _rgba(local_colours(world, region, mode, hillshade))
    error = default_error(axial_step_m(region)) if max_error_m is None else max_error_m
    return tile_mesh("region", pts, z, region.radius_m, error, ocean, colours, exaggeration, sea)


def _merged(meshes: list[TileMesh], frame: str):
    """Return one trimesh of all meshes in planet-centred metres, or in the first mesh's local frame."""
    import trimesh

    to_local = np.linalg.inv(meshes[0].transform) if frame == "local" else np.eye(4)
    parts = []
    for m in meshes:
        v = m.planet_vertices() @ to_local[:3, :3].T + to_local[:3, 3]
        parts.append(trimesh.Trimesh(v, m.faces, vertex_colors=m.colours, process=False))
    return trimesh.util.concatenate(parts)


# glTF is y-up: planet-centred (x, y, z = north) becomes (x, z, −y).
Y_UP = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])


def write_meshes(meshes: list[TileMesh], path: str | Path, frame: str = "planet") -> Path:
    """Write meshes to .glb/.gltf (a node per mesh with its transform, planet z turned to glTF's y-up), or
    merged to .ply/.obj in planet-centred metres (``frame='planet'``) or the first mesh's local frame."""
    import trimesh

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in MESH_FORMATS:
        raise ValueError(f"unknown mesh format '{suffix}'; use one of {', '.join(MESH_FORMATS)}")
    if not meshes:
        raise ValueError("no meshes to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in (".glb", ".gltf"):
        scene = trimesh.Scene()
        for m in meshes:
            # glTF local axes are (east, up, −north): rotate the ENU vertices to y-up within the node
            enu_to_gltf = Y_UP
            geometry = trimesh.Trimesh(m.vertices @ enu_to_gltf[:3, :3].T, m.faces, vertex_colors=m.colours,
                                       process=False)
            node = Y_UP @ m.transform @ np.linalg.inv(enu_to_gltf)
            scene.add_geometry(geometry, node_name=m.name, geom_name=m.name, transform=node)
        scene.export(path)
    else:
        _merged(meshes, frame).export(path)
    return path


def export_globe_mesh(world, path: str | Path, level: int = 2, field: str = "elevation",
                      max_error_m: Optional[float] = None, exaggeration: float = 1.0, sea: bool = True,
                      pyramid=None, epoch: Optional[float] = None) -> Path:
    """Write the globe as a mesh file (.glb, .gltf, .ply or .obj); see ``globe_meshes``."""
    return write_meshes(globe_meshes(world, level, field, max_error_m, exaggeration, sea, pyramid, epoch), path)


def export_region_mesh(world, region, path: str | Path, mode: str = "relief", max_error_m: Optional[float] = None,
                       exaggeration: float = 1.0, sea: bool = True, frame: str = "local") -> Path:
    """Write a zoomed region as a mesh; OBJ and PLY default to its local east–north–up frame."""
    return write_meshes(region_meshes(world, region, mode, max_error_m, exaggeration, sea), path, frame)


def export_pyramid_meshes(pyramid, world, folder: str | Path, levels=None, field: str = "elevation",
                          exaggeration: float = 1.0, sea: bool = True, suffix: str = ".glb") -> Path:
    """Write every tile of the chosen pyramid levels as ``L<level>/<face>/<x>_<y>.glb`` (terrain and sea nodes)."""
    folder = Path(folder)
    for level in (sorted(pyramid.levels) if levels is None else levels):
        meshes = pyramid_meshes(pyramid, world, level, field, None, exaggeration, sea)
        by_tile: dict = {}
        for m in meshes:
            by_tile.setdefault(m.name.rsplit("_", 1)[0], []).append(m)
        for name, group in by_tile.items():
            _, face, x, y = name.split("_")
            write_meshes(group, folder / f"L{level}" / face[1:] / f"{x}_{y}{suffix}")
    return folder
