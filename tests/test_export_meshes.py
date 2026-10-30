"""Meshes: RTIN simplification with locked borders, tile meshes, sea layer, frames and files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from worldgen.export import (add_chain, build_global, globe_meshes, pyramid_meshes, region_meshes,  # noqa: E402
                             write_meshes)
from worldgen.export.meshes import tile_mesh  # noqa: E402
from worldgen.export.rtin import is_rtin_size, rtin_errors, rtin_triangles  # noqa: E402
from worldgen.zoom import zoom_region  # noqa: E402
from worldgen.zoom.tiles import face_uv_to_unit  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def world(world_cache):
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def _area(tris, n):
    x, y = tris % n, tris // n
    return 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])).sum()


def test_rtin_covers_the_tile_and_keeps_locked_borders():
    n = 33
    flat = np.zeros((n, n))
    pts = np.zeros((n, n, 3))
    loose = rtin_triangles(rtin_errors(flat, pts, 0.0, False), 0.1)
    assert len(loose) == 2 and _area(loose, n) == (n - 1) ** 2
    locked = rtin_triangles(rtin_errors(flat, pts, 0.0, True), 0.1)
    used = np.unique(locked)
    border = [i for i in range(n * n) if i % n in (0, n - 1) or i // n in (0, n - 1)]
    assert set(border) <= set(used.tolist()) and _area(locked, n) == (n - 1) ** 2
    rough = np.random.default_rng(0).normal(size=(n, n))
    full = rtin_triangles(rtin_errors(rough, pts, 0.0, False), 0.0)
    assert len(full) == 2 * (n - 1) ** 2
    assert is_rtin_size(129) and is_rtin_size(1025) and not is_rtin_size(128) and not is_rtin_size(255)


def test_rtin_mesh_stays_within_the_tolerance_of_every_node():
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    from scipy.ndimage import zoom

    n = 129
    rng = np.random.default_rng(3)
    z = sum(zoom(rng.normal(size=(2 ** o + 1,) * 2), (n - 1) / 2 ** o + 1e-9, order=3)[:n, :n] * 500 * 2.0 ** (-0.8 * o)
            for o in range(1, 8))
    jj, ii = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    err = rtin_errors(z, np.zeros((n, n, 3)), 0.0, True)
    for tol in (5.0, 20.0, 50.0):
        tris = rtin_triangles(err, tol)
        mesh = LinearTriInterpolator(Triangulation(jj.ravel(), ii.ravel(), tris), z.ravel())(jj, ii)
        assert np.abs(mesh - z).max() <= tol * (1 + 1e-9)


def test_curvature_refines_a_flat_sea_on_the_sphere():
    n = 33
    s, t = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    pts = face_uv_to_unit(0, s, t)
    tris = rtin_triangles(rtin_errors(np.zeros((n, n)), pts, 6.4e6, False), 1000.0)
    assert len(tris) > 2


def test_tiles_face_outward_and_neighbours_share_their_border(world):
    meshes = globe_meshes(world, level=1, sea=True)
    terrain = {m.name: m for m in meshes if m.layer == "terrain"}
    for m in list(terrain.values())[:4]:
        p = m.planet_vertices()
        n = np.cross(p[m.faces[:, 1]] - p[m.faces[:, 0]], p[m.faces[:, 2]] - p[m.faces[:, 0]])
        assert (np.einsum("ij,ij->i", n, p[m.faces[:, 0]]) > 0).all()
    a = terrain["L1_f0_0_0_terrain"].planet_vertices()
    b = terrain["L1_f0_1_0_terrain"].planet_vertices()
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(b).query(a)
    shared = dist < 1.0
    assert shared.sum() >= 129                    # the whole shared edge
    assert dist[shared].max() < 1e-3


def test_sea_layer_repeats_wet_terrain_triangles_at_sea_level(world):
    meshes = globe_meshes(world, level=0, sea=True)
    sea = [m for m in meshes if m.layer == "sea"]
    assert sea
    radius = float(world.surface.attrs["radius_m"])
    r = np.linalg.norm(sea[0].planet_vertices(), axis=1)
    assert np.allclose(r, radius, atol=1.0)
    assert all(m.colours.shape[1] == 4 for m in meshes)


def test_exaggeration_scales_heights(world):
    plain = globe_meshes(world, level=0, sea=False, max_error_m=0.0)[0]
    tall = globe_meshes(world, level=0, sea=False, max_error_m=0.0, exaggeration=10.0)[0]
    radius = float(world.surface.attrs["radius_m"])
    h1 = np.linalg.norm(plain.planet_vertices(), axis=1) - radius
    h10 = np.linalg.norm(tall.planet_vertices(), axis=1) - radius
    assert np.allclose(h10, 10 * h1, atol=1.0)


def test_glb_nodes_place_vertices_in_planet_metres(world, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pyr = build_global(world, top_level=1)
    add_chain(pyr, world, 40.0, 20.0, 3)
    meshes = pyramid_meshes(pyr, world, 3)
    path = write_meshes(meshes, tmp_path / "t.glb")
    scene = trimesh.load(path)
    to_planet = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    for m in meshes:
        transform, _ = scene.graph[m.name]
        v = trimesh.transform_points(scene.geometry[m.name].vertices, transform) @ to_planet.T
        extent = np.ptp(m.vertices, axis=0).max()
        assert np.abs(v - m.planet_vertices()).max() < 1e-6 * extent + 1e-3


def test_region_meshes_keep_every_node_off_rtin_sizes(world, tmp_path):
    region = zoom_region(world, 40.0, 20.0, resolution_m=3000, half_width_km=40)
    (terrain, *_) = region_meshes(world, region)
    rows, cols = region.grid.shape
    assert len(terrain.faces) == 2 * (rows - 1) * (cols - 1)
    for suffix in (".ply", ".obj"):
        path = write_meshes(region_meshes(world, region, sea=False), tmp_path / f"r{suffix}", frame="local")
        assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="mesh format"):
        write_meshes([terrain], tmp_path / "r.stl")


def test_tile_mesh_simplifies_flat_ground():
    n = 129
    s, t = np.meshgrid(np.linspace(0.5, 0.51, n), np.linspace(0.5, 0.51, n))
    pts = face_uv_to_unit(0, s, t)
    (mesh,) = tile_mesh("flat", pts, np.full((n, n), 100.0), 6.4e6, 5.0)
    assert len(mesh.faces) < 0.2 * 2 * (n - 1) ** 2
