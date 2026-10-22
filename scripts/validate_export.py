"""Check the exports of a world: rasters, regions, engine heightmaps, the tile pyramid, meshes and timings.

Usage: python scripts/validate_export.py [--world DIR] [--resolution standard] [--pyramid] [--out DIR]

Uses a saved world (``--world``) or generates the Earth example. A world without a culture gets a modern
one, so the region rows can check culture layers and vectors. The pyramid rows run on a small pyramid
(global levels 0-1 and a chain to level 4) unless ``--pyramid`` builds the full one (levels 0-3 and a
chain to level 7, ~6.5 min more on an Earth-sized world). Files go to ``--out`` (default: a temporary
folder). Each row passes, fails or is reported without a verdict; the exit code is 1 if any row fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldgen import heuristics as h
from worldgen import load_world
from worldgen.export import (add_chain, build_global, engine_region, export_engine_heightmap, export_global,
                             export_pyramid, export_region, export_region_bundle, global_raster, globe_meshes,
                             pyramid_meshes, region_meshes, write_meshes)
from worldgen.export.fields import export_field
from worldgen.export.regions import ground_control_points
from worldgen.export.rtin import rtin_errors, rtin_triangles
from worldgen.export.sampling import point_sampler
from worldgen.export.writers import decode, read_sidecar
from worldgen.generate import generate_world
from worldgen.spec import load_spec
from worldgen.zoom import region_grid, zoom_region

EARTH_SPEC = Path(__file__).resolve().parent.parent / "examples" / "earth.yaml"
MODERN = {"cultures": [{"name": "Ana", "development": {"energy": "electric"}}]}


class Table:
    """Rows of a validation report: name, value, limit and verdict."""

    def __init__(self):
        self.rows = []

    def check(self, group: str, name: str, value: float, limit: float, fmt: str = "{:.3g}", op: str = "<=") -> None:
        """Add a row that passes when ``value op limit``."""
        ok = {"<=": value <= limit, "<": value < limit, ">=": value >= limit, "==": value == limit}[op]
        self.rows.append((group, name, fmt.format(value), f"{op} {fmt.format(limit)}", "PASS" if ok else "FAIL"))

    def report(self, group: str, name: str, value: float, fmt: str = "{:.1f}") -> None:
        """Add a row without a verdict."""
        self.rows.append((group, name, fmt.format(value), "", "-"))

    def print(self) -> int:
        """Print the table and return the number of failed rows."""
        widths = [max(len(r[i]) for r in self.rows) for i in range(4)]
        for r in self.rows:
            print(f"{r[0]:<{widths[0]}}  {r[1]:<{widths[1]}}  {r[2]:>{widths[2]}}  {r[3]:<{widths[3]}}  {r[4]}")
        return sum(r[4] == "FAIL" for r in self.rows)


def timed(table: Table, name: str, fn, *args, **kwargs):
    """Run a function, add its run time as a timing row and return its result."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    table.report("timing", f"{name} (s)", time.perf_counter() - start)
    return out


def read_band(path: Path) -> np.ndarray:
    """Return the first band of a GeoTIFF or a 16-bit PNG, or a RAW file shaped by its sidecar."""
    suffix = path.suffix.lower()
    if suffix == ".tif":
        import rasterio

        with warnings.catch_warnings(), rasterio.open(path) as src:
            warnings.simplefilter("ignore")
            return src.read(1)
    if suffix == ".png":
        from PIL import Image

        return np.asarray(Image.open(path))
    meta = read_sidecar(path)
    return np.fromfile(path, dtype="<u2").reshape(meta["height"], meta["width"])


def check_rasters(table: Table, world, out: Path) -> None:
    """Global rasters: file round trips, 16-bit decoding, values at cell centres and category labels."""
    raster = timed(table, "global raster 2048 × 1024", global_raster, world, "elevation", 2048)
    tif = export_global(world, out / "elevation.tif", width=2048)
    stored = read_band(tif)
    table.check("raster", "GeoTIFF round trip max |Δ| (m)", float(np.nanmax(np.abs(stored - raster.data.astype(
        stored.dtype)))), 0.0, op="==")
    for suffix in (".png", ".raw"):
        path = export_global(world, out / f"elevation{suffix}", width=2048)
        meta = read_sidecar(path)
        heights = decode(read_band(path), meta["encoding"])
        step = meta["encoding"]["scale"]
        table.check("raster", f"{suffix[1:].upper()} decode max error (steps)",
                    float(np.nanmax(np.abs(heights - raster.data)) / step), 0.5)
    spec = export_field(world, "elevation")
    at_cells = spec.sample(point_sampler(world.grid, world.grid.points), world.surface).ravel()
    table.check("raster", "values at cell centres vs grid max |Δ| (m)",
                float(np.abs(at_cells - world.surface["elevation"].values).max()), 1e-6)
    biomes = export_global(world, out / "biomes.tif", "biomes", width=1024)
    codes = np.unique(read_band(biomes))
    labels = {int(k) for k in read_sidecar(biomes)["labels"]}
    table.check("raster", "biome codes without a label", float(len(set(codes.tolist()) - labels)), 0.0, "{:.0f}",
                "==")


def check_regions(table: Table, world, place, out: Path) -> None:
    """Regions: GDAL's thin-plate warp of the native file against the LAEA export, control points, vectors and
    engine heightmaps."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    from worldgen.culture.local import zoom_culture

    region = timed(table, "zoom 120 km region at 2.5 km", zoom_region, world, *place, resolution_m=2500,
                   half_width_km=60)
    local = zoom_culture(world, region)
    folder = timed(table, "region bundle (GeoTIFF, culture, vectors)", export_region_bundle, world, region,
                   out / "region", culture=local)
    native = folder / "elevation.tif"
    laea = export_region(world, region, out / "region_laea.tif", crs="laea")
    with warnings.catch_warnings(), rasterio.open(native) as src, rasterio.open(laea) as dst:
        warnings.simplefilter("ignore")
        warped = np.full((dst.height, dst.width), np.nan, np.float32)
        reproject(rasterio.band(src, 1), warped, src_crs=src.gcps[1], gcps=src.gcps[0], dst_transform=dst.transform,
                  dst_crs=dst.crs, resampling=Resampling.bilinear, SRC_METHOD="GCP_TPS", dst_nodata=np.nan)
        ref = dst.read(1)
    ok = np.isfinite(warped) & np.isfinite(ref)
    rms = float(np.sqrt(np.mean((warped[ok] - ref[ok]) ** 2)))
    table.check("region", "TPS warp vs LAEA rms / relief std", rms / max(float(np.nanstd(ref)), 50.0), 0.02)
    table.report("region", "TPS warp vs LAEA rms (m)", rms, "{:.2f}")

    rows, cols = region.grid.shape
    lat = region.grid.lat.reshape(rows, cols)
    lon = region.grid.lon.reshape(rows, cols)
    worst = 0.0
    for r, c, x, y in ground_control_points(region):
        node = (rows - 1 - int(r - 0.5), int(c - 0.5))
        d = np.radians(np.hypot((x - lon[node]) * np.cos(np.radians(y)), y - lat[node])) * region.radius_m
        worst = max(worst, float(d))
    table.check("region", "control points off their nodes max (m)", worst, 1e-3)

    features = json.loads((folder / "settlements.geojson").read_text())["features"]
    if features:
        lo, la = np.radians(np.array([f["geometry"]["coordinates"] for f in features])).T
        vec = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        on_node = float(np.mean(region.grid.tree().query(vec)[1] == local.node))
        table.check("region", "settlement features on their nodes", on_node, 1.0, "{:.3f}", "==")
    else:
        table.report("region", "settlement features (none in the region)", 0, "{:.0f}")

    for engine, size in (("unity", 257), ("unreal", 253)):
        er = timed(table, f"{engine} {size} heightmap", engine_region, world, *place, engine, size)
        path = export_engine_heightmap(world, er, out / f"{engine}.r16", engine)
        meta = read_sidecar(path)
        codes = np.fromfile(path, dtype="<u2")
        table.check("engine", f"{engine} {size}: nodes written", float(codes.size), float(size * size), "{:.0f}", "==")
        heights = decode(codes.reshape(size, size), meta["encoding"])
        z = er.elevation.reshape(size, size)[::-1]
        table.check("engine", f"{engine} {size}: decode max error (steps)",
                    float(np.abs(heights - z).max() / meta["encoding"]["scale"]), 0.5)


def _block_points(block):
    """Return the unit vectors of a pyramid block's nodes."""
    x0, y0, x1, y1 = block.tiles
    return region_grid(block.face, block.level, x0, y0, x1, y1, nodes_per_tile=h.PYRAMID_NODES_PER_TILE).points


def _pit_share(block) -> float:
    """Return the share of a block's ice-free inland nodes lower than all their neighbours."""
    x0, y0, x1, y1 = block.tiles
    grid = region_grid(block.face, block.level, x0, y0, x1, y1, nodes_per_tile=h.PYRAMID_NODES_PER_TILE)
    z = block.fields["elevation"].ravel()
    outlet = grid.boundary().ravel()
    for name in ("ocean", "lake"):
        if name in block.fields:
            outlet = outlet | block.fields[name].ravel().astype(bool)
    adj = grid.neighbours.tocsr()
    lowest = np.minimum.reduceat(z[adj.indices], adj.indptr[:-1])
    inland = ~outlet & (block.fields["ice"].ravel() <= 0)
    return float(np.mean(z[inland] < lowest[inland])) if inland.any() else 0.0


def check_pyramid(table: Table, world, place, full: bool, out: Path):
    """Pyramid: faces meet, the top level keeps the global surface, decimation, chain edges, drainage and tiles."""
    top, finest = (3, 7) if full else (1, 4)
    pyramid = timed(table, f"global pyramid levels 0-{top}", build_global, world, top)
    timed(table, f"chain levels {top + 1}-{finest}", add_chain, pyramid, world, *place, finest)
    blocks = pyramid.blocks(top)
    points = [_block_points(b) for b in blocks]
    heights = [b.fields["elevation"].ravel() for b in blocks]
    worst = 0.0
    for i in range(6):
        tree = cKDTree(points[i])
        for j in range(i + 1, 6):
            dist, idx = tree.query(points[j], distance_upper_bound=1e-7)
            shared = np.isfinite(dist)
            if shared.any():
                worst = max(worst, float(np.abs(heights[j][shared] - heights[i][idx[shared]]).max()))
    table.check("pyramid", f"faces meet at level {top} max |Δ| (m)", worst, 1e-3)

    surface = world.surface["elevation"].values
    offsets = []
    for b, p in zip(blocks, points):
        under = point_sampler(world.grid, p).sample(surface).ravel()
        offsets.append((b.fields["elevation"].ravel() - under)[under > 0])
    table.check("pyramid", f"level {top} mean land offset from the global grid (m)",
                abs(float(np.concatenate(offsets).mean())), 10.0)

    worst = 0.0
    for level in range(top):
        for fine, coarse in zip(pyramid.blocks(level + 1), pyramid.blocks(level)):
            worst = max(worst, float(np.abs(coarse.fields["elevation"] - fine.fields["elevation"][::2, ::2]).max()))
    table.check("pyramid", "decimated levels vs every other node max |Δ| (m)", worst, 0.0, op="==")

    per = h.PYRAMID_NODES_PER_TILE - 1
    worst = 0.0
    for level in range(top + 1, finest + 1):
        for child in pyramid.blocks(level):
            x0, y0, _, _ = child.tiles
            parent = pyramid.block_at(child.face, level - 1, x0 // 2, y0 // 2)
            px0, py0, _, _ = parent.tiles
            r0, c0 = (y0 // 2 - py0) * per, (x0 // 2 - px0) * per
            under = parent.fields["elevation"][r0:r0 + per + 1, c0:c0 + per + 1]
            z = child.fields["elevation"]
            for edge, ref in ((z[0, ::2], under[0]), (z[-1, ::2], under[-1]), (z[::2, 0], under[:, 0]),
                              (z[::2, -1], under[:, -1])):
                worst = max(worst, float(np.abs(edge - ref).max()))
    table.check("pyramid", "chain edges vs parent max |Δ| (m)", worst, 1e-3)

    # closed basins deeper than the breach limit stay, as do basins under ice sheets (not counted)
    pits = max(_pit_share(b) for level in range(top + 1, finest + 1) for b in pyramid.blocks(level))
    table.check("pyramid", "ice-free inland pits, chain levels, worst block", pits, 1e-3, "{:.3%}")
    pits = max(_pit_share(b) for b in pyramid.blocks(top))
    table.report("pyramid", f"ice-free inland pits, level {top}, worst face", pits, "{:.3%}")

    folder = timed(table, "pyramid tiles (16-bit PNG)", export_pyramid, pyramid, world, out / "pyramid")
    index = json.loads((folder / "tileset.json").read_text())
    enc = index["elevation_encoding"]
    worst = 0.0
    for level, entry in index["levels"].items():
        tile = entry["tiles"][0]
        block = pyramid.block_at(tile["face"], int(level), tile["x"], tile["y"])
        decoded = enc["offset"] + enc["scale"] * read_band(folder / tile["files"]["elevation"]).astype(float)
        worst = max(worst, float(np.abs(decoded - block.tile(tile["x"], tile["y"])[::-1]).max() / enc["scale"]))
    table.check("pyramid", "tile files decode max error (steps)", worst, 0.5)
    return pyramid, finest


def _border_gap(meshes) -> tuple[float, int]:
    """Return the largest distance between border vertices that neighbouring tiles share, and how many pairs."""
    parts, owner = [], []
    for k, m in enumerate(meshes):
        p = m.planet_vertices()
        parts.append(p)
        owner.append(np.full(len(p), k))
    p, owner = np.concatenate(parts), np.concatenate(owner)
    tree = cKDTree(p)
    pairs = tree.query_pairs(r=1.0, output_type="ndarray")
    pairs = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
    if not len(pairs):
        return 0.0, 0
    return float(np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1).max()), len(pairs)


def check_meshes(table: Table, world, pyramid, finest: int, place, out: Path) -> None:
    """Meshes: shared borders, glTF placement and the simplification tolerance."""
    import trimesh

    globe = timed(table, "globe mesh level 2", globe_meshes, world, 2, sea=False)
    gap, pairs = _border_gap(globe)
    table.check("mesh", "globe level 2: shared border gap max (m)", gap, 1e-3)
    table.report("mesh", "globe level 2: shared border vertices", pairs, "{:,.0f}")
    chain = pyramid_meshes(pyramid, world, finest, sea=False)
    gap, pairs = _border_gap(chain)
    table.check("mesh", f"pyramid level {finest}: shared border gap max (m)", gap, 1e-3)

    path = write_meshes(chain, out / "chain.glb")
    scene = trimesh.load(path)
    to_planet = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    worst = 0.0
    for m in chain:
        transform, _ = scene.graph[m.name]
        v = trimesh.transform_points(scene.geometry[m.name].vertices, transform) @ to_planet.T
        worst = max(worst, float(np.abs(v - m.planet_vertices()).max() / np.ptp(m.vertices, axis=0).max()))
    table.check("mesh", "GLB placement error / tile extent", worst, 1e-6)

    from matplotlib.tri import LinearTriInterpolator, Triangulation

    block = pyramid.blocks(finest)[0]
    x0, y0, _, _ = block.tiles
    n = h.PYRAMID_NODES_PER_TILE
    z = block.tile(x0, y0)
    grid = region_grid(block.face, block.level, x0, y0, x0, y0, nodes_per_tile=n)
    tolerance = h.MESH_ERROR_FRACTION * grid.mean_spacing(pyramid.radius_m)
    tris = rtin_triangles(rtin_errors(z, grid.points.reshape(n, n, 3), pyramid.radius_m, True), tolerance)
    jj, ii = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    interp = LinearTriInterpolator(Triangulation(jj.ravel(), ii.ravel(), tris), z.ravel())(jj, ii)
    table.check("mesh", "RTIN height error max / tolerance", float(np.nanmax(np.abs(interp - z)) / tolerance), 1.0)
    table.report("mesh", "RTIN triangles kept (share of full)", len(tris) / (2 * (n - 1) ** 2), "{:.1%}")

    region = zoom_region(world, *place, resolution_m=2500, half_width_km=60)
    timed(table, "region mesh (GLB)", lambda: write_meshes(region_meshes(world, region), out / "region.glb",
                                                           frame="local"))


def check_globe(table: Table, world, out: Path) -> None:
    """Globe viewer: build time and one off-screen image."""
    try:
        import pyvista as pv
    except ImportError:
        table.report("timing", "globe viewer (pyvista not installed)", 0, "{:.0f}")
        return
    pv.OFF_SCREEN = True
    from worldgen.render import GlobeViewer

    viewer = timed(table, "globe viewer level 1", GlobeViewer, world, 1)
    timed(table, "globe image 1000 × 1000", viewer.screenshot, out / "globe.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default=None, help="saved world folder (default: generate the Earth example)")
    parser.add_argument("--resolution", default="standard")
    parser.add_argument("--pyramid", action="store_true", help="build the full pyramid (levels 0-3, chain to 7)")
    parser.add_argument("--out", default=None, help="folder for the files (default: a temporary folder)")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    table = Table()
    if args.world:
        world = load_world(args.world)
    else:
        world = timed(table, f"generate Earth ({args.resolution})", generate_world, load_spec(EARTH_SPEC),
                      args.resolution)
    if world.culture is None:
        from worldgen.culture import CultureSpec, apply_culture

        world = timed(table, "modern culture", apply_culture, world, CultureSpec.model_validate(MODERN))
    st = world.culture.settlements
    i = int(np.argsort(st.population)[len(st) // 2])
    place = (float(st.lat_deg[i]), float(st.lon_deg[i]))
    print(f"{world.state.name}: {world.grid.size:,} cells; region and chain around {place[0]:.2f}°, {place[1]:.2f}°")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(args.out) if args.out else Path(tmp)
        out.mkdir(parents=True, exist_ok=True)
        check_rasters(table, world, out)
        check_regions(table, world, place, out)
        pyramid, finest = check_pyramid(table, world, place, args.pyramid, out)
        check_meshes(table, world, pyramid, finest, place, out)
        check_globe(table, world, out)
    failed = table.print()
    print(f"\n{failed} of {sum(r[4] != '-' for r in table.rows)} checks failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
