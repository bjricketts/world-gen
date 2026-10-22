"""Region exports (task 10.2): node fields, native and LAEA rasters, georeferencing, bundles, vectors, engines."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from worldgen.export import (ENGINES, engine_region, engine_sizes, export_engine_heightmap, export_region,  # noqa: E402
                             export_region_bundle, region_field, region_raster, world_height_limits)
from worldgen.export.crs import laea_forward, laea_inverse  # noqa: E402
from worldgen.export.regions import ground_control_points, north_angle_deg  # noqa: E402
from worldgen.export.writers import read_sidecar  # noqa: E402
from worldgen.zoom import region_grid, zoom_region  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MODERN = {"cultures": [{"name": "Ana", "development": {"energy": "electric"}}]}


@pytest.fixture(scope="module")
def world(world_cache):
    """A preview Earth-spec world (heuristic plates) with a modern culture."""
    from worldgen.culture import CultureSpec, apply_culture
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return apply_culture(world_cache(spec), CultureSpec.model_validate(MODERN))


@pytest.fixture(scope="module")
def place(world):
    """A mid-sized settlement's position."""
    st = world.culture.settlements
    i = int(np.argsort(st.population)[len(st) // 2])
    return float(st.lat_deg[i]), float(st.lon_deg[i])


@pytest.fixture(scope="module")
def region(world, place):
    return zoom_region(world, *place, resolution_m=2500, half_width_km=60)


@pytest.fixture(scope="module")
def local(world, region):
    from worldgen.culture.local import zoom_culture

    return zoom_culture(world, region)


def test_node_fields_follow_the_region(region, local):
    assert np.array_equal(region_field(region, "elevation").values, region.elevation)
    ice = region.ice_thickness_m if region.ice_thickness_m is not None else 0.0
    assert np.allclose(region_field(region, "surface").values, region.elevation + ice)
    assert region_field(region, "biomes").labels[0] == "ocean"
    assert np.array_equal(region_field(region, "political", local).values, local.polity_map)
    roads = region_field(region, "roads", local)
    assert (roads.values >= 0).sum() == len({n for nodes, _, _ in local.roads for n in nodes})
    with pytest.raises(ValueError, match="needs the region's culture"):
        region_field(region, "political")
    with pytest.raises(ValueError, match="unknown region field"):
        region_field(region, "flavour")


def test_native_raster_puts_the_high_t_edge_on_top_with_gcps_at_nodes(world, region):
    raster = region_raster(world, region, "elevation")
    rows, cols = region.grid.shape
    assert np.array_equal(raster.data[0], region.elevation.reshape(rows, cols)[-1])
    lat = region.grid.lat.reshape(rows, cols)
    lon = region.grid.lon.reshape(rows, cols)
    gcps = ground_control_points(region)
    assert len(gcps) >= 17 * 17
    for r, c, x, y in gcps[::37]:
        node = (rows - 1 - int(r - 0.5), int(c - 0.5))
        assert np.isclose(x, lon[node]) and np.isclose(y, lat[node])


def test_north_angle_follows_the_cube_face():
    equator_x = region_grid(0, 6, 31, 31, 32, 32)      # face +x: t runs along +z, so north is up
    equator_y = region_grid(2, 6, 31, 31, 32, 32)      # face +y: s runs along +z, so north is to the right
    assert abs(north_angle_deg(type("R", (), {"grid": equator_x})())) < 1.0
    assert abs(north_angle_deg(type("R", (), {"grid": equator_y})()) - 90.0) < 1.0


def test_laea_inverse_inverts_forward_and_matches_proj():
    rng = np.random.default_rng(1)
    lat0, lon0, radius = np.radians(37.0), np.radians(112.0), 6.4e6
    x, y = rng.uniform(-8e5, 8e5, (2, 200))
    lat, lon = laea_inverse(x, y, radius, lat0, lon0)
    fx, fy = laea_forward(lat, lon, radius, lat0, lon0)
    assert np.allclose(fx, x, atol=1e-6) and np.allclose(fy, y, atol=1e-6)
    pyproj = pytest.importorskip("pyproj")
    t = pyproj.Transformer.from_crs(f"+proj=longlat +R={radius}", f"+proj=laea +lat_0=37 +lon_0=112 +R={radius}",
                                    always_xy=True)
    px, py = t.transform(np.degrees(lon), np.degrees(lat))
    assert np.allclose(px, x, atol=1e-3) and np.allclose(py, y, atol=1e-3)


def test_gdal_warp_of_the_native_file_matches_the_laea_export(world, region, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.warp import Resampling, reproject

    native = export_region(world, region, tmp_path / "n.tif")
    laea = export_region(world, region, tmp_path / "l.tif", crs="laea")
    with rasterio.open(native) as src, rasterio.open(laea) as dst:
        out = np.full((dst.height, dst.width), np.nan, np.float32)
        reproject(rasterio.band(src, 1), out, src_crs=src.gcps[1], gcps=src.gcps[0], dst_transform=dst.transform,
                  dst_crs=dst.crs, resampling=Resampling.bilinear, SRC_METHOD="GCP_TPS", dst_nodata=np.nan)
        ref = dst.read(1)
    ok = np.isfinite(out) & np.isfinite(ref)
    assert ok.mean() > 0.5
    assert np.sqrt(np.mean((out[ok] - ref[ok]) ** 2)) < 0.02 * max(float(np.nanstd(ref)), 50.0)
    assert read_sidecar(laea)["grid"] == "laea" and read_sidecar(native)["warp"] == "thin plate spline"


def test_heights_share_the_world_scale(world, region, tmp_path):
    path = export_region(world, region, tmp_path / "h.png")
    encoding = read_sidecar(path)["encoding"]
    lo, hi = world_height_limits(world)
    assert np.isclose(encoding["offset"], lo) and np.isclose(encoding["offset"] + 65535 * encoding["scale"], hi)
    with pytest.raises(ValueError, match="engine_region"):
        export_engine_heightmap(world, region, tmp_path / "x.r16")


def test_bundle_writes_layers_vectors_and_an_index(world, region, local, tmp_path):
    folder = export_region_bundle(world, region, tmp_path / "b", suffix=".png", culture=local)
    index = json.loads((folder / "region.json").read_text())
    for name in ("elevation", "ocean", "rivers", "political", "roads", "colour", "lonlat", "settlements_vectors",
                 "roads_vectors", "borders_vectors"):
        assert (folder / index["files"][name]).exists()
    settlements = json.loads((folder / "settlements.geojson").read_text())
    assert len(settlements["features"]) == len(local)
    if len(local):
        lon, lat = np.radians(np.array([f["geometry"]["coordinates"] for f in settlements["features"]])).T
        vec = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        assert np.array_equal(region.grid.tree().query(vec)[1], local.node)
    roads = json.loads((folder / "roads.geojson").read_text())
    assert len(roads["features"]) == len(local.roads)
    borders = json.loads((folder / "borders.geojson").read_text())
    owners = set(np.unique(local.polity_map).tolist())
    assert (len(borders["features"]) > 0) == (len(owners) > 1)
    lonlat = np.load(folder / "lonlat.npz")
    assert lonlat["lon"].shape == region.grid.shape


def test_engine_regions_have_engine_sizes_without_resampling(world, place, tmp_path):
    assert engine_sizes("unity")[:4] == [129, 257, 513, 1025]
    assert engine_sizes("unreal")[3:5] == [1009, 2017]
    with pytest.raises(ValueError, match="sizes are"):
        engine_region(world, *place, "unity", 1009)
    for engine, size in (("unity", 129), ("unreal", 253)):
        region = engine_region(world, *place, engine, size, resolution_m=3000)
        assert region.grid.shape == (size, size) and region.grid.nodes_per_tile == ENGINES[engine]
        path = export_engine_heightmap(world, region, tmp_path / f"{engine}.r16", engine)
        assert path.stat().st_size == 2 * size * size
        settings = read_sidecar(path)["engine_import"]
        lo, hi = world_height_limits(world)
        if engine == "unity":
            assert np.isclose(settings["terrain_height_m"], hi - lo)
        else:
            assert np.isclose(settings["z_scale"], (hi - lo) / 512 * 100)
