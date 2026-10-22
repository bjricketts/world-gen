"""Export core (task 10.1): planet CRS, samplers, field registry, raster writers and global rasters."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from worldgen.export import (EXPORT_FIELDS, equirect_geotransform, equirect_sampler, export_field,  # noqa: E402
                             export_global, global_colours, global_raster, planet_crs_wkt, point_sampler,
                             write_raster)
from worldgen.export.writers import decode, read_sidecar  # noqa: E402
from worldgen.render.raster import lonlat_vectors  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ORGANIC = {"history": {"start_years_ago": 2000, "epochs": [1000]},
           "cultures": [{"name": "Oru", "mode": "organic",
                         "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}}]}


@pytest.fixture(scope="module")
def earth(world_cache):
    """A preview Earth-spec world with heuristic plates."""
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return world_cache(spec)


def test_planet_crs_is_a_sphere_of_the_planet_radius():
    rasterio = pytest.importorskip("rasterio")
    crs = rasterio.crs.CRS.from_wkt(planet_crs_wkt('Kestrel "b"', 7.1e6))
    assert crs.is_geographic
    assert "7100000" in crs.to_wkt()


def test_geotransform_puts_pixel_centres_on_the_lonlat_lattice():
    lon, lat, _ = lonlat_vectors(8, 16)
    x0, dx, _, y0, _, dy = equirect_geotransform(8, 16)
    assert np.allclose(x0 + (np.arange(16) + 0.5) * dx, lon)
    assert np.allclose(y0 + (np.arange(8) + 0.5) * dy, lat)


def test_png_and_raw_codes_decode_to_the_values(tmp_path):
    data = np.linspace(-4000.0, 6000.0, 64 * 32).reshape(32, 64)
    data[3, 5] = np.nan
    for name in ("h.png", "h.r16"):
        path = write_raster(tmp_path / name, data, {"field": "elevation"})
        meta = read_sidecar(path)
        if name.endswith(".png"):
            from PIL import Image

            codes = np.asarray(Image.open(path))
        else:
            codes = np.fromfile(path, dtype="<u2").reshape(32, 64)
        values = decode(codes, meta["encoding"])
        assert np.isnan(values[3, 5])
        ok = np.isfinite(data)
        assert np.abs(values[ok] - data[ok]).max() <= meta["encoding"]["scale"] / 2 + 1e-9
        assert (meta["width"], meta["height"]) == (64, 32)


def test_category_codes_keep_their_values_and_mark_empty_cells(tmp_path):
    from PIL import Image

    codes = np.array([[0, 3, -1], [7, 254, 1]])
    path = write_raster(tmp_path / "c.png", codes, {"field": "biomes"}, kind="codes")
    read = np.asarray(Image.open(path))
    assert read.dtype == np.uint8 and read[0, 2] == 255 and read[1, 1] == 254
    wide = write_raster(tmp_path / "w.png", np.array([[0, 300, -1]]), {}, kind="codes")
    assert np.asarray(Image.open(wide)).max() == 65535
    with pytest.raises(ValueError, match="16 bits"):
        write_raster(tmp_path / "x.png", np.array([[70000]]), {}, kind="codes")
    with pytest.raises(ValueError, match="unknown raster format"):
        write_raster(tmp_path / "x.jpg", codes, {}, kind="codes")


def test_point_sampler_reproduces_grid_values_at_the_cells(earth):
    grid = earth.grid
    cells = np.arange(0, grid.size, 97)
    sampler = point_sampler(grid, grid.points[cells])
    elevation = earth.surface["elevation"].values
    assert np.allclose(sampler.sample(elevation).ravel(), elevation[cells], atol=1e-6)
    assert np.array_equal(sampler.sample(earth.surface["biome"].values, categorical=True).ravel(),
                          earth.surface["biome"].values[cells])


def test_every_physical_field_exports_on_a_living_world(earth):
    for name in EXPORT_FIELDS:
        try:
            field = export_field(earth, name)
        except ValueError as err:
            assert "culture" in str(err) or "no '" in str(err)
            continue
        assert field.values.shape == (earth.grid.size,)
        if field.categorical:
            assert field.labels is None or all(isinstance(v, str) for v in field.labels.values())
    july = export_field(earth, "rainfall:7")
    assert np.array_equal(july.values, earth.surface["monthly_precipitation"].values[6])
    with pytest.raises(ValueError, match="months"):
        export_field(earth, "temperature:13")
    with pytest.raises(ValueError, match="unknown export field"):
        export_field(earth, "flavour")


def test_global_geotiff_round_trip(earth, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    path = export_global(earth, tmp_path / "e.tif", "elevation", width=256)
    raster = global_raster(earth, "elevation", width=256)
    with rasterio.open(path) as ds:
        assert (ds.width, ds.height) == (256, 128)
        assert np.allclose(ds.read(1), raster.data.astype(np.float32))
        assert np.allclose(ds.transform.to_gdal(), equirect_geotransform(128, 256))
        assert ds.crs.is_geographic and ds.tags()["field"] == "elevation"
    meta = read_sidecar(path)
    assert meta["units"] == "m" and meta["grid"] == "equirectangular"


def test_sea_follows_the_blended_coast_and_categories_carry_labels(earth, tmp_path):
    ocean = global_raster(earth, "ocean", width=512)
    elevation = global_raster(earth, "elevation", width=512)
    assert not (ocean.data.astype(bool) & (elevation.data >= 0)).any()
    land_share = 1.0 - ocean.data.mean()
    assert abs(land_share - earth.surface.attrs["land_fraction"]) < 0.08
    path = export_global(earth, tmp_path / "b.png", "biomes", width=256)
    assert "tropical rainforest" in read_sidecar(path)["labels"].values()


def test_colour_rasters_match_the_map_colours(earth, tmp_path):
    raster = global_colours(earth, "biomes", width=256)
    assert raster.data.shape == (128, 256, 3) and raster.kind == "colour"
    assert 0.0 <= raster.data.min() and raster.data.max() <= 1.0
    path = export_global(earth, tmp_path / "e.png", "elevation", width=256, colour=True, hillshade=True)
    from PIL import Image

    assert Image.open(path).mode == "RGB"
    with pytest.raises(ValueError, match="RAW"):
        export_global(earth, tmp_path / "e.r16", "elevation", width=256, colour=True)


def test_culture_fields_export_at_kept_epochs(earth, tmp_path):
    from worldgen.culture import CultureSpec, apply_culture
    from worldgen.render.culture import _owner

    world = apply_culture(earth, CultureSpec.model_validate(ORGANIC))
    for epoch in (None, 1000.0):
        raster = global_raster(world, "political", width=256, epoch=epoch)
        civ = world.culture if epoch is None else world.culture.at_epoch(epoch)
        sampler = equirect_sampler(world.grid.size, 256)
        assert np.array_equal(raster.data, sampler.sample(_owner(civ), categorical=True))
    path = export_global(world, tmp_path / "energy.tif", "energy", width=256, epoch=1000.0)
    assert read_sidecar(path)["epoch_years_ago"] == 1000.0
    assert set(read_sidecar(path)["labels"].values()) >= {"foraging", "farming"}
    with pytest.raises(ValueError, match="overlays"):
        global_raster(world, "roads", width=64)
