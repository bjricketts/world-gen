"""Zooming into a world end to end: the request, saved regions, the cache, format-4 worlds and the CLI."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from worldgen import PlanetSpec, heuristics as h, load_world, save_world  # noqa: E402
from worldgen.cli import app  # noqa: E402
from worldgen.zoom import (erode_region, load_region, region_block, region_grid, save_region,  # noqa: E402
                           synthesize_region, zoom_region)
from worldgen.world import FORMAT_VERSION  # noqa: E402
from worldgen.zoom.assemble import NODE_FIELDS, OPTIONAL_FIELDS  # noqa: E402

SMALL = dict(level=7, half_width_km=0.0, nodes_per_tile=24)           # one small tile: fast


@pytest.fixture(scope="module")
def world(world_cache):
    return world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"}, surface={"tectonics": "heuristic"}))


def _land_place(world):
    """Return the latitude and longitude of an ice-free mountain cell."""
    s = world.surface
    ok = (~s["ocean"].values) & (s["ice_thickness"].values == 0.0)
    cell = np.flatnonzero(ok)[np.argmax(s["elevation"].values[ok])]
    return float(s.lat.values[cell]), float(s.lon.values[cell])


def _same(a, b):
    """Assert two eroded regions hold the same fields."""
    assert np.array_equal(a.grid.points, b.grid.points)
    for name in NODE_FIELDS + OPTIONAL_FIELDS + ("fabric",):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True), name
    assert a.categorical.keys() == b.categorical.keys()
    assert a.exported_m3 == b.exported_m3


def test_zoom_region_runs_the_whole_chain(world):
    lat, lon = _land_place(world)
    region = zoom_region(world, lat, lon, **SMALL)
    block = region_block(world, lat, lon, level=7, half_width_km=0.0, nodes_per_tile=24)
    manual = erode_region(world, synthesize_region(world, region_grid(*block, nodes_per_tile=24)))
    _same(region, manual)
    assert region.grid.contains(np.array([[np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                                           np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                                           np.sin(np.radians(lat))]]))[0]      # the place is inside the region


def test_resolution_picks_the_level(world):
    lat, lon = _land_place(world)
    coarse = region_block(world, lat, lon, resolution_m=2000.0)
    fine = region_block(world, lat, lon, resolution_m=500.0)
    assert fine[1] == coarse[1] + 2                                   # four times finer: two levels deeper


def test_requests_are_checked(world, monkeypatch):
    lat, lon = _land_place(world)
    with pytest.raises(ValueError):
        zoom_region(world, lat, lon, ruggedness="spiky", **SMALL)
    monkeypatch.setattr(h, "ZOOM_MAX_NODES", 100)
    with pytest.raises(ValueError):
        zoom_region(world, lat, lon, **SMALL)


def test_regions_round_trip_through_a_folder(world, tmp_path):
    region = zoom_region(world, *_land_place(world), **SMALL)
    _same(region, load_region(save_region(region, tmp_path / "r.zarr")))


def test_the_cache_reuses_regions_of_the_same_world(world, tmp_path):
    lat, lon = _land_place(world)
    first = zoom_region(world, lat, lon, cache_dir=tmp_path, **SMALL)
    stored = list(tmp_path.iterdir())
    assert len(stored) == 1
    stamp = stored[0].stat().st_mtime_ns
    second = zoom_region(world, lat, lon, cache_dir=tmp_path, **SMALL)
    _same(first, second)
    assert stored[0].stat().st_mtime_ns == stamp                     # loaded, not rebuilt
    zoom_region(world, lat, lon, cache_dir=tmp_path, ruggedness="gentle", **SMALL)
    assert len(list(tmp_path.iterdir())) == 2                         # a different request is a different entry


def test_format_4_worlds_load_with_their_fabric(world, tmp_path):
    folder = save_world(world, tmp_path / "w")
    assert yaml.safe_load((folder / "state.yaml").read_text(encoding="utf-8"))["format_version"] == FORMAT_VERSION
    data = yaml.safe_load((folder / "state.yaml").read_text(encoding="utf-8"))
    data["format_version"] = 4
    (folder / "state.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    surface = xr.open_zarr(folder / "surface.zarr", consolidated=False).load().drop_vars("fabric")
    surface.to_zarr(tmp_path / "old.zarr", mode="w", consolidated=False)
    import shutil
    shutil.rmtree(folder / "surface.zarr")
    shutil.move(tmp_path / "old.zarr", folder / "surface.zarr")
    loaded = load_world(folder)
    assert "fabric" in loaded.surface
    assert np.allclose(loaded.surface["fabric"].values, world.surface["fabric"].values, atol=1e-3)


@pytest.mark.slow
def test_cli_zoom(world, tmp_path):
    folder = save_world(world, tmp_path / "w")
    lat, lon = _land_place(world)
    runner = CliRunner()
    args = ["zoom", str(folder), str(tmp_path / "local.png"), f"--lat={lat}", f"--lon={lon}", "--level", "7",
            "--half-width", "0", "--cache"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "local.png").stat().st_size > 0
    assert len(list((folder / "regions").iterdir())) == 1
    result = runner.invoke(app, args[:3] + [f"--lat={lat}", f"--lon={lon}", "--field", "rainfall"])
    assert result.exit_code == 1
