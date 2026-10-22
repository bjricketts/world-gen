"""The export and globe commands (task 10.6): every subcommand writes its files, and bad requests fail cleanly."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from worldgen import save_world  # noqa: E402
from worldgen.cli import app  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ORGANIC = {"history": {"start_years_ago": 2000, "epochs": [1000]},
           "cultures": [{"name": "Oru", "mode": "organic",
                         "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}}]}
runner = CliRunner()


@pytest.fixture(scope="module")
def folder(world_cache, tmp_path_factory):
    from worldgen.culture import CultureSpec, apply_culture
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    world = apply_culture(world_cache(spec), CultureSpec.model_validate(ORGANIC))
    return save_world(world, tmp_path_factory.mktemp("cli") / "w")


def _place(folder):
    """Return the latitude and longitude of the highest land cell of the saved world."""
    from worldgen import load_world

    s = load_world(folder).surface
    land = np.flatnonzero(~s["ocean"].values & (s["ice_thickness"].values == 0.0))
    cell = land[np.argmax(s["elevation"].values[land])]
    return float(s.lat.values[cell]), float(s.lon.values[cell])


def _ok(args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def _fails(args, text):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code != 0
    assert text in result.output


def test_export_raster_and_fields(folder, tmp_path):
    listing = _ok(["export", "fields", folder])
    assert listing.splitlines()[0].startswith("elevation") and "political" in listing
    _ok(["export", "raster", folder, tmp_path / "e.tif", "--width", 256])
    _ok(["export", "raster", folder, tmp_path / "p.png", "-f", "political", "--colour", "--width", 256,
         "--epoch", 1000])
    meta = json.loads((tmp_path / "e.tif.json").read_text())
    assert meta["field"] == "elevation" and (meta["height"], meta["width"]) == (128, 256)
    _fails(["export", "raster", folder, tmp_path / "x.tif", "-f", "nope"], "nope")


def test_export_region_engine_and_mesh(folder, tmp_path):
    lat, lon = _place(folder)
    where = [f"--lat={lat}", f"--lon={lon}"]
    _ok(["export", "region", folder, tmp_path / "r", *where, "--level", 6, "--half-width", 0, "--format", ".png"])
    index = json.loads((tmp_path / "r" / "region.json").read_text())
    assert {"elevation", "rivers", "political", "settlements"} <= set(index["files"])
    _ok(["export", "engine", folder, tmp_path / "h.r16", *where, "--size", 129, "--level", 6])
    assert (tmp_path / "h.r16").stat().st_size == 129 * 129 * 2
    _fails(["export", "engine", folder, tmp_path / "h.r16", *where, "--size", 130], "sizes are")
    _ok(["export", "mesh", folder, tmp_path / "g.glb", "--level", 0])
    _ok(["export", "mesh", folder, tmp_path / "r.ply", *where, "--level", 6, "--half-width", 0, "-f", "biome"])
    assert (tmp_path / "g.glb").stat().st_size > 0 and (tmp_path / "r.ply").stat().st_size > 0
    _fails(["export", "mesh", folder, tmp_path / "x.glb", f"--lat={lat}"], "both --lat and --lon")
    _fails(["export", "mesh", folder, tmp_path / "x.glb", *where, "-f", "rainfall"], "relief or biome")


def test_export_pyramid(folder, tmp_path):
    lat, lon = _place(folder)
    _ok(["export", "pyramid", folder, tmp_path / "p", "--top-level", 0, "--place", f"{lat},{lon},1", "--meshes"])
    index = json.loads((tmp_path / "p" / "tileset.json").read_text())
    assert sorted(index["levels"], key=int) == ["0", "1"]
    assert len(list((tmp_path / "p" / "L0").rglob("*.png"))) == 6
    assert len(list((tmp_path / "p" / "meshes" / "L1").rglob("*.glb"))) == 4
    _fails(["export", "pyramid", folder, tmp_path / "q", "--place", "1,2"], "latitude,longitude,level")


def test_globe_writes_images_pages_and_folders(folder, tmp_path):
    pytest.importorskip("pyvista").OFF_SCREEN = True
    _ok(["globe", folder, tmp_path / "g.png", "--level", 0])
    assert (tmp_path / "g.png").stat().st_size > 0
    _ok(["globe", folder, tmp_path / "views", "--level", 0, "-f", "elevation,political", "--all-epochs"])
    assert sorted(p.name for p in (tmp_path / "views").iterdir()) == [
        "elevation_0ya.png", "elevation_1000ya.png", "political_0ya.png", "political_1000ya.png"]
    _fails(["globe", folder, tmp_path / "v2", "--level", 0, "-f", "elevation,nope"], "'nope' is not available")
    assert not (tmp_path / "v2").exists()
    _fails(["globe", folder, tmp_path / "g.jpg", "--level", 0], "unknown output type")
