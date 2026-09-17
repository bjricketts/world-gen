import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from worldgen import PlanetSpec, generate_world, load_world, save_world  # noqa: E402
from worldgen.cli import app  # noqa: E402
from worldgen.render import FIELDS, PROJECTIONS, plot_map, plot_overview, projection, save_figure, to_raster  # noqa: E402
from worldgen.render.raster import projected_sampler  # noqa: E402
from worldgen.report import state_to_plain  # noqa: E402


@pytest.fixture(scope="module")
def small_world():
    return generate_world(PlanetSpec(name="Testworld", seed=2, priors={"archetype": "temperate"}), "preview")


def test_save_and_load_round_trip(small_world, tmp_path):
    folder = save_world(small_world, tmp_path / "w")
    assert {p.name for p in folder.iterdir()} == {"spec.yaml", "state.yaml", "surface.zarr"}
    loaded = load_world(folder)
    assert loaded.spec == small_world.spec
    assert state_to_plain(loaded.state) == state_to_plain(small_world.state)
    for name in small_world.surface.data_vars:
        assert np.array_equal(loaded.surface[name].values, small_world.surface[name].values, equal_nan=True)
    assert loaded.surface.attrs["regime"] == small_world.surface.attrs["regime"]


def test_load_rejects_unknown_format(small_world, tmp_path):
    folder = save_world(small_world, tmp_path / "w")
    data = yaml.safe_load((folder / "state.yaml").read_text())
    data["format_version"] = 99
    (folder / "state.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_world(folder)


@pytest.mark.slow
def test_saving_a_world_without_surface(tmp_path):
    giant = generate_world(PlanetSpec(seed=1, priors={"archetype": "giant"}, body={"mass_mearth": 300.0}))
    folder = save_world(giant, tmp_path / "g")
    assert not (folder / "surface.zarr").exists()
    assert load_world(folder).surface is None


@pytest.mark.parametrize("name, fraction", [("mollweide", np.pi / 4), ("orthographic", np.pi / 4),
                                            ("equirectangular", 1.0)])
def test_sampler_covers_projection_outline(small_world, name, fraction):
    sampler = projected_sampler(small_world.grid, projection(name), 300)
    assert sampler.valid.mean() == pytest.approx(fraction, abs=0.02)
    image = sampler.sample(small_world.surface["elevation"].values)
    assert np.isnan(image).mean() == pytest.approx(1 - fraction, abs=0.02)


def test_equirectangular_raster(small_world):
    r = to_raster(small_world.grid, small_world.surface["elevation"].values, width=360)
    assert r.shape == (180, 360)
    assert np.isfinite(r).all()
    cat = to_raster(small_world.grid, small_world.surface["terrain"].values, width=360, categorical=True)
    assert set(np.unique(cat)) <= set(np.unique(small_world.surface["terrain"].values))


@pytest.mark.parametrize("field", FIELDS)
def test_every_field_renders(small_world, field):
    ax = plot_map(small_world, field, "mollweide", width=200)
    assert ax.get_title().endswith(field)
    matplotlib.pyplot.close(ax.figure)


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_every_projection_renders(small_world, proj):
    ax = plot_map(small_world, "elevation", proj, width=200)
    matplotlib.pyplot.close(ax.figure)


def test_overview_and_save(small_world, tmp_path):
    path = save_figure(plot_overview(small_world), tmp_path / "overview.png", dpi=40)
    assert path.stat().st_size > 10_000


def test_unknown_names_raise(small_world):
    with pytest.raises(ValueError):
        projection("gnomonic-ish")
    with pytest.raises(ValueError):
        plot_map(small_world, "soil_moisture", width=100)


@pytest.mark.slow
def test_cli_map_workflow(tmp_path):
    runner = CliRunner()
    world_dir = tmp_path / "world"
    overview = tmp_path / "overview.png"
    result = runner.invoke(app, ["random", "-a", "temperate", "--seed", "1", "-r", "preview",
                                 "--tectonics", "heuristic", "-m", str(overview), "-s", str(world_dir)])
    assert result.exit_code == 0, result.output
    assert overview.exists() and (world_dir / "surface.zarr").exists()
    globe = tmp_path / "globe.png"
    result = runner.invoke(app, ["map", str(world_dir), str(globe), "-f", "terrain", "-p", "orthographic"])
    assert result.exit_code == 0, result.output
    assert globe.exists()
    result = runner.invoke(app, ["report", str(world_dir)])
    assert result.exit_code == 0 and "Surface" in result.output


@pytest.mark.slow
def test_cli_tectonic_history_workflow(tmp_path):
    runner = CliRunner()
    world_dir = tmp_path / "world"
    result = runner.invoke(app, ["random", "-a", "temperate", "--seed", "4", "-r", "preview",
                                 "--duration", "60", "--start", "cratons", "--snapshots", "20",
                                 "-s", str(world_dir)])
    assert result.exit_code == 0, result.output
    for name in ("history.gif", "history.png"):
        result = runner.invoke(app, ["history", str(world_dir), str(tmp_path / name), "--panels", "4"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.slow
def test_cli_rejects_history_without_snapshots_and_bad_options(tmp_path):
    runner = CliRunner()
    world_dir = tmp_path / "world"
    result = runner.invoke(app, ["random", "-a", "temperate", "--seed", "4", "-r", "preview",
                                 "--tectonics", "heuristic", "-s", str(world_dir)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["history", str(world_dir), str(tmp_path / "h.gif")])
    assert result.exit_code == 1
    result = runner.invoke(app, ["random", "--tectonics", "magic"])
    assert result.exit_code == 1


def test_history_figure_needs_snapshots(small_world):
    from worldgen.render import has_history, plot_history

    assert not has_history(small_world)
    with pytest.raises(ValueError):
        plot_history(small_world)


def test_river_lines_follow_the_network(small_world):
    from worldgen.render.rivers import has_rivers, smooth_line, trace_rivers

    assert has_rivers(small_world)
    lines = trace_rivers(small_world)
    assert lines
    flow_to = small_world.surface["flow_to"].values
    points = small_world.grid.points
    first = lines[0]
    cells = [int(np.argmax(points @ p)) for p in first.points]
    assert all(flow_to[a] == b for a, b in zip(cells, cells[1:]))
    smooth = smooth_line(first.points)
    np.testing.assert_allclose(smooth[[0, -1]], first.points[[0, -1]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(smooth, axis=1), 1.0)


def test_water_maps_need_liquid_water(world_cache):
    dry = world_cache(PlanetSpec(seed=2, priors={"archetype": "hot_volcanic"}))
    for field in ("rainfall", "basins"):
        with pytest.raises(ValueError):
            plot_map(dry, field, width=100)
