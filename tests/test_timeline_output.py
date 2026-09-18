"""Timeline output: the saved world format, the report sections, the history figures and the CLI."""

import matplotlib
import pytest

matplotlib.use("Agg")

from typer.testing import CliRunner                                    # noqa: E402

from worldgen.cli import _epochs, app                                  # noqa: E402
from worldgen.generate import generate_world                           # noqa: E402
from worldgen.report import format_report, format_timeline, state_to_plain   # noqa: E402
from worldgen.spec import PlanetSpec, load_spec                        # noqa: E402
from worldgen.world import FORMAT_VERSION, load_world, save_world      # noqa: E402

from .conftest import EXAMPLES                                         # noqa: E402


@pytest.fixture(scope="module")
def earth_world():
    """Earth in history mode with a small surface, generated once."""
    spec = load_spec(EXAMPLES / "earth_history.yaml")
    spec = spec.model_copy(update={"history": spec.history.model_copy(update={"epochs_gyr": [1.0, 3.0]})})
    return generate_world(spec, "preview")


@pytest.fixture(scope="module")
def plain_world():
    """A small snapshot planet, for the cases that must not produce a history."""
    return generate_world(PlanetSpec(name="Testworld", seed=2, priors={"archetype": "temperate"}), "preview")


def test_epoch_option_reads_a_list_of_ages():
    """--epochs takes a comma-separated list and sorts it."""
    assert _epochs("2, 0.5,1") == [0.5, 1.0, 2.0]
    with pytest.raises(Exception):
        _epochs("soon")


@pytest.mark.slow
def test_saved_world_round_trips_its_timeline(earth_world, tmp_path):
    """A history world saves as format 4 with timeline.yaml and one file per epoch, and reloads unchanged."""
    folder = save_world(earth_world, tmp_path / "w")
    assert (folder / "timeline.yaml").exists()
    names = sorted(p.name for p in (folder / "epochs").iterdir())
    assert names == ["00_1.00gyr.yaml", "01_3.00gyr.yaml", "02_4.57gyr.yaml"]

    loaded = load_world(folder)
    assert state_to_plain(loaded.state) == state_to_plain(earth_world.state)
    saved, original = loaded.timeline, earth_world.timeline
    assert saved.times_s == pytest.approx(original.times_s)
    assert [e.kind for e in saved.events] == [e.kind for e in original.events]
    assert saved.series["surface_temperature_k"] == pytest.approx(original.series["surface_temperature_k"])
    assert [state_to_plain(s) for s in saved.states] == [state_to_plain(s) for s in original.states]

    (folder / "state.yaml").write_text(
        (folder / "state.yaml").read_text(encoding="utf-8").replace(f"format_version: {FORMAT_VERSION}",
                                                                    "format_version: 3"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_world(folder)


@pytest.mark.slow
def test_snapshot_world_saves_without_a_timeline(plain_world, tmp_path):
    """A snapshot planet has no history, so no timeline file is written."""
    folder = save_world(plain_world, tmp_path / "s")
    assert not (folder / "timeline.yaml").exists()
    assert load_world(folder).timeline is None


@pytest.mark.slow
def test_report_summarises_the_history(earth_world):
    """The main report gains a History section; the timeline report lists every event and epoch."""
    state, timeline = earth_world.state, earth_world.timeline
    report = format_report(state)
    assert "\nHistory\n" in report
    summary = report[report.index("\nHistory\n"):]
    assert "origin of life" in summary and "oxygenation" in summary

    full = format_timeline(timeline, state)
    for event in timeline.events:
        assert event.kind in full
    assert "Epochs" in full
    assert full.count("Gyr") >= len(timeline.events) + len(timeline.states)
    assert f"{state.atmosphere.surface_temperature_k:.0f}" in full


@pytest.mark.slow
def test_a_snapshot_planet_has_no_history_section(plain_world):
    """Snapshot mode leaves the report as it was."""
    assert "\nHistory\n" not in format_report(plain_world.state)


@pytest.mark.slow
def test_history_figures_cover_the_run(earth_world):
    """Each figure has four panels spanning the integrated history, with the events marked."""
    from worldgen.render import has_timeline, plot_timeline

    assert has_timeline(earth_world)
    series = earth_world.timeline.series
    for topic in ("climate", "interior", "life"):
        figure = plot_timeline(earth_world, topic)
        axes = figure.axes
        assert len(axes) >= 4
        assert axes[0].get_xlim() == pytest.approx((series["time_gyr"][0], series["time_gyr"][-1]))
        lines = [line for ax in axes for line in ax.get_lines()]
        assert any(len(line.get_xdata()) == len(series["time_gyr"]) for line in lines)
        vertical = sum(1 for ax in axes for line in ax.get_lines() if line.get_linestyle() == ":")
        assert vertical >= len(earth_world.timeline.events)
    with pytest.raises(ValueError):
        plot_timeline(earth_world, "weather")


@pytest.mark.slow
def test_figures_refuse_a_planet_without_a_history(plain_world):
    """A snapshot world has nothing to draw."""
    from worldgen.render import has_timeline, plot_climate_history

    assert not has_timeline(plain_world)
    with pytest.raises(ValueError):
        plot_climate_history(plain_world)


@pytest.mark.slow
def test_cli_history_workflow(tmp_path):
    """Generate a history world with chosen epochs, then draw and print its history."""
    runner = CliRunner()
    world_dir = tmp_path / "world"
    result = runner.invoke(app, ["generate", str(EXAMPLES / "earth_history.yaml"), "-r", "preview",
                                 "--epochs", "1,2", "-s", str(world_dir)])
    assert result.exit_code == 0, result.output
    assert "History" in result.output
    assert len(list((world_dir / "epochs").iterdir())) == 3

    figure = tmp_path / "life.png"
    result = runner.invoke(app, ["history", str(world_dir), str(figure), "-f", "life"])
    assert result.exit_code == 0, result.output
    assert figure.stat().st_size > 0

    result = runner.invoke(app, ["report", str(world_dir), "--timeline"])
    assert result.exit_code == 0 and "Events" in result.output and "origin_of_life" in result.output


@pytest.mark.slow
def test_cli_rejects_a_history_figure_without_a_history(tmp_path):
    """A snapshot world cannot produce history figures or a timeline report."""
    runner = CliRunner()
    world_dir = tmp_path / "world"
    result = runner.invoke(app, ["random", "-a", "temperate", "--seed", "4", "-r", "preview",
                                 "--tectonics", "heuristic", "-s", str(world_dir)])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["history", str(world_dir), str(tmp_path / "h.png")]).exit_code == 1
    assert runner.invoke(app, ["report", str(world_dir), "--timeline"]).exit_code == 1
