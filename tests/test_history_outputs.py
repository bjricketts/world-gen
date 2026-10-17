"""Outputs of an organic history: epochs on maps and reports, the chronicle, the series, figures and commands
(task 9.10)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from typer.testing import CliRunner

from worldgen import heuristics as h
from worldgen import load_world, save_world
from worldgen.cli import app
from worldgen.culture import CultureSpec, apply_culture, format_chronicle, world_at_epoch
from worldgen.culture import chronicle as ch
from worldgen.culture.model import ChronicleEntry
from worldgen.report import format_culture, format_history

SPEC = {"history": {"start_years_ago": 3000, "epochs": [2000, 1000]},
        "cultures": [{"name": "Oru", "mode": "organic", "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}},
                     {"name": "Settlers", "mode": "organic", "inhabitants": "colonist",
                      "development": {"energy": "fossil", "transport": "rail_and_steam", "communication": "telegraph"},
                      "colony": {"founded_years_ago": 500, "landing_sites": [{"lat_deg": 45, "lon_deg": 10}]}}]}


@pytest.fixture(scope="module")
def organic(world_cache):
    """A preview Earth-spec world with a 3000-year organic history and two kept epochs."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return apply_culture(world_cache(spec), CultureSpec.model_validate(SPEC))


@pytest.fixture(scope="module")
def saved(organic, tmp_path_factory):
    """The organic world saved to a folder."""
    folder = tmp_path_factory.mktemp("organic") / "world"
    save_world(organic, folder)
    return folder


def test_the_history_records_a_series_per_culture(organic):
    s = organic.culture.series
    assert len(s) and s.years_ago.max() < 3000 and s.years_ago.min() == 0.0
    assert set(np.unique(s.culture)) >= {0, 1}
    late = (s.culture == 1) & (s.years_ago == 0.0)
    assert s.people[late][0] == pytest.approx(organic.culture.culture(1).population, rel=0.05)
    assert np.all(s.years_ago[s.culture == 1] < 500)                       # the colony from its landing
    assert np.all(organic.culture.at_epoch(2000).series.years_ago >= 2000)


def test_the_series_survives_a_save(organic, saved):
    loaded = load_world(saved).culture
    assert np.array_equal(loaded.series.people, organic.culture.series.people)
    assert np.array_equal(loaded.epochs[0].series.years_ago, organic.culture.epochs[0].series.years_ago)


def test_a_world_shows_any_kept_epoch(organic):
    old = world_at_epoch(organic, 2000)
    assert old.culture.epoch_years_ago == 2000 and old.surface is organic.surface
    assert world_at_epoch(organic, 0).culture is organic.culture
    with pytest.raises(KeyError):
        world_at_epoch(organic, 1234)


def test_chronicle_entries_are_major_by_kind_or_size():
    e = lambda kind, **refs: ChronicleEntry(years_ago=100.0, kind=kind, text="", refs=refs)
    assert ch.is_major(e("independence")) and not ch.is_major(e("secession"))
    assert ch.is_major(e("war", states=h.CHRONICLE_MAJOR_WAR_STATES)) and not ch.is_major(e("war", states=3))
    assert ch.is_major(e("break_up", area_km2=h.CHRONICLE_MAJOR_KM2)) and not ch.is_major(e("break_up", area_km2=1))
    assert ch.is_major(e("pandemic", killed=2e7)) and not ch.is_major(e("pandemic", killed=2e6))
    assert ch.is_major(e("overseas_colony", new=True)) and not ch.is_major(e("overseas_colony", new=False))
    assert ch.period_of(4200.0) == (5000.0, 4000.0) and ch.period_of(1750.0) == (2000.0, 1500.0)
    assert ch.period_of(250.0) == (300.0, 200.0)


def test_the_chronicle_reads_by_period_with_names(organic):
    civ = organic.culture
    text = format_chronicle(civ)
    assert "3,000 years ago to" in text and "Oru live as foragers" in text
    listed = format_chronicle(civ, minor="list")
    hidden = format_chronicle(civ, minor="hide")
    assert len(listed.splitlines()) >= len(text.splitlines()) - text.count("also ") >= len(hidden.splitlines()) - 50
    assert "also" not in hidden
    colony = format_chronicle(civ, since=600, until=400, kinds=["landing"])
    assert "Settlers land" in colony and "Oru" not in colony
    states, cultures = ch.names(civ)
    named = ChronicleEntry(years_ago=10.0, kind="war", text=f"state {next(iter(states))} wins; Culture 0 loses")
    out = ch.named_text(named, states, cultures)
    assert next(iter(states.values())) in out and "Oru loses" in out


def test_the_report_shows_the_history(organic):
    text = format_culture(organic.culture, organic)
    assert "History" in text and "2,000 years ago" in text and "present" in text and "major" in text
    rows = format_history(organic.culture).splitlines()
    assert len([r for r in rows if "years ago" in r or "present" in r]) == 3


def test_epoch_figures_draw(organic, tmp_path):
    from worldgen.render import animate_epochs, epoch_dates, plot_chronicle, plot_epochs

    assert epoch_dates(organic) == [2000.0, 1000.0, 0.0]
    fig = plot_epochs(organic, "cultures")
    assert len(fig.axes) == 3 and not fig.texts and all(not ax.get_title() for ax in fig.axes)
    matplotlib.pyplot.close(fig)
    gif = animate_epochs(organic, tmp_path / "epochs.gif", "political", width=200)
    assert gif.exists() and gif.stat().st_size > 0
    fig = plot_chronicle(organic)
    assert len(fig.axes) == 3 and fig.axes[0].get_yscale() == "log"
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Oru" in labels and "Settlers" in labels
    matplotlib.pyplot.close(fig)


def test_states_and_cultures_keep_their_colour_across_epochs():
    from worldgen.render.culture import _id_colours

    a, b = _id_colours(np.array([5, 18943])), _id_colours(np.array([18943, 7, 5]))
    assert np.allclose(a[0], b[2]) and np.allclose(a[1], b[0])


def test_commands_show_epochs_and_the_chronicle(saved, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["chronicle", str(saved), "--culture", "Settlers"])
    assert result.exit_code == 0, result.output
    assert "Settlers" in result.output and "Oru live as foragers" not in result.output
    result = runner.invoke(app, ["report", str(saved), "--epoch", "1000"])
    assert result.exit_code == 0 and "1,000 years ago" in result.output
    result = runner.invoke(app, ["report", str(saved), "--epoch", "999"])
    assert result.exit_code == 1 and "no epoch" in result.output
    png = tmp_path / "old.png"
    result = runner.invoke(app, ["map", str(saved), str(png), "-f", "political", "-e", "2000"])
    assert result.exit_code == 0 and png.exists()
    for args in (["epochs", str(saved), str(tmp_path / "p.png"), "-f", "cultures"],
                 ["epochs", str(saved), str(tmp_path / "t.png"), "--timeline"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output


def test_format_two_civilisations_still_load(organic, saved, tmp_path):
    import shutil

    import yaml

    old = tmp_path / "old"
    shutil.copytree(saved, old)
    path = old / "culture" / "culture.yaml"
    data = yaml.safe_load(path.read_text())
    data["culture_format"] = 2
    path.write_text(yaml.safe_dump(data))
    assert load_world(old).culture.epoch_years_ago == 0.0
