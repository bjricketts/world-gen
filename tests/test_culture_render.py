"""Culture layers on maps, the culture report and the `worldgen culture` command."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
from typer.testing import CliRunner

from worldgen import save_world
from worldgen.cli import app
from worldgen.culture.build import apply_culture
from worldgen.culture.spec import CultureSpec, save_culture_spec
from worldgen.render import CULTURE_FIELDS, plot_culture, plot_map
from worldgen.render.culture import road_collections, split_field
from worldgen.report import format_culture

SPEC = CultureSpec(cultures=[{"name": "Testers", "development": {"energy": "fossil"}},
                             {"inhabitants": "colonist", "colony": {"founded_years_ago": 150}}])


@pytest.fixture(scope="module")
def cultured(world_cache):
    """A preview Earth-spec world on heuristic plates with a native and a colonist culture."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return apply_culture(world_cache(spec), SPEC)


@pytest.mark.parametrize("field", [f for f in CULTURE_FIELDS if f != "resources"] + ["resources:copper"])
def test_every_culture_field_draws(cultured, field):
    ax = plot_map(cultured, field, "mollweide", width=240)
    assert ax.get_title().endswith(field)
    matplotlib.pyplot.close(ax.figure)


def test_fields_explain_what_they_need(cultured, world_cache):
    with pytest.raises(ValueError, match="resources:<name>"):
        plot_map(cultured, "resources", width=120)
    bare = cultured.__class__(**{**cultured.__dict__, "culture": None})
    with pytest.raises(ValueError, match="no culture"):
        plot_map(bare, "political", width=120)
    assert split_field("resources:gold") == ("resources", "gold") and split_field("roads") == ("roads", None)


def test_roads_draw_in_every_projection(cultured):
    from worldgen.render import projection
    for name in ("mollweide", "orthographic"):
        collections = road_collections(cultured, projection(name))
        assert collections and sum(len(c.get_segments()) for c in collections) > 10


def test_culture_overview_and_report(cultured):
    fig = plot_culture(cultured)
    assert len(fig.axes) >= 4
    matplotlib.pyplot.close(fig)
    text = format_culture(cultured.culture, cultured)
    assert "Testers (native)" in text and "(colonist)" in text
    assert "largest settlements" in text and "space elevator" in text and "polities" in text


def test_culture_command_places_saves_and_maps(cultured, tmp_path):
    world_dir = tmp_path / "w"
    save_world(cultured.__class__(**{**cultured.__dict__, "culture": None}), world_dir)
    spec_path = tmp_path / "culture.yaml"
    save_culture_spec(SPEC, spec_path)
    out, png = tmp_path / "with_culture", tmp_path / "political.png"
    runner = CliRunner()
    result = runner.invoke(app, ["culture", str(world_dir), str(spec_path), "--save", str(out),
                                 "--map", str(png), "--field", "political"])
    assert result.exit_code == 0, result.output
    assert "Testers" in result.output and png.exists() and (out / "culture").exists()
    report = runner.invoke(app, ["report", str(out)])
    assert report.exit_code == 0 and "Culture" in report.output
