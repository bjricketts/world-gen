"""The notebook history explorer: fields, drawing, cell lookup, cell values and the widget."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from worldgen.culture import CultureSpec, apply_culture
from worldgen.render import HistoryExplorer

SPEC = {"history": {"start_years_ago": 3000, "epochs": [1500]},
        "cultures": [{"name": "Oru", "mode": "organic", "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}}]}


@pytest.fixture(scope="module")
def explorer(world_cache):
    """An explorer on a preview Earth-spec world with a 3000-year organic history and one kept epoch."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return HistoryExplorer(apply_culture(world_cache(spec), CultureSpec.model_validate(SPEC)))


def test_the_explorer_offers_epochs_and_fields(explorer):
    assert explorer.dates == [3000.0, 1500.0, 0.0] or explorer.dates[-2:] == [1500.0, 0.0]
    for f in ("elevation", "biomes", "political", "suitability", "energy", "livestock", "events", "land:arable"):
        assert f in explorer.fields
    assert len(set(explorer.fields)) == len(explorer.fields)


def test_a_world_without_culture_is_refused(world_cache):
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    world = world_cache(spec)
    if world.culture is not None:
        pytest.skip("the cached world carries a culture")
    with pytest.raises(ValueError, match="organic history"):
        HistoryExplorer(world)


@pytest.mark.parametrize("field", ["political", "energy", "events", "land:arable"])
def test_each_epoch_draws_without_a_title(explorer, field):
    fig = plt.figure(figsize=(6, 3.5))
    for years in explorer.dates:
        ax = explorer.draw(fig, field, years)
        assert ax.get_title() == "" and len(fig.axes) >= 1
        assert any("years ago" in t.get_text() or t.get_text() == "present" for t in ax.texts)
    plt.close(fig)


def test_the_cursor_finds_the_cell_under_it(explorer):
    import cartopy.crs as ccrs

    grid = explorer.world.grid
    for cell in np.random.default_rng(0).integers(0, grid.lat.size, 20):
        x, y = explorer.proj.transform_point(float(grid.lon[cell]), float(grid.lat[cell]), ccrs.PlateCarree())
        assert explorer.cell_at(x, y) == cell
    assert explorer.cell_at(1e9, 1e9) is None


def test_a_cell_is_described_at_each_epoch(explorer):
    civ = explorer.world.culture
    lived = np.flatnonzero(civ.layers["population_density"].values.sum(axis=0) > 0)
    text = explorer.describe(int(lived[0]), 0.0)
    assert "people/km²" in text and "elevation" in text and "arable" in text
    ocean = int(np.flatnonzero(explorer.world.surface["ocean"].values)[0])
    assert explorer.describe(ocean, explorer.dates[0]).endswith("ocean")


def test_the_widget_redraws_and_reports_the_hovered_cell(explorer):
    pytest.importorskip("ipywidgets")
    import cartopy.crs as ccrs
    from types import SimpleNamespace

    box = explorer.widget()
    w = explorer._widgets
    assert len(box.children) == 4
    w["field"].value = "energy"
    w["epoch"].value = explorer.dates[0]
    assert explorer.markers[0].get_xdata()[0] == explorer.dates[0]
    ax = w["figure"].axes[0]
    x, y = explorer.proj.transform_point(10.0, 45.0, ccrs.PlateCarree())
    w["hover"](SimpleNamespace(inaxes=ax, xdata=x, ydata=y))
    cell = explorer.cell_at(x, y)
    assert w["info"].value == "<br>".join(explorer.describe(cell, explorer.dates[0]).splitlines())
    plt.close("all")
