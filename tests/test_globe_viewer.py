"""The 3D globe: building, recolouring by field and epoch, draped regions, images, HTML, widget."""

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from worldgen.render import GlobeViewer, available_fields  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ORGANIC = {"history": {"start_years_ago": 2000, "epochs": [1000]},
           "cultures": [{"name": "Oru", "mode": "organic",
                         "homeland": {"lat_deg": 45, "lon_deg": 10, "radius_km": 3000}}]}


@pytest.fixture(scope="module")
def world(world_cache):
    from worldgen.culture import CultureSpec, apply_culture
    from worldgen.spec import load_spec

    spec = load_spec(EXAMPLES / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    return apply_culture(world_cache(spec), CultureSpec.model_validate(ORGANIC))


@pytest.fixture(scope="module")
def globe(world):
    return GlobeViewer(world, level=0)


def test_globe_offers_the_worlds_fields_and_epochs(globe, world):
    assert globe.fields == available_fields(world)
    assert "political" in globe.fields and "energy" in globe.fields
    assert globe.dates == [1000.0, 0.0]
    assert globe.terrain.n_points == len(globe.heights)
    assert np.isclose(np.abs(globe.heights).max(), np.abs(world.surface["elevation"].values).max(), rtol=0.2)


def test_recolouring_follows_field_and_epoch(globe):
    elevation = globe.colours("elevation")
    biomes = globe.colours("biomes")
    assert elevation.shape == biomes.shape == (globe.terrain.n_points, 4)
    assert not np.array_equal(elevation, biomes)
    globe.show_field("political", 900)               # snaps to the kept epoch 1000 years ago
    assert globe.epoch == 1000.0 and globe.field == "political"
    then = np.asarray(globe.terrain["rgba"]).copy()
    globe.show_field("political", 0)
    assert globe.epoch == 0.0 and then.shape == np.asarray(globe.terrain["rgba"]).shape
    with pytest.raises(ValueError, match="not available"):
        globe.show_field("flavour")


def test_regions_drape_above_the_globe(world, globe):
    from worldgen.zoom import zoom_region

    region = zoom_region(world, 45.0, 10.0, resolution_m=3000, half_width_km=40)
    poly = globe.add_region(region)
    r = np.linalg.norm(np.asarray(poly.points), axis=1)
    lifted = (region.elevation + (region.ice_thickness_m if region.ice_thickness_m is not None else 0.0))
    expected = globe.radius_m + globe.exaggeration * (lifted + 5.0)
    assert np.abs(np.sort(r) - np.sort(expected)).max() < 1.0
    globe.regions.clear()


def test_images_and_html_pages_are_written(globe, tmp_path):
    path = globe.screenshot(tmp_path / "g.png", "biomes", lat_deg=45, lon_deg=10, window_size=(300, 300))
    from PIL import Image

    image = np.asarray(Image.open(path))
    assert image.shape[:2] == (300, 300) and image.std() > 5
    pytest.importorskip("trame_vtk")
    pages = globe.export_views(tmp_path / "views", ("elevation",), epochs=[0.0])
    assert pages[0].exists() and pages[0].stat().st_size > 10_000


def test_widget_has_field_and_epoch_controls(globe):
    widgets = pytest.importorskip("ipywidgets")
    try:
        box = globe.widget(backend="static")
    except Exception as err:          # a backend PyVista cannot start here
        pytest.skip(f"no Jupyter backend: {err}")
    controls = box.children[0].children
    assert isinstance(controls[0], widgets.Dropdown) and isinstance(controls[1], widgets.SelectionSlider)
    controls[0].value = "biomes"
    assert globe.field == "biomes"
