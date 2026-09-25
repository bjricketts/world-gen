"""The stylised 2D local map renderer."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.collections import LineCollection

from worldgen import PlanetSpec
from worldgen.render.local import _river_lines, plot_local_map, save_local_map
from worldgen.zoom import erode_region, region_for, region_grid, synthesize_region


def _ice_free_peak(world):
    """Return the highest land cell with no ice and well below the snowline: mountains that rivers shape."""
    s = world.surface
    elevation = s["elevation"].values
    clear = (~s["ocean"].values) & (s["ice_thickness"].values == 0.0) & (elevation < s["snowline"].values - 1000.0)
    cells = np.flatnonzero(clear)
    return int(cells[np.argmax(elevation[cells])])


@pytest.fixture(scope="module")
def eroded(world_cache):
    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"}, surface={"tectonics": "heuristic"}))
    radius = float(world.surface.attrs["radius_m"])
    peak = _ice_free_peak(world)
    region = region_grid(*region_for(world.grid.points[peak], 7, radius, tiles_each_side=1), nodes_per_tile=24)
    return world, erode_region(world, synthesize_region(world, region))


def test_relief_map_draws_layers(eroded):
    world, region = eroded
    fig = plot_local_map(world, region, mode="relief")
    ax = fig.axes[0]
    assert ax.images                                           # the relief raster
    assert any(isinstance(c, LineCollection) for c in ax.collections)   # rivers
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_biome_map_renders(eroded):
    world, region = eroded
    fig = plot_local_map(world, region, mode="biome", graticule=False)
    assert fig.axes[0].images
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_stream_thresholds_thin_the_network(eroded):
    world, region = eroded
    dense, _, _ = _river_lines(region, 1, 0.0)
    fewer_order, _, _ = _river_lines(region, 3, 0.0)
    fewer_area, _, _ = _river_lines(region, 1, 100.0)
    fewer_flow, _, _ = _river_lines(region, 1, np.inf, 1.0)
    assert len(fewer_flow) < len(dense)                        # a discharge threshold draws fewer streams
    assert len(fewer_order) < len(dense)                       # a higher order keeps only trunks
    assert len(fewer_area) < len(dense)                        # a larger catchment threshold draws fewer streams


def test_save_local_map_writes_a_file(eroded, tmp_path):
    world, region = eroded
    path = save_local_map(world, region, tmp_path / "local.png")
    assert path.exists() and path.stat().st_size > 0


def test_shading_is_neutral_on_flat_ground_and_fixed_in_scale():
    from worldgen.render.local import _shade

    flat = _shade(np.full((20, 20), 300.0), 1000.0)
    assert np.allclose(flat, 1.0)                              # flat ground keeps its tint
    ramp = _shade(np.tile(np.arange(20.0) * 50.0, (20, 1)), 1000.0)
    spiky = np.tile(np.arange(20.0) * 50.0, (20, 1))
    spiky[0, 0] += 5000.0
    assert np.allclose(_shade(spiky, 1000.0)[10:, 10:], ramp[10:, 10:])   # an outlier does not rescale the rest


def test_land_colours_follow_the_world_scale(eroded):
    """A node's colour depends on its height, not on the highest point in the region."""
    from dataclasses import replace

    from worldgen.render.local import _base_rgb

    world, region = eroded
    raised = region.elevation.copy()
    top = int(np.argmax(raised))
    raised[top] += 1500.0
    before, _, _ = _base_rgb(world, region, biome=False)
    after, _, _ = _base_rgb(world, replace(region, elevation=raised), biome=False)
    keep = np.ones(region.grid.size, bool)
    keep[top] = False
    assert np.allclose(before.reshape(-1, 3)[keep], after.reshape(-1, 3)[keep])


def test_ice_is_drawn_white(eroded):
    """Ground under ice is drawn close to the ice colour, whatever its height."""
    from dataclasses import replace

    from worldgen.render.local import ICE_COLOUR
    from matplotlib.colors import to_rgb

    world, region = eroded
    ice = np.zeros(region.grid.size)
    ice[: region.grid.size // 2] = 300.0
    fig = plot_local_map(world, replace(region, ice_thickness_m=ice), hillshade=False, rivers=False, graticule=False)
    image = fig.axes[0].images[0].get_array().reshape(-1, 3)
    import matplotlib.pyplot as plt
    plt.close(fig)
    covered = (ice > 0.0) & ~region.ocean & ~region.lake
    assert np.abs(image[covered] - np.array(to_rgb(ICE_COLOUR))).max() < 0.3


def test_drawn_detail_follows_the_map_scale():
    """Zooming in by two lowers the drawn catchment by four, and the discharge threshold with it."""
    from worldgen.render.local import drawing_thresholds

    coarse_area, coarse_q = drawing_thresholds(2.0)
    fine_area, fine_q = drawing_thresholds(1.0)
    assert coarse_area == pytest.approx(4.0 * fine_area)
    assert coarse_q == pytest.approx(4.0 * fine_q)


def test_wider_lines_carry_more_water(eroded):
    world, region = eroded
    segments, widths, wets = _river_lines(region, 1, 0.0)
    assert min(widths) < max(widths)
