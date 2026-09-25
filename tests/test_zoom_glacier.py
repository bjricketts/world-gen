"""Glaciers, ice sheets and glacial erosion on zoomed regions."""

import numpy as np
import pytest

from worldgen import PlanetSpec, heuristics as h
from worldgen.climate.ice import ICE_DENSITY
from worldgen.constants import G_EARTH
from worldgen.hydrology.graph import priority_flood, spread_positive
from worldgen.zoom import erode_region, region_for, region_grid, synthesize_region
from worldgen.zoom.glacier import _plastic_surface, glacial_cooling_k, glaciate, ice_cover, mass_balance

RADIUS = 6.371e6


def _mountain(peak_m=3500.0, base_m=500.0):
    """Return a single-tile region with a cone-shaped mountain in its middle."""
    region = region_grid(4, 6, 20, 20, 20, 20, nodes_per_tile=48)
    rows, cols = region.shape
    r, c = np.divmod(np.arange(region.size), cols)
    distance = np.hypot(r - rows / 2, c - cols / 2) / (rows / 2)
    bed = base_m + (peak_m - base_m) * np.clip(1.0 - distance, 0.0, 1.0)
    return region, bed


def test_mass_balance_gains_above_the_snowline_and_melts_below():
    snowline = np.full(4, 2000.0)
    rain = np.full(4, 1.0)
    height = np.array([1000.0, 2000.0, 2000.0 + h.ZOOM_GLACIER_FULL_ACCUMULATION_M, 4000.0])
    balance = mass_balance(height, snowline, rain)
    assert balance[0] < 0.0 and balance[1] == 0.0
    assert balance[2] == pytest.approx(1.0) and balance[3] == pytest.approx(1.0)   # all precipitation kept
    # Melt depends on how far below the snowline the ice lies, not on the snowfall.
    assert mass_balance(height, snowline, 0.1 * rain)[0] == pytest.approx(balance[0])


def test_spread_flux_reaches_the_outlets_and_never_goes_negative():
    region, bed = _mountain()
    adj = region.neighbours
    edge = region.boundary()
    surface = priority_flood(bed, adj.indptr, adj.indices, edge, 1e-3)      # every node drains to the edge
    order = np.argsort(-surface, kind="stable")
    gain = np.where(edge, 0.0, 1.0)
    flux = spread_positive(order, adj.indptr, adj.indices, adj.data, surface, gain, 1.0, edge)
    assert (flux >= 0.0).all()
    assert flux[edge].sum() == pytest.approx(gain.sum(), rel=1e-9)            # every gain leaves at an outlet
    lossy = spread_positive(order, adj.indptr, adj.indices, adj.data, surface, gain - 2.0, 1.0, edge)
    assert (lossy >= 0.0).all()


def test_plastic_surface_follows_the_nye_profile_on_a_flat_bed():
    """Ice on a flat bed thickens as √(2 τ₀ L / ρg) with distance from its margin (Nye 1952)."""
    region = region_grid(4, 6, 20, 20, 20, 20, nodes_per_tile=48)
    rows, cols = region.shape
    _, c = np.divmod(np.arange(region.size), cols)
    ice = c > 0                                                  # bare along one side only
    adj = region.neighbours
    scale = h.ICE_YIELD_STRESS_PA / (ICE_DENSITY * G_EARTH)
    surface = _plastic_surface(adj.indptr, adj.indices, adj.data * RADIUS, np.zeros(region.size), ice,
                               np.full(region.size, np.inf), scale, 0.0)
    row = rows // 2
    line = region.points.reshape(rows, cols, 3)[row]
    length = np.linalg.norm(np.diff(line, axis=0), axis=1).sum() * RADIUS   # along the row from the margin
    far = row * cols + cols - 1
    assert surface[far] == pytest.approx(np.sqrt(2.0 * scale * length), rel=0.02)
    assert np.all(np.diff(surface[row * cols:(row + 1) * cols]) >= 0.0)      # rising away from the margin


def test_ice_covers_the_summit_and_not_the_lowlands():
    region, bed = _mountain()
    snowline = np.full(region.size, 2500.0)
    ice = ice_cover(region, RADIUS, bed, snowline, np.full(region.size, 1.0), np.zeros(region.size, bool),
                    region.boundary(), region.cell_area_m2(RADIUS))
    high = bed > 2500.0 + h.ZOOM_GLACIER_FULL_ACCUMULATION_M
    assert ice.covered[high].mean() > 0.5                        # most ground well above the snowline is under ice
    assert not ice.covered[bed < 1000.0].any()                   # far below the snowline the ice has melted
    assert (ice.thickness_m >= 0.0).all()
    dry = ice_cover(region, RADIUS, bed, snowline, None, np.zeros(region.size, bool), region.boundary(),
                    region.cell_area_m2(RADIUS))
    assert not dry.covered.any()                                 # no snowfall and no inherited sheet: no ice


def test_glaciers_only_lower_the_bed():
    region, bed = _mountain()
    carved, depth = glaciate(region, RADIUS, bed, np.full(region.size, 2500.0), np.full(region.size, 1.0),
                             np.zeros(region.size, bool), region.boundary(), region.cell_area_m2(RADIUS))
    assert (depth >= -1e-9).all() and depth.max() > 1.0
    assert np.allclose(carved, bed - depth)
    assert (depth[region.boundary()] == 0.0).all()               # the pinned edge is not cut


def test_glacial_cooling_reads_the_history_record():
    class _Summary:
        features = {"relict_glacial_cooling_k": 6.0}

    class _State:
        surface = _Summary()

    class _World:
        state = _State()

    assert glacial_cooling_k(_World()) == 6.0
    _State.surface = None
    assert glacial_cooling_k(_World()) == 0.0


@pytest.fixture(scope="module")
def icy(world_cache):
    """A temperate world zoomed on its highest ground, which carries ice."""
    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"}, surface={"tectonics": "heuristic"}))
    radius = float(world.surface.attrs["radius_m"])
    peak = int(np.argmax(world.surface["elevation"].values))
    region = region_grid(*region_for(world.grid.points[peak], 7, radius, tiles_each_side=1), nodes_per_tile=20)
    return world, synthesize_region(world, region)


def test_no_streams_or_lakes_are_mapped_under_ice(icy):
    world, detailed = icy
    out = erode_region(world, detailed)
    covered = out.ice_thickness_m > 0.0
    assert covered.any()
    assert not (covered & (out.river_order > 0)).any()
    assert not (covered & out.lake).any()


def test_a_colder_past_cuts_deeper(world_cache, monkeypatch):
    """A remembered glacial epoch lowers the snowline, so glaciers cover more ground and erode more."""
    import worldgen.zoom.erode as zoom_erode

    world = world_cache(PlanetSpec(seed=0, priors={"archetype": "temperate"}, surface={"tectonics": "heuristic"}))
    surface = world.surface
    gap = surface["snowline"].values - surface["elevation"].values
    near = np.flatnonzero((gap > 200.0) & (gap < 800.0) & ~surface["ocean"].values)
    cell = int(near[np.argmax(surface["precipitation"].values[near])])      # just below the snowline
    radius = float(surface.attrs["radius_m"])
    region = region_grid(*region_for(world.grid.points[cell], 7, radius, tiles_each_side=1), nodes_per_tile=20)
    detailed = synthesize_region(world, region)
    now = erode_region(world, detailed)
    monkeypatch.setattr(zoom_erode, "glacial_cooling_k", lambda _: 8.0)
    past = erode_region(world, detailed)
    assert past.glacial_m.sum() > now.glacial_m.sum()
