"""Culture on zoomed regions: villages, footprints, local roads and refined borders."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy import sparse
from typer.testing import CliRunner

from worldgen import save_world
from worldgen.cli import app
from worldgen.culture.build import apply_culture
from worldgen.culture.development import Development
from worldgen.culture.local import (_refine_borders, border_river_share, crosses, footprint_km2,
                                    truncated_power_sizes, zoom_culture)
from worldgen.culture.model import ROAD_KINDS, Culture
from worldgen.culture.spec import CultureSpec
from worldgen.render.local import plot_local_map
from worldgen.zoom import zoom_region

SPEC = CultureSpec(cultures=[{"name": "Testers", "development": {"energy": "fossil"}},
                             {"inhabitants": "colonist", "colony": {"founded_years_ago": 150}}])


@pytest.fixture(scope="module")
def zoomed(world_cache):
    """A cultured preview Earth and a region around a town next to a polity border."""
    from worldgen.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parent.parent / "examples" / "earth.yaml")
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": "heuristic"})})
    world = apply_culture(world_cache(spec), SPEC)
    civ = world.culture
    pol = civ.layers["polity"].values[0]
    adj = world.grid.neighbours.tocoo()
    edge = np.zeros(world.grid.size, dtype=bool)
    edge[adj.row[(pol[adj.row] != pol[adj.col]) & (pol[adj.row] >= 0) & (pol[adj.col] >= 0)]] = True
    st = civ.settlements
    candidates = np.flatnonzero(edge[st.cell] & (st.population > 2e4) & (st.population < 3e5) & (st.culture == 0))
    for k in candidates:
        region = zoom_region(world, float(st.lat_deg[k]), float(st.lon_deg[k]), half_width_km=40)
        local = zoom_culture(world, region)
        # the first town whose region holds a polity border, so the result does not depend on the generated world
        if not np.isnan(border_river_share(region, local)[0]):
            return world, region, local
    raise AssertionError("no candidate town has a polity border in its region")


def _culture(energy: int) -> Culture:
    return Culture(id=0, name="t", inhabitants="native", development=Development(energy, 2, 2))


def test_footprints_follow_ortman_and_modern_density():
    # Ortman et al. (2014): A(ha) = 0.265 N^0.652 below 5 000 people, 0.294 N^0.724 above.
    old = footprint_km2(np.array([1000.0, 20000.0]), _culture(2))
    assert old[0] == pytest.approx(0.265 * 1000**0.652 / 100)
    assert old[1] == pytest.approx(0.294 * 20000**0.724 / 100)
    # Angel et al. (2011): 60 people/ha at the median large city (209 468), +16% per doubling.
    modern = footprint_km2(np.array([209_468.0, 2 * 209_468.0]), _culture(4))
    assert 209_468 / modern[0] == pytest.approx(6000.0)
    assert (2 * 209_468 / modern[1]) / 6000.0 == pytest.approx(1.16, rel=1e-3)


def test_village_sizes_add_up_and_follow_the_power_law():
    rng = np.random.default_rng(3)
    sizes = truncated_power_sizes(2e6, 50.0, 5000.0, 1.0, rng)
    assert 2e6 - 25.0 <= sizes.sum() <= 2e6          # a remainder under half the smallest size is dropped
    assert sizes.min() >= 25.0 and sizes[:-1].max() <= 5000.0
    # P(S > s) for a pdf ∝ s^-2 on [50, 5000]: (50/s − 50/5000) / (1 − 50/5000)
    s = 500.0
    assert np.mean(sizes > s) == pytest.approx((50 / s - 0.01) / 0.99, abs=0.01)
    assert truncated_power_sizes(30.0, 50.0, 5000.0, 1.0, rng).tolist() == [30.0]


def test_diagonal_steps_between_river_nodes_cross_the_river():
    shape = (3, 3)
    river = np.zeros(9, dtype=bool)
    river[[2, 4, 6]] = True                            # a river running along the anti-diagonal
    row = np.array([1, 0, 0, 3])
    col = np.array([5, 4, 1, 1])
    # 1→5 slips diagonally between river nodes 2 and 4; 0→4 steps onto the river; 0→1 stays on one bank;
    # 3→1 passes corners 0 and 4, of which only one is river
    assert crosses(shape, row, col, river).tolist() == [True, True, False, False]


def test_borders_settle_on_a_river_inside_the_band():
    rows, cols = 40, 80
    idx = np.arange(rows * cols).reshape(rows, cols)
    src, dst = [], []
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        src.append(idx[r0:r1, c0:c1].ravel())
        dst.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel())
    i, j = np.concatenate(src), np.concatenate(dst)
    spacing_km = 1.0
    nb = sparse.csr_matrix((np.full(i.size, 1e-3), (i, j)), shape=(rows * cols,) * 2)
    region = SimpleNamespace(grid=SimpleNamespace(neighbours=nb, shape=(rows, cols)))
    label = np.where(np.arange(rows * cols) % cols < 40, 0, 1)
    river = np.arange(rows * cols) % cols == 30          # a river 10 km inside polity 0
    hours = np.where(river[j], 100.0, 1.0)
    out = _refine_borders(region, label, sparse.csr_matrix((hours, (i, j)), shape=nb.shape), spacing_km)
    col = np.arange(rows * cols) % cols
    assert np.all(out[col > 30] == 1) and np.all(out[col < 30] == 0)


def test_zoomed_culture_conserves_people_and_stands_on_land(zoomed):
    world, region, local = zoomed
    land = ~region.ocean & ~region.lake
    assert len(local) > 100
    assert land[local.node].all()
    civ = world.culture
    listed = local.listed >= 0
    smallest = civ.settlements.population[civ.settlements.culture == 0].min()
    villages = ~listed & (local.culture == 0)
    assert local.population[villages].max() <= smallest * 1.0001
    # people: listed settlements plus the dispersed population of the cells the region covers
    _, parent = world.grid.tree.query(region.grid.points)
    node_km2 = (float(np.median(region.grid.neighbours.data)) * region.radius_m / 1e3) ** 2
    counted = np.bincount(parent, minlength=world.grid.size)
    expected = np.sum(civ.layers["dispersed_density"].values.sum(axis=0) * counted) * node_km2
    assert local.population[~listed].sum() == pytest.approx(expected, rel=0.1)


def test_zoomed_culture_names_forms_and_footprints(zoomed):
    world, region, local = zoomed
    st = world.culture.settlements
    by_id = dict(zip(st.id.tolist(), st.name.tolist()))
    listed = np.flatnonzero(local.listed >= 0)
    assert all(local.name[k] == by_id[int(local.listed[k])] for k in listed)
    for cid in np.unique(local.culture):
        names = local.name[local.culture == cid]
        assert len(set(names)) == names.size and all(names)
    big = np.argmax(local.population)
    assert local.footprint[local.node[big]] == big
    assert local.radius_km[big] > local.radius_km[np.argmin(local.population)]


def test_local_roads_run_on_land_along_lattice_edges(zoomed):
    _, region, local = zoomed
    land = ~region.ocean & ~region.lake
    cols = region.grid.shape[1]
    kinds = {k for _, k, _ in local.roads}
    assert {ROAD_KINDS.index("track"), ROAD_KINDS.index("road")} & kinds
    assert ROAD_KINDS.index("sea_route") not in kinds
    for nodes, _, _ in local.roads[:2000]:
        assert land[nodes].all()
        r, c = np.divmod(nodes, cols)
        assert np.all(np.maximum(np.abs(np.diff(r)), np.abs(np.diff(c))) == 1)
    served = np.zeros(region.grid.size, dtype=bool)
    for nodes, _, _ in local.roads:
        served[nodes] = True
    assert served[local.node[local.culture == 0]].mean() > 0.9


def test_refined_borders_follow_rivers_and_skip_water(zoomed):
    _, region, local = zoomed
    assert np.all(local.polity_map[region.ocean | region.lake] == -1)
    border, everywhere = border_river_share(region, local)
    assert border > 2 * everywhere


def test_zoomed_culture_is_deterministic(zoomed):
    world, region, local = zoomed
    again = zoom_culture(world, region)
    np.testing.assert_array_equal(again.node, local.node)
    np.testing.assert_array_equal(again.population, local.population)
    np.testing.assert_array_equal(again.polity_map, local.polity_map)
    assert again.name.tolist() == local.name.tolist() and len(again.roads) == len(local.roads)


def test_local_map_draws_culture(zoomed):
    world, region, local = zoomed
    bare = plot_local_map(world, region)
    drawn = plot_local_map(world, region, culture=local)
    assert len(drawn.axes[0].collections) > len(bare.axes[0].collections)
    assert len(drawn.axes[0].texts) > len(bare.axes[0].texts)
    matplotlib.pyplot.close("all")


def test_zoom_command_draws_culture(zoomed, tmp_path):
    world, region, _ = zoomed
    save_world(world, tmp_path / "w")
    x, y, z = region.grid.points[region.grid.size // 2 + region.grid.shape[1] // 2]
    lat, lon = np.degrees(np.arcsin(z)), np.degrees(np.arctan2(y, x))
    args = ["zoom", str(tmp_path / "w"), str(tmp_path / "m.png"), "--lat", f"{lat}", "--lon", f"{lon}",
            "--half-width", "40"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "settlements" in result.output and (tmp_path / "m.png").exists()
    bare = CliRunner().invoke(app, args + ["--no-culture"])
    assert bare.exit_code == 0 and "settlements" not in bare.output
