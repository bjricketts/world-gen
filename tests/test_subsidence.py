"""Tectonic subsidence of continental crust and the sediment that fills it."""

import numpy as np
import pytest

from worldgen import heuristics as h
from worldgen.grid import build_grid
from worldgen.hydrology.erosion import erode
from worldgen.tectonics.crust import Crust
from worldgen.tectonics.subsidence import (E0_M, cool_margins, flex_forelands, initial_subsidence, stretch_ratio,
                                           thermal_subsidence, unfilled_subsidence)


@pytest.mark.parametrize("beta, table_m", [(1.25, 740), (1.5, 1309), (2.0, 2015), (4.0, 2850), (10.0, 3114)])
def test_full_thermal_subsidence_matches_mckenzie(beta, table_m):
    # McKenzie (1978), Table 2: e(0) of the approximate expression, E0 r.
    assert E0_M * stretch_ratio(np.array(beta)) == pytest.approx(table_m, rel=0.02)


def test_thermal_subsidence_approaches_its_limit_on_the_thermal_time():
    beta = np.array([2.0])
    full = E0_M * stretch_ratio(beta)
    assert thermal_subsidence(beta, np.array([62.8])) == pytest.approx(full * (1 - np.exp(-1)))
    assert thermal_subsidence(beta, np.array([1000.0])) == pytest.approx(full, rel=1e-6)


def test_initial_subsidence_changes_sign_near_18_km_of_crust():
    # McKenzie (1978): S_i is positive if the crust is thicker than ~18 km.
    beta = np.array([2.0])
    assert initial_subsidence(beta, 20e3) > 0 > initial_subsidence(beta, 16e3)
    assert initial_subsidence(np.array([1.0]), 35e3) == 0.0


def _crust_on(grid):
    crust = Crust.empty(grid.points)
    crust.continental[:] = True
    crust.cont_elev[:] = h.CONTINENT_BASE_M
    return crust


def test_foreland_bends_down_beside_a_belt_and_rebounds_when_it_erodes():
    grid = build_grid(10_000)
    crust = _crust_on(grid)
    belt = grid.points[:, 2] > 0.95
    crust.cont_elev[belt] += 3000.0
    before = crust.cont_elev.copy()
    flex_forelands(crust, grid, 6371.0)
    dip = before - crust.cont_elev
    ring = (grid.points[:, 2] < 0.95) & (grid.points[:, 2] > 0.93)
    far = grid.points[:, 2] < 0.5
    assert dip[ring].max() > 500 and dip.min() >= 0 and np.all(dip[far] == 0) and np.all(dip[belt] == 0)
    crust.cont_elev[belt] = h.CONTINENT_BASE_M        # the belt is worn away
    flex_forelands(crust, grid, 6371.0)
    np.testing.assert_allclose(crust.cont_elev[~belt], before[~belt])


def test_margins_sink_as_they_cool_and_report_their_unfilled_space():
    grid = build_grid(10_000)
    crust = _crust_on(grid)
    crust.stretch[:100] = 1.0
    cool_margins(crust, 62.8)
    sunk = h.CONTINENT_BASE_M - crust.cont_elev[:100]
    np.testing.assert_allclose(sunk, thermal_subsidence(np.full(100, 2.0), np.full(100, 62.8)))
    crust.basin_fill[:100] = 500.0
    room = unfilled_subsidence(crust)
    np.testing.assert_allclose(room[:100], initial_subsidence(np.full(100, 2.0)) + sunk - 500.0)
    assert np.all(room[100:] == 0)


def test_river_sediment_fills_subsided_ground_and_is_conserved():
    grid = build_grid(10_000)
    z = 200.0 + 4000.0 * np.clip(grid.points[:, 2] - 0.6, 0, None)     # a dome sloping to a low plain
    ocean = grid.points[:, 2] < -0.3
    z[ocean] = -3000.0
    room = np.where((grid.points[:, 2] > -0.2) & (grid.points[:, 2] < 0.4), 300.0, 0.0)
    runoff = np.full(grid.size, 0.3)
    without = erode(grid, 6.371e6, z, ocean, runoff, 10.0, 1.0)
    with_room = erode(grid, 6.371e6, z, ocean, runoff, 10.0, 1.0, accommodation_m=room)
    kept = with_room.deposited_m[room > 0].sum()
    assert kept > 2 * without.deposited_m[room > 0].sum()
    assert np.all(with_room.deposited_m[room > 0] <= room[room > 0] + 1e-6)
    # Mass: what was eroded is deposited somewhere (whole planet, no open edge).
    assert with_room.deposited_m.sum() == pytest.approx(with_room.eroded_m.sum() - (without.eroded_m.sum() -
                                                        without.deposited_m.sum()) * 0 - 0, rel=0.02)
