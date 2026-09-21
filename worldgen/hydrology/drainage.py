"""Where water goes on land: flow directions, drainage basins, lakes and rivers.

Closed depressions are filled with the priority-flood algorithm (Barnes et
al. 2014) so that every land cell drains to the ocean through a flow path.
A depression holds a lake whose size balances inflow against evaporation;
a lake that fills to its spill point overflows and the river continues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .. import constants as c
from .. import heuristics as h
from ..grid import SphereGrid
from .graph import accumulate, basin_roots, priority_flood, steepest_receivers, upstream_order
from .rainfall import Rainfall

FILL_EPSILON_M = 1e-3       # slope added across filled flats so water keeps moving
LAKE_MIN_DEPTH_M = 1.0      # depressions shallower than this are treated as flats


@dataclass
class Drainage:
    """Flow network of a surface."""

    receiver: np.ndarray        # cell each cell drains to (itself at the ocean and in closed lakes)
    distance_m: np.ndarray      # length of the flow step to the receiver
    order: np.ndarray           # cells ordered from outlets upstream
    discharge_m3_s: np.ndarray  # mean flow leaving each cell
    basin: np.ndarray           # drainage basin id (0 = ocean), numbered by size
    lake: np.ndarray            # covered by a lake
    lake_level_m: np.ndarray    # lake surface elevation, NaN outside lakes
    endorheic: np.ndarray       # land draining to a lake without an outlet
    river_order: np.ndarray     # Strahler order of river cells, 0 elsewhere
    filled_m: np.ndarray        # elevation with depressions filled to their spill level


@dataclass
class _Lakes:
    """Closed depressions of a filled surface."""

    label: np.ndarray           # lake id per cell, −1 outside depressions
    exit: np.ndarray            # per lake: the cell where water leaves it
    cells: list[np.ndarray]     # per lake: its cells, lowest first


def _find_depressions(grid: SphereGrid, z: np.ndarray, filled: np.ndarray, receiver: np.ndarray,
                      land: np.ndarray) -> _Lakes:
    """Return the closed depressions and make each drain through a single exit cell."""
    deep = land & (filled - z > LAKE_MIN_DEPTH_M)
    label = np.full(grid.size, -1, dtype=np.int64)
    if not deep.any():
        return _Lakes(label, np.empty(0, dtype=np.int64), [])
    i, j = grid.edges()
    both = deep[i] & deep[j]
    graph = coo_matrix((np.ones(both.sum()), (i[both], j[both])), shape=(grid.size, grid.size))
    _, comp = connected_components(graph, directed=False)
    ids = comp[deep]
    _, compact = np.unique(ids, return_inverse=True)
    label[deep] = compact
    count = compact.max() + 1

    exits = np.full(count, -1, dtype=np.int64)
    outside = deep & (label[receiver] != label)
    for cell in np.flatnonzero(outside):
        lake = label[cell]
        if exits[lake] < 0 or filled[cell] < filled[exits[lake]]:
            exits[lake] = cell
    if (exits < 0).any():
        # Depressions without a way out cannot occur on a filled surface; drop them if they do.
        label[np.isin(label, np.flatnonzero(exits < 0))] = -1
        return _find_depressions(grid, z, np.where(label >= 0, filled, np.minimum(filled, z)), receiver, land)
    # Other cells that leave the lake are sent to the exit, so all its water passes one place.
    redirect = outside & (exits[label.clip(0)] != np.arange(grid.size))
    receiver[redirect] = exits[label[redirect]]

    members = np.argsort(label, kind="stable")[np.count_nonzero(label < 0):]
    starts = np.searchsorted(label[members], np.arange(count + 1))
    cells = []
    for k in range(count):
        group = members[starts[k]:starts[k + 1]]
        cells.append(group[np.argsort(z[group], kind="stable")])
    return _Lakes(label, exits, cells)


@njit(cache=True)
def _route(order, receiver, inflow, exit_lake, lake_cells, lake_start, lake_loss, z):
    """Accumulate flow downstream; lakes lose water to evaporation and pass on only their overflow.

    Returns the outflow of each cell (m³/yr) and each lake's fill level
    (index into its lowest-first cell list, −1 for a full lake).
    """
    flow = inflow.copy()
    n_lakes = lake_start.size - 1
    filled_to = np.full(n_lakes, -1, dtype=np.int64)
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        lake = exit_lake[i]
        if lake >= 0:
            capacity = 0.0
            for m in range(lake_start[lake], lake_start[lake + 1]):
                capacity += lake_loss[lake_cells[m]]
            if flow[i] <= capacity:
                used = 0.0
                last = lake_start[lake]
                for m in range(lake_start[lake], lake_start[lake + 1]):
                    used += lake_loss[lake_cells[m]]
                    last = m
                    if used >= flow[i]:
                        break
                filled_to[lake] = last - lake_start[lake]
                flow[i] = 0.0
            else:
                flow[i] -= capacity
        r = receiver[i]
        if r != i:
            flow[r] += flow[i]
    return flow, filled_to


@njit(cache=True)
def _strahler(order, receiver, river):
    """Return the Strahler order of river cells."""
    n = receiver.size
    strahler = np.zeros(n, dtype=np.int8)
    top = np.zeros(n, dtype=np.int8)
    count = np.zeros(n, dtype=np.int8)
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        if not river[i]:
            continue
        if count[i] == 0:
            strahler[i] = 1
        elif count[i] >= 2:
            strahler[i] = top[i] + 1
        else:
            strahler[i] = top[i]
        r = receiver[i]
        if r != i and river[r]:
            if strahler[i] > top[r]:
                top[r] = strahler[i]
                count[r] = 1
            elif strahler[i] == top[r]:
                count[r] += 1
    return strahler


def build_drainage(grid: SphereGrid, radius_m: float, height_m: np.ndarray, ocean: np.ndarray,
                   rain: Rainfall, cell_area_m2: float | None = None) -> Drainage:
    """Return the drainage network of a surface with an ocean.

    ``height_m`` is elevation above sea level. ``cell_area_m2`` overrides the
    per-cell area (used by local zoom grids, which are not the whole sphere).
    """
    adj = grid.neighbours
    land = ~ocean
    z = height_m.astype(float)
    filled = priority_flood(z, adj.indptr, adj.indices, ocean, FILL_EPSILON_M)
    receiver, distance = steepest_receivers(filled, adj.indptr, adj.indices, adj.data, ocean)
    lakes = _find_depressions(grid, z, filled, receiver, land)
    order = upstream_order(receiver)

    area = 4 * np.pi * radius_m**2 / grid.size if cell_area_m2 is None else cell_area_m2
    in_lake = lakes.label >= 0
    # Lakes receive rain directly and lose water at the open-water evaporation rate.
    inflow = np.where(land & ~in_lake, rain.runoff_m, 0.0) * area
    lake_loss = np.where(in_lake, np.maximum(rain.evaporation_m - rain.precipitation_m, 0.0), 0.0) * area
    inflow += np.where(in_lake, np.maximum(rain.precipitation_m - rain.evaporation_m, 0.0), 0.0) * area

    exit_lake = np.full(grid.size, -1, dtype=np.int64)
    exit_lake[lakes.exit] = np.arange(lakes.exit.size)
    lake_start = np.concatenate([[0], np.cumsum([len(cl) for cl in lakes.cells])]).astype(np.int64)
    lake_cells = np.concatenate(lakes.cells) if lakes.cells else np.empty(0, dtype=np.int64)
    flow, filled_to = _route(order, receiver, inflow, exit_lake, lake_cells, lake_start, lake_loss, z)

    # Lake extents and levels; closed lakes stop the flow.
    lake = np.zeros(grid.size, dtype=bool)
    level = np.full(grid.size, np.nan)
    closed_exits = []
    for k, cells in enumerate(lakes.cells):
        if filled_to[k] < 0:
            wet, surface = cells, filled[lakes.exit[k]]
        else:
            wet = cells[:filled_to[k] + 1]
            surface = z[wet[-1]]
            closed_exits.append(lakes.exit[k])
        lake[wet] = True
        level[wet] = surface
    receiver = receiver.copy()
    receiver[np.asarray(closed_exits, dtype=np.int64)] = np.asarray(closed_exits, dtype=np.int64)
    order = upstream_order(receiver)

    roots = basin_roots(order, receiver)
    endorheic = land & ~ocean[roots]
    basin = _number_basins(roots, land)

    discharge = flow / c.SECONDS_PER_YEAR
    upstream_cells = accumulate(order, receiver, np.ones(grid.size))
    min_cells = max(h.RIVER_MIN_AREA_KM2 * 1e6 / area, h.RIVER_MIN_CELLS)
    river = land & ~lake & (discharge >= h.RIVER_MIN_DISCHARGE_M3_S) & (upstream_cells >= min_cells)
    strahler = _strahler(order, receiver, river)
    return Drainage(receiver=receiver, distance_m=distance * radius_m, order=order, discharge_m3_s=discharge,
                    basin=basin, lake=lake, lake_level_m=level, endorheic=endorheic, river_order=strahler,
                    filled_m=filled)


def _number_basins(roots: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Return basin ids for land cells, 1 for the largest basin; 0 for the ocean."""
    basin = np.zeros(roots.size, dtype=np.int32)
    ids, inverse, counts = np.unique(roots[land], return_inverse=True, return_counts=True)
    rank = np.empty(ids.size, dtype=np.int32)
    rank[np.argsort(-counts, kind="stable")] = np.arange(1, ids.size + 1)
    basin[land] = rank[inverse]
    return basin
