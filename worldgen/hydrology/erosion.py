"""River erosion, hillslope creep and sediment deposition over geological time.

Rivers cut down at a rate set by the stream-power law, E = K Q^m S, solved
implicitly along the flow network (Braun & Willett 2013). Slopes also creep
(diffusion). Eroded material travels downstream and fills closed
depressions, then builds shelves and deltas offshore up to a little below
sea level; what remains continues down the sea floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .. import heuristics as h
from ..grid import SphereGrid
from .graph import accumulate, priority_flood, steepest_receivers, upstream_order


@dataclass
class ErosionResult:
    """Change of the surface over an erosion run."""

    height_m: np.ndarray        # new elevation above sea level
    eroded_m: np.ndarray        # total lowering by erosion (positive)
    deposited_m: np.ndarray     # total sediment thickness added
    exported_m3: float = 0.0    # sediment carried out across an open edge of the domain


@njit(cache=True)
def _stream_power(order, receiver, distance, power, z, floor):
    """Lower each cell toward its receiver; ``power`` is K·dt·Q^m per cell (m).

    Cells never drop below their receiver's new height or below ``floor``.
    """
    for k in range(order.size):
        i = order[k]
        r = receiver[i]
        if r == i or power[i] <= 0.0 or z[i] <= z[r]:
            continue
        f = power[i] / distance[i]
        new = (z[i] + f * z[r]) / (1.0 + f)
        z[i] = max(new, z[r], floor[i])
    return z


@njit(cache=True)
def _deposit(order, receiver, supply, capacity):
    """Carry sediment downstream, filling each cell's capacity.

    Returns the volume left in each cell and, per cell, the volume that
    reached it as a pit or outlet with no room left.
    """
    flux = supply.copy()
    left = np.zeros(supply.size)
    spare = np.zeros(supply.size)
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        take = min(flux[i], capacity[i])
        left[i] = take
        flux[i] -= take
        r = receiver[i]
        if r != i:
            flux[r] += flux[i]
        else:
            spare[i] += flux[i]
        flux[i] = 0.0
    return left, spare


@njit(cache=True)
def _creep(z, indptr, indices, lengths, rate, steps, fixed):
    """Diffuse elevations along the grid graph; ``rate`` is κ·dt per step (rad²)."""
    n = z.size
    for _ in range(steps):
        change = np.zeros(n)
        for i in range(n):
            if fixed[i]:
                continue
            total = 0.0
            for k in range(indptr[i], indptr[i + 1]):
                total += (z[indices[k]] - z[i]) / (lengths[k] * lengths[k])
            change[i] = rate * total / (indptr[i + 1] - indptr[i]) * 4.0
        z += change
    return z


def _flow(grid: SphereGrid, z: np.ndarray, outlet: np.ndarray, sea: np.ndarray):
    """Return receivers and step lengths (rad) over land and sea floor, and the upstream order.

    Water leaves the land at ``outlet`` cells (the sea, and any open edge of the domain).
    """
    adj = grid.neighbours
    filled = priority_flood(z, adj.indptr, adj.indices, outlet, 1e-3)
    land_rec, land_dist = steepest_receivers(filled, adj.indptr, adj.indices, adj.data, outlet)
    # On the sea floor, sediment only moves between sea cells.
    sea_rec, sea_dist = steepest_receivers(np.where(sea, z, np.inf), adj.indptr, adj.indices, adj.data, ~sea)
    receiver = np.where(sea, sea_rec, land_rec)
    distance = np.where(sea, sea_dist, land_dist)
    return receiver, distance, upstream_order(receiver), filled


def erode(grid: SphereGrid, radius_m: float, height_m: np.ndarray, ocean: np.ndarray, runoff_m: np.ndarray,
          duration_myr: float, relief: float, step_myr: float = h.EROSION_STEP_MYR,
          cell_area_m2: float | None = None, creep_m2_per_myr: float = h.CREEP_M2_PER_MYR,
          edge: np.ndarray | None = None) -> ErosionResult:
    """Return the surface after ``duration_myr`` of river erosion, creep and deposition.

    ``height_m`` is elevation above sea level and ``runoff_m`` the yearly
    runoff depth on land. The ocean mask stays fixed during the run.
    ``cell_area_m2`` overrides the per-cell area and ``creep_m2_per_myr`` the
    hillslope diffusivity (local zoom grids are not the whole sphere, and at
    their resolution creep takes its physical value rather than the global
    grid's effective one). ``edge`` marks an open boundary of a local grid:
    water and sediment reaching it leave the domain.
    """
    steps = max(int(np.ceil(duration_myr / step_myr)), 1)
    dt = duration_myr / steps
    area = 4 * np.pi * radius_m**2 / grid.size if cell_area_m2 is None else cell_area_m2
    adj = grid.neighbours
    z = height_m.astype(float).copy()
    edge = np.zeros(grid.size, dtype=bool) if edge is None else edge
    outlet = ocean | edge
    sea = ocean & ~edge
    land = ~outlet
    exported = 0.0
    shelf_top = -h.SHELF_DEPTH_M * relief
    eroded = np.zeros(grid.size)
    deposited = np.zeros(grid.size)
    creep_rad2 = creep_m2_per_myr * dt / radius_m**2
    creep_steps = max(int(np.ceil(creep_rad2 * 8.0 / np.min(adj.data) ** 2)), 1)

    for _ in range(steps):
        receiver, distance, order, filled = _flow(grid, z, outlet, sea)
        discharge = accumulate(order, receiver, np.where(land, runoff_m, 0.0) * area)
        power = np.where(land & (filled - z < 1.0),
                         h.STREAM_POWER_K * dt * discharge ** h.STREAM_POWER_M, 0.0)
        before = z.copy()
        z = _stream_power(order, receiver, distance * radius_m, power, z, np.where(land, 0.0, -np.inf))
        cut = before - z
        eroded += cut
        # Creep moves material between neighbours without removing it.
        z = _creep(z, adj.indptr, adj.indices, adj.data, creep_rad2 / creep_steps, creep_steps, outlet)

        # River sediment fills depressions on land and builds shelves offshore.
        hollow = np.where(power > 0.0, 0.0, filled - before)
        capacity = np.where(land, hollow, np.where(sea, shelf_top - z, 0.0)).clip(0.0) * area
        left, stuck = _deposit(order, receiver, cut * area, capacity)
        # Sediment reaching an open edge leaves. On the whole planet, sediment that finds no room settles
        # thinly over the sea floor; on a local grid its sea belongs to the ocean beyond, so it leaves too.
        exported += float(stuck[edge].sum())
        spare = float(stuck[~edge].sum())
        if edge.any():
            exported += spare
        elif sea.any():
            left[sea] += spare / sea.sum()
        z += left / area
        deposited += left / area

    return ErosionResult(height_m=z, eroded_m=eroded, deposited_m=deposited, exported_m3=exported)
