"""Re-mapping crust onto the global grid, creating new sea floor, rifting and removing tiny plates."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .. import heuristics as h
from ..grid import SphereGrid
from ..noise import warp
from ..util import named_rng
from .crust import Crust, Plates
from .processes import StepContext, StepStats, count_feedback, plate_sizes


def remap(crust: Crust, plates: Plates, grid: SphereGrid, elapsed_myr: float, ctx: StepContext) -> Crust:
    """Return crust sampled at the grid cells.

    Each cell takes the properties of the nearest live crust point. Cells
    farther than the gap distance from any crust lie in newly opened sea
    floor: they join the plate on their nearer side and get an age that
    decreases toward the middle of the gap.
    """
    live = np.flatnonzero(crust.alive)
    dist, nearest = cKDTree(crust.points[live]).query(grid.points)
    source = live[nearest]
    new = crust.take(source)
    new.points = grid.points.copy()
    new.alive[:] = True

    gap = dist > h.GAP_FACTOR * ctx.edge_rad
    if gap.any():
        omega = plates.omega[crust.plate[source[gap]]]
        speed = np.linalg.norm(np.cross(omega, crust.points[source[gap]]), axis=1) * ctx.radius_m
        opened_m = (dist[gap] - h.GAP_FACTOR * ctx.edge_rad) * ctx.radius_m
        age = elapsed_myr - opened_m / np.maximum(speed, 1.0)
        new.continental[gap] = False
        new.ocean_age[gap] = np.clip(age, 0.0, elapsed_myr)
        new.ocean_extra[gap] = 0.0
        new.sediment[gap] = 0.0
        new.cont_elev[gap] = 0.0
        new.orogeny_age[gap] = np.nan
        new.orogeny_kind[gap] = 0
        new.volcanic[gap] = False
        new.front_kind[gap] = 0
        new.front_age[gap] = np.inf
        new.collision_age[gap] = np.inf
    return new


BALANCE_TOLERANCE = 0.002     # continental area may drift by this fraction of the surface before correction


def _neighbour_plate_counts(grid: SphereGrid, plate: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Return, for each cell, how many of its valid neighbours belong to each plate (n × k)."""
    onehot = np.zeros((grid.size, k))
    onehot[np.flatnonzero(valid), plate[valid]] = 1.0
    adj = grid.neighbours.copy()
    adj.data[:] = 1.0
    return np.asarray(adj @ onehot)


def tidy_plates(crust: Crust, plates: Plates, grid: SphereGrid) -> None:
    """Smooth plate outlines on the grid: isolated cells and small detached fragments join the plate around them."""
    k = len(plates.active)
    everywhere = np.ones(grid.size, dtype=bool)
    counts = _neighbour_plate_counts(grid, crust.plate, everywhere, k)
    own = counts[np.arange(grid.size), crust.plate]
    best = counts.argmax(axis=1)
    stray = (own <= 1) & (counts.max(axis=1) >= 3)
    crust.plate[stray] = best[stray]

    i, j = grid.edges()
    same = crust.plate[i] == crust.plate[j]
    graph = coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(grid.size, grid.size))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = np.zeros(k, dtype=np.int64)
    np.maximum.at(largest, crust.plate, sizes[labels])
    fragment = (sizes[labels] < largest[crust.plate]) & (sizes[labels] < h.MIN_PLATE_FRACTION * grid.size)
    assigned = ~fragment
    for _ in range(50):
        if assigned.all():
            break
        counts = _neighbour_plate_counts(grid, crust.plate, assigned, k)
        ready = ~assigned & (counts.max(axis=1) > 0)
        if not ready.any():
            break
        crust.plate[ready] = counts[ready].argmax(axis=1)
        assigned |= ready


def balance_continents(crust: Crust, grid: SphereGrid, target_fraction: float) -> None:
    """Keep the continental area at its initial value, as re-mapping gains or loses coastal cells.

    Missing area is added by accreting the highest oceanic relief (island arcs,
    volcanic plateaus) along coasts; surplus area is removed from the lowest
    coastal lowlands.
    """
    i, j = grid.edges()
    for _ in range(5):
        cont = crust.continental
        excess = int(round((cont.mean() - target_fraction) * grid.size))
        if abs(excess) < BALANCE_TOLERANCE * grid.size:
            return
        mixed = cont[i] != cont[j]
        coast = np.zeros(grid.size, dtype=bool)
        coast[i[mixed]] = True
        coast[j[mixed]] = True
        if excess < 0:
            candidates = np.flatnonzero(coast & ~cont)
            chosen = candidates[np.argsort(-crust.ocean_extra[candidates], kind="stable")[:-excess]]
            crust.continental[chosen] = True
            crust.cont_elev[chosen] = h.CONTINENT_BASE_M + np.maximum(crust.ocean_extra[chosen]
                                                                      + h.OCEAN_RIDGE_DEPTH_M, 0.0)
            crust.sediment[chosen] = 0.0
        else:
            candidates = np.flatnonzero(coast & cont)
            chosen = candidates[np.argsort(crust.cont_elev[candidates], kind="stable")[:excess]]
            crust.continental[chosen] = False
            crust.ocean_extra[chosen] = 0.0
            neighbour_age = np.zeros(grid.size)
            ocean_i, ocean_j = ~cont[i], ~cont[j]
            np.maximum.at(neighbour_age, j[ocean_i], crust.ocean_age[i[ocean_i]])
            np.maximum.at(neighbour_age, i[ocean_j], crust.ocean_age[j[ocean_j]])
            crust.ocean_age[chosen] = neighbour_age[chosen]


def absorb_small_plates(crust: Crust, plates: Plates, grid: SphereGrid) -> None:
    """Merge plates below the minimum size into the plate of their nearest foreign cell."""
    sizes = plate_sizes(crust, plates)
    minimum = h.MIN_PLATE_FRACTION * grid.size
    for p in np.flatnonzero(plates.active & (sizes < minimum)):
        members = crust.plate == p
        others = ~members
        if not members.any():
            plates.active[p] = False
            plates.omega[p] = 0.0
            continue
        if not others.any():
            continue
        _, idx = cKDTree(crust.points[others]).query(crust.points[members])
        target = np.bincount(crust.plate[others][idx]).argmax()
        crust.plate[members] = target
        plates.active[p] = False
        plates.omega[p] = 0.0
    if not plates.active.any():
        plates.active[0] = True


def rift(crust: Crust, plates: Plates, grid: SphereGrid, interval_myr: float, ctx: StepContext,
         stats: StepStats, target_count: int, seed: int, step: int) -> None:
    """Split plates, mostly those carrying large continents, with the halves moving apart."""
    rng = named_rng(seed, f"tectonics.rift.{step}")
    sizes = plate_sizes(crust, plates)
    active = np.flatnonzero(plates.active)
    if len(active) >= h.PLATES_RANGE[1]:
        return
    mean_size = sizes[active].mean()
    feedback = count_feedback(plates.count, target_count)
    for p in active:
        members = np.flatnonzero(crust.plate == p)
        if members.size < 4 * h.MIN_PLATE_FRACTION * grid.size:
            continue
        continental = crust.continental[members].mean()
        weight = continental + (1.0 - continental) * h.OCEANIC_RIFT_FACTOR
        chance = min(interval_myr / h.RIFT_TIMESCALE_MYR * weight * sizes[p] / mean_size * feedback, 0.9)
        if rng.random() >= chance or plates.count >= h.PLATES_RANGE[1]:
            continue
        pts = crust.points[members]
        a = pts[rng.integers(members.size)]
        b = pts[np.argmin(pts @ a)]
        warped = warp(pts, seed + step, f"tectonics.rift.{p}", 0.15)
        side_b = (warped @ b) > (warped @ a)
        if side_b.all() or not side_b.any():
            continue
        axis = np.cross(a, b)
        axis /= max(np.linalg.norm(axis), 1e-12)
        opening = h.RIFT_SPEED_FACTOR * ctx.typical_speed / ctx.radius_m
        base = plates.omega[p].copy()
        plates.omega[p] = base - 0.5 * opening * axis
        new_id = plates.add(base + 0.5 * opening * axis)
        crust.plate[members[side_b]] = new_id
        stats.rifts += 1


def _largest_component(grid: SphereGrid, cells: np.ndarray) -> np.ndarray:
    """Return the cells of the largest connected group among ``cells``."""
    sub = grid.neighbours[cells][:, cells]
    _, labels = connected_components(sub, directed=False)
    return cells[labels == np.bincount(labels).argmax()]


def initiate_subduction(crust: Crust, plates: Plates, grid: SphereGrid, interval_myr: float,
                        ctx: StepContext, stats: StepStats, target_count: int, seed: int, step: int) -> None:
    """Detach old sea floor from plates that also carry continents and set it moving toward the continent."""
    rng = named_rng(seed, f"tectonics.initiation.{step}")
    for p in np.flatnonzero(plates.active):
        if plates.count >= h.PLATES_RANGE[1]:
            return
        members = np.flatnonzero(crust.plate == p)
        continental = crust.continental[members]
        if continental.all() or not continental.any():
            continue
        oceanic = members[~continental]
        old = oceanic[crust.ocean_age[oceanic] > h.OLD_OCEAN_MYR]
        old_fraction = old.size / oceanic.size
        chance = min(interval_myr / h.SUBDUCTION_INITIATION_TIMESCALE_MYR * old_fraction, 0.9)
        if old.size == 0 or rng.random() >= chance:
            continue
        basin = _largest_component(grid, old)
        if basin.size < 2 * h.MIN_PLATE_FRACTION * grid.size:
            continue
        c_ocean = crust.points[basin].mean(axis=0)
        c_cont = crust.points[members[continental]].mean(axis=0)
        axis = np.cross(c_ocean, c_cont)
        if np.linalg.norm(axis) < 1e-9:
            continue
        axis /= np.linalg.norm(axis)
        closing = h.RIFT_SPEED_FACTOR * ctx.typical_speed / ctx.radius_m
        new_id = plates.add(plates.omega[p] + closing * axis)
        crust.plate[basin] = new_id
        stats.initiations += 1
