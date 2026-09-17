"""Processes applied at each simulation step: motion, subduction, collision, uplift, erosion, hotspots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .. import heuristics as h
from .crust import (
    FRONT_ANDEAN,
    FRONT_NONE,
    OROGENY_ANDEAN,
    OROGENY_ARC,
    OROGENY_COLLISION,
    Crust,
    Plates,
)
from .kernels import find_contacts, rotate_points

NEIGHBOURS = 6
FRONT_MEMORY_MYR = 10.0
PROFILE_WIDTHS = 1.5         # uplift profiles are cut off beyond this many widths from their peak


@dataclass
class StepContext:
    """Planet constants used by every step."""

    radius_m: float
    edge_rad: float            # mean grid edge length (radians)
    relief: float              # relief factor from gravity
    typical_speed: float       # typical plate speed (m/Myr)
    dt: float                  # step length (Myr)

    @property
    def edge_km(self) -> float:
        """Return the mean grid edge length in km."""
        return self.edge_rad * self.radius_m / 1e3


@dataclass
class StepStats:
    """Counters accumulated over the simulation."""

    subducted: int = 0
    collisions: int = 0
    merges: int = 0
    rifts: int = 0
    initiations: int = 0


def move(crust: Crust, plates: Plates, dt: float) -> None:
    """Rotate every crust point with its plate for one step."""
    crust.points = rotate_points(crust.points, crust.plate, plates.omega * dt)


def interact(crust: Crust, plates: Plates, ctx: StepContext, stats: StepStats) -> np.ndarray:
    """Consume subducting crust, record subduction fronts and collisions; return plate-pair collision counts."""
    alive = crust.alive
    tree = cKDTree(crust.points[alive])
    alive_index = np.flatnonzero(alive)
    _, nbr = tree.query(crust.points, k=NEIGHBOURS + 1, distance_upper_bound=h.CONTACT_FACTOR * ctx.edge_rad)
    missing = nbr >= len(alive_index)
    nbr = np.where(missing, crust.size, alive_index[np.minimum(nbr, len(alive_index) - 1)])
    consumed, front_kind, front_speed, collision_speed, pairs = find_contacts(
        crust.points, nbr, crust.plate, crust.continental, crust.ocean_age, alive,
        plates.omega, ctx.radius_m, len(plates.active))

    crust.alive &= ~consumed
    stats.subducted += int(consumed.sum())

    new_front = front_kind != FRONT_NONE
    crust.front_kind[new_front] = np.maximum(crust.front_kind[new_front], front_kind[new_front])
    crust.front_speed[new_front] = front_speed[new_front]
    crust.front_age[new_front] = 0.0
    colliding = collision_speed > 0
    crust.collision_speed[colliding] = collision_speed[colliding]
    crust.collision_age[colliding] = 0.0
    stats.collisions += int(colliding.sum())
    return pairs


def _speed_factor(speed: np.ndarray, ctx: StepContext) -> np.ndarray:
    return np.minimum(speed / ctx.typical_speed, h.UPLIFT_SPEED_CAP)


def _profile(x: np.ndarray) -> np.ndarray:
    """Return a bell-shaped uplift profile in units of its width, zero beyond the cutoff."""
    return np.where(np.abs(x) <= PROFILE_WIDTHS, np.exp(-x**2), 0.0)


def _cap_factor(height: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(1.0 - height / cap, 0.0, 1.0) ** 2


def _record_orogeny(crust: Crust, index: np.ndarray, rate: np.ndarray, kind: int) -> None:
    active = rate > h.OROGENY_MIN_RATE_M_MYR
    crust.orogeny_age[index[active]] = 0.0
    crust.orogeny_kind[index[active]] = kind


def uplift(crust: Crust, ctx: StepContext) -> None:
    """Raise mountains and island arcs behind active subduction fronts and at collisions; deepen trenches."""
    alive = crust.alive
    cap = h.MAX_ELEVATION_EARTH_M * ctx.relief
    km = ctx.radius_m / 1e3

    fronts = np.flatnonzero(alive & (crust.front_kind != FRONT_NONE) & (crust.front_age <= FRONT_MEMORY_MYR))
    if fronts.size:
        reach_km = max(h.ANDEAN_OFFSET_KM + PROFILE_WIDTHS * h.ANDEAN_WIDTH_KM, 2 * ctx.edge_km)
        tree = cKDTree(crust.points[fronts])
        targets = np.flatnonzero(alive)
        dist, near = tree.query(crust.points[targets], distance_upper_bound=reach_km / km)
        ok = np.isfinite(dist)
        targets, dist, near = targets[ok], dist[ok] * km, near[ok]
        front = fronts[near]
        same_plate = crust.plate[targets] == crust.plate[front]
        speed = _speed_factor(crust.front_speed[front], ctx)

        # Overriding plate: Andean belts on continents, island arcs on oceanic crust.
        andean = same_plate & (crust.front_kind[front] == FRONT_ANDEAN) & crust.continental[targets]
        idx = targets[andean]
        profile = _profile((dist[andean] - h.ANDEAN_OFFSET_KM) / h.ANDEAN_WIDTH_KM)
        rate = h.SUBDUCTION_UPLIFT_M_MYR * ctx.relief * speed[andean] * profile
        rate *= _cap_factor(crust.cont_elev[idx], cap)
        crust.cont_elev[idx] += rate * ctx.dt
        _record_orogeny(crust, idx, rate, OROGENY_ANDEAN)

        arc = same_plate & ~crust.continental[targets]
        idx = targets[arc]
        profile = _profile(dist[arc] / h.ARC_WIDTH_KM)
        arc_cap = (h.ARC_HEIGHT_M - h.OCEAN_RIDGE_DEPTH_M) * ctx.relief
        rate = h.ARC_UPLIFT_M_MYR * ctx.relief * speed[arc] * profile * _cap_factor(crust.ocean_extra[idx], arc_cap)
        crust.ocean_extra[idx] += rate * ctx.dt
        _record_orogeny(crust, idx, rate, OROGENY_ARC)

        # Subducting plate: trench along the front.
        trench = ~same_plate & ~crust.continental[targets] & (dist < 3 * h.TRENCH_WIDTH_KM)
        idx = targets[trench]
        depth = -h.TRENCH_DEPTH_M * np.exp(-((dist[trench] / h.TRENCH_WIDTH_KM) ** 2))
        crust.ocean_extra[idx] = np.minimum(crust.ocean_extra[idx], depth)

    colliding = np.flatnonzero(alive & (crust.collision_age <= FRONT_MEMORY_MYR))
    if colliding.size:
        reach_km = max(PROFILE_WIDTHS * h.COLLISION_WIDTH_KM, 2 * ctx.edge_km)
        tree = cKDTree(crust.points[colliding])
        targets = np.flatnonzero(alive & crust.continental)
        dist, near = tree.query(crust.points[targets], distance_upper_bound=reach_km / km)
        ok = np.isfinite(dist)
        targets, dist, near = targets[ok], dist[ok] * km, near[ok]
        speed = _speed_factor(crust.collision_speed[colliding[near]], ctx)
        profile = _profile(dist / h.COLLISION_WIDTH_KM)
        rate = h.COLLISION_UPLIFT_M_MYR * ctx.relief * speed * profile * _cap_factor(crust.cont_elev[targets], cap)
        crust.cont_elev[targets] += rate * ctx.dt
        _record_orogeny(crust, targets, rate, OROGENY_COLLISION)


def hotspots(crust: Crust, positions: np.ndarray, ctx: StepContext) -> None:
    """Build volcanoes where plates pass over fixed mantle hotspots."""
    if len(positions) == 0:
        return
    km = ctx.radius_m / 1e3
    radius_km = max(h.HOTSPOT_RADIUS_KM, 0.8 * ctx.edge_km)
    alive = np.flatnonzero(crust.alive)
    tree = cKDTree(crust.points[alive])
    hits = tree.query_ball_point(positions, radius_km / km)
    idx = np.unique(np.concatenate([alive[np.asarray(hh, dtype=np.int64)] for hh in hits]))
    if idx.size == 0:
        return
    cap = h.VOLCANO_MAX_HEIGHT_EARTH_M * ctx.relief
    rate = h.HOTSPOT_UPLIFT_M_MYR * ctx.relief
    cont = crust.continental[idx]
    ci, oi = idx[cont], idx[~cont]
    crust.cont_elev[ci] += rate * ctx.dt * _cap_factor(crust.cont_elev[ci] - h.CONTINENT_BASE_M, cap)
    crust.ocean_extra[oi] += rate * ctx.dt * _cap_factor(crust.ocean_extra[oi], cap - h.OCEAN_RIDGE_DEPTH_M * ctx.relief)
    crust.volcanic[idx] = True


def age_and_erode(crust: Crust, ctx: StepContext, rivers: bool) -> None:
    """Age oceanic crust and fronts and let oceanic relief subside.

    Without rivers, continental relief also wears down uniformly; with rivers,
    erosion is applied when crust is re-mapped.
    """
    dt = ctx.dt
    crust.ocean_age += dt
    crust.orogeny_age += dt
    crust.front_age += dt
    crust.collision_age += dt
    if not rivers:
        above = crust.cont_elev - h.CONTINENT_BASE_M
        crust.cont_elev -= np.where(above > 0, above, 0.0) * (1.0 - np.exp(-dt / h.CONTINENT_EROSION_TIMESCALE_MYR))
    crust.ocean_extra *= np.exp(-dt / h.OCEAN_RELIEF_TIMESCALE_MYR)
    stale = crust.front_age > FRONT_MEMORY_MYR
    crust.front_kind[stale] = FRONT_NONE


def plate_sizes(crust: Crust, plates: Plates) -> np.ndarray:
    """Return the number of live points on each plate."""
    return np.bincount(crust.plate[crust.alive], minlength=len(plates.active))


def update_motion(crust: Crust, plates: Plates, pairs: np.ndarray, alive_before: np.ndarray,
                  ctx: StepContext) -> None:
    """Turn plates toward their subduction zones (slab pull) and lock colliding plates together.

    ``alive_before`` marks the points that were alive before this step's subduction.
    """
    omega_typ = ctx.typical_speed / ctx.radius_m
    sunk = alive_before & ~crust.alive
    if sunk.any():
        for p in np.unique(crust.plate[sunk]):
            members = crust.alive & (crust.plate == p)
            if not members.any():
                continue
            centre = crust.points[members].mean(axis=0)
            centre /= np.linalg.norm(centre)
            pull = np.cross(centre, crust.points[sunk & (crust.plate == p)])
            norms = np.linalg.norm(pull, axis=1, keepdims=True)
            direction = (pull / np.maximum(norms, 1e-12)).sum(axis=0)
            if np.linalg.norm(direction) > 0:
                plates.omega[p] += h.SLAB_PULL_RATE * ctx.dt * omega_typ * direction / np.linalg.norm(direction)

    sizes = plate_sizes(crust, plates).astype(float)
    coupling = min(h.COLLISION_COUPLING_PER_MYR * ctx.dt, 1.0)
    for a, b in zip(*np.nonzero(np.triu(pairs, k=1))):
        total = sizes[a] + sizes[b]
        if total == 0:
            continue
        shared = (plates.omega[a] * sizes[a] + plates.omega[b] * sizes[b]) / total
        plates.omega[a] += coupling * (shared - plates.omega[a])
        plates.omega[b] += coupling * (shared - plates.omega[b])

    limit = h.PLATE_SPEED_MAX_FACTOR * omega_typ
    speed = np.linalg.norm(plates.omega, axis=1)
    too_fast = speed > limit
    plates.omega[too_fast] *= (limit / speed[too_fast])[:, None]


def merge_colliding(crust: Crust, plates: Plates, history: np.ndarray, target_count: int,
                    stats: StepStats) -> None:
    """Fuse plates whose continents have collided over a wide front for a sustained time.

    ``history`` holds the accumulated collision contacts between plate pairs.
    """
    sizes = plate_sizes(crust, plates)
    threshold = h.MERGE_CONTACT_FRACTION * count_feedback(plates.count, target_count)
    a_idx, b_idx = np.nonzero(np.triu(history, k=1))
    for a, b in sorted(zip(a_idx, b_idx), key=lambda ab: -history[ab[0], ab[1]]):
        if not (plates.active[a] and plates.active[b]):
            continue
        smaller = min(sizes[a], sizes[b])
        if smaller == 0 or history[a, b] < threshold * smaller:
            continue
        keep, gone = (a, b) if sizes[a] >= sizes[b] else (b, a)
        total = sizes[keep] + sizes[gone]
        plates.omega[keep] = (plates.omega[keep] * sizes[keep] + plates.omega[gone] * sizes[gone]) / total
        crust.plate[crust.plate == gone] = keep
        plates.active[gone] = False
        plates.omega[gone] = 0.0
        sizes[keep], sizes[gone] = total, 0
        history[keep] += history[gone]
        history[:, keep] += history[:, gone]
        history[gone] = 0.0
        history[:, gone] = 0.0
        history[keep, keep] = 0.0
        stats.merges += 1


def count_feedback(count: int, target: int) -> float:
    """Return a factor above 1 when there are fewer plates than expected, below 1 when there are more."""
    lo, hi = h.PLATE_COUNT_FEEDBACK
    return float(np.clip(target / max(count, 1), lo, hi))
