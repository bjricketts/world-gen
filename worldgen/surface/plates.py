"""Heuristic plate-tectonic terrain for mobile-lid planets.

Plates are drawn once with present-day motions; mountain belts, trenches,
ridges and rifts are placed from the boundary types. No time evolution:
milestone 3 replaces this with a plate simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..noise import fbm, ridged, warp
from ..util import named_rng
from .distance import distance_from, smooth
from .fields import Boundary, Crust, SurfaceFields, Terrain
from .landforms import add_volcano, random_unit_vectors

KM_PER_MYR_PER_CM_YR = 10.0


@dataclass
class Plates:
    """Plate layout and motion."""

    count: int
    seeds: np.ndarray            # (k, 3) plate centres
    euler_poles: np.ndarray      # (k, 3) rotation axes
    angular_speed: np.ndarray    # (k,) rad/yr
    cell_plate: np.ndarray       # (n,) plate index per cell


def seafloor_depth(age_myr: np.ndarray, relief: float = 1.0) -> np.ndarray:
    """Return sea-floor depth (m, negative) for oceanic crust of a given age (Myr).

    Depths scale with the relief factor: basins are shallower on high-gravity
    planets (Cowan & Abbot 2014).
    """
    age = np.maximum(age_myr, 0.0)
    young = h.OCEAN_RIDGE_DEPTH_M + h.OCEAN_SUBSIDENCE_M_PER_SQRT_MYR * np.sqrt(age)
    old = h.OCEAN_MAX_DEPTH_M + h.OCEAN_PLATE_DECAY_M * np.exp(-age / h.OCEAN_PLATE_TIMESCALE_MYR)
    return relief * np.where(age < h.OCEAN_PLATE_TRANSITION_MYR, young, old)


def plate_count(radius_m: float, activity: float, rng: np.random.Generator) -> int:
    """Return the number of plates: more for larger and more active planets."""
    expected = h.PLATES_EARTH * (radius_m / c.R_EARTH) ** 2 * max(activity, 0.05) ** 0.3
    lo, hi = h.PLATES_RANGE
    return int(np.clip(round(expected * rng.uniform(0.7, 1.3)), lo, hi))


def make_plates(fields: SurfaceFields, activity: float, seed: int) -> Plates:
    """Return plates with irregular boundaries and random rotation poles."""
    rng = named_rng(seed, "plates.layout")
    k = plate_count(fields.radius_m, activity, rng)
    seeds = random_unit_vectors(rng, k)
    weights = rng.uniform(0.0, 0.35, size=k)    # larger weight -> larger plate
    warped = warp(fields.grid.points, seed, "plates.warp", h.PLATE_WARP_RAD)
    angle = np.arccos(np.clip(warped @ seeds.T, -1, 1))
    cell_plate = np.argmin(angle - weights, axis=1).astype(np.int16)

    speed_m_yr = h.PLATE_SPEED_EARTH_CM_YR / 100 * max(activity, 0.05) ** 0.5
    angular = speed_m_yr / fields.radius_m * rng.uniform(0.3, 1.0, size=k)
    return Plates(count=k, seeds=seeds, euler_poles=random_unit_vectors(rng, k),
                  angular_speed=angular, cell_plate=cell_plate)


def continental_crust(fields: SurfaceFields, land_fraction: float, spreading_distance_km: np.ndarray,
                      seed: int) -> np.ndarray:
    """Return a boolean mask of continental crust covering the target area.

    Continents are placed away from spreading boundaries, so ridges sit in
    ocean basins and continental margins are old.
    """
    target = float(np.clip(land_fraction + h.CONTINENTAL_SHELF_EXTRA, 0.02, h.CONTINENTAL_MAX_FRACTION))
    potential = fbm(fields.grid.points, seed, "continents", frequency=1.2, octaves=5)
    potential += h.CONTINENT_RIDGE_AVOIDANCE * np.tanh(np.nan_to_num(spreading_distance_km, posinf=1e9)
                                                       / h.CONTINENT_RIDGE_SCALE_KM)
    return potential >= np.quantile(potential, 1.0 - target)


def cell_velocity(fields: SurfaceFields, plates: Plates) -> np.ndarray:
    """Return each cell's surface velocity vector (m/yr)."""
    omega = plates.euler_poles[plates.cell_plate] * plates.angular_speed[plates.cell_plate, None]
    return np.cross(omega, fields.grid.points) * fields.radius_m


def classify_boundaries(fields: SurfaceFields, plates: Plates, velocity: np.ndarray):
    """Return boundary edges with their type and closing speed (m/yr, positive = converging)."""
    i, j = fields.grid.edges()
    cross = plates.cell_plate[i] != plates.cell_plate[j]
    i, j = i[cross], j[cross]
    p = fields.grid.points
    direction = p[j] - p[i]
    direction -= np.einsum("ij,ij->i", direction, p[i])[:, None] * p[i]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    relative = velocity[i] - velocity[j]
    closing = np.einsum("ij,ij->i", relative, direction)
    total = np.linalg.norm(relative, axis=1) + 1e-12
    kind = np.full(i.size, Boundary.TRANSFORM, dtype=np.int8)
    kind[closing > h.BOUNDARY_NORMAL_RATIO * total] = Boundary.CONVERGENT
    kind[closing < -h.BOUNDARY_NORMAL_RATIO * total] = Boundary.DIVERGENT
    return i, j, kind, closing


def _uplift(fields, sources, strength, height_m, width_km, offset_km=0.0, limit_factor=3.0):
    """Return a mountain-belt profile around source cells, scaled by per-source strength."""
    n = fields.grid.size
    if len(sources) == 0:
        return np.zeros(n), np.full(n, np.inf)
    radius_km = fields.radius_m / 1e3
    dist, nearest = distance_from(fields.grid, sources, radius_km, limit_km=offset_km + limit_factor * width_km)
    out = np.zeros(n)
    ok = nearest >= 0
    out[ok] = height_m * strength[nearest[ok]] * np.exp(-(((dist[ok] - offset_km) / width_km) ** 2))
    return out, dist


def build_plate_terrain(fields: SurfaceFields, activity: float, land_fraction: float,
                        relief: float, age_gyr: float, seed: int) -> dict:
    """Fill the surface fields for a plate-tectonic planet; returns summary attributes."""
    n = fields.grid.size
    pts = fields.grid.points
    radius_km = fields.radius_m / 1e3
    plates = make_plates(fields, activity, seed)
    fields.plate[:] = plates.cell_plate
    velocity = cell_velocity(fields, plates)
    i, j, kind, closing = classify_boundaries(fields, plates, velocity)
    div = kind == Boundary.DIVERGENT
    spreading_dist, _ = distance_from(fields.grid, np.concatenate([i[div], j[div]]), radius_km)
    continental = continental_crust(fields, land_fraction, spreading_dist, seed)
    fields.crust[:] = np.where(continental, Crust.CONTINENTAL, Crust.OCEANIC)
    speed_scale = h.PLATE_SPEED_EARTH_CM_YR / 100
    strength_edge = np.clip(np.abs(closing) / speed_scale, 0.2, 1.5)

    # Per-cell strongest boundary type.
    for code in (Boundary.TRANSFORM, Boundary.DIVERGENT, Boundary.CONVERGENT):
        sel = kind == code
        fields.boundary[i[sel]] = code
        fields.boundary[j[sel]] = code

    strength = np.zeros(n)
    np.maximum.at(strength, i, strength_edge)
    np.maximum.at(strength, j, strength_edge)

    cont_i, cont_j = continental[i], continental[j]
    conv = kind == Boundary.CONVERGENT

    # Convergent boundaries: who overrides whom.
    collision = conv & cont_i & cont_j
    andean_i = conv & cont_i & ~cont_j      # i continental overrides oceanic j
    andean_j = conv & ~cont_i & cont_j
    oceanic = conv & ~cont_i & ~cont_j
    flip = named_rng(seed, "plates.subduction").random(i.size) < 0.5
    arc_cells = np.concatenate([i[oceanic & flip], j[oceanic & ~flip]])
    trench_cells = np.concatenate([j[andean_i], i[andean_j], j[oceanic & flip], i[oceanic & ~flip]])
    andean_cells = np.concatenate([i[andean_i], j[andean_j]])
    collision_cells = np.concatenate([i[collision], j[collision]])
    ridge_cells = np.concatenate([i[div & ~cont_i], j[div & ~cont_j]])
    rift_cells = np.concatenate([i[div & cont_i], j[div & cont_j]])

    # Oceanic crust age from distance to ridges, and depth from age.
    half_rate_km_myr = max(speed_scale * 100 * max(activity, 0.05) ** 0.5 / 2, 0.1) * KM_PER_MYR_PER_CM_YR
    ridge_dist, _ = distance_from(fields.grid, ridge_cells, radius_km)
    age_myr = np.minimum(ridge_dist / half_rate_km_myr, age_gyr * 1e3)
    age_myr[~np.isfinite(age_myr)] = min(200.0, age_gyr * 1e3)
    ocean_depth = seafloor_depth(age_myr, relief)
    fields.crust_age_myr[:] = np.where(continental, np.nan, age_myr)

    continent_height = h.CONTINENT_BASE_M + h.CONTINENT_NOISE_M * fbm(pts, seed, "continent.detail", 3.0, 6)
    base = np.where(continental, continent_height, ocean_depth)
    base = smooth(fields.grid, base, h.MARGIN_SMOOTHING_STEPS)
    base += np.where(continental, 0.0, h.OCEAN_NOISE_M * fbm(pts, seed, "ocean.detail", 4.0, 5))

    # Mountain belts, arcs, trenches and rifts.
    rough = 0.55 + 0.45 * ridged(pts, seed, "mountains", 6.0, 5)
    collision_up, d_col = _uplift(fields, collision_cells, strength, h.COLLISION_HEIGHT_M * relief, h.COLLISION_WIDTH_KM)
    andean_up, d_and = _uplift(fields, andean_cells, strength, h.ANDEAN_HEIGHT_M * relief, h.ANDEAN_WIDTH_KM,
                               offset_km=h.ANDEAN_OFFSET_KM)
    arc_up, d_arc = _uplift(fields, arc_cells, strength, h.ARC_HEIGHT_M * relief, h.ARC_WIDTH_KM)
    trench, d_tr = _uplift(fields, trench_cells, strength, h.TRENCH_DEPTH_M, h.TRENCH_WIDTH_KM)
    rift, d_rift = _uplift(fields, rift_cells, strength, h.RIFT_DEPTH_M, h.RIFT_WIDTH_KM)

    # Uplift only affects the overriding / continental side.
    andean_up *= continental
    trench *= ~continental
    mountains = np.maximum(collision_up * continental, andean_up) * rough
    arc = arc_up * ~continental * rough
    elevation = base + mountains + arc - trench - rift

    cap = h.MAX_ELEVATION_EARTH_M * relief
    elevation = np.where(elevation > 0.6 * cap, 0.6 * cap + 0.4 * cap * np.tanh((elevation - 0.6 * cap) / (0.4 * cap)),
                         elevation)
    fields.elevation[:] = elevation

    fields.terrain[:] = Terrain.PLAIN
    fields.terrain[continental & (elevation > 1500)] = Terrain.HIGHLAND
    fields.terrain[~continental & (ridge_dist < 1.5 * fields.spacing_km)] = Terrain.RIDGE
    fields.terrain[mountains > 1500 * relief] = Terrain.MOUNTAIN
    fields.terrain[arc > 1000 * relief] = Terrain.MOUNTAIN
    fields.terrain[trench > 0.4 * h.TRENCH_DEPTH_M] = Terrain.TRENCH
    fields.terrain[rift > 0.4 * h.RIFT_DEPTH_M] = Terrain.RIFT

    # Hotspot volcanoes, mostly oceanic.
    rng = named_rng(seed, "plates.hotspots")
    expected = h.HOTSPOTS_EARTH * max(activity, 0.05) ** 0.5 * (fields.radius_m / c.R_EARTH) ** 2
    n_hot = min(rng.poisson(expected), h.HOTSPOTS_MAX)
    for centre in random_unit_vectors(rng, n_hot):
        add_volcano(fields, centre, h.VOLCANO_MAX_HEIGHT_EARTH_M * relief * rng.uniform(0.3, 0.8))

    return {
        "plates": plates.count,
        "hotspots": int(n_hot),
        "continental_crust_fraction": float(continental.mean()),
    }
