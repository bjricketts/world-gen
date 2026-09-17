"""Individual landforms stamped onto the surface: volcanoes, craters and rifts."""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from .fields import SurfaceFields, Terrain


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return ``n`` uniformly distributed unit vectors."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cells_within(fields: SurfaceFields, centre: np.ndarray, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the cells within a surface distance of ``centre`` and their distances (km)."""
    angle = min(radius_km * 1e3 / fields.radius_m, np.pi)
    chord = 2.0 * np.sin(angle / 2.0)
    idx = np.asarray(fields.grid.tree.query_ball_point(centre, chord), dtype=np.int64)
    if idx.size == 0:
        return idx, np.empty(0)
    cosang = np.clip(fields.grid.points[idx] @ centre, -1.0, 1.0)
    return idx, np.arccos(cosang) * fields.radius_m / 1e3


def add_volcano(fields: SurfaceFields, centre: np.ndarray, height_m: float) -> None:
    """Add a shield volcano with a summit caldera."""
    radius_km = 0.5 * h.VOLCANO_WIDTH_PER_HEIGHT * height_m / 1e3
    radius_km = max(radius_km, 1.2 * fields.spacing_km)
    idx, d = _cells_within(fields, centre, radius_km)
    if idx.size == 0:
        return
    x = d / radius_km
    profile = height_m * (1.0 - x**2) ** 1.5
    caldera = x < h.CALDERA_FRACTION
    profile[caldera] -= 0.08 * height_m * (1.0 - (x[caldera] / h.CALDERA_FRACTION) ** 2)
    fields.elevation[idx] += profile
    fields.terrain[idx[x < 0.6]] = Terrain.VOLCANO


def crater_depth(diameter_km: float, relief_factor: float) -> float:
    """Return crater depth (m): simple craters are 0.2 D deep, larger ones proportionally shallower."""
    transition = h.CRATER_TRANSITION_EARTH_KM * relief_factor
    if diameter_km <= transition:
        return h.CRATER_SIMPLE_DEPTH_RATIO * diameter_km * 1e3
    return h.CRATER_SIMPLE_DEPTH_RATIO * transition * 1e3 * (diameter_km / transition) ** 0.3


def add_crater(fields: SurfaceFields, centre: np.ndarray, diameter_km: float, relief_factor: float) -> None:
    """Add an impact crater: bowl, raised rim and ejecta falling off outside."""
    r = diameter_km / 2.0
    idx, d = _cells_within(fields, centre, 2.5 * r)
    if idx.size == 0:
        return
    depth = crater_depth(diameter_km, relief_factor)
    rim = 0.25 * depth
    x = d / r
    inside = x < 1.0
    profile = np.where(inside, -depth * (1.0 - x**2) + rim * x**4, rim * np.clip(x, 1, None) ** -3)
    # The crater floor replaces older terrain, so overlapping craters overprint each other.
    base = np.median(fields.elevation[idx])
    fields.elevation[idx[inside]] = base + profile[inside]
    fields.elevation[idx[~inside]] += profile[~inside]
    fields.terrain[idx[x < 1.1]] = Terrain.CRATER


def sample_crater_diameters(rng: np.random.Generator, area_km2: float, surface_age_gyr: float,
                            d_min_km: float, d_max_km: float) -> np.ndarray:
    """Return crater diameters (km) following N(>D) ∝ D⁻² for a surface of the given age."""
    if d_min_km >= d_max_km or surface_age_gyr <= 0:
        return np.empty(0)
    expected = h.CRATER_DENSITY * (surface_age_gyr / 4.5) * area_km2 * (d_min_km**-2 - d_max_km**-2)
    count = min(int(rng.poisson(expected)), h.CRATER_MAX_COUNT)
    u = rng.uniform(size=count)
    # Inverse CDF of the truncated power law.
    inv = d_min_km**-2 - u * (d_min_km**-2 - d_max_km**-2)
    return np.sort(inv ** -0.5)[::-1]


def add_crater_field(fields: SurfaceFields, rng: np.random.Generator, surface_age_gyr: float,
                     relief_factor: float, mask: np.ndarray | None = None) -> int:
    """Add a population of craters for a surface age; returns the number added.

    Largest craters are placed first so smaller ones overprint them. If
    ``mask`` is given, craters are only centred on cells where it is True.
    """
    radius_km = fields.radius_m / 1e3
    area = 4.0 * np.pi * radius_km**2
    if mask is not None:
        area *= mask.mean()
        if not mask.any():
            return 0
    d_min = h.CRATER_MIN_CELLS * fields.spacing_km
    d_max = h.CRATER_MAX_DIAMETER_FRACTION * radius_km
    diameters = sample_crater_diameters(rng, area, surface_age_gyr, d_min, d_max)
    if diameters.size == 0:
        return 0
    if mask is None:
        centres = random_unit_vectors(rng, diameters.size)
    else:
        centres = fields.grid.points[rng.choice(np.flatnonzero(mask), size=diameters.size)]
    for centre, diameter in zip(centres, diameters):
        add_crater(fields, centre, diameter, relief_factor)
    return int(diameters.size)


def add_rift(fields: SurfaceFields, start: np.ndarray, end: np.ndarray, width_km: float, depth_m: float) -> None:
    """Add a rift valley along the great-circle arc from ``start`` to ``end``, with raised shoulders."""
    normal = np.cross(start, end)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return
    normal /= norm
    arc = np.arccos(np.clip(start @ end, -1, 1))
    mid = start + end
    mid /= np.linalg.norm(mid)
    half_length_km = arc / 2 * fields.radius_m / 1e3
    idx, _ = _cells_within(fields, mid, half_length_km + 3 * width_km)
    if idx.size == 0:
        return
    p = fields.grid.points[idx]
    across_km = np.abs(np.arcsin(np.clip(p @ normal, -1, 1))) * fields.radius_m / 1e3
    # Position along the arc, measured from the midpoint.
    tangent = np.cross(normal, mid)
    along = np.arctan2(p @ tangent, p @ mid)
    beyond_km = np.clip(np.abs(along) - arc / 2, 0, None) * fields.radius_m / 1e3
    d = np.hypot(across_km, beyond_km)
    x = d / width_km
    profile = -depth_m * np.exp(-x**2) + 0.25 * depth_m * np.exp(-((x - 1.8) / 0.6) ** 2)
    fields.elevation[idx] += profile
    fields.terrain[idx[x < 0.8]] = Terrain.RIFT
