"""Terrain for planets without plate tectonics: stagnant lid, episodic, heat pipe and inactive."""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from ..noise import fbm, ridged, warp
from ..util import named_rng
from .distance import smooth
from .fields import SurfaceFields, Terrain
from .landforms import add_crater_field, add_rift, add_volcano, random_unit_vectors


def _base_relief(fields: SurfaceFields, relief: float, seed: int, dichotomy: float) -> np.ndarray:
    """Return broad topography: low-order noise plus an optional hemispheric step."""
    pts = fields.grid.points
    base = h.LID_BASE_RELIEF_M * relief * fbm(pts, seed, "lid.base", 1.5, 7)
    if dichotomy > 0:
        rng = named_rng(seed, "lid.dichotomy")
        axis = random_unit_vectors(rng, 1)[0]
        tilted = warp(pts, seed, "lid.dichotomy.warp", 0.35)
        base += h.DICHOTOMY_HEIGHT_M * relief * dichotomy * np.tanh(3.0 * (tilted @ axis))
    return base


def _volcanic_rise(fields: SurfaceFields, rng: np.random.Generator, relief: float) -> np.ndarray:
    """Return the centre of a broad volcanic rise after adding it to the elevation."""
    centre = random_unit_vectors(rng, 1)[0]
    angle = np.arccos(np.clip(fields.grid.points @ centre, -1, 1))
    fields.elevation += h.RISE_HEIGHT_M * relief * np.exp(-((angle / h.RISE_RADIUS_RAD) ** 2))
    return centre


def _offset_point(rng: np.random.Generator, centre: np.ndarray, max_angle: float) -> np.ndarray:
    """Return a random point within ``max_angle`` radians of ``centre``."""
    v = centre + rng.normal(size=3) * max_angle
    return v / np.linalg.norm(v)


def _volcano_height(rng: np.random.Generator, relief: float, low: float = 0.25, high: float = 1.0) -> float:
    """Return a random volcano height, skewed toward smaller edifices."""
    return h.VOLCANO_MAX_HEIGHT_EARTH_M * relief * rng.uniform(low, high) ** 1.5


def build_stagnant_lid(fields: SurfaceFields, activity: float, relief: float, age_gyr: float,
                       water_erosion: float, seed: int) -> dict:
    """Mars-like surface: dichotomy, a volcanic rise with giant shields, radial rifts and old craters."""
    rng = named_rng(seed, "lid.stagnant")
    fields.elevation[:] = _base_relief(fields, relief, seed, dichotomy=rng.uniform(0.3, 1.0))
    fields.terrain[fields.elevation > np.quantile(fields.elevation, 0.6)] = Terrain.HIGHLAND
    rise = _volcanic_rise(fields, rng, relief)

    n_volc = max(1, rng.poisson(h.STAGNANT_VOLCANOES * max(activity, 0.1)))
    for k in range(n_volc):
        centre = _offset_point(rng, rise, 0.35) if k < n_volc * 0.7 else random_unit_vectors(rng, 1)[0]
        add_volcano(fields, centre, _volcano_height(rng, relief))

    n_rift = rng.integers(1, 4)
    for _ in range(n_rift):
        start = _offset_point(rng, rise, 0.25)
        end = _offset_point(rng, start, 0.6)
        add_rift(fields, start, end, h.STAGNANT_RIFT_WIDTH_KM, h.STAGNANT_RIFT_DEPTH_M * relief * rng.uniform(0.4, 1.0))

    # More active interiors resurface faster, leaving younger, less cratered terrain.
    surface_age = h.STAGNANT_SURFACE_AGE_FRACTION * age_gyr * water_erosion / max(activity, 1.0)
    craters = add_crater_field(fields, rng, surface_age, relief, mask=fields.terrain != Terrain.VOLCANO)
    return {"volcanoes": int(n_volc), "rifts": int(n_rift), "craters": craters}


def build_episodic(fields: SurfaceFields, activity: float, relief: float, water_erosion: float, seed: int) -> dict:
    """Venus-like surface: mostly young volcanic plains with rough highland blocks and scattered volcanoes."""
    rng = named_rng(seed, "lid.episodic")
    pts = fields.grid.points
    base = _base_relief(fields, relief, seed, dichotomy=0.0)
    lo, hi = h.EPISODIC_PLAINS_FRACTION
    plains_fraction = rng.uniform(lo, hi)
    selector = fbm(pts, seed, "episodic.plains", 1.8, 5)
    highland = selector > np.quantile(selector, plains_fraction)
    rough = 0.6 * h.LID_BASE_RELIEF_M * relief * ridged(pts, seed, "episodic.tesserae", 8.0, 5)
    flattened = smooth(fields.grid, base, 6) * 0.4
    fields.elevation[:] = np.where(highland, base + 0.8 * h.LID_BASE_RELIEF_M * relief + rough, flattened)
    fields.elevation[:] = smooth(fields.grid, fields.elevation, 1)
    fields.terrain[:] = np.where(highland, Terrain.HIGHLAND, Terrain.VOLCANIC_PLAIN)

    n_volc = max(1, rng.poisson(h.EPISODIC_VOLCANOES * max(activity, 0.1)))
    for _ in range(n_volc):
        add_volcano(fields, random_unit_vectors(rng, 1)[0], _volcano_height(rng, relief, 0.15, 0.6))

    craters = add_crater_field(fields, rng, h.EPISODIC_SURFACE_AGE_GYR * water_erosion, relief)
    return {"volcanoes": int(n_volc), "plains_fraction": float(plains_fraction), "craters": craters}


def build_heat_pipe(fields: SurfaceFields, relief: float, seed: int) -> dict:
    """Io-like surface: smooth lava plains, many volcanic centres and isolated tectonic massifs."""
    rng = named_rng(seed, "lid.heatpipe")
    pts = fields.grid.points
    base = 0.4 * h.LID_BASE_RELIEF_M * relief * fbm(pts, seed, "heatpipe.base", 2.0, 5)
    massif_mask = fbm(pts, seed, "heatpipe.massifs", 2.5, 4) > 0.45
    massifs = massif_mask * 2.0 * h.LID_BASE_RELIEF_M * relief * ridged(pts, seed, "heatpipe.ridges", 10.0, 4)
    fields.elevation[:] = base + massifs
    fields.terrain[:] = Terrain.VOLCANIC_PLAIN
    fields.terrain[massifs > 500 * relief] = Terrain.MOUNTAIN

    n_volc = rng.poisson(h.HEATPIPE_VOLCANOES)
    for centre in random_unit_vectors(rng, n_volc):
        add_volcano(fields, centre, h.HEATPIPE_VOLCANO_HEIGHT_M * relief * rng.uniform(0.2, 1.0))
    return {"volcanoes": int(n_volc), "craters": 0}


def build_inactive(fields: SurfaceFields, relief: float, age_gyr: float, water_erosion: float, seed: int) -> dict:
    """Moon-like surface: cratered highlands with low, smooth mare plains."""
    rng = named_rng(seed, "lid.inactive")
    pts = fields.grid.points
    fields.elevation[:] = _base_relief(fields, relief, seed, dichotomy=rng.uniform(0.0, 0.6))
    fields.terrain[:] = Terrain.HIGHLAND
    craters = add_crater_field(fields, rng, age_gyr * water_erosion, relief)

    # Flood-basalt maria fill the lowest basins after most cratering.
    selector = fields.elevation + 0.3 * h.LID_BASE_RELIEF_M * relief * fbm(pts, seed, "maria", 3.0, 4)
    maria = selector < np.quantile(selector, h.INACTIVE_MARIA_FRACTION)
    level = np.quantile(fields.elevation[maria], 0.7) if maria.any() else 0.0
    fields.elevation[maria] = np.minimum(fields.elevation[maria], level)
    fields.terrain[maria] = Terrain.VOLCANIC_PLAIN
    late = add_crater_field(fields, rng, 0.1 * age_gyr * water_erosion, relief)
    return {"craters": craters + late, "maria_fraction": float(maria.mean())}
