"""Synthesise sub-grid relief when a region is zoomed (milestone 7).

The global surface resolves relief down to the grid spacing; zoom adds the
detail below it. Detail is fractal noise (``worldgen.noise``) evaluated at the
region's own points, so it is a pure function of position and the planet's
seed: it continues the global surface seamlessly and any region regenerates
identically. Landform type sets its amplitude — rough hills on land, gentler
texture on the sea floor — and the inherited structural fabric orients it, so
mountain and abyssal-hill relief runs along the grain rather than as isotropic
blobs. The coastline is refined for free: the detailed elevation crossing sea
level gives a crenellated shore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..noise import fbm, ridged
from .inherit import InheritedRegion, downscale_climate, inherit_region
from .tiles import RegionGrid


def _detail_octaves(global_spacing_rad: float, region_spacing_rad: float) -> int:
    """Return how many octaves of detail to add: more the deeper the zoom."""
    ratio = max(global_spacing_rad / max(region_spacing_rad, 1e-30), 1.0)
    octaves = np.log2(ratio) + h.ZOOM_DETAIL_BASE_OCTAVES
    return int(np.clip(octaves, h.ZOOM_DETAIL_BASE_OCTAVES, h.ZOOM_DETAIL_MAX_OCTAVES))


def _elongate(region: RegionGrid, values: np.ndarray, fabric: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Return the field smoothed along the structural grain, which stretches features into ridges.

    Each pass blends a node with its neighbours in proportion to how well the
    edge to them aligns with the grain, weighted by the grain strength, so
    relief becomes constant along strike and stays sharp across it.
    """
    adj = region.neighbours
    rows = np.repeat(np.arange(region.size), np.diff(adj.indptr))
    cols = adj.indices
    edge = region.points[cols] - region.points[rows]
    edge -= np.einsum("ij,ij->i", edge, region.points[rows])[:, None] * region.points[rows]
    edge /= np.maximum(np.linalg.norm(edge, axis=1, keepdims=True), 1e-30)
    weight = np.einsum("ij,ij->i", edge, fabric[rows]) ** 2       # fabric carries the grain strength
    blend = np.clip(strength, 0.0, 1.0)
    out = values.copy()
    for _ in range(h.ZOOM_DETAIL_SMOOTH_STEPS):
        num = np.bincount(rows, weights=weight * out[cols], minlength=region.size)
        den = np.bincount(rows, weights=weight, minlength=region.size)
        # Include the node itself, weighted as much as its grain-aligned neighbours, for a true low-pass.
        aligned = np.where(den > 1e-12, (num + den * out) / (2.0 * np.where(den > 1e-12, den, 1.0)), out)
        out = out + blend * (aligned - out)
    return out


def synthesize_detail(world, region: RegionGrid, inherited: InheritedRegion) -> np.ndarray:
    """Return the sub-grid relief (m) to add to the inherited coarse elevation."""
    seed = world.state.draw_seed
    relief = float(world.surface.attrs.get("relief_factor", 1.0))
    points = region.points
    octaves = _detail_octaves(world.grid.mean_spacing(), region.mean_spacing())
    base_frequency = h.ZOOM_DETAIL_BASE_CELLS / world.grid.mean_spacing()

    rough = fbm(points, seed, "zoom.rough", frequency=base_frequency, octaves=octaves)
    ridge = ridged(points, seed, "zoom.ridge", frequency=base_frequency, octaves=max(octaves - 1, 3))
    strength = np.linalg.norm(inherited.fabric, axis=1)
    ridge = _elongate(region, ridge, inherited.fabric, strength)

    land = ~inherited.ocean
    amp_rough = h.ZOOM_DETAIL_ROUGH_M * relief * np.where(land, 1.0, h.ZOOM_OCEAN_ROUGH_FACTOR)
    amp_ridge = h.ZOOM_DETAIL_RIDGE_M * relief * strength
    return amp_rough * rough + amp_ridge * (ridge - h.ZOOM_RIDGED_MEDIAN)


@dataclass
class DetailedRegion:
    """A zoomed region with its synthesised detail, refined coastline and downscaled climate."""

    grid: RegionGrid
    radius_m: float
    seed: int
    elevation: np.ndarray                        # coarse elevation plus detail, sea level = 0
    detail: np.ndarray                           # the added sub-grid relief
    ocean: np.ndarray                            # refined coastline: detailed elevation below sea level
    fabric: np.ndarray
    temperature_k: np.ndarray
    wind: np.ndarray
    precipitation_m: Optional[np.ndarray] = None
    evaporation_m: Optional[np.ndarray] = None
    categorical: Optional[dict] = None


def synthesize_region(world, region: RegionGrid, inherited: Optional[InheritedRegion] = None) -> DetailedRegion:
    """Return the zoomed region with sub-grid relief added and its climate downscaled to it."""
    if inherited is None:
        inherited = inherit_region(world, region)
    detail = synthesize_detail(world, region, inherited)
    elevation = inherited.elevation + detail
    temperature, precipitation, evaporation = downscale_climate(inherited, elevation)
    return DetailedRegion(
        grid=region,
        radius_m=inherited.radius_m,
        seed=world.state.draw_seed,
        elevation=elevation,
        detail=detail,
        ocean=elevation < 0.0,
        fabric=inherited.fabric,
        temperature_k=temperature,
        wind=inherited.wind,
        precipitation_m=precipitation,
        evaporation_m=evaporation,
        categorical=inherited.categorical,
    )
