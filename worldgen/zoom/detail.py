"""Synthesise sub-grid relief when a region is zoomed (milestone 7).

The global surface resolves relief down to its grid spacing; zoom continues its
relief spectrum below that, one octave at a time down to the region's own
resolution. An octave of wavelength λ has a standard deviation that grows as
λ^H and is set by how rugged the ground is (mountains hundreds of metres,
plains a few), so a region shows the relief a real landscape of that kind has
at those scales. The octaves are fixed functions of position and the planet's
seed, starting at the global grid spacing whatever the zoom level: regions
agree where they meet, any region regenerates identically, and zooming deeper
only adds finer octaves to the same field. The sample positions are
domain-warped so valleys meander instead of following the lattice, orogens
take a ridged character, and the inherited structural fabric stretches relief
along the grain. The coastline is refined for free: the detailed elevation
crossing sea level gives a crenellated shore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..noise import fbm, warp
from .inherit import InheritedRegion, connected_sea, downscale_climate, inherit_region
from .tiles import RegionGrid

# Amplitude of the synthesised relief relative to the Earth-like calibration.
RUGGEDNESS = {"earth": 1.0, "dramatic": 1.9, "gentle": 0.45}
# Statistics of one noise octave (Perlin / 0.7), used to give every octave unit variance.
OCTAVE_STD = 0.384
RIDGE_MEAN, RIDGE_STD = 0.687, 0.223


def _detail_octaves(coarse_rad: float, fine_rad: float) -> int:
    """Return how many octaves fit between a coarse wavelength and twice a lattice spacing."""
    octaves = int(np.floor(np.log2(max(coarse_rad / (2.0 * max(fine_rad, 1e-30)), 1e-30)))) + 1
    return int(np.clip(octaves, h.ZOOM_DETAIL_MIN_OCTAVES, h.ZOOM_DETAIL_MAX_OCTAVES))


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


def synthesize_detail(world, region: RegionGrid, inherited: InheritedRegion, ruggedness: str = "earth") -> np.ndarray:
    """Return the sub-grid relief (m) to add to the inherited coarse elevation.

    ``ruggedness`` ('earth', 'dramatic' or 'gentle') scales the relief.
    """
    seed = world.state.draw_seed
    radius_km = inherited.radius_m / 1e3
    scale = RUGGEDNESS.get(ruggedness, 1.0) * float(world.surface.attrs.get("relief_factor", 1.0))
    coarse = world.grid.mean_spacing()                        # the first octave's wavelength (radians)
    # The nominal lattice spacing depends only on the level, so regions at one level add the same octaves.
    spacing = (np.pi / 2.0) / (1 << region.level) / (region.nodes_per_tile - 1)
    octaves = _detail_octaves(coarse, spacing)
    points = warp(region.points, seed, "zoom.warp", strength=h.ZOOM_WARP_FRACTION * coarse, frequency=1.0 / coarse)

    rugged = inherited.ruggedness
    sigma_ref = np.where(inherited.ocean,
                         h.ZOOM_RELIEF_ABYSSAL_M + (h.ZOOM_RELIEF_SEAFLOOR_M - h.ZOOM_RELIEF_ABYSSAL_M) * rugged,
                         h.ZOOM_RELIEF_PLAIN_M + (h.ZOOM_RELIEF_MOUNTAIN_M - h.ZOOM_RELIEF_PLAIN_M) * rugged)
    strength = np.clip(np.linalg.norm(inherited.fabric, axis=1), 0.0, 1.0)
    detail = np.zeros(region.size)
    for k in range(octaves):
        wavelength = coarse / 2**k
        noise = fbm(points, seed, f"zoom.octave.{k}", frequency=1.0 / wavelength, octaves=1)
        smooth = noise / OCTAVE_STD
        crest = ((1.0 - np.abs(noise)) - RIDGE_MEAN) / RIDGE_STD      # sharp crests where there is a grain
        sigma = sigma_ref * (wavelength * radius_km / h.ZOOM_RELIEF_REF_KM) ** h.ZOOM_HURST
        detail += sigma * ((1.0 - strength) * smooth + strength * crest)
    return scale * _elongate(region, detail, inherited.fabric, strength)


@dataclass
class DetailedRegion:
    """A zoomed region with its synthesised detail, refined coastline and downscaled climate."""

    grid: RegionGrid
    radius_m: float
    seed: int
    elevation: np.ndarray                        # coarse elevation plus detail, sea level = 0
    detail: np.ndarray                           # the added sub-grid relief
    ocean: np.ndarray                            # refined coastline: detailed ground below sea level open to the sea
    fabric: np.ndarray
    temperature_k: np.ndarray
    wind: np.ndarray
    precipitation_m: Optional[np.ndarray] = None
    evaporation_m: Optional[np.ndarray] = None
    categorical: Optional[dict] = None
    ice_sheet: Optional[np.ndarray] = None       # share of the surrounding global cells under an ice sheet (0–1)
    ice_top_m: Optional[np.ndarray] = None       # global ice-sheet surface above sea level
    snowline_m: Optional[np.ndarray] = None      # present equilibrium-line altitude
    sea_ice: Optional[np.ndarray] = None


HALO = h.ZOOM_DETAIL_SMOOTH_STEPS + 1     # lattice nodes beyond the region that its edge values depend on


def synthesize_region(world, region: RegionGrid, ruggedness: str = "earth") -> DetailedRegion:
    """Return the zoomed region with sub-grid relief added and its climate downscaled to it.

    The work is done on the region plus a halo of lattice nodes and then
    cropped: the grain smoothing and the local rain shadow look a few nodes
    around each node, and the halo gives the region's edge the same
    neighbourhood a neighbouring region sees, so two regions agree exactly
    where they meet.
    """
    padded = region.padded(HALO)
    inner = region.inner_index(HALO)
    inherited = inherit_region(world, padded)
    detail = synthesize_detail(world, padded, inherited, ruggedness)
    elevation = inherited.elevation + detail
    temperature, precipitation, evaporation = downscale_climate(inherited, elevation)

    def crop(values):
        return None if values is None else values[inner]

    return DetailedRegion(
        grid=region,
        radius_m=inherited.radius_m,
        seed=world.state.draw_seed,
        elevation=elevation[inner],
        detail=detail[inner],
        ocean=connected_sea(padded, elevation, inherited.sea)[inner],
        fabric=inherited.fabric[inner],
        temperature_k=temperature[inner],
        wind=inherited.wind[inner],
        precipitation_m=crop(precipitation),
        evaporation_m=crop(evaporation),
        categorical={name: values[inner] for name, values in inherited.categorical.items()},
        ice_sheet=crop(inherited.ice_sheet),
        ice_top_m=crop(inherited.ice_top_m),
        snowline_m=crop(inherited.snowline_m),
        sea_ice=crop(inherited.sea_ice),
    )
