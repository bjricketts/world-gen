"""Glaciers, ice sheets and glacial erosion on a zoomed region (milestone 7).

Ice is built on the region from the local mass balance, which gains ice above
the snowline and loses it below (Hergarten 2021). The ice flux is carried down
the ice surface to every lower neighbour, as in balance-flux calculations for
ice sheets (Le Brocq et al. 2006), so an ice cap spreads to a lobed margin and
feeds many outlet glaciers rather than one. Ground is under ice wherever flux
reaches it, and ice fills each valley across to the height of its surface, so
a tongue melts over its full width. Where the planet grid has an ice sheet,
its surface is a floor for the local ice surface, blended in with the share of
global cells under the sheet so its margin follows the local ice rather than
the planet grid's triangles; detailed bedrock above it shows as nunataks.

The ice surface is the perfectly plastic profile (Nye 1952), built inward from
the ice margins as in glacier reconstructions (Benn & Hulton 2010) and taking
the lowest surface any margin allows. It rises steeply near a margin and
flattens over thick ice, so neighbouring glaciers and ice caps share one
continuous surface.

Glacial erosion follows the glacial stream-power law, E = K_g Q_iᵐ S, where
S is the slope of the ice surface (Hergarten 2021; Liebl et al. 2023). Because
the ice surface and not the bed sets the slope, a glacier can deepen its bed
below the ground further down its valley, most where the ice thins toward its
snout; those overdeepenings hold lakes once the ice has gone. Their depth is
bounded because an adverse bed slope much steeper than the ice surface stops
eroding (Alley et al. 2003).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit

from .. import heuristics as h
from ..climate.ice import ICE_DENSITY
from ..constants import G_EARTH
from ..hydrology.graph import accumulate, priority_flood, spread_positive, steepest_receivers, upstream_order
from .tiles import RegionGrid



@dataclass
class IceCover:
    """Ice on a region's lattice, and the ice-flow network under it."""

    thickness_m: np.ndarray      # ice above the bed (0 where the ground is bare)
    flux_m3: np.ndarray          # ice flux through each node (m³ water equivalent per year)
    surface_m: np.ndarray        # ice surface where there is ice, the bed elsewhere (depressions filled)
    receiver: np.ndarray         # where each node's ice flows
    distance_m: np.ndarray       # length of that step
    order: np.ndarray            # nodes ordered so each comes after its receiver

    @property
    def covered(self) -> np.ndarray:
        """Return the nodes under ice."""
        return self.thickness_m > 0.0


def mass_balance(surface_m: np.ndarray, snowline_m: np.ndarray, precipitation_m: np.ndarray) -> np.ndarray:
    """Return the yearly ice gain (m of water; negative where ice melts) at each height.

    Above the snowline the share of precipitation kept as ice rises linearly
    to all of it ``ZOOM_GLACIER_FULL_ACCUMULATION_M`` higher (Hergarten 2021).
    Below it ice melts by degree days, faster the lower it lies.
    """
    above = surface_m - snowline_m
    gain = precipitation_m * np.clip(above / h.ZOOM_GLACIER_FULL_ACCUMULATION_M, 0.0, 1.0)
    melt_per_m = h.DEGREE_DAY_FACTOR_M * h.LAPSE_RATE_K_PER_M * h.ZOOM_MELT_SEASON_DAYS
    return np.where(above >= 0.0, gain, melt_per_m * above)


def glacial_cooling_k(world) -> float:
    """Return how much colder than now the coldest remembered epoch was (0 without a history record)."""
    summary = getattr(world.state, "surface", None)
    features = getattr(summary, "features", None) or {}
    return float(features.get("relict_glacial_cooling_k", 0.0))


@njit(cache=True)
def _plastic_surface(indptr, indices, lengths, bed, ice, start, scale, floor):
    """Return the lowest perfectly plastic ice surface over the ``ice`` nodes that rises from their margins.

    Along each step of length L the thickness grows as H² → H² + 2 (τ₀/ρg) L
    (``scale`` = τ₀/ρg). The surface spreads inward from bare ground and from
    the ``start`` surface given at ice nodes where ice leaves the region, always
    taking the lowest height any neighbour allows, so it has no steps where
    flow paths from different margins meet. The ice is at least ``floor`` thick.
    """
    n = bed.size
    surface = np.full(n, np.inf)
    done = np.zeros(n, dtype=np.bool_)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for i in range(n):
        if not ice[i]:
            surface[i] = bed[i]
            heapq.heappush(heap, (bed[i], np.int64(i)))
        elif np.isfinite(start[i]):
            surface[i] = start[i]
            heapq.heappush(heap, (start[i], np.int64(i)))
    while heap:
        level, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        below = level - bed[i] if ice[i] else 0.0
        for m in range(indptr[i], indptr[i + 1]):
            j = indices[m]
            if done[j] or not ice[j]:
                continue
            rise = np.sqrt(below * below + 2.0 * scale * lengths[m]) - below
            cand = max(level + rise, bed[j] + floor)
            if cand < surface[j]:
                surface[j] = cand
                heapq.heappush(heap, (cand, np.int64(j)))
    for i in range(n):
        if not np.isfinite(surface[i]):
            surface[i] = bed[i] + floor
    return surface


@njit(cache=True)
def _fill_across(indptr, indices, bed, top, covered, floor, passes):
    """Return the ice cover and surface after ice fills each valley across to the height of its surface.

    A bare node beside ice, higher than that ice's bed but more than ``floor``
    below its surface, is on the valley side under that ice. Ice reaches lower
    ground only by flowing there. ``passes`` bounds how far across a valley the
    ice reaches.
    """
    cover = covered.copy()
    surface = top.copy()
    for _ in range(passes):
        grown = cover.copy()
        level = surface.copy()
        for i in range(bed.size):
            if cover[i]:
                continue
            for m in range(indptr[i], indptr[i + 1]):
                j = indices[m]
                if cover[j] and bed[i] >= bed[j] and surface[j] - bed[i] > floor:
                    grown[i] = True
                    level[i] = max(level[i], surface[j])
        cover = grown
        surface = level
    return cover, surface


@njit(cache=True)
def _smooth_over(indptr, indices, values, mask, passes):
    """Return ``values`` averaged with their ``mask`` neighbours over ``passes`` passes, only on ``mask`` nodes."""
    out = values.copy()
    for _ in range(passes):
        prev = out.copy()
        for i in range(values.size):
            if not mask[i]:
                continue
            total = prev[i]
            count = 1
            for m in range(indptr[i], indptr[i + 1]):
                j = indices[m]
                if mask[j]:
                    total += prev[j]
                    count += 1
            out[i] = total / count
    return out


@njit(cache=True)
def _glacial_power(order, receiver, distance, power, z, surface, ice, adverse):
    """Lower each glaciated node's bed so its ice surface falls toward its receiver's.

    ``power`` is K_g·dt·Q^m (m); ``surface`` is the ice surface and ``ice``
    the thickness, both held fixed over the step. Solved implicitly downstream-first (Braun & Willett 2013).
    Unlike river incision the bed may end below its receiver's, but an adverse
    bed slope steeper than ``adverse`` times the ice-surface slope stops
    eroding, as meltwater freezes on climbing it (Alley et al. 2003).
    """
    for k in range(order.size):
        i = order[k]
        r = receiver[i]
        drop = surface[i] - surface[r]
        if r == i or power[i] <= 0.0 or drop <= 0.0:
            continue
        f = power[i] / distance[i]
        new = (z[i] + f * (z[r] + ice[r] - ice[i])) / (1.0 + f)
        z[i] = min(z[i], max(new, z[r] - adverse * drop))
    return z


def ice_cover(region: RegionGrid, radius_m: float, bed_m: np.ndarray, snowline_m: np.ndarray,
              precipitation_m: Optional[np.ndarray], ocean: np.ndarray, edge: np.ndarray, cell_area_m2: float,
              sheet: Optional[np.ndarray] = None, sheet_top_m: Optional[np.ndarray] = None,
              gravity_m_s2: float = G_EARTH) -> IceCover:
    """Return the ice on a region from the local mass balance, floored by any inherited ice sheet.

    Ice flows down the ice surface and leaves at the sea (calving) and across
    the region's edge. ``sheet`` is the share of global cells under an ice
    sheet and ``sheet_top_m`` that sheet's surface. The ice surface feeds back
    on the mass balance and the flow directions, so both are solved together
    over a few passes. Without precipitation only the inherited sheet is kept.
    """
    adj = region.neighbours
    outlet = ocean | edge
    scale = h.ICE_YIELD_STRESS_PA / (ICE_DENSITY * gravity_m_s2)
    inherited = np.zeros(region.size)
    if sheet is not None:
        # The planet grid's sheet surface is linear within each of its triangles; smooth away the creases.
        # Toward the sheet's edge its surface is blended with the broad shape of the bed (not the bed itself,
        # which would print every valley into the ice surface).
        everywhere = np.ones(region.size, dtype=bool)
        passes = h.ZOOM_ICE_SHEET_SMOOTH_PASSES
        floor_top = _smooth_over(adj.indptr, adj.indices, sheet_top_m, everywhere, passes)
        broad_bed = _smooth_over(adj.indptr, adj.indices, bed_m, everywhere, passes)
        inherited = np.maximum(sheet * floor_top + (1.0 - sheet) * broad_bed - bed_m, 0.0) * (sheet > 0.0)
    under_sheet = (inherited >= h.ZOOM_GLACIER_MIN_THICKNESS_M) & ~ocean
    if sheet is not None:
        under_sheet &= sheet >= 0.5
    covered = under_sheet.copy()
    top = bed_m + inherited
    flux = np.zeros(region.size)
    receiver = distance = order = None
    for _ in range(h.ZOOM_ICE_ITERATIONS):
        surface = priority_flood(np.where(covered, top, bed_m), adj.indptr, adj.indices, outlet, 1e-3)
        receiver, distance = steepest_receivers(surface, adj.indptr, adj.indices, adj.data, outlet)
        distance = distance * radius_m
        order = upstream_order(receiver)
        if precipitation_m is None:
            break
        gain = np.where(outlet, 0.0, mass_balance(surface, snowline_m, precipitation_m) * cell_area_m2)
        flux = spread_positive(np.argsort(-surface, kind="stable"), adj.indptr, adj.indices, adj.data, surface, gain,
                               h.ZOOM_ICE_SPREAD_EXPONENT, outlet)
        flux[ocean] = 0.0                                        # ice reaching the sea calves
        covered = ((flux > 0.0) & ~ocean) | under_sheet
        # Ice leaving across the edge starts there as thin as at a margin, unless an ice sheet continues beyond.
        start = np.where(edge & covered, bed_m + np.maximum(inherited, h.ZOOM_GLACIER_MIN_THICKNESS_M), np.inf)
        top = _plastic_surface(adj.indptr, adj.indices, adj.data * radius_m, bed_m, covered, start, scale,
                               h.ZOOM_GLACIER_MIN_THICKNESS_M)
        top = np.maximum(top, bed_m + inherited)
        covered, top = _fill_across(adj.indptr, adj.indices, bed_m, top, covered & ~ocean,
                                    h.ZOOM_GLACIER_MIN_THICKNESS_M, h.ZOOM_ICE_FILL_PASSES)
        covered &= ~ocean
    thickness = np.where(covered, np.maximum(top - bed_m, 0.0), 0.0)
    surface = priority_flood(bed_m + thickness, adj.indptr, adj.indices, outlet, 1e-3)
    return IceCover(thickness_m=thickness, flux_m3=flux, surface_m=surface, receiver=receiver,
                    distance_m=distance, order=order)


def glaciate(region: RegionGrid, radius_m: float, bed_m: np.ndarray, snowline_m: np.ndarray,
             precipitation_m: np.ndarray, ocean: np.ndarray, edge: np.ndarray, cell_area_m2: float,
             duration_myr: float = h.ZOOM_GLACIAL_MYR, step_myr: float = h.ZOOM_GLACIAL_STEP_MYR,
             sheet: Optional[np.ndarray] = None, sheet_top_m: Optional[np.ndarray] = None,
             gravity_m_s2: float = G_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """Return the bed after ``duration_myr`` of glacial erosion at ``snowline_m``, and the depth eroded.

    The eroded rock leaves as glacial sediment; moraines are not built.
    """
    steps = max(int(np.ceil(duration_myr / step_myr)), 1)
    dt = duration_myr / steps
    z = bed_m.astype(float).copy()
    for _ in range(steps):
        ice = ice_cover(region, radius_m, z, snowline_m, precipitation_m, ocean, edge, cell_area_m2, sheet, sheet_top_m,
                        gravity_m_s2)
        glacier = ice.flux_m3 > 0.0
        if not glacier.any():
            break
        # The water flux weights the ice flux so that parallel swaths do not merge into sheets (Hergarten 2021).
        water = accumulate(ice.order, ice.receiver, np.maximum(precipitation_m, 0.0) * cell_area_m2)
        share = h.ZOOM_GLACIAL_WATER_SHARE
        effective = water ** share * ice.flux_m3 ** (1.0 - share)
        power = np.where(glacier & ~ocean & ~edge, h.ZOOM_GLACIAL_K * dt * effective ** h.STREAM_POWER_M, 0.0)
        z = _glacial_power(ice.order, ice.receiver, ice.distance_m, power, z, ice.surface_m, ice.surface_m - z,
                           h.ZOOM_ADVERSE_SLOPE_RATIO)
    return z, bed_m - z
