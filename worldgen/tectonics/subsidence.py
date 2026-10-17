"""Tectonic subsidence of continental crust: stretched margins and forelands of mountain belts.

Rivers can only fill space that exists. Two processes make that space on
continental crust:

- Stretched margins (McKenzie 1978). When a continent rifts, the margin is
  stretched by β, which thins the crust (a fault-controlled subsidence S_i)
  and heats the lithosphere, which then cools and sinks over a thermal time
  constant τ = 62.8 Myr: S_t(t) = E₀ r (1 − e^(−t/τ)), r = (β/π) sin(π/β),
  E₀ = 3.2 km. Stretching is largest at the ocean and fades inland.
- Forelands (DeCelles & Giles 1996). The topographic load of a mountain belt
  bends the lithosphere beside it into a foredeep 100–300 km wide, the trough
  of a flexural profile w(x) = w₀ e^(−x/α) cos(x/α), α 26–150 km. The
  deflection follows the load, so the foreland rebounds as the belt erodes.

Both lower the continental surface; the river erosion that follows fills the
space with sediment, which is where the basin fill of the crust comes from.
"""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from ..grid import SphereGrid
from ..surface.distance import distance_from
from .crust import Crust

# Lithosphere and densities of McKenzie (1978, Table 1).
LITHOSPHERE_M = 125e3
RHO_MANTLE = 3330.0
RHO_CRUST = 2800.0
RHO_WATER = 1000.0
EXPANSION = 3.28e-5
T_ASTHENOSPHERE = 1333.0
THERMAL_TIME_MYR = 62.8
E0_M = 3200.0


def stretch_ratio(beta: np.ndarray) -> np.ndarray:
    """Return r = (β/π) sin(π/β), the share of the full thermal subsidence a stretch of β produces."""
    beta = np.maximum(beta, 1.0)
    return beta / np.pi * np.sin(np.pi / beta)


def initial_subsidence(beta: np.ndarray, crust_m: float = h.CONTINENTAL_CRUST_M) -> np.ndarray:
    """Return the fault-controlled subsidence S_i (m, water-loaded) of crust of thickness ``crust_m`` stretched by β."""
    a, tc, at = LITHOSPHERE_M, crust_m, EXPANSION * T_ASTHENOSPHERE
    numerator = a * ((RHO_MANTLE - RHO_CRUST) * tc / a * (1 - at * tc / (2 * a)) - at * RHO_MANTLE / 2)
    return numerator * (1 - 1 / np.maximum(beta, 1.0)) / (RHO_MANTLE * (1 - at) - RHO_WATER)


def thermal_subsidence(beta: np.ndarray, age_myr: np.ndarray) -> np.ndarray:
    """Return the thermal subsidence S_t (m, water-loaded) of a margin stretched by β, ``age_myr`` after stretching."""
    return E0_M * stretch_ratio(beta) * (1 - np.exp(-np.maximum(age_myr, 0.0) / THERMAL_TIME_MYR))


def _nearest_ocean(grid: SphereGrid, crust: Crust, radius_km: float):
    """Return, for each point, the distance (km) to the nearest oceanic point and that point's index."""
    ocean = np.flatnonzero(~crust.continental)
    return distance_from(grid, ocean, radius_km, h.MARGIN_STRETCH_REACH_KM)


def stretch_margins(crust: Crust, grid: SphereGrid, radius_km: float, interval_myr: float, first: bool) -> None:
    """Mark newly rifted continental margins as stretched and give them their fault-controlled subsidence.

    A margin is new when the sea floor beside it is younger than the re-map
    interval; on the first call every passive margin counts, dated by the sea
    floor beside it. Margins above an active subduction front are not stretched.
    """
    dist, nearest = _nearest_ocean(grid, crust, radius_km)
    cont = crust.continental
    reach = cont & np.isfinite(dist) & (crust.stretch == 0)
    ocean_age = np.where(nearest >= 0, crust.ocean_age[np.maximum(nearest, 0)], np.inf)
    new = reach & ((ocean_age <= interval_myr) | first)
    new &= ~(crust.front_age <= h.MARGIN_ACTIVE_FRONT_MYR)
    if not new.any():
        return
    stretch = (h.MARGIN_BETA - 1.0) * np.exp(-dist[new] / h.MARGIN_STRETCH_WIDTH_KM)
    keep = stretch >= h.MARGIN_MIN_STRETCH
    idx = np.flatnonzero(new)[keep]
    crust.stretch[idx] = stretch[keep]
    crust.margin_age[idx] = np.where(first, ocean_age[idx], 0.0)
    crust.cont_elev[idx] -= initial_subsidence(1.0 + crust.stretch[idx])


def cool_margins(crust: Crust, interval_myr: float) -> None:
    """Let stretched margins age and sink as their lithosphere cools."""
    idx = np.flatnonzero(crust.continental & (crust.stretch > 0))
    crust.margin_age[idx] += interval_myr
    target = thermal_subsidence(1.0 + crust.stretch[idx], crust.margin_age[idx])
    crust.cont_elev[idx] -= target - crust.thermal_subsidence[idx]
    crust.thermal_subsidence[idx] = target


def _trough(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Return the mean of e^(−u) cos u over [x0, x1] (in units of α), clipped to the trough u ≤ π/2."""
    lo = np.clip(x0, 0.0, np.pi / 2)
    hi = np.clip(x1, 0.0, np.pi / 2)

    def primitive(u):
        return np.exp(-u) * (np.sin(u) - np.cos(u)) / 2

    width = np.maximum(x1 - x0, 1e-12)
    return (primitive(hi) - primitive(lo)) / width


def flex_forelands(crust: Crust, grid: SphereGrid, radius_km: float) -> None:
    """Bend continental crust beside mountain belts under their load, and let it rebound as the load goes.

    The flexural profile is averaged over each cell's width, so coarse grids
    keep the trough that lies within their first ring of cells.
    """
    cont = crust.continental
    load = np.where(cont, crust.cont_elev - h.CONTINENT_BASE_M, 0.0)
    belt = cont & (load > h.FLEXURE_MIN_LOAD_M)
    alpha = h.FLEXURAL_PARAMETER_KM
    spacing = grid.mean_spacing(radius_km)
    reach = np.pi / 2 * alpha + spacing
    dist, nearest = distance_from(grid, np.flatnonzero(belt), radius_km, reach)
    ok = cont & ~belt & (nearest >= 0)
    # A cell a distance x from the centre of the nearest belt cell spans x − s to x from the belt's edge.
    x = dist[ok]
    target = np.zeros(crust.size)
    target[ok] = h.FLEXURE_LOAD_RATIO * load[nearest[ok]] * _trough((x - spacing) / alpha, x / alpha)
    change = target - crust.flexure
    crust.cont_elev[cont] -= change[cont]
    crust.flexure[cont] = target[cont]


def unfilled_subsidence(crust: Crust) -> np.ndarray:
    """Return the subsidence (m) on continental crust that sediment has not yet filled."""
    made = np.where(crust.stretch > 0, initial_subsidence(1.0 + crust.stretch), 0.0)
    made = made + crust.thermal_subsidence + np.maximum(crust.flexure, 0.0)
    return np.where(crust.continental, np.maximum(made - crust.basin_fill, 0.0), 0.0)


def subside(crust: Crust, grid: SphereGrid, radius_m: float, interval_myr: float, first: bool) -> np.ndarray:
    """Apply one re-map interval of margin stretching, thermal subsidence and foreland flexure; return the lowering (m)."""
    radius_km = radius_m / 1e3
    before = crust.cont_elev.copy()
    stretch_margins(crust, grid, radius_km, interval_myr, first)
    cool_margins(crust, interval_myr)
    flex_forelands(crust, grid, radius_km)
    return np.where(crust.continental, before - crust.cont_elev, 0.0)
