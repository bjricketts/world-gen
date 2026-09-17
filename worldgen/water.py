"""Water inventory: how a planet's water divides between mantle and surface, and how much land it leaves.

The spec's ``body.water_mass_fraction`` is the planet's total water. On
plate-tectonic planets the mantle share follows the steady-state water cycle
of Cowan & Abbot (2014); other planets keep a fixed heuristic share in the
mantle. The land fraction estimate uses a reference Earth-like hypsometry
and is refined once the surface is built.
"""

from __future__ import annotations

import numpy as np

from . import constants as c
from . import heuristics as h
from .state import BulkState, WaterState

# Reference hypsometries: land fraction against the ocean volume per unit
# planet area (m) at Earth gravity. The plate-tectonic curve comes from
# simulated, eroded Earth-like surfaces; the others from the heuristic regime
# surfaces (three seeds each, standard resolution).
_REF_LAND = {
    "mobile_lid": [0.0, 0.05, 0.09, 0.11, 0.13, 0.17, 0.23, 0.28, 0.36, 0.46, 0.64, 0.80, 0.86, 0.88, 0.96, 1.0],
}
_REF_DEPTH_M = {
    "mobile_lid": [7500.0, 6000.0, 5000.0, 4300.0, 3800.0, 3400.0, 3000.0, 2600.0, 2200.0, 1800.0, 1400.0,
                   1000.0, 700.0, 400.0, 150.0, 0.0],
}
_LID_LAND = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
_LID_DEPTH_M = {
    "stagnant_lid": [4500.0, 3600.0, 3200.0, 2780.0, 2440.0, 2150.0, 1970.0, 1780.0, 1550.0, 1330.0, 830.0, 500.0,
                     320.0, 190.0, 50.0, 20.0, 0.0],
    "episodic": [5500.0, 4680.0, 4040.0, 3420.0, 2720.0, 2000.0, 1700.0, 1360.0, 610.0, 385.0, 350.0, 310.0,
                 160.0, 27.0, 21.0, 2.0, 0.0],
    "heat_pipe": [4500.0, 3490.0, 750.0, 570.0, 470.0, 400.0, 340.0, 290.0, 250.0, 220.0, 160.0, 110.0, 94.0,
                  57.0, 22.0, 16.0, 0.0],
    "inactive": [3500.0, 2750.0, 2340.0, 1950.0, 1700.0, 1440.0, 1220.0, 1080.0, 930.0, 810.0, 580.0, 420.0,
                 290.0, 180.0, 61.0, 29.0, 0.0],
}
for _regime, _depths in _LID_DEPTH_M.items():
    _REF_LAND[_regime] = _LID_LAND
    _REF_DEPTH_M[_regime] = _depths


def gravity_ratio(bulk: BulkState) -> float:
    """Return surface gravity relative to Earth's."""
    return bulk.surface_gravity_m_s2 / c.G_EARTH


def mantle_water(total: float, g_ratio: float, plate_cycling: bool) -> float:
    """Return the water held in the mantle as a fraction of planet mass."""
    if total <= 0.0:
        return 0.0
    fm = h.MANTLE_MASS_FRACTION
    if not plate_cycling:
        return min(h.NONPLATE_MANTLE_SHARE * total, h.MANTLE_WATER_EARTH * fm)

    def excess(x: float) -> float:
        """Return how far a mantle water content exceeds its steady-state value."""
        pressure = g_ratio**2 * max(total - fm * x, 0.0) / h.SURFACE_WATER_EARTH
        return x - h.MANTLE_WATER_EARTH * pressure**h.WATER_PRESSURE_EXPONENT

    lo, hi = 0.0, min(h.MANTLE_WATER_MAX, total / fm)
    if excess(hi) <= 0.0:
        return hi * fm
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi) * fm


def total_water_for_surface(surface: float, g_ratio: float, plate_cycling: bool) -> float:
    """Return the total water fraction that leaves ``surface`` (fraction of planet mass) at the surface."""
    if surface <= 0.0:
        return 0.0
    if not plate_cycling:
        share = h.NONPLATE_MANTLE_SHARE
        cap = h.MANTLE_WATER_EARTH * h.MANTLE_MASS_FRACTION
        return surface + min(surface * share / (1.0 - share), cap)
    pressure = g_ratio**2 * surface / h.SURFACE_WATER_EARTH
    x = min(h.MANTLE_WATER_EARTH * pressure**h.WATER_PRESSURE_EXPONENT, h.MANTLE_WATER_MAX)
    return surface + h.MANTLE_MASS_FRACTION * x


def _reference(regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference land fractions and ocean depths for a tectonic regime."""
    key = regime if regime in _REF_LAND else "mobile_lid"
    return np.asarray(_REF_LAND[key]), np.asarray(_REF_DEPTH_M[key])


def land_fraction_estimate(ocean_volume_m3: float, radius_m: float, relief: float,
                           regime: str = "mobile_lid") -> float:
    """Return the land fraction an ocean of this volume leaves on a typical surface of the given regime.

    Basin depths scale with the relief factor (1 at Earth gravity).
    """
    land, depths = _reference(regime)
    depth = ocean_volume_m3 / (4 * np.pi * radius_m**2) / max(relief, 1e-6)
    return float(np.interp(depth, depths[::-1], land[::-1]))


def volume_for_land_fraction(land_fraction: float, radius_m: float, relief: float,
                             regime: str = "mobile_lid") -> float:
    """Return the ocean volume (m³) that leaves ``land_fraction`` dry on a typical surface of the given regime."""
    land, depths = _reference(regime)
    depth = float(np.interp(land_fraction, land, depths))
    return depth * max(relief, 1e-6) * 4 * np.pi * radius_m**2


def build_water(bulk: BulkState, regime: str, relief: float) -> WaterState:
    """Return the water inventory of a planet with the given tectonic regime."""
    plate_cycling = regime == "mobile_lid"
    total = bulk.water_mass_fraction
    g_ratio = gravity_ratio(bulk)
    mantle = mantle_water(total, g_ratio, plate_cycling)
    surface = max(total - mantle, 0.0)
    volume = surface * bulk.mass_kg / h.SEAWATER_DENSITY
    return WaterState(
        total_mass_fraction=total,
        mantle_mass_fraction=mantle,
        surface_mass_kg=surface * bulk.mass_kg,
        ocean_volume_m3=volume,
        seafloor_pressure_ratio=g_ratio**2 * surface / h.SURFACE_WATER_EARTH,
        plate_cycling=plate_cycling,
        land_fraction_estimate=land_fraction_estimate(volume, bulk.radius_m, relief, regime)
        if surface > h.SURFACE_WATER_MIN_FRACTION else 1.0,
    )


def has_surface_water(water: WaterState, bulk: BulkState) -> bool:
    """Return whether the planet has enough surface water to count."""
    return water.surface_mass_kg / bulk.mass_kg > h.SURFACE_WATER_MIN_FRACTION
