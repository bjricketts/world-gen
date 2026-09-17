"""Vegetation on land: productivity, cover, albedo and moisture recycling.

Net primary productivity follows the Miami model (Lieth 1973):
NPP = min(3000 (1 − e^(−0.000664 P)), 3000 / (1 + e^(1.315 − 0.119 T))) in
g m⁻² yr⁻¹, with P in mm/yr and T in °C. For other biochemistries the
temperature is taken relative to their growth optimum, and growth stops
beyond their tolerance. Light scales productivity weakly. Cover rises with
productivity; vegetation darkens the land and lets moisture travel further
inland through transpiration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import heuristics as h
from ..state import BiosphereState

FREEZING_K = 273.15


@dataclass
class Vegetation:
    """Vegetation on the surface grid."""

    productivity: np.ndarray         # net primary productivity (g m⁻² yr⁻¹), 0 at sea and under ice
    cover: np.ndarray                # 0–1
    land_albedo: np.ndarray          # snow-free land albedo
    moisture_decay_km: np.ndarray    # distance over which air loses its moisture


def miami_productivity(temperature_k: np.ndarray, precipitation_m: np.ndarray,
                       optimum_k: float = h.EARTH_OPTIMUM_K, tolerance_k: float = h.EARTH_TOLERANCE_K) -> np.ndarray:
    """Return net primary productivity (g m⁻² yr⁻¹) from annual temperature and precipitation."""
    t_c = temperature_k - FREEZING_K - (optimum_k - h.EARTH_OPTIMUM_K)
    from_t = h.NPP_MAX / (1.0 + np.exp(np.clip(1.315 - 0.119 * t_c, -50, 50)))
    from_p = h.NPP_MAX * (1.0 - np.exp(-0.000664 * np.maximum(precipitation_m, 0.0) * 1000.0))
    heat = 1.0 / (1.0 + np.exp((temperature_k - optimum_k - tolerance_k) / 2.0))
    return np.minimum(from_t, from_p) * heat


def light_factor(instellation_earth: float) -> float:
    """Return the productivity scale from the light the planet receives."""
    lo, hi = h.LIGHT_FACTOR_RANGE
    return float(np.clip(instellation_earth ** h.LIGHT_EXPONENT, lo, hi))


def bare_albedo(precipitation_m: np.ndarray, evaporation_m: np.ndarray) -> np.ndarray:
    """Return the albedo of ground without plants: brighter where it is dry."""
    aridity = np.clip(1.0 - precipitation_m / np.maximum(evaporation_m, 1e-6), 0.0, 1.0)
    return h.SOIL_ALBEDO + h.DESERT_BRIGHTENING * aridity


def build_vegetation(temperature_k: np.ndarray, precipitation_m: np.ndarray, evaporation_m: np.ndarray,
                     ocean: np.ndarray, glacier: np.ndarray, life: BiosphereState | None,
                     instellation_earth: float) -> Vegetation:
    """Return the vegetation of a surface from its annual climate; without surface life, bare ground."""
    bare = bare_albedo(precipitation_m, evaporation_m)
    land = ~ocean & ~glacier
    if life is None or life.life != "surface":
        zero = np.zeros(ocean.size)
        return Vegetation(productivity=zero, cover=zero, land_albedo=bare,
                          moisture_decay_km=np.full(ocean.size, h.MOISTURE_DECAY_BARE_KM))
    npp = miami_productivity(temperature_k, precipitation_m, life.optimum_temperature_k,
                             life.temperature_tolerance_k) * light_factor(instellation_earth)
    npp = np.where(land, npp, 0.0)
    cover = np.clip(npp / h.NPP_FULL_COVER, 0.0, 1.0)
    albedo = bare + (life.vegetation_albedo - bare) * cover
    decay = h.MOISTURE_DECAY_BARE_KM + (h.MOISTURE_DECAY_VEGETATED_KM - h.MOISTURE_DECAY_BARE_KM) * cover
    return Vegetation(productivity=npp, cover=cover, land_albedo=albedo, moisture_decay_km=decay)


def productive_share(vegetation: Vegetation, ocean: np.ndarray) -> float:
    """Return the share of land with productivity above the productive threshold."""
    land = ~ocean
    if not land.any():
        return 0.0
    return float((vegetation.productivity[land] > h.PRODUCTIVE_NPP).mean())
