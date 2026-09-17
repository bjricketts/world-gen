"""Climate classes and biomes.

Köppen–Geiger classes follow the rules of Peel et al. (2007) on monthly
temperature and precipitation. Biomes follow a Whittaker-type diagram of
annual temperature and precipitation; for other biochemistries the
temperature axis is shifted by the difference between their growth
optimum and Earth's. Lifeless land is barren, and ice sheets form their
own class in both schemes.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np

from .. import heuristics as h
from ..state import BiosphereState

FREEZING_K = 273.15

KOPPEN_CODES = ("", "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk",
                "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc",
                "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb", "Dfc", "Dfd",
                "ET", "EF")
KOPPEN_INDEX = {code: i for i, code in enumerate(KOPPEN_CODES)}


class Biome(IntEnum):
    """Biome classes (0 is the ocean)."""

    OCEAN = 0
    ICE_SHEET = 1
    TUNDRA = 2
    BOREAL_FOREST = 3
    COLD_DESERT = 4
    TEMPERATE_GRASSLAND = 5
    WOODLAND_SHRUBLAND = 6
    TEMPERATE_FOREST = 7
    TEMPERATE_RAINFOREST = 8
    HOT_DESERT = 9
    SAVANNA = 10
    TROPICAL_SEASONAL_FOREST = 11
    TROPICAL_RAINFOREST = 12
    BARREN = 13
    POLAR_DESERT = 14


def summer_months(northern: np.ndarray, months: int = 12) -> np.ndarray:
    """Return a (months, cells) mask of the warmer half-year: months 4–9 in the north, the rest in the south."""
    north = (np.arange(months) >= 3) & (np.arange(months) < 9)
    return np.where(northern[None, :], north[:, None], ~north[:, None])


def koppen(monthly_k: np.ndarray, monthly_p_m: np.ndarray, northern: np.ndarray, glaciated: np.ndarray) -> np.ndarray:
    """Return Köppen–Geiger class indices (``KOPPEN_CODES``) from monthly temperature and precipitation rates."""
    t = monthly_k - FREEZING_K
    p = monthly_p_m * 1000.0 / 12.0                 # mm per month
    mat = t.mean(axis=0)
    map_ = p.sum(axis=0)
    t_hot, t_cold = t.max(axis=0), t.min(axis=0)
    warm_months = (t >= 10.0).sum(axis=0)
    summer = summer_months(northern, t.shape[0])
    p_summer = np.where(summer, p, 0.0)
    p_winter = np.where(~summer, p, 0.0)
    ps_total, pw_total = p_summer.sum(axis=0), p_winter.sum(axis=0)
    ps_dry = np.where(summer, p, np.inf).min(axis=0)
    ps_wet = p_summer.max(axis=0)
    pw_dry = np.where(~summer, p, np.inf).min(axis=0)
    pw_wet = p_winter.max(axis=0)
    p_dry = p.min(axis=0)

    threshold = np.where(pw_total >= 0.7 * map_, 2 * mat,
                         np.where(ps_total >= 0.7 * map_, 2 * mat + 28, 2 * mat + 14))
    out = np.zeros(t.shape[1], dtype=np.int8)

    # Temperate and cold: second letter from dry season, third from summer warmth.
    dry_summer = (ps_dry < 40) & (ps_dry < pw_wet / 3)
    dry_winter = pw_dry < ps_wet / 10
    season = np.where(dry_summer, "s", np.where(dry_winter, "w", "f"))
    warmth = np.where(t_hot >= 22, "a", np.where(warm_months >= 4, "b", "c"))
    cold_warmth = np.where((warmth == "c") & (t_cold < -38), "d", warmth)
    for i in range(t.shape[1]):
        if t_hot[i] < 10:
            code = "ET" if t_hot[i] > 0 else "EF"
        elif map_[i] < 10 * threshold[i]:
            code = ("BW" if map_[i] < 5 * threshold[i] else "BS") + ("h" if mat[i] >= 18 else "k")
        elif t_cold[i] >= 18:
            code = "Af" if p_dry[i] >= 60 else ("Am" if p_dry[i] >= 100 - map_[i] / 25 else "Aw")
        elif t_cold[i] > 0:
            code = "C" + season[i] + warmth[i]
        else:
            code = "D" + season[i] + cold_warmth[i]
        out[i] = KOPPEN_INDEX[code]
    out[glaciated] = KOPPEN_INDEX["EF"]
    return out


def whittaker(temperature_k: np.ndarray, precipitation_m: np.ndarray, warmest_k: np.ndarray,
              optimum_k: float = h.EARTH_OPTIMUM_K) -> np.ndarray:
    """Return biome classes of vegetated land from annual temperature, precipitation and the warmest month."""
    t = temperature_k - FREEZING_K - (optimum_k - h.EARTH_OPTIMUM_K)
    hot = warmest_k - FREEZING_K - (optimum_k - h.EARTH_OPTIMUM_K)
    p = precipitation_m * 100.0                     # cm per year
    out = np.full(t.shape, Biome.TEMPERATE_FOREST, dtype=np.int8)
    temperate = (t >= 3) & (t < 20)
    tropical = t >= 20
    out[temperate & (p < 25)] = Biome.COLD_DESERT
    out[temperate & (p >= 25) & (p < 50)] = Biome.TEMPERATE_GRASSLAND
    out[temperate & (p >= 50) & (p < 100)] = Biome.WOODLAND_SHRUBLAND
    out[temperate & (p >= 100) & (p < 225)] = Biome.TEMPERATE_FOREST
    out[temperate & (p >= 225)] = Biome.TEMPERATE_RAINFOREST
    out[temperate & (t >= 15) & (p < 25)] = Biome.HOT_DESERT
    out[tropical & (p < 25)] = Biome.HOT_DESERT
    out[tropical & (p >= 25) & (p < 75)] = Biome.SAVANNA
    out[tropical & (p >= 75) & (p < 175)] = Biome.TROPICAL_SEASONAL_FOREST
    out[tropical & (p >= 175)] = Biome.TROPICAL_RAINFOREST
    cold = t < 3
    out[cold & (p >= 30)] = Biome.BOREAL_FOREST
    out[cold & (p < 30)] = Biome.COLD_DESERT
    out[hot < 10] = Biome.TUNDRA
    out[(hot < 10) & (p < 15)] = Biome.POLAR_DESERT
    return out


def classify_biomes(temperature_k: np.ndarray, precipitation_m: np.ndarray, monthly_k: np.ndarray,
                    ocean: np.ndarray, glaciated: np.ndarray, biosphere: Optional[BiosphereState]) -> np.ndarray:
    """Return the biome of every cell: vegetated classes with surface life, barren or polar land without."""
    warmest = monthly_k.max(axis=0)
    if biosphere is not None and biosphere.life == "surface":
        out = whittaker(temperature_k, precipitation_m, warmest, biosphere.optimum_temperature_k)
        too_hot = temperature_k > biosphere.optimum_temperature_k + biosphere.temperature_tolerance_k
        out[too_hot] = Biome.BARREN
    else:
        out = np.full(temperature_k.shape, Biome.BARREN, dtype=np.int8)
        out[warmest < FREEZING_K + 10] = Biome.POLAR_DESERT
    out[glaciated] = Biome.ICE_SHEET
    out[ocean] = Biome.OCEAN
    return out
