"""The biosphere in time: productivity, O₂ from a redox balance, and biotic methane.

O₂ rises when organic burial (the oxygen source) exceeds the flux of
volcanic reductants, which declines as the mantle oxidises; oxidative
weathering ∝ √O₂ sets the level once it is high (after Goldblatt et al.
2006). Ocean life starts burial; land life adds to it after a colonisation
delay. Methanogens keep CH₄ high while the air is anoxic.
"""

from __future__ import annotations

import math

from .. import heuristics as h
from .params import HistoryParams

EARTH_OXIDATIVE = (h.OXYGEN_BURIAL_EARTH * (1.0 - h.REDUCTANT_EARTH_SHARE)) / math.sqrt(h.OXYGEN_MASS_EARTH)


def land_productivity(p: HistoryParams, mean_k: float, land: float, open_ocean: float) -> float:
    """Return land productivity relative to Earth's for a global mean temperature and land share."""
    optimum = p.optimum_k - (298.0 - 288.0)    # Earth's global mean sits 10 K below the growth optimum
    climate = math.exp(-(((mean_k - optimum) / h.LAND_PRODUCTIVITY_WIDTH_K) ** 2))
    return climate * min(land / 0.29, 2.0) * min(open_ocean / 0.9, 1.0)


def productivity(p: HistoryParams, life: str, land_colonised: float, mean_k: float, land: float,
                 open_ocean: float) -> tuple[float, float]:
    """Return (total, land) productivity relative to Earth's for the current life and climate."""
    if life not in ("surface", "ocean"):
        return 0.0, 0.0
    ocean = h.OCEAN_PRODUCTIVITY_SHARE * min(open_ocean / 0.9, 1.0) * min((1.0 - land) / 0.71, 1.4)
    on_land = 0.0
    if life == "surface":
        on_land = land_colonised * land_productivity(p, mean_k, land, open_ocean)
    return ocean + (1.0 - h.OCEAN_PRODUCTIVITY_SHARE) * on_land, on_land


def oxygen_rate(p: HistoryParams, t_gyr: float, oxygen: float, total_productivity: float, melt: float,
                surface_k: float, land: float, heat_ratio: float, spreading: float = 0.0) -> float:
    """Return dO₂/dt (10¹⁸ kg per Gyr) from burial, reductants, oxidative weathering and hot surfaces."""
    burial = 0.0
    if p.product == "o2":
        burial = h.OXYGEN_BURIAL_EARTH * p.burial * total_productivity * p.area_ratio
    reductants = (h.OXYGEN_BURIAL_EARTH * h.REDUCTANT_EARTH_SHARE * math.sqrt(max(melt, 0.0))
                  * math.exp((4.57 - t_gyr) / p.reductant_decay_gyr))
    present = oxygen / (oxygen + h.OXYGEN_SMALL)
    root = math.sqrt(max(oxygen, 0.0))
    # Oxygen is consumed by weathering exposed rock and by fresh sea floor (hydrothermal alteration),
    # so an ocean world with no land still has a sink.
    exposure = ((max(land / 0.29, 0.05) * max(heat_ratio, 0.1) ** 0.5
                 + h.SEAFLOOR_OXIDATION * spreading) * p.area_ratio)
    weathering = EARTH_OXIDATIVE * root * exposure
    hot = 1.0 / (1.0 + math.exp(-(surface_k - h.HOT_OXIDATION_K) / 20.0))
    return burial - present * reductants - weathering - hot * oxygen / h.OXYGEN_CRUST_SINK_GYR


def methane_fraction(p: HistoryParams, life: str, o2_fraction: float) -> float:
    """Return the CH₄ mole fraction kept up by methanogens."""
    if life == "none":
        return 0.0
    if p.product == "ch4":
        return h.METHANE_BIOTIC
    if p.product == "o2" or p.biochemistry == "oxygenic":
        return h.METHANE_BIOTIC / (1.0 + (o2_fraction / h.METHANE_ANOXIC_O2) ** 2)
    return 0.0
