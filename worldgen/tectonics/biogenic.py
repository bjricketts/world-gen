"""Sediment the sea makes itself: carbonate platforms, shelf and slope carbonate, and pelagic rain.

Rates are long-term accumulation, not production:

- Tropical platforms (Schlager 2003: the tropical factory ends where the
  coldest month falls below ~20 °C). Production in lit water reaches several
  mm/yr (Bosscher & Schlager 1992), about 40 times faster than platforms
  accumulate over millions of years (Opdyke & Wilkinson 1988: 1000 against
  25 m/Myr), so a platform keeps pace with subsidence and fills its space up
  to just below sea level. Ground that was within the lit zone before this
  interval's subsidence counts, so platforms keep up instead of drowning.
- Other shelves, slopes and the deep sea accumulate at the rates of the
  Holocene carbonate budget of Milliman (1993): shelves ~16, slopes ~18,
  deep sea ~3.8 g m⁻² yr⁻¹, converted to thickness at 1.2 t m⁻³ as Milliman
  does (1200 g m⁻² yr⁻¹ ≈ 1 m kyr⁻¹). Below the carbonate compensation depth
  pelagic accumulation is slower.

Biogenic sediment needs sea life; its strength ramps up with the age of the
marine biosphere.
"""

from __future__ import annotations

import numpy as np

from .. import heuristics as h
from .crust import Crust


def biogenic_thickness(depth_m: np.ndarray, depth_before_m: np.ndarray, warm: np.ndarray, continental: np.ndarray,
                       duration_myr: float, strength: float) -> np.ndarray:
    """Return the biogenic sediment thickness (m) laid on each sea cell over ``duration_myr``.

    ``depth_m`` is water depth after this interval's subsidence (0 or less on
    land), ``depth_before_m`` the depth before it, ``warm`` where the tropical
    factory works. Shelf and slope rates apply on continental crust; oceanic
    crust, including shallow ridges and volcanic islands' flanks, takes pelagic
    rain unless a platform grows on it.
    """
    sea = depth_m > 0
    room = np.maximum(depth_m - h.CARBONATE_PLATFORM_TOP_M, 0.0) / h.SEDIMENT_ISOSTATIC_FACTOR
    platform = sea & warm & (depth_before_m <= h.CARBONATE_PHOTIC_DEPTH_M)
    margin = sea & ~platform & continental
    shelf = margin & (depth_m <= h.SHELF_EDGE_DEPTH_M)
    slope = margin & (depth_m > h.SHELF_EDGE_DEPTH_M) & (depth_m <= h.SLOPE_BASE_DEPTH_M)
    deep = sea & ~platform & ~shelf & ~slope
    rate = np.zeros(depth_m.shape)
    rate[shelf] = h.SHELF_CARBONATE_M_MYR
    rate[slope] = h.SLOPE_CARBONATE_M_MYR
    rate[deep] = np.where(depth_m[deep] < h.CARBONATE_COMPENSATION_DEPTH_M, h.PELAGIC_ABOVE_CCD_M_MYR,
                          h.PELAGIC_BELOW_CCD_M_MYR)
    thickness = np.where(platform, room, np.minimum(rate * duration_myr, np.where(shelf, room, np.inf)))
    return strength * np.where(sea, thickness, 0.0)


def lay_biogenic(crust: Crust, height_m: np.ndarray, sank_m: np.ndarray, temperature_k: np.ndarray,
                 duration_myr: float, strength: float) -> None:
    """Add one interval of biogenic sediment to the crust (``height_m`` relative to sea level)."""
    if strength <= 0.0:
        return
    depth = -height_m
    warm = temperature_k >= h.CARBONATE_TROPICAL_ANNUAL_K
    thickness = biogenic_thickness(depth, depth - sank_m, warm, crust.continental, duration_myr, strength)
    rise = thickness * h.SEDIMENT_ISOSTATIC_FACTOR
    cont = crust.continental
    crust.cont_elev[cont] += rise[cont]
    crust.basin_fill[cont] += thickness[cont]
    crust.sediment[~cont] += rise[~cont]
