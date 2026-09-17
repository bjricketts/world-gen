"""Water balance inputs for drainage and erosion: precipitation, evaporation, runoff and temperature.

The fields come from the Tier 1 climate model (``worldgen.climate``).
Runoff on land follows the Fu–Budyko curve (Zhang et al. 2004) from annual
precipitation and potential evaporation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import heuristics as h


@dataclass
class Rainfall:
    """Per-cell water balance inputs (all per year)."""

    precipitation_m: np.ndarray
    evaporation_m: np.ndarray        # potential evaporation from open water or wet ground
    runoff_m: np.ndarray             # precipitation minus actual evaporation on land
    temperature_k: np.ndarray        # annual mean near-surface air temperature
    monthly_temperature_k: Optional[np.ndarray] = None     # (12, cells)
    monthly_precipitation_m: Optional[np.ndarray] = None   # (12, cells), as annual rates


def potential_evaporation(temperature_k: np.ndarray) -> np.ndarray:
    """Return potential evaporation (m/yr) from air temperature."""
    return h.EVAPORATION_M * np.clip((temperature_k - h.EVAPORATION_ZERO_K) / 40.0, 0.0, 2.0)


def fu_runoff(precipitation: np.ndarray, evaporation: np.ndarray) -> np.ndarray:
    """Return runoff (m/yr) from the Fu–Budyko curve (Zhang et al. 2004)."""
    w = h.BUDYKO_W
    p = np.maximum(precipitation, 1e-9)
    ratio = evaporation / p
    actual = p * (1.0 + ratio - (1.0 + ratio**w) ** (1.0 / w))
    return np.maximum(p - actual, 0.0)
