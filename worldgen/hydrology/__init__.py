"""Water on the surface: drainage, lakes, rivers and erosion, driven by the climate model.

``WaterSetting`` carries what these steps need to know about the planet.
Rivers and erosion run only on planets with liquid surface water.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..grid import SphereGrid
from .drainage import Drainage, build_drainage
from .erosion import ErosionResult, erode
from .rainfall import Rainfall

if TYPE_CHECKING:
    from ..climate import ClimateSetting, SurfaceClimate, TemperatureControl, ZonalClimate

__all__ = ["Drainage", "ErosionResult", "Rainfall", "WaterSetting", "build_drainage", "erode", "erode_surface"]


@dataclass
class WaterSetting:
    """Planet properties used by the climate, drainage and erosion steps."""

    radius_m: float
    ocean_volume_m3: float
    climate: "ClimateSetting"
    zonal: "ZonalClimate"
    relief: float
    rivers: bool = True               # liquid surface water: rivers erode the land
    land_albedo: Optional[np.ndarray] = field(default=None, repr=False)   # snow-free, on the surface grid
    moisture_decay_km: Optional[np.ndarray | float] = field(default=None, repr=False)
    control: Optional["TemperatureControl"] = None   # mean temperature held by the greenhouse (weathering)

    def surface_climate(self, grid: SphereGrid, height_m: np.ndarray, ocean: np.ndarray, seasonal: bool = True,
                        glacier: Optional[np.ndarray] = None) -> "SurfaceClimate":
        """Return the Tier 1 climate of a surface (``height_m`` above sea level)."""
        from ..climate import run_climate   # imported here: the climate package builds on this one

        return run_climate(grid, self.radius_m, height_m, ocean, self.climate, self.zonal, seasonal=seasonal,
                           land_albedo=_on_grid(self.land_albedo, grid), glacier=glacier,
                           moisture_decay_km=_on_grid(self.moisture_decay_km, grid), control=self.control)

    def rainfall(self, grid: SphereGrid, height_m: np.ndarray, ocean: np.ndarray, seasonal: bool = False) -> Rainfall:
        """Return the water balance of a surface; annual-mean climate unless ``seasonal``."""
        return self.surface_climate(grid, height_m, ocean, seasonal=seasonal).rainfall


def erode_surface(grid: SphereGrid, setting: WaterSetting, height_m: np.ndarray, ocean: np.ndarray,
                  duration_myr: float, step_myr: float) -> ErosionResult:
    """Return a surface (``height_m`` above sea level) after erosion under its own annual climate.

    Land far above the snowline is worn down toward it; with liquid water,
    rivers then erode and deposit.
    """
    from ..climate.ice import buzzsaw   # imported here: the climate package builds on this one

    climate = setting.surface_climate(grid, height_m, ocean, seasonal=False)
    lowered = buzzsaw(height_m, ocean, climate.ela_m, duration_myr)
    if not setting.rivers:
        zero = np.zeros(grid.size)
        return ErosionResult(height_m=lowered, eroded_m=height_m - lowered, deposited_m=zero)
    result = erode(grid, setting.radius_m, lowered, ocean, climate.rainfall.runoff_m, duration_myr, setting.relief,
                   step_myr)
    result.eroded_m += height_m - lowered
    return result


def _on_grid(values, grid: SphereGrid):
    """Return a per-cell field if it belongs to this grid (or is a single value), else None."""
    if values is None or np.ndim(values) == 0:
        return values
    return values if np.size(values) == grid.size else None
