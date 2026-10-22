"""Fields a world can export: per-cell values, how they are sampled, their units and category labels.

Field names follow the map fields (``elevation``, ``biomes``, ``rainfall``,
``political``, ``resources:copper`` and so on), plus fields only exports need
(``ocean``, ``lake``, ``rivers``, ``discharge``, ``surface`` and others).
Continuous fields are blended linearly between cells, rivers and discharge
take the nearest cell, and categories carry labels for their codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..biosphere.biomes import KOPPEN_CODES, Biome
from ..render.culture import CULTURE_FIELDS, HISTORY_LAYER_FIELDS, culture_values, split_field
from ..surface.fields import Terrain
from ..surface.relicts import Relict

LINEAR, NEAREST, CATEGORICAL = "linear", "nearest", "categorical"

# name: (surface variable, sampling, units, description)
PHYSICAL = {
    "elevation": ("elevation", LINEAR, "m", "ground height above sea level (below ice)"),
    "surface": (None, LINEAR, "m", "height of the top surface above sea level, ice included"),
    "ocean": (None, CATEGORICAL, "", "1 where the sea lies"),
    "lake": ("lake", CATEGORICAL, "", "1 where a lake lies"),
    "rivers": ("river_order", NEAREST, "Strahler order", "river order (0: no river)"),
    "discharge": ("discharge", NEAREST, "m3/s", "mean river discharge"),
    "basins": ("drainage_basin", CATEGORICAL, "basin id", "drainage basin (0: drains to no basin)"),
    "temperature": ("air_temperature", LINEAR, "K", "annual mean air temperature"),
    "rainfall": ("precipitation", LINEAR, "m/yr", "annual precipitation"),
    "runoff": ("runoff", LINEAR, "m/yr", "annual runoff"),
    "ice": ("ice_thickness", LINEAR, "m", "ice-sheet thickness"),
    "sea_ice": ("sea_ice", LINEAR, "fraction", "sea-ice cover"),
    "vegetation": ("vegetation_cover", LINEAR, "fraction", "vegetation cover"),
    "sediment": ("sediment", LINEAR, "m", "sediment thickness"),
    "biomes": ("biome", CATEGORICAL, "biome", "biome"),
    "koppen": ("koppen", CATEGORICAL, "Koppen class", "Köppen–Geiger class"),
    "terrain": ("terrain", CATEGORICAL, "terrain", "landform class"),
    "plates": ("plate", CATEGORICAL, "plate id", "tectonic plate"),
    "crust_age": ("crust_age", LINEAR, "Myr", "oceanic crust age (none on continents)"),
    "orogeny_age": ("orogeny_age", LINEAR, "Myr", "time since mountain building"),
    "relicts": ("relict", CATEGORICAL, "relict", "relict feature"),
}
MONTHLY = {"temperature": "monthly_temperature", "rainfall": "monthly_precipitation"}
OVERLAYS = ("settlements", "roads")          # drawn over maps; no per-cell values
EXPORT_FIELDS = tuple(PHYSICAL) + tuple(f for f in CULTURE_FIELDS if f not in OVERLAYS) + HISTORY_LAYER_FIELDS


def _labels(name: str) -> Optional[dict]:
    """Return the code labels of a physical category field."""
    if name == "biomes":
        return {int(b): b.name.lower().replace("_", " ") for b in Biome}
    if name == "koppen":
        return {k: code for k, code in enumerate(KOPPEN_CODES)}
    if name == "terrain":
        return {int(t): t.name.lower().replace("_", " ") for t in Terrain}
    if name == "relicts":
        return {int(r): r.name.lower() for r in Relict}
    if name in ("ocean", "lake"):
        return {0: "no", 1: "yes"}
    return None


@dataclass(frozen=True)
class ExportField:
    """One field of a world ready to sample: per-cell values, sampling rule, units and labels."""

    name: str
    values: np.ndarray            # per grid cell
    sampling: str                 # 'linear', 'nearest' or 'categorical'
    units: str
    description: str
    labels: Optional[dict] = None  # category code → label
    nodata: Optional[float] = None  # code that marks empty cells of a category field

    @property
    def categorical(self) -> bool:
        """Return whether the values are category codes."""
        return self.sampling == CATEGORICAL

    def sample(self, sampler, surface=None) -> np.ndarray:
        """Return the field at a sampler's points; the sea is drawn where the blended elevation is below sea level."""
        if self.name == "ocean" and surface is not None:
            elevation = sampler.sample(surface["elevation"].values)
            near_sea = sampler.sample(surface["ocean"].values, categorical=True).astype(bool)
            return ((elevation < 0.0) & near_sea).astype(np.int16)
        return sampler.sample(self.values, categorical=self.sampling != LINEAR)


def export_field(world, field: str) -> ExportField:
    """Return a world's field by map name, e.g. ``elevation``, ``rainfall:7`` (month 7), ``political``."""
    ds = world.surface
    if ds is None:
        raise ValueError(f"{world.state.name} has no solid surface to export")
    name, option = split_field(field)
    if name in PHYSICAL:
        variable, sampling, units, description = PHYSICAL[name]
        if option is not None and name in MONTHLY:
            month = int(option)
            if not 1 <= month <= 12:
                raise ValueError("months run from 1 to 12")
            variable, description = MONTHLY[name], f"{description.replace('annual', 'month ' + option)}"
            values = ds[variable].values[month - 1]
        elif name == "surface":
            ice = ds["ice_surface"].values if "ice_surface" in ds else 0.0
            thick = ds["ice_thickness"].values if "ice_thickness" in ds else 0.0
            values = ds["elevation"].values + np.where(np.asarray(thick) > 0.0, ice, 0.0)
        elif name == "ocean":
            values = ds["ocean"].values.astype(np.int16)
        else:
            if variable not in ds:
                raise ValueError(f"{world.state.name} has no '{name}' field (no '{variable}' on its surface)")
            values = ds[variable].values
        if sampling == CATEGORICAL or name == "rivers":
            values = np.asarray(values).astype(np.int32)
        return ExportField(field, np.asarray(values), sampling, units, description, _labels(name),
                           -1 if sampling == CATEGORICAL else None)
    if name in CULTURE_FIELDS + HISTORY_LAYER_FIELDS:
        values, categorical, units, labels = culture_values(world, field)
        values = np.asarray(values)
        return ExportField(field, values.astype(np.int32) if categorical else values.astype(float),
                           CATEGORICAL if categorical else LINEAR, units, f"culture layer '{field}'", labels,
                           -1 if categorical else None)
    raise ValueError(f"unknown export field '{field}'; choose from {', '.join(EXPORT_FIELDS)}")
