"""Global surface fields and their conversion to an xarray Dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import xarray as xr

from ..grid import SphereGrid


class Terrain(IntEnum):
    """Landform class of each grid cell."""

    PLAIN = 0
    HIGHLAND = 1
    MOUNTAIN = 2
    VOLCANO = 3
    RIFT = 4
    TRENCH = 5
    RIDGE = 6
    CRATER = 7
    VOLCANIC_PLAIN = 8
    SHELF = 9
    ABYSSAL_PLAIN = 10
    LAKE = 11


class Crust(IntEnum):
    """Crust type of each grid cell."""

    OCEANIC = 0
    CONTINENTAL = 1
    PRIMARY = 2        # single-plate (non-plate-tectonic) crust


class Boundary(IntEnum):
    """Plate boundary type of each grid cell."""

    NONE = 0
    CONVERGENT = 1
    DIVERGENT = 2
    TRANSFORM = 3


def _enum_attrs(enum: type[IntEnum]) -> dict[str, str]:
    """Return CF-style flag attributes describing an enum."""
    return {"flag_values": " ".join(str(int(e)) for e in enum),
            "flag_meanings": " ".join(e.name.lower() for e in enum)}


@dataclass
class SurfaceFields:
    """Per-cell surface arrays being built for one planet."""

    grid: SphereGrid
    radius_m: float
    elevation: np.ndarray = field(init=False)
    terrain: np.ndarray = field(init=False)
    crust: np.ndarray = field(init=False)
    plate: np.ndarray = field(init=False)
    boundary: np.ndarray = field(init=False)
    crust_age_myr: np.ndarray = field(init=False)
    orogeny_age_myr: np.ndarray = field(init=False)
    fabric: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Allocate empty per-cell arrays."""
        n = self.grid.size
        self.elevation = np.zeros(n)
        self.terrain = np.full(n, Terrain.PLAIN, dtype=np.int8)
        self.crust = np.full(n, Crust.PRIMARY, dtype=np.int8)
        self.plate = np.full(n, -1, dtype=np.int16)
        self.boundary = np.full(n, Boundary.NONE, dtype=np.int8)
        self.crust_age_myr = np.full(n, np.nan)
        self.orogeny_age_myr = np.full(n, np.nan)
        self.fabric = np.zeros((n, 3))

    @property
    def spacing_km(self) -> float:
        """Return the mean grid spacing in km."""
        return self.grid.mean_spacing(self.radius_m / 1e3)

    def to_dataset(self, ocean: np.ndarray, attrs: dict) -> xr.Dataset:
        """Return the fields as an xarray Dataset on the ``cell`` dimension."""
        g = self.grid
        return xr.Dataset(
            data_vars={
                "elevation": ("cell", self.elevation.astype(np.float32),
                              {"units": "m", "long_name": "elevation above sea level (or mean datum)"}),
                "ocean": ("cell", ocean, {"long_name": "covered by ocean (liquid or frozen)"}),
                "terrain": ("cell", self.terrain, {"long_name": "landform class", **_enum_attrs(Terrain)}),
                "crust": ("cell", self.crust, {"long_name": "crust type", **_enum_attrs(Crust)}),
                "plate": ("cell", self.plate, {"long_name": "plate index (-1: no plates)"}),
                "boundary": ("cell", self.boundary, {"long_name": "plate boundary type", **_enum_attrs(Boundary)}),
                "crust_age": ("cell", self.crust_age_myr.astype(np.float32),
                              {"units": "Myr", "long_name": "oceanic crust age"}),
                "orogeny_age": ("cell", self.orogeny_age_myr.astype(np.float32),
                                {"units": "Myr", "long_name": "time since last mountain building"}),
                "fabric": (("vec", "cell"), self.fabric.T.astype(np.float32),
                           {"long_name": "structural grain as a tangent vector (x, y, z in the grid frame); "
                                         "length 0 to 1 is the grain strength, 0 where there is no grain"}),
            },
            coords={
                "lat": ("cell", g.lat.astype(np.float32), {"units": "degrees_north"}),
                "lon": ("cell", g.lon.astype(np.float32), {"units": "degrees_east"}),
                "vec": ("vec", np.arange(3, dtype=np.int8), {"long_name": "Cartesian component (x, y, z)"}),
            },
            attrs={"grid": "fibonacci", "grid_size": g.size, "radius_m": self.radius_m, **attrs},
        )
