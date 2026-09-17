"""Crust points and plates: the state carried through the tectonic simulation.

Each crust point belongs to one plate and moves rigidly with it. Time is in
Myr and angular velocities in rad/Myr throughout this package.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .. import heuristics as h
from ..surface.plates import seafloor_depth

FRONT_NONE, FRONT_ANDEAN, FRONT_ARC = 0, 1, 2
OROGENY_NONE, OROGENY_ANDEAN, OROGENY_ARC, OROGENY_COLLISION = 0, 1, 2, 3


@dataclass
class Crust:
    """Per-point crust properties; all arrays have one entry per point."""

    points: np.ndarray          # (n, 3) unit vectors
    plate: np.ndarray           # plate index
    continental: np.ndarray     # continental (True) or oceanic crust
    cont_elev: np.ndarray       # elevation of continental crust (m)
    ocean_extra: np.ndarray     # relief on oceanic crust above its age-depth (m)
    ocean_age: np.ndarray       # age of oceanic crust (Myr)
    sediment: np.ndarray        # sediment on oceanic crust (m)
    orogeny_age: np.ndarray     # time since last mountain building (Myr), NaN if none
    orogeny_kind: np.ndarray    # OROGENY_* code of the last mountain building
    volcanic: np.ndarray        # built by a hotspot
    front_kind: np.ndarray      # FRONT_* code if the point overrides a subduction zone
    front_speed: np.ndarray     # convergence speed at that front (m/Myr)
    front_age: np.ndarray       # time since the front was last active (Myr)
    collision_speed: np.ndarray  # convergence speed if in continental collision (m/Myr)
    collision_age: np.ndarray   # time since the collision was last active (Myr)
    alive: np.ndarray           # False once consumed by subduction

    @property
    def size(self) -> int:
        """Return the number of points, including consumed ones."""
        return len(self.points)

    @classmethod
    def empty(cls, points: np.ndarray) -> "Crust":
        """Return oceanic crust of age zero at the given points, with no plate assigned."""
        n = len(points)
        return cls(
            points=points.copy(),
            plate=np.zeros(n, dtype=np.int32),
            continental=np.zeros(n, dtype=bool),
            cont_elev=np.zeros(n),
            ocean_extra=np.zeros(n),
            ocean_age=np.zeros(n),
            sediment=np.zeros(n),
            orogeny_age=np.full(n, np.nan),
            orogeny_kind=np.zeros(n, dtype=np.int8),
            volcanic=np.zeros(n, dtype=bool),
            front_kind=np.zeros(n, dtype=np.int8),
            front_speed=np.zeros(n),
            front_age=np.full(n, np.inf),
            collision_speed=np.zeros(n),
            collision_age=np.full(n, np.inf),
            alive=np.ones(n, dtype=bool),
        )

    def take(self, index: np.ndarray) -> "Crust":
        """Return a new crust made of the points at ``index`` (with repeats allowed)."""
        return Crust(**{f.name: getattr(self, f.name)[index].copy() for f in fields(self)})


@dataclass
class Plates:
    """Rotation vectors of all plates; inactive plates have been merged away."""

    omega: np.ndarray       # (k, 3) rotation axis × angular speed (rad/Myr)
    active: np.ndarray      # (k,) bool

    @property
    def count(self) -> int:
        """Return the number of active plates."""
        return int(self.active.sum())

    def add(self, omega: np.ndarray) -> int:
        """Append a plate and return its index."""
        self.omega = np.vstack([self.omega, omega[None, :]])
        self.active = np.append(self.active, True)
        return len(self.active) - 1


def ocean_depth(age_myr: np.ndarray, relief: float = 1.0) -> np.ndarray:
    """Return sea-floor depth (m, negative) for oceanic crust of a given age."""
    return seafloor_depth(age_myr, relief)


def elevation(crust: Crust, relief: float = 1.0) -> np.ndarray:
    """Return the elevation (m) of every point before sea level is applied."""
    return np.where(crust.continental, crust.cont_elev,
                    ocean_depth(crust.ocean_age, relief) + crust.ocean_extra + crust.sediment)


def typical_speed_m_myr(activity: float) -> float:
    """Return the typical plate speed (m/Myr) for an interior activity index."""
    return h.PLATE_SPEED_EARTH_CM_YR / 100.0 * 1e6 * max(activity, 0.05) ** 0.5
