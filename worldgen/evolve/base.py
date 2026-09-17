"""Common interface for the snapshot and history evolvers."""

from __future__ import annotations

from typing import Optional, Protocol

from ..spec import PlanetSpec
from ..state import PlanetState, Timeline


class Evolver(Protocol):
    """Produces a planet's global state at a target epoch from a spec."""

    def evolve(self, spec: PlanetSpec, t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
        """Return the planet state at ``t_target_gyr`` (default: the star's age) and its timeline."""
        ...
