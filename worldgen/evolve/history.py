"""History mode: integrate the planet's global state from formation (milestone 6)."""

from __future__ import annotations

from typing import Optional

from ..spec import PlanetSpec
from ..state import PlanetState, Timeline


class HistoryEvolver:
    """Placeholder for the time-integrated evolver."""

    def evolve(self, spec: PlanetSpec, t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
        """Not yet available."""
        raise NotImplementedError("history mode is planned for milestone 6; use mode: snapshot")
