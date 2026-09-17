"""Evolvers that turn a spec into a planet state: snapshot now, history in milestone 6."""

from __future__ import annotations

from typing import Optional

from ..spec import PlanetSpec
from ..state import PlanetState, Timeline
from .base import Evolver
from .history import HistoryEvolver
from .snapshot import SnapshotEvolver

EVOLVERS: dict[str, type] = {"snapshot": SnapshotEvolver, "history": HistoryEvolver}


def evolve(spec: PlanetSpec, t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
    """Return the planet state at the target epoch using the evolver named in ``spec.mode``."""
    return EVOLVERS[spec.mode]().evolve(spec, t_target_gyr)


__all__ = ["Evolver", "SnapshotEvolver", "HistoryEvolver", "evolve"]
