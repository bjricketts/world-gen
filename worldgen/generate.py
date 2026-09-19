"""Top-level entry points: generate one planet, or rank several candidates."""

from __future__ import annotations

from typing import Optional

from .evolve import evolve
from .spec import PlanetSpec
from .state import PlanetState, Timeline
from .surface import build_surface
from .world import World


def generate(spec: PlanetSpec, t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
    """Return the planet described by a spec, and its timeline."""
    return evolve(spec, t_target_gyr)


def generate_world(spec: PlanetSpec, resolution: int | str = "standard",
                   t_target_gyr: Optional[float] = None,
                   snapshot_interval_myr: Optional[float] = None) -> World:
    """Return a planet with its global surface at the given grid resolution.

    ``snapshot_interval_myr`` stores the tectonic history at that interval
    (simulated plate tectonics only).
    """
    state, timeline = evolve(spec, t_target_gyr)
    surface = build_surface(state, resolution, snapshot_interval_myr, timeline)
    return World(spec=spec, state=state, surface=surface, timeline=timeline)


def candidates(spec: PlanetSpec, n: int) -> list[PlanetState]:
    """Return ``n`` planets drawn from the same spec with different seeds, best occupiability first.

    Planets whose report contains errors are ranked last.
    """
    planets = []
    for k in range(n):
        seed = spec.seed + k
        variant = spec.model_copy(update={"seed": seed, "name": f"{spec.name} #{k + 1} (seed {seed})"})
        state, _ = evolve(variant)
        planets.append(state)

    def rank_key(state: PlanetState) -> tuple[bool, float]:
        """Return the sort key: planets without errors first, then higher scores."""
        has_error = any(i.level == "error" for i in state.issues)
        return has_error, -state.occupiability.score

    return sorted(planets, key=rank_key)
