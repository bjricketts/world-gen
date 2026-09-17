"""Archetype outcome constraints and the redraw loop that enforces them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .. import heuristics as h
from ..spec import PlanetSpec
from ..state import Issue, PlanetState
from ..util import stable_hash


@dataclass(frozen=True)
class Constraint:
    """A required outcome for planets of an archetype."""

    description: str
    check: Callable[[PlanetState], bool]

    def __call__(self, state: PlanetState) -> bool:
        """Return whether the planet satisfies this constraint."""
        return self.check(state)


NOT_LOCKED = Constraint("rotates freely (not tidally locked)",
                        lambda s: s.orbit.spin_state.startswith("free"))
SYNCHRONOUS = Constraint("synchronously rotating (tidally locked)",
                         lambda s: s.orbit.spin_state == "synchronous")
LIQUID_WATER = Constraint("liquid surface water", lambda s: s.atmosphere.surface_water == "liquid")
MOSTLY_OPEN_OCEAN = Constraint(
    f"liquid surface water with at least {h.OPEN_OCEAN_TEMPERATE:.0%} of the ocean free of ice",
    lambda s: s.atmosphere.surface_water == "liquid"
    and (s.climate is None or s.climate.open_ocean_fraction >= h.OPEN_OCEAN_TEMPERATE))
ICE_SURFACE = Constraint("frozen surface water", lambda s: s.atmosphere.surface_water == "ice")
WET_SURFACE = Constraint("surface water as liquid or ice",
                         lambda s: s.atmosphere.surface_water in ("liquid", "ice"))
MOSTLY_LAND = Constraint("at least half the surface is land (estimated from the water inventory)",
                         lambda s: s.water is None or s.water.land_fraction_estimate >= 0.5)
NOT_RUNAWAY = Constraint("not in a runaway greenhouse", lambda s: not s.atmosphere.runaway_greenhouse)


def surface_temperature_between(low_k: float, high_k: float) -> Constraint:
    """Return a constraint on the global mean surface temperature."""
    return Constraint(f"surface temperature {low_k:.0f}–{high_k:.0f} K",
                      lambda s: low_k <= s.atmosphere.surface_temperature_k <= high_k)


def dayside_temperature_between(low_k: float, high_k: float) -> Constraint:
    """Return a constraint on the day-side mean temperature of a synchronous planet (the global mean otherwise)."""
    def check(s: PlanetState) -> bool:
        """Return whether the day-side temperature lies in the band."""
        day = s.climate.dayside_temperature_k if s.climate is not None else None
        t = s.atmosphere.surface_temperature_k if day is None else day
        return low_k <= t <= high_k

    return Constraint(f"day-side temperature {low_k:.0f}–{high_k:.0f} K", check)


def attempt_seed(seed: int, attempt: int) -> int:
    """Return the deterministic seed for a redraw; attempt 0 keeps the spec's seed."""
    if attempt == 0:
        return seed
    return stable_hash("redraw", seed, attempt) & 0x7FFFFFFF


def failed_constraints(state: PlanetState) -> list[Constraint]:
    """Return the constraints of the planet's archetype that it does not satisfy."""
    from .archetypes import ARCHETYPES

    if state.archetype is None:
        return []
    return [c for c in ARCHETYPES[state.archetype].constraints if not c(state)]


def _times(n: int) -> str:
    """Return 'once' or 'N times'."""
    return "once" if n == 1 else f"{n} times"


def draw_until_satisfied(spec: PlanetSpec, build: Callable[[int], PlanetState]) -> PlanetState:
    """Return the first draw whose archetype constraints hold, redrawing with derived seeds.

    ``build(seed)`` generates one planet from the spec using ``seed`` for all
    random draws. If no draw satisfies the constraints within
    ``spec.priors.max_attempts``, the draw with the fewest failures is
    returned and the report says which constraints it misses.
    """
    attempts = spec.priors.max_attempts if spec.priors.enforce_constraints else 1
    best, best_failures, tried = None, None, 0
    for attempt in range(attempts):
        tried = attempt + 1
        state = build(attempt_seed(spec.seed, attempt))
        failures = failed_constraints(state)
        if best is None or len(failures) < len(best_failures):
            best, best_failures = state, failures
        if not failures:
            break

    best.attempts = tried
    best.constraints_met = not best_failures
    if best_failures and spec.priors.enforce_constraints:
        missed = "; ".join(c.description for c in best_failures)
        best.issues.insert(0, Issue(
            "warning", "conflict", "priors",
            f"no draw in {best.attempts} attempts met the '{best.archetype}' archetype; missing: {missed}"))
    elif best.attempts > 1:
        best.issues.insert(0, Issue(
            "info", "note", "priors",
            f"redrawn {_times(best.attempts - 1)} to meet the '{best.archetype}' archetype"))
    return best
