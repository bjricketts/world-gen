"""Run an organic history: start the organic cultures, step them to the present, keep the requested epochs.

The steps apply a list of processes in order. Each process is a callable
``process(ctx, state, years)`` that advances the state by ``years``; later
tasks add growth and spread, farming and development, polities, colonists and
events. A colony lands with its founders and its state (``colonists.py``). The history is deterministic for a given world, spec and
seed: every random draw comes from a stream named by the process and step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from ...util import named_rng
from ..density import colony_age, population_density
from ..development import AXES, Development
from ..model import Culture
from ..spec import HistorySpec
from ..suitability import Covariates
from .clock import next_date, step_years, stops
from .colonists import found_colony_state
from .state import Chronicle, HistoryState
from .suitability import Suitability

Process = Callable[["HistoryContext", HistoryState, float], None]


@dataclass
class HistoryContext:
    """What every process may read: the world, the organic cultures, the history spec, covariates, suitability and the chronicle."""

    world: object
    cultures: list[Culture]
    history: HistorySpec
    seed: int
    cov: Covariates
    suit: Suitability | None = None
    resources: object = None                             # resource potential layer (resource × cell), if any
    chronicle: Chronicle = field(default_factory=Chronicle)
    cache: dict = field(default_factory=dict)            # data a process builds once per history
    next_id: int = 0                                     # the id the next culture arising in the history takes

    def new_culture_id(self) -> int:
        """Return a culture id not used by any culture of the spec or the history."""
        self.next_id = max(self.next_id, max((c.id for c in self.cultures), default=-1) + 1)
        self.next_id += 1
        return self.next_id - 1

    def rng(self, name: str, step: int) -> np.random.Generator:
        """Return the random stream of one process at one step."""
        return named_rng(self.seed, f"history.{name}.{step}")


@dataclass
class HistoryResult:
    """The states kept at the requested epochs (oldest first, the present last) and the chronicle."""

    snapshots: list[HistoryState]
    chronicle: Chronicle
    steps: int
    cultures: list[Culture]                  # the organic cultures, with colonies' landing sites filled
    cells: object = None                     # the cell record (``continuity.CellRecord``), if kept
    series: object = None                    # the series record (``series.SeriesRecord``), if kept


def _levels(dev: Development) -> np.ndarray:
    """Return a development as an array of levels in axis order."""
    return np.array([getattr(dev, axis) for axis in AXES], dtype=float)


def founding_date(culture: Culture, history: HistorySpec) -> float:
    """Return when an organic culture enters the history (years ago): natives at the start, colonists at their landing."""
    if culture.inhabitants == "colonist":
        return min(colony_age(culture), history.start_years_ago)
    return history.start_years_ago


def initial_state(ctx: HistoryContext) -> HistoryState:
    """Return the state at the start: organic natives placed as foragers, colonists not yet landed."""
    n, size = len(ctx.cultures), ctx.cov.land.size
    population = np.zeros((n, size))
    development = np.zeros((n, len(AXES), size))
    active = np.zeros(n, dtype=bool)
    natives = [i for i, c in enumerate(ctx.cultures) if c.inhabitants == "native"]
    if natives:
        density, *_ = population_density(ctx.world.grid.points, ctx.cov, [ctx.cultures[i] for i in natives], ctx.seed)
        for k, i in enumerate(natives):
            population[i] = density[k] * ctx.cov.cell_area_km2
            development[i] = _levels(ctx.cultures[i].development)[:, None]
            active[i] = True
            ctx.chronicle.add(ctx.history.start_years_ago, "origin",
                              f"{ctx.cultures[i].name or f'Culture {ctx.cultures[i].id}'} live as foragers",
                              culture=ctx.cultures[i].id)
    return HistoryState(years_ago=ctx.history.start_years_ago, step=0, culture_ids=[c.id for c in ctx.cultures],
                        population=population, development=development, active=active)


def land_colonists(ctx: HistoryContext, state: HistoryState) -> None:
    """Place the founders of every organic colony whose landing date has come."""
    for i, c in enumerate(ctx.cultures):
        if state.active[i] or c.inhabitants != "colonist":
            continue
        if state.years_ago > founding_date(c, ctx.history) + 1e-9:
            continue
        arriving = replace(c, founded_years_ago=0.0)
        density, _, _, sites = population_density(ctx.world.grid.points, ctx.cov, [arriving], ctx.seed)
        state.population[i] = density[0] * ctx.cov.cell_area_km2
        state.development[i] = _levels(c.development)[:, None]
        state.active[i] = True
        ctx.cultures[i] = replace(c, landing_sites=sites[0])
        where = ", ".join(f"{s['lat_deg']:.1f}°, {s['lon_deg']:.1f}°" for s in sites[0])
        ctx.chronicle.add(state.years_ago, "landing", f"{c.name or f'Culture {c.id}'} land at {where}", culture=c.id)
        found_colony_state(ctx, state, i)


def add_culture(ctx: HistoryContext, state: HistoryState, culture: Culture, development: np.ndarray) -> int:
    """Add a culture that arises during the history (an overseas colony or a creole); return its row.

    The state's arrays gain a row (no people yet, the given development
    levels per axis and cell) and every process cache with an ``add_row``
    method is told.
    """
    ctx.cultures.append(culture)
    n = state.population.shape[1]
    state.population = np.vstack([state.population, np.zeros((1, n))])
    state.development = np.concatenate([state.development, development[None]], axis=0)
    state.active = np.r_[state.active, True]
    state.culture_ids.append(culture.id)
    for cached in ctx.cache.values():
        if hasattr(cached, "add_row"):
            cached.add_row(ctx, culture)
    return len(ctx.cultures) - 1


def run_history(world, cultures: list[Culture], history: HistorySpec, seed: int, cov: Covariates,
                processes: Sequence[Process] = (), suit: Suitability | None = None,
                resources=None, reserved_ids: Sequence[int] = ()) -> HistoryResult:
    """Return the history of the organic cultures from the start to the present, with the states at the kept epochs.

    Cultures arising in the history take ids above those of ``cultures`` and ``reserved_ids`` (the scoped cultures).
    """
    ctx = HistoryContext(world=world, cultures=list(cultures), history=history, seed=seed, cov=cov, suit=suit,
                         resources=resources, next_id=max(reserved_ids, default=-1) + 1)
    landings = tuple(founding_date(c, history) for c in cultures if c.inhabitants == "colonist")
    dates = stops(history, landings)
    state = initial_state(ctx)
    land_colonists(ctx, state)
    kept = [state.copy()] if state.years_ago in history.epochs else []
    while state.years_ago > 0:
        now = state.years_ago
        after = next_date(now, step_years(history, state.top_energy()), dates)
        for process in processes:
            process(ctx, state, now - after)
        state.years_ago = after
        state.step += 1
        land_colonists(ctx, state)
        if after in history.epochs or after == 0.0:
            kept.append(state.copy())
    return HistoryResult(snapshots=kept, chronicle=ctx.chronicle, steps=state.step, cultures=ctx.cultures,
                         cells=ctx.cache.get("cells"), series=ctx.cache.get("series"))


def default_processes() -> list[Process]:
    """Return the processes an organic history applies each step, in order."""
    from .development import knowledge_step, level_spread_step, origin_step
    from .colonists import epidemic_step, home_step, immigration_step, settler_mortality_step
    from .contact import shift_step, transfer_step
    from .overseas import overseas_step
    from .continuity import record_step
    from .events import events_step
    from .polities import polity_step
    from .population import grow_step, spread_step
    from .series import series_step
    return [events_step, origin_step, knowledge_step, level_spread_step, grow_step, spread_step, immigration_step,
            settler_mortality_step, epidemic_step, overseas_step, home_step, transfer_step, shift_step, polity_step,
            record_step, series_step]


