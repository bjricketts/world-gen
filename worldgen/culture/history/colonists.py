"""Colonists in an organic history: their state at landing, later immigrants, and contact epidemics among natives.

- *Colonial state.* A colony founds a state (type colony) at the landing cell
  holding most of its founders, taking that cell from any state there.
- *Immigration.* An organic colony's ``immigrants_per_year`` arrive where its
  founders landed, in the founders' proportions, for ``immigration_years``
  after landing (or to the present).
- *Contact epidemics* (after Koch et al. 2019). When a colonist culture first
  lives on a landmass, an epidemic spreads from its cells there at
  EPIDEMIC_SPEED_KM_YR; natives in each cell it reaches die at an extra rate
  that removes EPIDEMIC_DEATH_SHARE of them over EPIDEMIC_YEARS, and they do
  not grow meanwhile. A native culture meets each landmass's epidemic once.
- *Settler mortality* (after Acemoglu, Johnson & Robinson 2001). Colonists
  die at the excess settler mortality of their cells, rising with malaria
  suitability, so they settle temperate land and stay few in the wet tropics.
- *Home power.* A colony stays tied to its home power (a state on the
  planet, or one off it) until independence, at a rate that rises with
  historical time and earlier independences (Strang 1991) and that turmoil at
  home (war, the home state's fall) hastens; while tied its states gain the home's power and
  a home state on the planet sends emigrants.
"""

from __future__ import annotations

import math

import numpy as np

from ... import heuristics as h
from .population import landmasses
from .state import HistoryPolity


def found_colony_state(ctx, state, row: int) -> None:
    """Found a colony's state at the cell holding most of its founders; remember where the founders landed."""
    people = state.population[row]
    if people.sum() <= 0:
        return
    cells = np.flatnonzero(people > 0)
    ctx.cache.setdefault("landings", {})[row] = (cells, people[cells] / people[cells].sum())
    cell = int(np.argmax(people))
    culture = ctx.cultures[row]
    polity = HistoryPolity(id=len(state.polities), culture=culture.id, capital=cell,
                           founded_years_ago=state.years_ago, cohesion=h.POLITY_COHESION_START)
    state.polities.append(polity)
    state.owner[cell] = polity.id
    lat, lon = ctx.world.grid.lat[cell], ctx.world.grid.lon[cell]
    ctx.chronicle.add(state.years_ago, "colony_state",
                      f"{culture.name or f'Culture {culture.id}'} found a colony at {lat:.1f}°, {lon:.1f}°",
                      culture=culture.id, polity=polity.id, cell=cell)


def immigration_step(ctx, state, years: float) -> None:
    """Process: add each organic colony's immigrants at its landing cells."""
    from .run import founding_date

    for row, c in enumerate(ctx.cultures):
        if not state.active[row] or c.inhabitants != "colonist" or c.immigrants_per_year <= 0:
            continue
        landed = founding_date(c, ctx.history)
        until = landed - c.immigration_years if c.immigration_years is not None else 0.0
        span = min(years, state.years_ago - until)
        if span <= 0:
            continue
        cells, share = ctx.cache.get("landings", {}).get(row, (np.empty(0, dtype=np.int64), np.empty(0)))
        if cells.size:
            state.population[row, cells] += c.immigrants_per_year * span * share


class Epidemics:
    """Contact dates per native culture and cell, and the landmasses each has met colonists on."""

    def __init__(self, ctx):
        """Label the landmasses; no contact yet."""
        n = ctx.cov.land.size
        self.mass = landmasses(ctx.world)
        self.contact = {i: np.full(n, np.nan) for i, c in enumerate(ctx.cultures) if c.inhabitants == "native"}
        self.met: set[int] = set()

    def running(self, years_ago: float, years: float) -> dict:
        """Return, per native culture row, the cells where an epidemic runs during the step starting ``years_ago``."""
        out = {}
        for row, contact in self.contact.items():
            since = contact - years_ago
            sick = np.isfinite(since) & (since + years > 0.0) & (since < h.EPIDEMIC_YEARS)
            if sick.any():
                out[row] = sick
        return out

    def step(self, ctx, state, years: float) -> None:
        """Start epidemics on newly reached landmasses and apply the deaths of the step."""
        colonists = [i for i, c in enumerate(ctx.cultures) if c.inhabitants == "colonist" and state.active[i]]
        if not colonists or not self.contact:
            return
        present = state.population[colonists].sum(axis=0) > 0
        pts = ctx.world.grid.points
        for m in sorted(set(np.unique(self.mass[present]).tolist()) - self.met - {-1}):
            self.met.add(m)
            first = np.flatnonzero(present & (self.mass == m))
            on = np.flatnonzero(self.mass == m)
            d = ctx.cov.radius_km * np.arccos(np.clip(pts[on] @ pts[first].T, -1.0, 1.0)).min(axis=1)
            arrival = state.years_ago - d / h.EPIDEMIC_SPEED_KM_YR
            hit = 0.0
            sources = {ctx.cultures[i].home_culture for i in colonists}
            for row, contact in self.contact.items():
                if ctx.cultures[row].id in sources:                  # the colonists' own people share their diseases
                    continue
                contact[on] = np.where(np.isnan(contact[on]), arrival, contact[on])
                hit += state.population[row][on].sum()
            if hit >= h.EPIDEMIC_CHRONICLE_PEOPLE:
                c = ctx.cultures[colonists[0]]
                ctx.chronicle.add(state.years_ago, "epidemic",
                                  f"Diseases brought by {c.name or f'Culture {c.id}'} spread among {hit:,.0f} "
                                  f"natives on landmass {m + 1}", culture=c.id, landmass=m)
        rate = -math.log(1.0 - h.EPIDEMIC_DEATH_SHARE) / h.EPIDEMIC_YEARS
        for row, contact in self.contact.items():
            since = contact - state.years_ago                  # years since the epidemic reached the cell
            start = np.clip(since, 0.0, None)
            end = np.clip(since + years, None, h.EPIDEMIC_YEARS)
            exposed = np.where(np.isfinite(since), np.clip(end - start, 0.0, None), 0.0)
            state.population[row] *= np.exp(-rate * exposed)


def home_polity(ctx, state, culture) -> int:
    """Return the id of a colony's home state: the largest living state of the culture its origin names, or −1
    (a home off the planet)."""
    if culture.home_culture is not None:
        home = culture.home_culture
    else:
        named = [c.id for c in ctx.cultures if c.name and culture.origin and c.name.lower() == culture.origin.lower()
                 and c.id != culture.id]
        if not named:
            return -1
        home = named[0]
    owned = np.bincount(state.owner[state.owner >= 0], minlength=len(state.polities))
    states = [p for p in state.polities if p.culture == home and p.alive and owned[p.id] > 0]
    return max(states, key=lambda p: owned[p.id]).id if states else -1


def home_step(ctx, state, years: float) -> None:
    """Process: keep each organic colony tied to its home power until independence; send emigrants from home.

    A tied colony becomes independent at the rate of ``independence_rate``,
    TURMOIL_WAR_FACTOR times faster while its home state is at war (a home off
    the planet is at war in a share ABSTRACT_HOME_WAR_SHARE of the years, drawn
    per step), or at once when its home state ends. A given
    ``colony.independence_years`` overrides this. While tied, a home state on
    the planet sends COLONY_EMIGRATION_RATE of its people a year.
    """
    from .run import founding_date

    homes = ctx.cache.setdefault("homes", {})
    linked = ctx.cache.setdefault("linked", set())
    freed = ctx.cache.setdefault("independent", set())
    rng = ctx.rng("independence", state.step)
    wars = ctx.cache["polities"].wars.active if "polities" in ctx.cache else {}
    fighting = {x for k in wars for x in (int(k >> 32), int(k & 0xFFFFFFFF))}
    colonies = [(row, c) for row, c in enumerate(ctx.cultures) if c.inhabitants == "colonist" and state.active[row]]
    if not colonies:
        return
    first = max(founding_date(c, ctx.history) for _, c in colonies)
    first = max(first, ctx.cache.setdefault("first_colony", first))
    ctx.cache["first_colony"] = first
    for row, c in colonies:
        if c.id in freed or state.population[row].sum() <= 0:
            continue
        if c.id not in homes:
            homes[c.id] = home_polity(ctx, state, c)
        home = homes[c.id]
        elapsed = founding_date(c, ctx.history) - state.years_ago
        cause = None
        if c.independence_years is not None:
            if elapsed >= c.independence_years:
                cause = "at the appointed time"
        elif home >= 0 and not state.polities[home].alive:
            cause = f"as their home state {home} falls"
        else:
            war = home in fighting if home >= 0 else rng.random() < h.ABSTRACT_HOME_WAR_SHARE
            rate = independence_rate(ctx, c, first - state.years_ago) * (h.TURMOIL_WAR_FACTOR if war else 1.0)
            if rng.random() < -np.expm1(-rate * years):
                cause = "while their home state is at war" if war else "in a quiet time at home"
        if cause is None:
            linked.add(c.id)
            if home >= 0:
                _emigrate(ctx, state, row, home, years)
            continue
        linked.discard(c.id)
        freed.add(c.id)
        ctx.chronicle.add(state.years_ago, "independence",
                          f"{c.name or f'Culture {c.id}'} break with their home power {cause}, after "
                          f"{elapsed:,.0f} years and {state.population[row].sum():,.0f} strong", culture=c.id)


def empire(ctx, culture) -> int:
    """Return the id of the culture at the root of a colonist culture's line of homes (itself if it has none)."""
    by_id = {c.id: c for c in ctx.cultures}
    seen = set()
    while culture.home_culture is not None and culture.home_culture in by_id and culture.id not in seen:
        seen.add(culture.id)
        culture = by_id[culture.home_culture]
    return culture.id


def independence_rate(ctx, culture, since_first: float) -> float:
    """Return a tied colony's independence rate per year in quiet times (Strang 1991, model A).

    exp(DECOLONISATION_LOG_RATE + DECOLONISATION_TIME_SLOPE × years since the
    history's first colony (at most DECOLONISATION_TIME_MAX) + DECOLONISATION_WITHIN_EMPIRE × earlier
    independences in its empire + DECOLONISATION_OUTSIDE_EMPIRE × the others; each count at most
    DECOLONISATION_EVENTS_MAX).
    """
    freed = ctx.cache.get("independent", set())
    by_id = {c.id: c for c in ctx.cultures}
    mine = empire(ctx, culture)
    within = sum(1 for f in freed if f in by_id and empire(ctx, by_id[f]) == mine)
    outside = len(freed) - within
    within, outside = min(within, h.DECOLONISATION_EVENTS_MAX), min(outside, h.DECOLONISATION_EVENTS_MAX)
    t = min(max(since_first, 0.0), h.DECOLONISATION_TIME_MAX)
    return math.exp(h.DECOLONISATION_LOG_RATE + h.DECOLONISATION_TIME_SLOPE * t
                    + h.DECOLONISATION_WITHIN_EMPIRE * within + h.DECOLONISATION_OUTSIDE_EMPIRE * outside)


def _emigrate(ctx, state, row: int, home: int, years: float) -> None:
    """Move COLONY_EMIGRATION_RATE of the home state's people of its culture to the colony's landing cells."""
    cells, share = ctx.cache.get("landings", {}).get(row, (np.empty(0, dtype=np.int64), np.empty(0)))
    if not cells.size:
        return
    home_row = state.index(state.polities[home].culture)
    inside = state.owner == home
    leaving = state.population[home_row] * inside * min(h.COLONY_EMIGRATION_RATE * years, 0.5)
    total = leaving.sum()
    if total <= 0:
        return
    state.population[home_row] -= leaving
    state.population[row, cells] += total * share


def settler_mortality(world, cov) -> np.ndarray:
    """Return the settler mortality of every cell (deaths per 1000 per year) from its malaria suitability.

    Suitability is a thermal response (0 at SETTLER_MALARIA_T_C[0], 1 at the
    optimum [1], 0 again at [2]) times a wetness ramp of precipitation over
    potential evaporation (SETTLER_MALARIA_WETNESS); mortality runs
    log-linearly from SETTLER_MORTALITY_MIN at 0 to SETTLER_MORTALITY_MAX at 1.
    """
    from .suitability import ramp

    t0, t1, t2 = h.SETTLER_MALARIA_T_C
    t = cov.temperature_c
    thermal = np.where(t <= t1, ramp(t, t0, t1), 1.0 - ramp(t, t1, t2))
    s = world.surface
    if "precipitation" in s and "potential_evaporation" in s:
        wet = ramp(s["precipitation"].values / np.maximum(s["potential_evaporation"].values, 1e-6),
                   *h.SETTLER_MALARIA_WETNESS)
    else:
        wet = np.zeros(t.size)
    lo, hi = math.log(h.SETTLER_MORTALITY_MIN), math.log(h.SETTLER_MORTALITY_MAX)
    return np.exp(lo + (hi - lo) * thermal * wet)


def settler_mortality_step(ctx, state, years: float) -> None:
    """Process: colonists die at the settler mortality of their cells above the temperate baseline."""
    if "settler_mortality" not in ctx.cache:
        m = settler_mortality(ctx.world, ctx.cov)
        ctx.cache["settler_mortality"] = np.clip(m - h.SETTLER_MORTALITY_BASELINE, 0.0, None) / 1000.0
    excess = ctx.cache["settler_mortality"]
    for row, c in enumerate(ctx.cultures):
        if c.inhabitants == "colonist" and state.active[row]:
            state.population[row] *= np.exp(-excess * years)


def epidemic_step(ctx, state, years: float) -> None:
    """Process: contact epidemics among natives."""
    if "epidemics" not in ctx.cache:
        ctx.cache["epidemics"] = Epidemics(ctx)
    ctx.cache["epidemics"].step(ctx, state, years)
