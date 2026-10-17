"""Settlements and roads carried through the kept epochs of an organic history: stable ids and founding dates.

- *The cell record.* While the history runs it records, for every cell, the
  date of its first farming settlement (a culture with farming in force lives
  there) and its people every SETTLEMENT_RECORD_YEARS.
- *Identity.* A settlement is known by its culture, cell and order of size in
  the cell. It keeps its id at every kept epoch that has a settlement with the
  same key, from the oldest epoch on; a key that goes missing at an epoch and
  returns later is a new settlement.
- *Founding.* A cell's largest settlement dates from the cell's first farming
  settlement (Bosker & Buringh's city seeds). Any other new settlement dates
  from the first recorded time after the previous kept epoch (or the start)
  at which the cell's people passed halfway from their number then to their
  number at the epoch.
- *Names.* A settlement or state that keeps its id keeps its first name; a
  new one whose name repeats a kept one is renamed.
- *Roads* date from the first kept epoch at which every edge they follow was a
  road.
"""

from __future__ import annotations

import numpy as np

from ... import heuristics as h
from ..names import Namer
from .state import in_force


class CellRecord:
    """Each cell's first farming settlement (years ago, NaN if none) and people every SETTLEMENT_RECORD_YEARS."""

    def __init__(self, size: int):
        """Start with no record."""
        self.settled_since = np.full(size, np.nan)
        self.times: list[float] = []
        self.people: list[np.ndarray] = []

    def update(self, state, when: float) -> None:
        """Record the state as it stands ``when`` years ago."""
        live = state.active
        people = state.population[live].sum(axis=0)
        farming = ((state.population[live] > 0) & (in_force(state.development[live, 0]) >= 1)).any(axis=0)
        self.settled_since[farming & np.isnan(self.settled_since)] = when
        if not self.times or self.times[-1] - when >= h.SETTLEMENT_RECORD_YEARS - 1e-9 or when == 0.0:
            self.times.append(float(when))
            self.people.append(people.astype(np.float32))

    def people_at(self, when: float) -> np.ndarray:
        """Return each cell's people at the last record at or before a date (zeros before the first)."""
        times = np.asarray(self.times)
        past = np.flatnonzero(times >= when - 1e-9)
        return self.people[past[-1]].astype(float) if past.size else np.zeros_like(self.settled_since)

    def passed(self, cell: int, level: float, after: float, until: float) -> float:
        """Return the first recorded date later than ``after`` (years ago) at which a cell's people reached
        ``level``, or ``until`` if none did by then."""
        for t, people in zip(self.times, self.people):
            if t < after - 1e-9 and t >= until - 1e-9 and people[cell] >= level:
                return t
        return until


def record_step(ctx, state, years: float) -> None:
    """Process: record the cells at the end of the step."""
    record = ctx.cache.setdefault("cells", CellRecord(state.population.shape[1]))
    record.update(state, state.years_ago - years)


def in_cell_order(settlements) -> np.ndarray:
    """Return each settlement's order of size within its culture and cell (0: the largest)."""
    n = len(settlements)
    order = np.lexsort((-settlements.population, settlements.cell, settlements.culture))
    key = np.stack([settlements.culture[order], settlements.cell[order]], axis=1)
    start = np.r_[True, (key[1:] != key[:-1]).any(axis=1)] if n else np.zeros(0, dtype=bool)
    group = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - first[group]
    return rank


def carry_through(civs, record: CellRecord, organic_ids: set, start_years_ago: float) -> None:
    """Give the settlements and roads of the kept epochs (oldest first) stable ids and founding dates, in place."""
    registry: dict[tuple, tuple[int, float]] = {}
    names: dict[int, str] = {}                       # settlement id -> name
    polity_names: dict[int, str] = {}                # organic state id -> name
    seen: set[tuple] = set()
    previous: set[tuple] = set()
    road_since: dict[tuple, float] = {}
    next_id = 0
    before = start_years_ago
    for civ in civs:
        now = civ.epoch_years_ago
        st = civ.settlements
        rank = in_cell_order(st)
        ids = np.empty(len(st), dtype=np.int64)
        founded = st.founded_years_ago.copy()
        then = record.people_at(before) if before < start_years_ago else np.zeros_like(record.settled_since)
        here = record.people_at(now)
        keys = []
        for k in range(len(st)):
            key = (int(st.culture[k]), int(st.cell[k]), int(rank[k]))
            keys.append(key)
            if key in previous:
                ids[k], founded[k] = registry[key]
                continue
            ids[k], next_id = next_id, next_id + 1
            if int(st.culture[k]) in organic_ids:
                c = key[1]
                if key[2] == 0 and key not in seen and np.isfinite(record.settled_since[c]):
                    founded[k] = max(float(record.settled_since[c]), now)
                else:
                    founded[k] = record.passed(c, 0.5 * (then[c] + here[c]), before, now)
            registry[key] = (int(ids[k]), float(founded[k]))
        seen.update(keys)
        previous = set(keys)
        _renumber(civ, ids)
        st.founded_years_ago = founded
        _keep_names(civ, names, polity_names, organic_ids)
        road_since = _date_roads(civ.roads, now, road_since)
        before = now


def _keep_names(civ, names: dict, polity_names: dict, organic_ids: set) -> None:
    """Give carried settlements and states their earlier names; rename new ones that would repeat a name."""
    st = civ.settlements
    new = np.array([int(i) not in names for i in st.id], dtype=bool)
    for k in np.flatnonzero(~new):
        st.name[k] = names[int(st.id[k])]
    carried_polity = [i for i, p in enumerate(civ.polities) if p.culture in organic_ids and p.id in polity_names]
    for i in carried_polity:
        civ.polities[i].name = polity_names[civ.polities[i].id]
    for c in {int(x) for x in st.culture}:
        mine = st.culture == c
        taken = set(st.name[mine & ~new]) | {civ.polities[i].name for i in carried_polity if civ.polities[i].culture == c}
        clash = np.flatnonzero(mine & new & np.isin(st.name, list(taken)))
        clash_p = [i for i, p in enumerate(civ.polities)
                   if p.culture == c and i not in carried_polity and p.name in taken]
        if not (clash.size or clash_p):
            continue
        culture = civ.culture(c)
        namer = Namer(culture.phonology, civ.seed, f"culture.{c}.epoch.{civ.epoch_years_ago:g}.renames")
        namer.used = set(st.name[mine]) | {p.name for p in civ.polities if p.culture == c}
        for k in clash:
            st.name[k] = namer.root()
        for i in clash_p:
            civ.polities[i].name = namer.root()
    for k in range(len(st)):
        names.setdefault(int(st.id[k]), str(st.name[k]))
    for p in civ.polities:
        if p.culture in organic_ids:
            polity_names.setdefault(p.id, p.name)


def _renumber(civ, ids: np.ndarray) -> None:
    """Replace settlement ids (their table positions until now) everywhere they are referenced."""
    def remap(values):
        values = np.asarray(values)
        return np.where(values >= 0, ids[np.maximum(values, 0)], -1)

    civ.settlements.id = ids
    civ.roads.a, civ.roads.b = remap(civ.roads.a), remap(civ.roads.b)
    civ.trade.a, civ.trade.b = remap(civ.trade.a), remap(civ.trade.b)
    for i, p in enumerate(civ.polities):
        if p.capital >= 0:
            civ.polities[i].capital = int(ids[p.capital])


def _date_roads(roads, now: float, since: dict) -> dict:
    """Date each road from the first epoch at which all its edges were roads; return the edge dates so far."""
    edges_now = {}
    for i in range(len(roads)):
        p = roads.path(i)
        pairs = [(int(min(a, b)), int(max(a, b))) for a, b in zip(p[:-1], p[1:])]
        dates = [since.get(e, now) for e in pairs]
        roads.founded_years_ago[i] = min(dates) if dates else now
        for e, d in zip(pairs, dates):
            edges_now[e] = d
    return edges_now
