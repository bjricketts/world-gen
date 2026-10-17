"""Cultures in contact in an organic history: technology transfer and language shift.

- *Supremacy* (heuristic). The first time a culture's states rule people of
  another, it is drawn whether they rule them as a people apart, more likely
  the further ahead the rulers are. Supremacist rulers pass on no technology,
  and their subjects shift language towards them without bilingualism.
- *Technology transfer* (after Comin & Hobijn 2010). Where a culture lives in
  a cell ruled by a state of another culture with a higher level in force
  there, it takes that level (at its bottom) at a rate
  1 / TECH_ADOPTION_LAG_YEARS[level] per year; its own level fronts then carry
  it on. Natives take energy (their transport and communication follow it);
  colonists take each axis.
- *Language shift* (Abrams & Strogatz 2003). In a cell where two cultures
  live, the share x of one among the two changes as
  dx/dt = c·(y·x^a·s − x·y^a·(1 − s)), s its status: LANGUAGE_STATUS for the
  culture at the higher energy level in the cell, or between equals for the
  one ruling the cell, and 0.5 otherwise. People who shift join the other culture in their
  cell. Every pair of cultures in a cell shifts in turn.
- *Bilingualism and creoles* (Mira & Paredes 2005). In cells a culture's
  states rule without supremacy over people of another culture living there,
  a share BILINGUAL_EASE of those taking up the other language become
  bilingual (``Contact._may_mix`` says which pairs can). The bilinguals of
  the pair form a new creole culture once they number CREOLE_MIN_PEOPLE;
  until then they are held apart and do not grow, and those left outside
  the pair's bilingual cells return to the two cultures. The creole takes the
  higher level of its parents in each cell and is a native culture.
"""

from __future__ import annotations

import math

import numpy as np

from ... import heuristics as h
from ..development import AXES, Development
from ..model import Culture
from ..names import make_phonology
from .state import in_force


class Contact:
    """The technology transfer and language shift steps."""

    def __init__(self, ctx):
        """Start with no supremacy drawn."""
        self.supremacist: dict[tuple[int, int], bool] = {}         # (ruling culture id, ruled culture id)
        self.pools: dict[tuple[int, int], np.ndarray] = {}          # (culture id, culture id) -> bilinguals per cell
        self.creoles: dict[tuple[int, int], int] = {}               # (culture id, culture id) -> creole culture id

    def draw_supremacy(self, ctx, state, rng) -> None:
        """Decide, the first time a culture's states rule people of another, whether it rules them as supremacists.

        The chance is SUPREMACY_BASE + SUPREMACY_PER_LEVEL × the rulers' lead in
        energy level (their people's mean over the mean of the ruled, in the
        ruled cells), within 0–1.
        """
        live = np.flatnonzero(state.active)
        if live.size < 2:
            return
        ruler = self._ruling_rows(state)
        for a in live:
            under = ruler == a
            if not under.any():
                continue
            for b in live:
                key = (state.culture_ids[a], state.culture_ids[b])
                if b == a or key in self.supremacist:
                    continue
                cells = np.flatnonzero(under & (state.population[b] > 0))
                if not cells.size:
                    continue
                lead = (np.average(in_force(state.development[a, 0, cells]), weights=state.population[a, cells] + 1e-12)
                        - np.average(in_force(state.development[b, 0, cells]), weights=state.population[b, cells]))
                chance = min(max(h.SUPREMACY_BASE + h.SUPREMACY_PER_LEVEL * max(lead, 0.0), 0.0), 1.0)
                self.supremacist[key] = bool(rng.random() < chance)
                if self.supremacist[key]:
                    ruling, ruled = ctx.cultures[a], ctx.cultures[b]
                    ctx.chronicle.add(state.years_ago, "supremacy",
                                      f"{ruling.name or f'Culture {ruling.id}'} rule "
                                      f"{ruled.name or f'Culture {ruled.id}'} as a people apart",
                                      culture=ruling.id, subject=ruled.id)

    def excluded(self, state, a: int, b: int) -> bool:
        """Return whether culture row ``a`` rules culture row ``b`` as supremacists."""
        return self.supremacist.get((state.culture_ids[a], state.culture_ids[b]), False)

    def transfer(self, ctx, state, rng, years: float) -> None:
        """Let cultures take higher levels from other cultures in or next to their cells."""
        live = np.flatnonzero(state.active)
        if live.size < 2:
            return
        lag = np.asarray(h.TECH_ADOPTION_LAG_YEARS)
        ruler = self._ruling_rows(state)
        ruled = np.flatnonzero(ruler >= 0)
        if not ruled.size:
            return
        r = ruler[ruled]
        present = state.population[r, ruled] > 0
        for i in live:
            ok = present & (r != i) & (state.population[i, ruled] > 0)
            for j in np.unique(r[ok]):
                if self.excluded(state, j, i):                          # supremacists share nothing
                    ok &= r != j
            if not ok.any():
                continue
            cells, rows = ruled[ok], r[ok]
            axes = (0, 1, 2) if ctx.cultures[i].inhabitants == "colonist" else (0,)
            for axis in axes:
                best = in_force(state.development[rows, axis, cells])
                mine = in_force(state.development[i, axis, cells])
                behind = best > mine
                if not behind.any():
                    continue
                target = best[behind]
                p = -np.expm1(-years / lag[np.clip(target, 0, lag.size - 1)])
                take = rng.random(target.size) < p
                where = cells[behind][take]
                state.development[i, axis, where] = np.maximum(state.development[i, axis, where], target[take] - 0.5)

    @staticmethod
    def _ruling_rows(state) -> np.ndarray:
        """Return the culture row of the state ruling each cell (−1 where none)."""
        owner = state.owner
        rows = {c: r for r, c in enumerate(state.culture_ids)}
        ruling = np.full(owner.size, -1)
        ruled = owner >= 0
        if ruled.any():
            culture_of = np.fromiter((rows[p.culture] for p in state.polities), dtype=np.int64,
                                     count=len(state.polities))
            ruling[ruled] = culture_of[owner[ruled]]
        return ruling

    def shift(self, ctx, state, years: float) -> None:
        """Move people between cultures that share cells by language shift, bilingualism and creoles."""
        live = np.flatnonzero(state.active)
        if live.size < 2:
            return
        ruling = self._ruling_rows(state)
        substeps = max(1, int(math.ceil(years)))
        rows = {c: r for r, c in enumerate(state.culture_ids)}
        parents = {rows[cid]: tuple(sorted(rows[x] for x in pair)) for pair, cid in self.creoles.items() if cid in rows}
        bilingual: dict[tuple[int, int], np.ndarray] = {}
        for k, i in enumerate(live):
            for j in live[k + 1:]:
                if j in parents.get(i, ()) or i in parents.get(j, ()):      # a creole and its parent
                    continue
                if not self._may_mix(ctx, state, parents, i, j):
                    continue
                cells = self._bilingual_cells(state, ruling, i, j)
                if cells.size:
                    bilingual[(i, j)] = cells
                    self._bilingual(ctx, state, ruling, i, j, cells, years, substeps)
        self._release(state, bilingual)
        for k, i in enumerate(live):
            for j in live[k + 1:]:
                both = (state.population[i] > 0) & (state.population[j] > 0)
                pair = (parents[i] if j in parents.get(i, ()) else
                        parents[j] if i in parents.get(j, ()) else (i, j))
                if pair in bilingual:
                    both[bilingual[pair]] = False
                both = np.flatnonzero(both)
                if both.size:
                    self._monolingual(state, ruling, i, j, both, years, substeps)
        self._found_creoles(ctx, state)

    def _may_mix(self, ctx, state, parents, i: int, j: int) -> bool:
        """Return whether a pair of culture rows can have bilinguals.

        Never two cultures of one language (colonists and their home). Beyond
        that CREOLE_CONTACT decides: "any" pair; "once", cultures that have no
        creole with a third culture, and no creole itself; "colonial", pairs
        with a colonist culture.
        """
        from .colonists import empire

        if empire(ctx, ctx.cultures[i]) == empire(ctx, ctx.cultures[j]):   # one language: colonists and their home
            return False
        mode = h.CREOLE_CONTACT
        if mode == "colonial":
            return "colonist" in (ctx.cultures[i].inhabitants, ctx.cultures[j].inhabitants)
        if mode == "once":
            if i in parents or j in parents:
                return False
            key = self._pair(state, i, j)
            joined = {x for pair in self.creoles for x in pair if pair != key}
            return state.culture_ids[i] not in joined and state.culture_ids[j] not in joined
        return True

    def _status(self, state, ruling, i: int, j: int, cells: np.ndarray) -> np.ndarray:
        """Return the status of culture row ``i`` against ``j`` in the cells: by energy level, else by rule."""
        ei = in_force(state.development[i, 0, cells])
        ej = in_force(state.development[j, 0, cells])
        rule = ruling[cells]
        by_rule = np.where(rule == i, h.LANGUAGE_STATUS, np.where(rule == j, 1.0 - h.LANGUAGE_STATUS, 0.5))
        by_level = np.where(ei > ej, h.LANGUAGE_STATUS, np.where(ei < ej, 1.0 - h.LANGUAGE_STATUS, 0.5))
        return np.where(by_level != 0.5, by_level, by_rule)

    def _monolingual(self, state, ruling, i: int, j: int, cells: np.ndarray, years: float, substeps: int) -> None:
        """Shift people between two cultures in the cells (Abrams & Strogatz)."""
        a, c = h.LANGUAGE_SHIFT_EXPONENT, h.LANGUAGE_SHIFT_RATE
        s = self._status(state, ruling, i, j, cells)
        total = state.population[i, cells] + state.population[j, cells]
        x = state.population[i, cells] / total
        dt = years / substeps
        for _ in range(substeps):
            y = 1.0 - x
            x = np.clip(x + dt * c * (y * x**a * s - x * y**a * (1.0 - s)), 0.0, 1.0)
        state.population[i, cells] = x * total
        state.population[j, cells] = (1.0 - x) * total

    def _pair(self, state, i: int, j: int) -> tuple[int, int]:
        """Return the key of a pair of culture rows."""
        a, b = state.culture_ids[i], state.culture_ids[j]
        return (a, b) if a < b else (b, a)

    def _bilinguals(self, state, i: int, j: int) -> np.ndarray:
        """Return the bilinguals of a pair per cell: its creole's people, or the pool held apart (a view)."""
        key = self._pair(state, i, j)
        if key in self.creoles:
            return state.population[state.index(self.creoles[key])]
        if key not in self.pools:
            self.pools[key] = np.zeros(state.population.shape[1])
        return self.pools[key]

    def _bilingual_cells(self, state, ruling, i: int, j: int) -> np.ndarray:
        """Return the cells where one of the pair rules the other without supremacy and two of the three groups live."""
        ruled = np.zeros(ruling.size, dtype=bool)
        if not self.excluded(state, i, j):
            ruled |= ruling == i
        if not self.excluded(state, j, i):
            ruled |= ruling == j
        if not ruled.any():
            return np.empty(0, dtype=np.int64)
        key = self._pair(state, i, j)
        pi, pj = state.population[i] > 0, state.population[j] > 0
        if key in self.creoles or key in self.pools:
            b = self._bilinguals(state, i, j) > 0
            two = (pi & pj) | (b & (pi | pj))
        else:
            two = pi & pj
        return np.flatnonzero(ruled & two)

    def _bilingual(self, ctx, state, ruling, i: int, j: int, cells: np.ndarray, years: float, substeps: int) -> None:
        """Shift people among two cultures and their bilinguals in the cells (Mira & Paredes 2005)."""
        a, c, k = h.LANGUAGE_SHIFT_EXPONENT, h.LANGUAGE_SHIFT_RATE, h.BILINGUAL_EASE
        bil = self._bilinguals(state, i, j)
        sx = self._status(state, ruling, i, j, cells)
        sy = 1.0 - sx
        X, Y, B = state.population[i, cells], state.population[j, cells], bil[cells]
        total = X + Y + B
        x, y = X / total, Y / total
        dt = years / substeps
        for _ in range(substeps):
            b = np.clip(1.0 - x - y, 0.0, 1.0)
            pxy, pxb = c * (1 - k) * sy * (1 - x) ** a, c * k * sy * (1 - x) ** a
            pyx, pyb = c * (1 - k) * sx * (1 - y) ** a, c * k * sx * (1 - y) ** a
            dx = y * pyx + b * pyx - x * (pxy + pxb)
            dy = x * pxy + b * pxy - y * (pyx + pyb)
            x = np.clip(x + dt * dx, 0.0, 1.0)
            y = np.clip(y + dt * dy, 0.0, 1.0 - x)
        state.population[i, cells] = x * total
        state.population[j, cells] = y * total
        bil[cells] = np.clip(1.0 - x - y, 0.0, 1.0) * total

    def _release(self, state, bilingual: dict) -> None:
        """Return bilinguals held apart outside this step's bilingual cells to their two cultures, half each."""
        done = {self._pair(state, i, j): cells for (i, j), cells in bilingual.items()}
        for key, pool in self.pools.items():
            loose = pool > 0
            if key in done:
                loose[done[key]] = False
            if not loose.any():
                continue
            for cid in key:
                state.population[state.index(cid), loose] += 0.5 * pool[loose]
            pool[loose] = 0.0

    def _found_creoles(self, ctx, state) -> None:
        """Make each pool of bilinguals of CREOLE_MIN_PEOPLE or more a creole culture."""
        from .run import add_culture

        for key in sorted(self.pools):
            pool = self.pools[key]
            if pool.sum() < h.CREOLE_MIN_PEOPLE:
                continue
            i, j = state.index(key[0]), state.index(key[1])
            development = np.maximum(state.development[i], state.development[j])
            cell = int(np.argmax(pool))
            new_id = ctx.new_culture_id()
            levels = development[:, cell]
            culture = Culture(id=new_id, name=None, inhabitants="native",
                              development=Development(**{a: int(in_force(levels[n])) for n, a in enumerate(AXES)}),
                              mode="organic", founded_years_ago=state.years_ago, creole_of=list(key),
                              phonology=make_phonology(ctx.seed, f"culture.{new_id}.phonology"))
            row = add_culture(ctx, state, culture, development)
            state.population[row] = pool
            del self.pools[key]
            self.creoles[key] = new_id
            first, second = ctx.cultures[i], ctx.cultures[j]
            ctx.chronicle.add(state.years_ago, "creole",
                              f"A creole people forms from {first.name or f'Culture {first.id}'} and "
                              f"{second.name or f'Culture {second.id}'} around {ctx.world.grid.lat[cell]:.1f}°, "
                              f"{ctx.world.grid.lon[cell]:.1f}°", culture=new_id, cell=cell)


def contact(ctx) -> Contact:
    """Return the history's contact processes, created on first use."""
    if "contact" not in ctx.cache:
        ctx.cache["contact"] = Contact(ctx)
    return ctx.cache["contact"]


def transfer_step(ctx, state, years: float) -> None:
    """Process: supremacy drawn for new rulings, then technology transfer under rule."""
    contact(ctx).draw_supremacy(ctx, state, ctx.rng("supremacy", state.step))
    contact(ctx).transfer(ctx, state, ctx.rng("transfer", state.step), years)


def shift_step(ctx, state, years: float) -> None:
    """Process: language shift between cultures sharing cells."""
    contact(ctx).shift(ctx, state, years)
