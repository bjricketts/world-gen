"""Polities in an organic history: states that form, expand, fight each other, pass through successions and break apart.

Every cell has at most one state, whatever cultures live there; a state
belongs to the culture that founded it and may rule people of others.

- *Before states.* Occupied cells without a state hold sub-cell polities
  (bands, tribes, chiefdoms) as a count per culture and cell; they enter the
  effective number and the epoch maps but are not tracked one by one.
- *Formation.* A state forms in a cell without one where a culture has
  writing and lives densely enough, at a small rate per year. On steppe next
  to a state's farmland, herders with horses form confederations without
  writing or density, with full cohesion (after Turchin 2009's mirror empires);
  states with their capital on the steppe ride faster over it.
- *Expansion* (after Turchin et al. 2013). Each year every border cell of a
  state may attack a neighbouring occupied cell without a state. The
  attacker's power is 1 + β·c·S, S its area in Turchin's 100 km × 100 km cells
  and c its cohesion, weakened by exp(−d / reach) with d the travel time from
  its capital over the travel graph of its transport level, in km at that
  level's flat-ground speed (so slopes, rivers and seas weigh as they do for
  travellers), and the reach rising with the communication level at its
  capital (log-linear between the levels, so it grows within an era); the
  defender's is 1 plus γ times the cell's defensibility (Turchin's elevation
  term). The attack succeeds with probability (P_att − P_def)/(P_att + P_def).
- *Wars between states* (``wars.py``, after Cederman 2003).
- *Cohesion* (0–1, a heuristic stand-in for Turchin's ultrasocial traits)
  grows with the share of a state's cells on its frontier and falls with its size.
- *Secession* (heuristic). Connected cells of a state where another
  literate culture holds SECESSION_SHARE of the people break away as a state
  of that culture.
- *Successions and break-up* (rulers in ``rulers.py``). A state can break up
  only at the successions of its rulers, with a chance per 25 years of
  δ₀ + δ_S·S·(r₁/r)^k − δ_c·c (r the reach of its capital's communication, r₁
  that of writing), lower once it has primogeniture. It splits into its
  provinces: the capital's continues as the heir, the others become states.

A culture's first state, states passing POLITY_CHRONICLE_KM2 (not the heirs
of states that did), large wars, conquests of such states' capitals and their
break-ups enter the chronicle.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from ... import heuristics as h
from ..roads import TravelGraph, tobler_kmh
from .rulers import province_count, provinces, reign_years, succession_times
from .state import HistoryPolity, in_force


def subpolity_km2(energy: np.ndarray) -> np.ndarray:
    """Return the typical area of a sub-cell polity: bands and tribes for foragers, chiefdoms once farming."""
    return np.where(in_force(energy) >= 1, h.SUBPOLITY_FARMING_KM2, h.SUBPOLITY_FORAGING_KM2)


def subpolity_counts(people: np.ndarray, owner: np.ndarray, energy: np.ndarray, cell_area_km2: float) -> np.ndarray:
    """Return the number of sub-cell polities in each occupied cell without a state (0 elsewhere)."""
    free = (people > 0) & (owner < 0)
    return np.where(free, np.maximum(cell_area_km2 / subpolity_km2(energy), 1.0), 0.0)


def effective_number_with_subpolities(owner: np.ndarray, people: np.ndarray, energy: np.ndarray,
                                      cell_area_km2: float) -> float:
    """Return the effective number of polities, (Σ A)² / Σ A², counting states and sub-cell polities."""
    live = (owner >= 0) & (people > 0)
    areas = np.bincount(owner[live]).astype(float) * cell_area_km2 if live.any() else np.zeros(0)
    areas = areas[areas > 0]
    counts = subpolity_counts(people, owner, energy, cell_area_km2)
    sub = counts > 0
    sub_area = cell_area_km2 / counts[sub]
    total = areas.sum() + cell_area_km2 * sub.sum()
    squares = (areas**2).sum() + (counts[sub] * sub_area**2).sum()
    return float(total**2 / squares) if squares > 0 else 0.0


def steppe_cells(world) -> np.ndarray:
    """Return which land cells have a steppe climate (Köppen BSk or BSh); none on a world without climate classes."""
    from ...biosphere import KOPPEN_CODES

    s = world.surface
    if "koppen" not in s:
        return np.zeros(world.grid.size, dtype=bool)
    codes = [KOPPEN_CODES.index(c) for c in h.STEPPE_KOPPEN]
    return np.isin(s["koppen"].values, codes) & ~s["ocean"].values.astype(bool)


def reach_km(communication: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Return the reach (km) at real-valued communication levels: the anchors apply at the integers, log-linear between."""
    c = np.clip(np.asarray(communication, dtype=float), 0.0, anchors.size - 1)
    return np.exp(np.interp(c, np.arange(anchors.size), np.log(anchors)))


def arc_km(points: np.ndarray, a, b, radius_km: float) -> np.ndarray:
    """Return great-circle distances between cells ``a`` and ``b`` (arrays of indices)."""
    return radius_km * np.arccos(np.clip((points[a] * points[b]).sum(axis=-1), -1.0, 1.0))


class Polities:
    """The polity processes of an organic history, with the grid's land edges."""

    def __init__(self, ctx):
        """Prepare the land edges, cell positions, defensibility and the war state."""
        from .wars import Wars

        world, cov = ctx.world, ctx.cov
        self.stress = None
        coo = world.grid.neighbours.tocoo()
        keep = cov.land[coo.row] & cov.land[coo.col]
        self.er, self.ec = coo.row[keep], coo.col[keep]
        self.points = world.grid.points
        self.lat, self.lon = world.grid.lat, world.grid.lon
        self.radius_km = cov.radius_km
        self.area = cov.cell_area_km2
        self.defence = ctx.suit.defensibility if ctx.suit is not None else np.zeros(cov.land.size)
        self.reach = np.asarray(h.POLITY_REACH_KM, dtype=float)
        self.world = world
        self.steppe = steppe_cells(world)
        self._graphs: dict[tuple[int, bool], tuple] = {}  # (transport, mounted) -> (travel-time matrix, flat km/h)
        self._travel: dict[tuple[int, int], np.ndarray] = {}   # (polity, transport) -> travel km from its capital
        self.wars = Wars(self)
        self._capitals = np.zeros(0, dtype=np.int64)     # per polity id; capitals and cultures never change
        self._cultures = np.zeros(0, dtype=np.int64)
        self._founders: set[int] = set()                 # cultures that have founded a state
        self.linked: set[int] = set()                    # colonist cultures tied to a home power
        self.homes: dict[int, int] = {}                  # colonist culture id -> home state id (−1: off the planet)

    def step(self, ctx, state, years: float) -> None:
        """Form, expand, fight, change cohesion and pass the successions of the states."""
        rng = ctx.rng("polity", state.step)
        self.linked = ctx.cache.get("linked", set())
        self.homes = ctx.cache.get("homes", {})
        self.stress = ctx.cache.get("stress")
        people = state.population[state.active].sum(axis=0)
        state.owner[people <= 0] = -1
        for row in np.flatnonzero(state.active):
            self._form(ctx, state, int(row), rng, years)
        if not (state.owner >= 0).any():
            return
        self._expand(ctx, state, rng, years, people)
        self.wars.step(ctx, state, rng, years)
        self._cohesion(ctx, state, years)
        self._successions(ctx, state, rng, years)
        if state.active.sum() > 1:
            self._secessions(ctx, state)

    def _form(self, ctx, state, row: int, rng, years: float) -> None:
        """Found single-cell states in dense, literate cells without a state."""
        owner, people = state.owner, state.population[row]
        communication = in_force(state.development[row, 2])
        dense = people / self.area >= h.POLITY_FORM_DENSITY
        free = (owner < 0) & (communication >= h.POLITY_FORM_COMMUNICATION) & dense
        new = np.flatnonzero(free & (rng.random(owner.size) < -np.expm1(-h.POLITY_FORM_RATE * years)))
        culture = ctx.cultures[row]
        first = len(state.polities) if culture.id not in self._founders else -1
        if new.size:
            self._founders.add(culture.id)
        nomads = np.setdiff1d(self._confederations(state, row, rng, years), new)
        if nomads.size:
            self._founders.add(culture.id)
        cohesion = np.r_[np.full(new.size, h.POLITY_COHESION_START), np.full(nomads.size, h.STEPPE_CONFEDERATION_COHESION)]
        for c, start in zip(np.r_[new, nomads].astype(int), cohesion):
            polity = HistoryPolity(id=len(state.polities), culture=culture.id, capital=int(c),
                                   founded_years_ago=state.years_ago, cohesion=float(start))
            state.polities.append(polity)
            owner[c] = polity.id
            if polity.id == first:
                ctx.chronicle.add(state.years_ago, "first_state",
                                  f"The first state arises at {self.lat[c]:.1f}°, {self.lon[c]:.1f}°",
                                  culture=culture.id, polity=polity.id, cell=int(c))

    def _confederations(self, state, row: int, rng, years: float) -> np.ndarray:
        """Return the steppe cells where a culture's herders form a confederation this step (after Turchin 2009's
        mirror empires): stateless occupied steppe next to a state's farmland, with horses (transport level
        STEPPE_RIDE_TRANSPORT) and energy STEPPE_CONFEDERATION_ENERGY, at STEPPE_CONFEDERATION_RATE per year,
        needing neither writing nor density."""
        owner, people = state.owner, state.population[row]
        if not self.steppe.any():
            return np.zeros(0, dtype=np.int64)
        able = (in_force(state.development[row, 1]) >= h.STEPPE_RIDE_TRANSPORT) & \
               (state.development[row, 0] >= h.STEPPE_CONFEDERATION_ENERGY)
        near = np.zeros(owner.size, dtype=bool)
        edge = (owner[self.er] >= 0) & ~self.steppe[self.er]
        near[self.ec[edge]] = True
        cand = (owner < 0) & self.steppe & (people > 0) & able & near
        return np.flatnonzero(cand & (rng.random(owner.size) < -np.expm1(-h.STEPPE_CONFEDERATION_RATE * years)))

    def sizes(self, state) -> np.ndarray:
        """Return each polity's area S in POLITY_UNIT_KM2 (indexed by polity id)."""
        owner = state.owner
        return np.bincount(owner[owner >= 0], minlength=len(state.polities)) * (self.area / h.POLITY_UNIT_KM2)

    def powers(self, state) -> np.ndarray:
        """Return each polity's β·c·S plus the support of a home power (indexed by polity id)."""
        cohesion = np.fromiter((p.cohesion for p in state.polities), dtype=float, count=len(state.polities))
        return h.POLITY_POWER_BETA * cohesion * self.sizes(state) + self.home_support(state)

    def home_support(self, state) -> np.ndarray:
        """Return, for the states of colonies still tied to a home power, the home's power projected to their
        capital ((1 + β·c·S)·exp(−d / reach) of the home state), or HOME_POWER for a home off the planet; else 0."""
        out = np.zeros(len(state.polities))
        if not self.linked:
            return out
        tied = [p for p in state.polities if p.alive and p.culture in self.linked]
        if not tied:
            return out
        cohesion = np.fromiter((p.cohesion for p in state.polities), dtype=float, count=len(state.polities))
        own = h.POLITY_POWER_BETA * cohesion * self.sizes(state)
        capitals, reach = self.capital_reach(state)
        for p in tied:
            home = self.homes.get(p.culture, -1)
            if home < 0:
                out[p.id] = h.HOME_POWER
            elif state.polities[home].alive:
                d = arc_km(self.points, capitals[home], capitals[p.id], self.radius_km)
                out[p.id] = (1.0 + own[home]) * np.exp(-d / reach[home])
        return out

    def capital_reach(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Return each polity's capital cell and the reach (km) of the communication there."""
        n = len(state.polities)
        if self._capitals.size < n:
            rows_of = {c: i for i, c in enumerate(state.culture_ids)}
            new = state.polities[self._capitals.size:]
            self._capitals = np.r_[self._capitals, [p.capital for p in new]].astype(np.int64)
            self._cultures = np.r_[self._cultures, [rows_of[p.culture] for p in new]].astype(np.int64)
        capitals, rows = self._capitals[:n], self._cultures[:n]
        return capitals, reach_km(state.development[rows, 2, capitals], self.reach)

    def projected(self, state, power, attackers, targets, capital_reach=None) -> np.ndarray:
        """Return attackers' power at target cells: (1 + β·c·S)·exp(−d / reach), d the travel distance from the
        capital (``travel_km``)."""
        capitals, reach = capital_reach if capital_reach is not None else self.capital_reach(state)
        out = np.empty(np.size(attackers))
        rows = self._cultures[:len(state.polities)]
        for a in np.unique(attackers):
            mine = attackers == a
            transport = int(np.clip(in_force(state.development[rows[a], 1, capitals[a]]), 0, len(h.LAND_SPEED_FACTOR) - 1))
            d = self.travel_km(state, int(a), int(capitals[a]), transport, float(reach[a]))[targets[mine]]
            out[mine] = (1.0 + power[a]) * np.exp(-d / reach[a])
        return out

    def travel_km(self, state, polity: int, capital: int, transport: int, reach: float) -> np.ndarray:
        """Return the travel time from a capital to every cell at a transport level, as km at the level's
        flat-ground speed (slopes, river crossings and sea passages lengthen it), out to
        POLITY_TRAVEL_LIMIT_REACHES reaches (beyond: infinite). A state with its capital on the steppe rides
        STEPPE_RIDE_FACTOR times faster between steppe cells from transport level STEPPE_RIDE_TRANSPORT on.
        Kept per polity and level while the polity lives."""
        key = (polity, transport)
        if key not in self._travel:
            mounted = bool(self.steppe[capital]) and transport >= h.STEPPE_RIDE_TRANSPORT
            if (transport, mounted) not in self._graphs:
                g = TravelGraph(self.world, transport)
                hours = g.base_hours
                if mounted:
                    hours = np.where(self.steppe[g.row] & self.steppe[g.col], hours / h.STEPPE_RIDE_FACTOR, hours)
                self._graphs[transport, mounted] = (g.matrix(hours),
                                                    float(tobler_kmh(np.array(0.0))) * h.LAND_SPEED_FACTOR[transport])
            matrix, speed = self._graphs[transport, mounted]
            hours = dijkstra(matrix, indices=capital, limit=h.POLITY_TRAVEL_LIMIT_REACHES * reach / speed)
            self._travel[key] = (hours * speed).astype(np.float32)
            if len(self._travel) > h.POLITY_TRAVEL_CACHE:
                alive = {p.id for p in state.polities if p.alive}
                for k in [k for k in self._travel if k[0] not in alive]:
                    del self._travel[k]
        return self._travel[key]

    def annex(self, ctx, state, cells, winners) -> None:
        """Give cells to their conquerors; taking a state's capital annexes the whole state."""
        owner = state.owner
        for cell, winner in zip(cells.tolist(), winners.tolist()):
            if not state.polities[winner].alive:          # absorbed earlier in this batch
                continue
            loser = int(owner[cell])
            if loser == winner:                            # came with a state annexed earlier in this batch
                continue
            owner[cell] = winner
            q = state.polities[loser] if loser >= 0 else None
            if q is None or not q.alive or q.capital != cell:
                continue
            area = float((owner == loser).sum() + 1) * self.area
            owner[owner == loser] = winner
            self._end(state, int(loser))
            self.wars.polity_ended(int(loser))
            if area >= h.POLITY_CHRONICLE_KM2:
                ctx.chronicle.add(state.years_ago, "conquest",
                                  f"State {winner} takes the capital of state {loser} and its {area:,.0f} km²",
                                  culture=state.polities[winner].culture, polity=int(winner), loser=int(loser))

    def _expand(self, ctx, state, rng, years: float, people) -> None:
        """Let border cells of states attack neighbouring occupied cells without a state."""
        owner = state.owner
        oa, ob = owner[self.er], owner[self.ec]
        cand = (oa >= 0) & (ob < 0) & (people[self.ec] > 0)
        ea, eb = self.er[cand], self.ec[cand]
        tries = rng.random(ea.size) < -np.expm1(-h.POLITY_ATTACK_RATE * years)
        ea, eb = ea[tries], eb[tries]
        if not ea.size:
            return
        att = owner[ea]
        p_att = self.projected(state, self.powers(state), att, eb)
        p_def = 1.0 + h.POLITY_DEFENCE_GAMMA * self.defence[eb]
        win = rng.random(ea.size) < (p_att - p_def) / (p_att + p_def)
        taken, by = first_claims(rng, eb[win], att[win])
        owner[taken] = by

    def _cohesion(self, ctx, state, years: float) -> None:
        """Raise cohesion with a state's frontier share and lower it with its size; record peak areas."""
        owner = state.owner
        size = self.sizes(state)
        cells = np.bincount(owner[owner >= 0], minlength=size.size)
        border_cells = np.unique(self.er[(owner[self.er] >= 0) & (owner[self.er] != owner[self.ec])])
        border = np.bincount(owner[border_cells], minlength=size.size)
        frontier = np.where(cells > 0, border / np.maximum(cells, 1), 0.0)
        periods = years / h.POLITY_PERIOD_YEARS
        for p in [q for q in state.polities if q.alive]:
            if size[p.id] <= 0:
                self._end(state, p.id)
                self.wars.polity_ended(p.id)
                continue
            area = float(size[p.id] * h.POLITY_UNIT_KM2)
            if p.peak_km2 < h.POLITY_CHRONICLE_KM2 <= area and not self._great_ancestor(state, p):
                ctx.chronicle.add(state.years_ago, "great_power",
                                  f"State {p.id}, founded {p.founded_years_ago:,.0f} years ago with its capital at "
                                  f"{self.lat[p.capital]:.1f}°, {self.lon[p.capital]:.1f}°, holds {area / 1e6:.1f} million km²",
                                  culture=p.culture, polity=p.id, cell=p.capital)
            p.peak_km2 = max(p.peak_km2, area)
            change = h.POLITY_COHESION_GAIN * frontier[p.id] - h.POLITY_COHESION_LOSS * np.log1p(size[p.id])
            p.cohesion = float(np.clip(p.cohesion + periods * change, h.POLITY_COHESION_MIN, 1.0))

    def _successions(self, ctx, state, rng, years: float) -> None:
        """Pass the rulers' successions; at a succession a state may break into its provinces.

        The break-up chance per 25 years is δ₀ + δ_S·S·(r₁/r)^k − δ_c·c, times
        1 + CONFLICT_PER_SIGMA × the state's mean climate anomaly in the step; it is
        spread over the successions (one expected per mean reign) and cut by
        PRIMOGENITURE_BREAKUP_CUT once primogeniture is adopted.
        """
        size = self.sizes(state)
        _, reach = self.capital_reach(state)
        written = self.reach[h.POLITY_FORM_COMMUNICATION]
        shocked = self._shocked(state)
        for p in [q for q in state.polities if q.alive]:
            if not p.primogeniture and reach[p.id] >= written:
                p.primogeniture = bool(rng.random() < -np.expm1(-h.PRIMOGENITURE_ADOPT_RATE * years))
            mean_reign = reign_years(p.primogeniture)
            times = succession_times(rng, mean_reign, years)
            if not times.size:
                continue
            p.rulers += times.size
            p.reign_started_years_ago = state.years_ago - times[-1]
            self.wars.succession(p.id, rng)
            span = (written / reach[p.id]) ** h.POLITY_BREAKUP_REACH_EXP
            chance = np.clip((h.POLITY_BREAKUP_BASE + h.POLITY_BREAKUP_SIZE * size[p.id] * span
                              - h.POLITY_BREAKUP_COHESION * p.cohesion) * shocked[p.id], 0.0, 0.999)
            cut = 1.0 - h.PRIMOGENITURE_BREAKUP_CUT * p.primogeniture
            per_succession = -np.expm1(np.log1p(-chance) * mean_reign / h.POLITY_PERIOD_YEARS * cut)
            if rng.random() >= -np.expm1(np.log1p(-per_succession) * times.size):
                continue
            self._split(ctx, state, p, size[p.id], state.years_ago - times[0])

    def _shocked(self, state) -> np.ndarray:
        """Return each state's break-up factor from the step's climate anomaly over its cells."""
        n = len(state.polities)
        if self.stress is None or not self.stress.sigma.any():
            return np.ones(n)
        ruled = state.owner >= 0
        cells = np.bincount(state.owner[ruled], minlength=n)
        sigma = np.bincount(state.owner[ruled], weights=self.stress.sigma[ruled], minlength=n)
        return 1.0 + h.CONFLICT_PER_SIGMA * sigma / np.maximum(cells, 1)

    def _split(self, ctx, state, p, size: float, when: float) -> None:
        """Break a state into its provinces: the capital's continues as the heir, the others become states."""
        owner = state.owner
        cells = np.flatnonzero(owner == p.id)
        self._end(state, p.id)
        self.wars.polity_ended(p.id)
        prov = provinces(self.points, cells, p.capital, province_count(cells.size, self.area))
        people = state.population[state.index(p.culture)]
        parts = []
        for j in range(prov.max() + 1):
            part = cells[prov == j]
            if not part.size:
                continue
            capital = p.capital if j == 0 and owner[p.capital] == p.id else int(part[np.argmax(people[part])])
            new = HistoryPolity(id=len(state.polities), culture=p.culture, capital=capital,
                                founded_years_ago=state.years_ago, parent=p.id, cohesion=h.POLITY_COHESION_START,
                                primogeniture=p.primogeniture and j == 0)
            state.polities.append(new)
            owner[part] = new.id
            parts.append(new.id)
        if size * h.POLITY_UNIT_KM2 >= h.POLITY_CHRONICLE_KM2:
            ctx.chronicle.add(when, "break_up",
                              f"At a succession, state {p.id} breaks into {len(parts)} states after "
                              f"{p.founded_years_ago - when:,.0f} years and {p.rulers} rulers",
                              culture=p.culture, polity=p.id, parts=len(parts),
                              area_km2=round(float(size * h.POLITY_UNIT_KM2)))

    def _secessions(self, ctx, state) -> None:
        """Let connected cells of a state where another literate culture has SECESSION_SHARE of the people
        break away as a state of that culture."""
        owner, people = state.owner, state.population
        total = people.sum(axis=0)
        ruled = owner >= 0
        if not ruled.any():
            return
        self.capital_reach(state)                                  # fills the culture rows of the polities
        ruling = np.where(ruled, self._cultures[np.maximum(owner, 0)], -1)
        top = people.argmax(axis=0)
        share = people.max(axis=0) / np.maximum(total, 1e-12)
        cells = np.arange(owner.size)
        literate = in_force(state.development[top, 2, cells]) >= h.POLITY_FORM_COMMUNICATION
        cand = ruled & (top != ruling) & (share >= h.SECESSION_SHARE) & literate
        if cand.sum() < h.SECESSION_MIN_CELLS:
            return
        same = cand[self.er] & cand[self.ec] & (owner[self.er] == owner[self.ec]) & (top[self.er] == top[self.ec])
        graph = sparse.coo_matrix((np.ones(same.sum()), (self.er[same], self.ec[same])), shape=(owner.size,) * 2)
        _, label = connected_components(graph, directed=False)
        for group in np.unique(label[cand]):
            part = np.flatnonzero((label == group) & cand)
            if part.size < h.SECESSION_MIN_CELLS:
                continue
            row = int(top[part[0]])
            capital = int(part[np.argmax(people[row, part])])
            parent = int(owner[capital])
            new = HistoryPolity(id=len(state.polities), culture=state.culture_ids[row], capital=capital,
                                founded_years_ago=state.years_ago, parent=parent, cohesion=h.POLITY_COHESION_START)
            state.polities.append(new)
            owner[part] = new.id
            area = part.size * self.area
            if area >= h.POLITY_CHRONICLE_KM2 / 4:
                name = ctx.cultures[row].name or f"Culture {new.culture}"
                ctx.chronicle.add(state.years_ago, "secession",
                                  f"{name} of {area:,.0f} km² break away from state {parent}",
                                  culture=new.culture, polity=new.id, loser=parent)

    @staticmethod
    def _great_ancestor(state, polity) -> bool:
        """Return whether a state split from one that passed POLITY_CHRONICLE_KM2."""
        while polity.parent >= 0:
            polity = state.polities[polity.parent]
            if polity.peak_km2 >= h.POLITY_CHRONICLE_KM2:
                return True
        return False

    @staticmethod
    def _end(state, polity_id: int) -> None:
        """Mark a polity as ended at the state's date."""
        p = state.polities[polity_id]
        if p.alive:
            p.ended_years_ago = state.years_ago


def first_claims(rng, targets: np.ndarray, claimants: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each target cell once, with one of its claimants drawn at random."""
    if not targets.size:
        return targets, claimants
    order = np.lexsort((rng.random(targets.size), targets))
    first = np.r_[True, targets[order][1:] != targets[order][:-1]]
    return targets[order][first], claimants[order][first]


def polities(ctx) -> Polities:
    """Return the history's polity processes, created on first use."""
    if "polities" not in ctx.cache:
        ctx.cache["polities"] = Polities(ctx)
    return ctx.cache["polities"]


def polity_step(ctx, state, years: float) -> None:
    """Process: formation, expansion, wars, cohesion and successions of states."""
    polities(ctx).step(ctx, state, years)
