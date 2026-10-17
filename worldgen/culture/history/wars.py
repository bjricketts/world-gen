"""Wars between states in an organic history, after Cederman (2003), with heuristic defensive alliances.

Wars are fought in rounds of WAR_ROUND_YEARS within each history step.

- *Onset.* A state at peace with a neighbour starts a war on it at
  WAR_ONSET_RATE per year (every round once on alert), with probability
  1 / (1 + (b / WAR_THRESHOLD)^(−WAR_THRESHOLD_SLOPE)) (Cederman's
  logistic threshold, 3:1), b the ratio of the two states' powers projected
  to their border, (1 + β·c·S)·exp(−d / reach) as in ``polities.py``.
- *Alert* (Cederman's context activation). States at war and their
  neighbours are on alert; a state leaves alert at WAR_ALERT_DECAY per year
  once no neighbour fights.
- *Battles.* Each round every border edge between belligerents carries an
  attack with probability WAR_BATTLE_RATE per year, won as in expansion,
  with the defender's power 1 + β·c·S + γ·defensibility. Taking a capital
  annexes the state.
- *End.* A war ends at WAR_END_RATE per year (Cederman's dropped
  campaigns), when the two states no longer border each other, or when one of
  them ends.
- *Size.* A war joins the conflict of any war its states are already in
  (Cederman's clusters); a conflict ends when its last war does. Its
  severity is WAR_DAMAGE times the people living in cells on the fighting
  fronts, per year of fighting (a stand-in for battle deaths).
- *Alliances* (heuristic). Two neighbouring states at peace that each border
  a state more than ALLIANCE_THREAT times as powerful ally at ALLIANCE_RATE
  per year. Allies of an attacked state that border the attacker join the
  war. An alliance ends at a succession in either state with probability
  ALLIANCE_BREAK_AT_SUCCESSION, when the two fight or when one ends.

Conflicts in which POLITY_CHRONICLE_KM2 of land changes hands or WAR_CHRONICLE_STATES states fight enter
the chronicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import heuristics as h

SHIFT = np.int64(1) << 32


def pair_key(a, b):
    """Return the key of an unordered pair of polity ids."""
    a, b = np.minimum(a, b), np.maximum(a, b)
    return np.asarray(a, dtype=np.int64) * SHIFT + np.asarray(b, dtype=np.int64)


def key_pair(key: int) -> tuple[int, int]:
    """Return the two polity ids of a pair key."""
    return int(key // SHIFT), int(key % SHIFT)


@dataclass
class Conflict:
    """A cluster of linked wars: when it began, the states drawn in, its severity and the cells that changed hands."""

    start: float
    states: set = field(default_factory=set)
    wars: int = 0                    # wars still going
    severity: float = 0.0
    cells: int = 0
    largest_km2: float = 0.0


@dataclass
class ConflictRecord:
    """A finished conflict."""

    start: float
    end: float
    states: int
    severity: float
    cells: int
    largest_km2: float


class Wars:
    """The wars, alerts and alliances between the states of a history."""

    def __init__(self, polities):
        """Start at peace."""
        self.p = polities
        self.active: dict[int, int] = {}          # pair key -> conflict id
        self.conflicts: dict[int, Conflict] = {}
        self.next_conflict = 0
        self.alert: set[int] = set()
        self.allies: set[int] = set()
        self.records: list[ConflictRecord] = []

    # -- events from the polity processes ------------------------------------------------------
    def polity_ended(self, polity: int) -> None:
        """Close the wars and alliances of a state that ended (its conflict stays open while others fight)."""
        for key in [k for k in self.active if polity in key_pair(k)]:
            self._close(key)
        self.allies = {k for k in self.allies if polity not in key_pair(k)}
        self.alert.discard(polity)

    def succession(self, polity: int, rng) -> None:
        """Let a new ruler drop each alliance with probability ALLIANCE_BREAK_AT_SUCCESSION."""
        for key in sorted(k for k in self.allies if polity in key_pair(k)):
            if rng.random() < h.ALLIANCE_BREAK_AT_SUCCESSION:
                self.allies.discard(key)

    # -- the step ------------------------------------------------------------------------------
    def step(self, ctx, state, rng, years: float) -> None:
        """Fight the step's wars in yearly rounds, then form alliances."""
        rounds = max(1, int(round(years / h.WAR_ROUND_YEARS)))
        dt = years / rounds
        people = state.population[state.active].sum(axis=0)
        owned = state.owner >= 0                          # fighting moves cells between states only
        keep = owned[self.p.er] & owned[self.p.ec]
        self.er, self.ec = self.p.er[keep], self.p.ec[keep]
        self.cr = self.p.capital_reach(state)
        self.cohesion = np.fromiter((p.cohesion for p in state.polities), dtype=float, count=len(state.polities))
        r = 0
        while r < rounds:
            forced = None
            if not self.active and not self.alert:        # at peace: jump to the next war's onset
                wait, forced = self._next_onset(state, rng)
                if not wait < (rounds - r) * dt:
                    break
                r += int(wait // dt)
            when = state.years_ago - r * dt
            if forced is not None:
                self._open(*forced, when, self.p.sizes(state) * h.POLITY_UNIT_KM2)
            self._round(ctx, state, rng, dt, people, when)
            self._close_finished(ctx, state, when - dt)
            r += 1
        self._ally(state, rng, years)

    def _onset_rates(self, state, ea, eb, a, b, keys, power):
        """Return the edge indices of neighbouring pairs at peace, their attacker, defender and war-onset rate per year
        (raised by climate anomalies on the border)."""
        pairs, first = np.unique(keys, return_index=True)
        busy = np.isin(pairs, self._keys(self.active)) | np.isin(pairs, self._keys(self.allies))
        idx = first[~busy]
        pa = self.p.projected(state, power, a[idx], eb[idx], self.cr)
        pb = self.p.projected(state, power, b[idx], ea[idx], self.cr)
        strong = np.where(pa >= pb, a[idx], b[idx])
        weak = np.where(pa >= pb, b[idx], a[idx])
        ratio = np.maximum(pa, pb) / np.maximum(np.minimum(pa, pb), 1e-300)
        alerted = np.isin(strong, self._keys(self.alert))
        likely = 1.0 / (1.0 + (ratio / h.WAR_THRESHOLD) ** (-h.WAR_THRESHOLD_SLOPE))
        onset = h.WAR_ONSET_RATE * self._conflict(ea, eb, keys, pairs)[~busy]
        return strong, weak, np.where(alerted, 1.0, onset) * likely

    def _conflict(self, ea, eb, keys, pairs) -> np.ndarray:
        """Return each pair's war-onset factor, 1 + CONFLICT_PER_SIGMA × the largest climate anomaly (in standard
        deviations) on its border in the step (Hsiang et al. 2013)."""
        stress = getattr(self.p, "stress", None)
        if stress is None or not stress.sigma.any():
            return np.ones(pairs.size)
        edge = np.maximum(stress.sigma[ea], stress.sigma[eb])
        worst = np.zeros(pairs.size)
        np.maximum.at(worst, np.searchsorted(pairs, keys), edge)
        return 1.0 + h.CONFLICT_PER_SIGMA * worst

    def _next_onset(self, state, rng):
        """Return the years until the next war begins in a world at peace and its attacker and defender."""
        ea, eb, a, b, keys = self._borders(state)
        if not keys.size:
            return np.inf, None
        strong, weak, rate = self._onset_rates(state, ea, eb, a, b, keys, self._power(state))
        total = rate.sum()
        if total <= 0:
            return np.inf, None
        k = int(np.searchsorted(np.cumsum(rate), rng.random() * total))
        return rng.exponential(1.0 / total), (int(strong[min(k, rate.size - 1)]), int(weak[min(k, rate.size - 1)]))

    def _power(self, state) -> np.ndarray:
        """Return each polity's β·c·S with the cohesion fixed for the step."""
        return h.POLITY_POWER_BETA * self.cohesion * self.p.sizes(state) + self.p.home_support(state)

    def _borders(self, state):
        """Return the directed land edges between different states and their pair keys."""
        owner = state.owner
        oa, ob = owner[self.er], owner[self.ec]
        inter = oa != ob
        ea, eb = self.er[inter], self.ec[inter]
        return ea, eb, owner[ea], owner[eb], pair_key(owner[ea], owner[eb])

    def _round(self, ctx, state, rng, dt: float, people, when: float) -> None:
        """One year of war (starting ``when`` years ago): end, start, fight, measure and alert."""
        ea, eb, a, b, keys = self._borders(state)
        if self.active:
            live = self._keys(self.active)
            for key in live[~np.isin(live, keys)].tolist():
                self._close(key)
        for key in [k for k in self.active if rng.random() < -np.expm1(-h.WAR_END_RATE * dt)]:
            self._close(key)
        if not keys.size:
            return
        power = self._power(state)
        self._start(state, rng, dt, ea, eb, a, b, keys, power, when)
        fighting = np.isin(keys, self._keys(self.active))
        if not fighting.any():
            self._cool(rng, dt, a, b, keys, fighting)
            return
        self._battle(ctx, state, rng, dt, ea[fighting], eb[fighting], keys[fighting], power, people)
        self._cool(rng, dt, a, b, keys, fighting)

    def _start(self, state, rng, dt, ea, eb, a, b, keys, power, when: float) -> None:
        """Start wars between neighbours at peace, the stronger attacking; defenders' allies join."""
        strong, weak, rate = self._onset_rates(state, ea, eb, a, b, keys, power)
        go = rng.random(rate.size) < -np.expm1(-rate * dt)
        if not go.any():
            return
        size = self.p.sizes(state) * h.POLITY_UNIT_KM2
        neighbours = None
        for attacker, defender in zip(strong[go], weak[go]):
            attacker, defender = int(attacker), int(defender)
            if int(pair_key(attacker, defender)) in self.active:
                continue
            conflict = self._open(attacker, defender, when, size)
            if neighbours is None:
                neighbours = set(np.unique(keys).tolist())
            for key in sorted(k for k in self.allies if defender in key_pair(k)):
                ally = sum(key_pair(key)) - defender
                joint = int(pair_key(ally, attacker))
                if joint in neighbours and joint not in self.active:
                    self.allies.discard(joint)
                    self._open(attacker, ally, when, size, conflict)

    def _battle(self, ctx, state, rng, dt, ea, eb, keys, power, people) -> None:
        """Fight along the fronts of the wars; winners annex the cells they take."""
        owner = state.owner
        front = np.unique(np.stack([np.r_[keys, keys], np.r_[ea, eb]]), axis=1)
        heads = np.zeros(front.shape[1])
        heads[:] = people[front[1]]
        pk, inv = np.unique(front[0], return_inverse=True)
        load = np.bincount(inv, weights=heads)
        for k, n in zip(pk, load):
            c = self.conflicts[self.active[int(k)]]
            c.severity += h.WAR_DAMAGE * n * dt
        tries = rng.random(ea.size) < -np.expm1(-h.WAR_BATTLE_RATE * dt)
        ea, eb, keys = ea[tries], eb[tries], keys[tries]
        if not ea.size:
            return
        att, dfn = owner[ea], owner[eb]
        p_att = self.p.projected(state, power, att, eb, self.cr)
        p_def = 1.0 + power[dfn] + h.POLITY_DEFENCE_GAMMA * self.p.defence[eb]
        win = rng.random(ea.size) < (p_att - p_def) / (p_att + p_def)
        if not win.any():
            return
        from .polities import first_claims

        taken, by = first_claims(rng, eb[win], att[win])
        lost = owner[taken]
        for k in pair_key(by, lost):
            if int(k) in self.active:
                self.conflicts[self.active[int(k)]].cells += 1
        self.p.annex(ctx, state, taken, by)

    def _cool(self, rng, dt, a, b, keys, fighting) -> None:
        """Put states at war and their neighbours on alert; let the others leave it."""
        at_war = set()
        for k in self.active:
            at_war.update(key_pair(k))
        near = at_war | set(b[np.isin(a, np.fromiter(at_war, dtype=np.int64, count=len(at_war)))].tolist())
        leave = -np.expm1(-h.WAR_ALERT_DECAY * dt)
        self.alert = {s for s in sorted(self.alert) if s in near or rng.random() >= leave} | near

    def _ally(self, state, rng, years: float) -> None:
        """Let threatened neighbours at peace ally."""
        ea, eb, a, b, keys = self._borders(state)
        if not keys.size:
            return
        strength = 1.0 + self.p.powers(state)
        threat = np.zeros(strength.size)
        np.maximum.at(threat, a, strength[b])
        threatened = threat > h.ALLIANCE_THREAT * strength
        pairs, first = np.unique(keys, return_index=True)
        pa, pb = a[first], b[first]
        cand = threatened[pa] & threatened[pb]
        cand &= ~(np.isin(pairs, self._keys(self.active)) | np.isin(pairs, self._keys(self.allies)))
        new = pairs[cand & (rng.random(pairs.size) < -np.expm1(-h.ALLIANCE_RATE * years))]
        self.allies.update(int(k) for k in new)

    @staticmethod
    def _keys(collection) -> np.ndarray:
        """Return the members (or dict keys) of a collection of ints as an array."""
        return np.fromiter(collection, dtype=np.int64, count=len(collection))

    # -- conflicts ---------------------------------------------------------------------------------
    def _open(self, attacker: int, defender: int, when: float, size, conflict: int | None = None) -> int:
        """Open a war, in the given conflict, the conflict of either state, or a new one."""
        if conflict is None:
            for k, cid in self.active.items():
                if attacker in key_pair(k) or defender in key_pair(k):
                    conflict = cid
                    break
        if conflict is None:
            conflict = self.next_conflict
            self.next_conflict += 1
            self.conflicts[conflict] = Conflict(start=when)
        c = self.conflicts[conflict]
        key = int(pair_key(attacker, defender))
        self.active[key] = conflict
        self.allies.discard(key)
        c.wars += 1
        c.states.update((int(attacker), int(defender)))
        c.largest_km2 = max(c.largest_km2, float(size[attacker]), float(size[defender]))
        return conflict

    def _close(self, key: int) -> None:
        """End one war."""
        cid = self.active.pop(key)
        self.conflicts[cid].wars -= 1

    def _close_finished(self, ctx, state, when: float) -> None:
        """Record the conflicts whose wars have all ended; large ones enter the chronicle."""
        for cid in [c for c, v in self.conflicts.items() if v.wars <= 0]:
            c = self.conflicts.pop(cid)
            record = ConflictRecord(start=c.start, end=when, states=len(c.states), severity=c.severity,
                                    cells=c.cells, largest_km2=c.largest_km2)
            self.records.append(record)
            if c.cells * self.p.area >= h.POLITY_CHRONICLE_KM2 or len(c.states) >= h.WAR_CHRONICLE_STATES:
                big = max(c.states, key=lambda s: state.polities[s].peak_km2)
                ctx.chronicle.add(c.start, "war",
                                  f"A war of {len(c.states)} states, state {big} the largest, lasts "
                                  f"{max(c.start - when, 1.0):,.0f} years; {c.cells} cells change hands",
                                  culture=state.polities[big].culture, polity=int(big), states=len(c.states),
                                  area_km2=round(float(c.cells * self.p.area)))
