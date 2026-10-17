"""Overseas colonies founded by states during an organic history (an Age of Sail on any world).

A state with a coast whose capital has reached OVERSEAS_TRANSPORT and
OVERSEAS_COMMUNICATION founds a colony at a rate OVERSEAS_RATE per year. The colony goes to
the coastal land cell on another landmass, within OVERSEAS_REACH_KM of the
state's coast, that holds most colonists after settler mortality; cells ruled
by a state at the same or a higher energy level are passed over. The colonists
are a new culture: they speak their home culture's language, bring the
development of the home capital, take COLONY_FOUNDERS people from it, found a
colonial state and stay tied to the founding state (``colonists.home_step``).
Later colonies from the same line of home cultures on the same landmass add
to that colonist culture and found further colonial states. Only landmasses of
OVERSEAS_MIN_LANDMASS_KM2 or more are colonised.
"""

from __future__ import annotations

import numpy as np

from ... import heuristics as h
from ..development import AXES, Development
from ..model import Culture
from .colonists import empire, found_colony_state
from .population import capacity_by_level, cell_capacity, landmasses
from .state import in_force


class Overseas:
    """Coastal cells, landmasses and what a colony could hold, for founding colonies."""

    def __init__(self, ctx):
        """Find the coastal land cells and the settler mortality."""
        from .colonists import settler_mortality

        world, cov = ctx.world, ctx.cov
        ocean = world.surface["ocean"].values.astype(bool)
        coo = world.grid.neighbours.tocoo()
        near_sea = np.zeros(ocean.size, dtype=bool)
        near_sea[coo.row[ocean[coo.col]]] = True
        self.coast = cov.land & near_sea
        self.mass = landmasses(world)
        area = np.bincount(self.mass[self.mass >= 0]) * cov.cell_area_km2
        self.large = (self.mass >= 0) & (area[np.maximum(self.mass, 0)] >= h.OVERSEAS_MIN_LANDMASS_KM2)
        self.points = world.grid.points
        self.radius_km = cov.radius_km
        excess = np.clip(settler_mortality(world, cov) - h.SETTLER_MORTALITY_BASELINE, 0.0, None) / 1000.0
        self.survival = np.exp(-excess * h.OVERSEAS_MORTALITY_HORIZON_YEARS)
        self.colonies: dict[tuple[int, int], int] = {}     # (home culture id, landmass) -> colonist culture id
        template = Culture(id=-1, name=None, inhabitants="colonist", development=Development(0, 0, 0), mode="organic")
        self.capacity = capacity_by_level(template, cov)

    def step(self, ctx, state, years: float) -> None:
        """Let seafaring states found colonies."""
        from .run import add_culture

        rng = ctx.rng("overseas", state.step)
        owner = state.owner
        polities = [p for p in state.polities if p.alive]
        if not polities:
            return
        rows = {c: r for r, c in enumerate(state.culture_ids)}
        for p in polities:
            row = rows[p.culture]
            if (in_force(state.development[row, 1, p.capital]) < h.OVERSEAS_TRANSPORT
                    or in_force(state.development[row, 2, p.capital]) < h.OVERSEAS_COMMUNICATION):
                continue
            shore = np.flatnonzero((owner == p.id) & self.coast)
            if not shore.size:
                continue
            if rng.random() >= -np.expm1(-h.OVERSEAS_RATE * years):
                continue
            target = self._target(state, p, row, shore, rng)
            if target < 0:
                continue
            levels = state.development[row, :, p.capital].copy()
            home = ctx.cultures[row]
            key = (empire(ctx, home), int(self.mass[target]))
            founders = min(h.COLONY_FOUNDERS, 0.5 * state.population[row, p.capital])
            if key in self.colonies:                                  # the home culture's colonists there already
                new = state.index(self.colonies[key])
                state.population[row, p.capital] -= founders
                state.population[new, target] += founders
                state.development[new, :, target] = np.maximum(state.development[new, :, target], levels)
                self._found_state(ctx, state, new, target, p)
                continue
            culture = Culture(id=ctx.new_culture_id(), name=None, inhabitants="colonist",
                              development=Development(**{a: int(in_force(levels[k])) for k, a in enumerate(AXES)}),
                              mode="organic", founded_years_ago=state.years_ago, origin=home.name,
                              home_culture=home.id, phonology=home.phonology,
                              landing_sites=[{"lat_deg": float(ctx.world.grid.lat[target]),
                                              "lon_deg": float(ctx.world.grid.lon[target])}])
            new = add_culture(ctx, state, culture, np.repeat(levels[:, None], owner.size, axis=1))
            self.colonies[key] = culture.id
            state.population[row, p.capital] -= founders
            state.population[new, target] = founders
            ctx.cache.setdefault("homes", {})[culture.id] = p.id
            found_colony_state(ctx, state, new)
            self._chronicle(ctx, state, new, target, p, True)

    def _found_state(self, ctx, state, row: int, cell: int, home) -> None:
        """Found another colonial state of an existing colonist culture at a cell."""
        from .state import HistoryPolity

        culture = ctx.cultures[row]
        polity = HistoryPolity(id=len(state.polities), culture=culture.id, capital=cell,
                               founded_years_ago=state.years_ago, cohesion=h.POLITY_COHESION_START)
        state.polities.append(polity)
        state.owner[cell] = polity.id
        self._chronicle(ctx, state, row, cell, home)

    @staticmethod
    def _chronicle(ctx, state, row: int, cell: int, home, new: bool = False) -> None:
        """Record an overseas colony (``new``: the first of its colonist culture)."""
        origin = ctx.cultures[state.index(home.culture)]
        ctx.chronicle.add(state.years_ago, "overseas_colony",
                          f"State {home.id} of {origin.name or f'Culture {origin.id}'} sends colonists overseas to "
                          f"{ctx.world.grid.lat[cell]:.1f}°, {ctx.world.grid.lon[cell]:.1f}°",
                          culture=ctx.cultures[row].id, polity=home.id, cell=cell, new=new)

    def _target(self, state, polity, row: int, shore: np.ndarray, rng) -> int:
        """Return the best coastal cell for a colony of a state, or −1."""
        transport = int(min(in_force(state.development[row, 1, polity.capital]), len(h.OVERSEAS_REACH_KM) - 1))
        reach = h.OVERSEAS_REACH_KM[transport]
        home_mass = self.mass[polity.capital]
        cand = np.flatnonzero(self.coast & (self.mass != home_mass) & self.large & (state.owner != polity.id))
        if not cand.size:
            return -1
        cos = self.points[cand] @ self.points[shore].T
        near = np.arccos(np.clip(cos.max(axis=1), -1.0, 1.0)) * self.radius_km <= reach
        cand = cand[near]
        if not cand.size:
            return -1
        energy = state.development[row, 0, polity.capital]
        mine = in_force(energy)
        ruled = state.owner[cand] >= 0
        if ruled.any():
            rows = {c: r for r, c in enumerate(state.culture_ids)}
            their = np.array([in_force(state.development[rows[state.polities[o].culture], 0,
                                                         state.polities[o].capital]) if o >= 0 else -1
                              for o in state.owner[cand]])
            cand = cand[their < mine]
            if not cand.size:
                return -1
        hold = cell_capacity(self.capacity, np.full(self.capacity.shape[1], energy))[cand] * self.survival[cand]
        if hold.max() <= 0:
            return -1
        score = np.log(np.maximum(hold, 1e-300)) + rng.gumbel(size=cand.size)
        return int(cand[np.argmax(score)])


def overseas(ctx) -> Overseas:
    """Return the history's overseas process, created on first use."""
    if "overseas" not in ctx.cache:
        ctx.cache["overseas"] = Overseas(ctx)
    return ctx.cache["overseas"]


def overseas_step(ctx, state, years: float) -> None:
    """Process: seafaring states found overseas colonies."""
    overseas(ctx).step(ctx, state, years)
