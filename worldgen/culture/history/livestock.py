"""Livestock in an organic history: which farmers keep herd animals, and so carry and withstand crowd diseases.

Wolfe, Dunavan & Diamond (2007): 8 of the 15 major temperate diseases probably
reached humans from domestic animals; crowd diseases need several hundred
thousand people to persist; 13 of the 14 major livestock species are from the
Old World, and the New World's llama gave humans no known pathogen. So where
farmers kept herds, their descendants carry crowd diseases and withstand them,
and where they did not, contact with herders' descendants kills.

- *Origins* (heuristic draw). A farming origin has herd animals with
  probability LIVESTOCK_P_CLOSED + (LIVESTOCK_P_OPEN − LIVESTOCK_P_CLOSED) × the
  share of open land (LIVESTOCK_OPEN_BIOMES: grassland, savanna, shrubland,
  where large grazers live) within LIVESTOCK_NEIGHBOURHOOD_KM of it.
- *Spread.* Herds pass from a farming cell that has them to neighbouring
  farming cells of any culture, at the population front speed times the
  transport speed factor of the cell's leading culture and the destination's
  arable component to the power FARMING_FRONT_ARABLE_EXPONENT, along the
  spread links of that transport level (sea routes from pack_and_boat).
- *Settlers.* Colonists whose pool carries herds bring them to the cells
  they settle. A pool carries them if most people of its home states live
  with livestock; a colony from off the planet only if its ``crowd_diseases``
  setting says so.

A cell's livestock (``HistoryState.livestock``, kept in the epochs) is also the immunity of everyone living there,
except herds settlers have brought (``immune``): those do not protect the people the settlers meet. Herds spread on
from settlers' cells count as kept.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ... import heuristics as h
from .population import front_speed_km_yr, population
from .state import LIVESTOCK_BROUGHT, LIVESTOCK_KEPT, in_force


class Livestock:
    """Which cells keep herd animals, with the fronts carrying them on."""

    def __init__(self, ctx):
        """Start without livestock; prepare the open-land share around each cell for origins."""
        from ...biosphere.biomes import Biome

        world, cov = ctx.world, ctx.cov
        self.progress = np.zeros(cov.land.size)
        biome = world.surface["biome"].values if "biome" in world.surface else np.zeros(cov.land.size, dtype=int)
        codes = [int(getattr(Biome, name)) for name in h.LIVESTOCK_OPEN_BIOMES]
        self.open = np.isin(biome, codes) & cov.land
        self.land = np.flatnonzero(cov.land)
        self.tree = cKDTree(world.grid.points[self.land])
        self.points = world.grid.points
        self.radius_km = cov.radius_km
        self.arable = ctx.suit.arable if ctx.suit is not None else np.where(cov.land, 1.0, 0.0)

    def open_share(self, cell: int) -> float:
        """Return the share of open land among land cells within LIVESTOCK_NEIGHBOURHOOD_KM of a cell."""
        chord = 2 * np.sin(h.LIVESTOCK_NEIGHBOURHOOD_KM / self.radius_km / 2)
        near = self.land[self.tree.query_ball_point(self.points[cell], chord)]
        return float(self.open[near].mean()) if near.size else 0.0

    @staticmethod
    def immune(state) -> np.ndarray:
        """Cells whose people withstand crowd diseases: those keeping herds, unless settlers brought them."""
        return state.livestock == LIVESTOCK_KEPT

    def originate(self, state, cell: int, rng) -> bool:
        """Draw whether a new farming origin keeps herd animals; return the draw."""
        p = h.LIVESTOCK_P_CLOSED + (h.LIVESTOCK_P_OPEN - h.LIVESTOCK_P_CLOSED) * self.open_share(cell)
        herds = bool(rng.random() < p)
        if herds:
            state.livestock[cell] = LIVESTOCK_KEPT
        return herds

    def carrier(self, ctx, state, pool: int) -> float:
        """Return whether a pool of colonists carries crowd diseases (1) or not (0): a colony from off the planet if
        its ``crowd_diseases`` setting says so; one from this planet if most people of its home states (or, without
        one, of its root culture) live with livestock."""
        rows = {c.id: r for r, c in enumerate(ctx.cultures)}
        root = ctx.cultures[rows[pool]] if pool in rows else None
        if root is None:
            return 0.0
        if root.inhabitants == "colonist" and root.home_culture is None:     # a colony from off the planet
            return 1.0 if root.crowd_diseases else 0.0
        from .colonists import empire

        homes = ctx.cache.get("homes", {})
        states = {homes[c.id] for c in ctx.cultures if c.id in homes and homes[c.id] >= 0 and empire(ctx, c) == pool}
        people = state.population[rows[pool]]
        if states:
            people = np.where(np.isin(state.owner, list(states)), state.population.sum(axis=0), 0.0)
        total = people.sum()
        return float(people[state.livestock > 0].sum() / total >= 0.5) if total > 0 else 0.0

    def step(self, ctx, state, years: float) -> None:
        """Bring herds to the cells carrier colonists settle and advance the livestock fronts."""
        from .colonists import empire

        P, E = state.population, state.development
        for row, c in enumerate(ctx.cultures):
            if c.inhabitants == "colonist" and state.active[row] and P[row].any():
                if self.carrier(ctx, state, empire(ctx, c)):
                    state.livestock[(P[row] > 0) & (state.livestock == 0)] = LIVESTOCK_BROUGHT
        occupied = P > 0
        farmed = ((E[:, 0] >= h.FARMING_START_ENERGY) & occupied).any(axis=0)
        has = state.livestock > 0
        source = has & farmed
        target = farmed & ~has
        if not source.any() or not target.any():
            self.progress[:] = 0.0
            return
        transport = np.where(occupied, in_force(E[:, 1]), -1).max(axis=0)
        best = np.zeros(has.size)
        pop = population(ctx)
        for t in np.unique(transport[source]):
            t = int(np.clip(t, 0, len(h.LAND_SPEED_FACTOR) - 1))
            links = pop.links_for(t)
            live = source[links.rows] & (transport[links.rows] == t) & target[links.cols]
            src, dst = links.rows[live], links.cols[live]
            factor = h.LAND_SPEED_FACTOR[t] / h.LAND_SPEED_FACTOR[0]
            gain = (front_speed_km_yr() * factor * np.clip(self.arable[dst], 0.0, 1.0) ** h.FARMING_FRONT_ARABLE_EXPONENT
                    * years / links.flat_km[live])
            np.maximum.at(best, dst, gain)
        self.progress = np.where(target, self.progress + best, 0.0)
        ready = self.progress >= 1.0
        state.livestock[ready] = LIVESTOCK_KEPT
        self.progress[ready] = 0.0


def livestock(ctx) -> Livestock:
    """Return the history's livestock, created on first use."""
    if "livestock" not in ctx.cache:
        ctx.cache["livestock"] = Livestock(ctx)
    return ctx.cache["livestock"]


def livestock_step(ctx, state, years: float) -> None:
    """Process: settlers' herds and the spread of livestock."""
    livestock(ctx).step(ctx, state, years)
