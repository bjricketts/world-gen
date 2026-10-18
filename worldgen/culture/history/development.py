"""Farming and development in an organic history: where farming begins, how knowledge grows and how it spreads.

- *Origins.* Foragers begin farming at a rate proportional to their numbers
  and the arable land they hold (the surplus view of Kavanagh et al. 2018),
  except where farmers already live within the interaction sphere; a new origin
  starts at early farming (energy 0.5) and draws whether it keeps herd animals
  (``livestock.py``).
- *Growth of knowledge.* In farming cells, energy rises as knowledge does in
  Kremer's (1993) model with Jones's damping: the log of knowledge grows at
  g · (people in the interaction sphere)^λ · A^(φ−1), and a cell's energy
  moves by the log of capacity that knowledge buys between the level anchors.
  Past water_wind, energy rises only where coal or hydrocarbons lie within the
  sphere, at a faster industrial rate (without fuel a cell's own knowledge
  stops just short of fossil); electric follows fossil.
- *Loss of knowledge* (after Henrich 2004). Above farming, a cell whose
  sphere holds fewer people than its level needs loses energy, faster the
  larger the shortfall.
- *Spread.* A cell takes a neighbour's higher level in force once a front,
  advancing at the population front speed times the transport level's speed
  factor, has crossed the link between them (demic and cultural diffusion);
  into forager land the front slows with the destination's arable component.
- Transport and communication follow energy along Earth's pairing of levels.

New origins and each culture's first cell at each level enter the chronicle.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ... import heuristics as h
from ..development import COMMUNICATION_LEVELS, ENERGY_LEVELS, TRANSPORT_LEVELS
from .livestock import livestock
from .population import front_speed_km_yr, population
from .state import in_force

# Without fuel a cell's own knowledge stops just short of fossil in force (which begins at INDUSTRIAL_FROM_ENERGY).
WITHOUT_FUEL_MAX = float(np.nextafter(h.INDUSTRIAL_FROM_ENERGY, 0.0))
LN_ANCHORS = np.log(np.array([h.POPULATION_ANCHOR[name] for name in ENERGY_LEVELS]))


def companion_levels(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the transport and communication levels that go with real-valued energy levels."""
    e, t, c = (np.array(x, dtype=float) for x in zip(*h.DEVELOPMENT_PAIRING))
    return np.interp(energy, e, t), np.interp(energy, e, c)


def capacity_gap(energy: np.ndarray) -> np.ndarray:
    """Return the log step of capacity between the anchors around each energy level."""
    lo = np.clip(np.floor(energy).astype(int), 0, len(LN_ANCHORS) - 2)
    return LN_ANCHORS[lo + 1] - LN_ANCHORS[lo]


def knowledge_log(energy: np.ndarray) -> np.ndarray:
    """Return the log of capacity gained since foraging at each energy level (Kremer's ln A up to a constant)."""
    return np.interp(energy, np.arange(len(LN_ANCHORS)), LN_ANCHORS - LN_ANCHORS[0])


def energy_rate(energy: np.ndarray, sphere_people: np.ndarray, fuel: np.ndarray, pace: float = 1.0) -> np.ndarray:
    """Return how fast each cell's energy level rises (per year); ``pace`` scales the knowledge rate.

    Farming cells rise at g (S/10⁶)^λ A^(φ−1) / Δ ln K; past the industrial
    threshold only with fuel within the sphere, and faster by the industrial factor.
    """
    farming = energy >= h.FARMING_START_ENERGY
    a = knowledge_log(energy)
    rate = pace * h.KNOWLEDGE_RATE * np.power(np.maximum(sphere_people, 0.0) / 1e6, h.KNOWLEDGE_POPULATION_EXPONENT)
    rate = rate * np.exp(-(1.0 - h.KNOWLEDGE_DAMPING_PHI) * a) / capacity_gap(energy)
    industrial = energy >= h.INDUSTRIAL_FROM_ENERGY
    rate = np.where(industrial, rate * h.INDUSTRIAL_RATE_FACTOR * (fuel >= h.INDUSTRIAL_FUEL_MIN), rate)
    rate = np.where(energy >= h.HISTORY_MAX_ENERGY, 0.0, rate)
    return np.where(farming, rate, 0.0)


def advance_levels(level: np.ndarray, progress: np.ndarray, credit: np.ndarray, links,
                   years: float, arable: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy, front progress, time credit and the cells that took a neighbour's higher level after ``years``.

    A front leaves each cell toward neighbours at a lower level in force, at the
    population front speed times the speed factor of the source's transport
    level; into forager land it runs at that speed times the destination's
    arable component to the power FARMING_FRONT_ARABLE_EXPONENT (after Silva &
    Steele 2014), so it stalls where crops grow badly. Once it has crossed the
    link the neighbour starts at the bottom of the source's level, and keeps the
    years left over in the step as credit for its own fronts.
    """
    now = in_force(level)
    src, dst = links.rows, links.cols
    higher = now[src] > now[dst]
    src, dst = src[higher], dst[higher]
    transport, _ = companion_levels(level[src])
    factor = np.asarray(h.LAND_SPEED_FACTOR)[np.clip(in_force(transport), 0, 5)] / h.LAND_SPEED_FACTOR[0]
    if arable is not None:
        factor = np.where(now[dst] == 0, factor * np.clip(arable[dst], 0.0, 1.0) ** h.FARMING_FRONT_ARABLE_EXPONENT,
                          factor)
    gain = front_speed_km_yr() * factor * (years + credit[src]) / links.flat_km[higher]
    best = np.zeros(level.size)
    np.maximum.at(best, dst, gain)
    progress = np.where(best > 0, progress + best, 0.0)
    credit = np.zeros(level.size)
    ready = progress >= 1.0
    if not ready.any():
        return level, progress, credit, np.flatnonzero(ready)
    top = np.full(level.size, -1.0)
    np.maximum.at(top, dst, now[src] - 0.5)
    credit = np.where(ready, np.minimum((progress - 1.0) / np.maximum(best, 1e-12) * years, years), 0.0)
    level = np.where(ready, np.maximum(level, top), level)
    progress = np.where(ready, 0.0, progress)
    return level, progress, credit, np.flatnonzero(ready)


def sphere_kernel(points: np.ndarray, land: np.ndarray, length_km: float, radius_km: float, size: int) -> sparse.csr_matrix:
    """Return the interaction weights exp(−d / length) between land cells within KNOWLEDGE_SPHERE_REACH lengths."""
    reach = h.KNOWLEDGE_SPHERE_REACH * length_km
    tree = cKDTree(points[land])
    chord = 2 * np.sin(reach / radius_km / 2)
    pairs = tree.sparse_distance_matrix(tree, chord, output_type="coo_matrix")
    arc = 2 * np.arcsin(np.clip(pairs.data / 2, 0, 1)) * radius_km
    k = sparse.csr_matrix((np.exp(-arc / length_km), (land[pairs.row], land[pairs.col])), shape=(size, size))
    k.setdiag(np.where(np.isin(np.arange(size), land), 1.0, 0.0))
    return k.tocsr()


def sphere_length_km(transport: int) -> float:
    """Return the interaction length of a transport level: longer as travel gets faster."""
    return h.KNOWLEDGE_SPHERE_KM * (h.LAND_SPEED_FACTOR[transport] / h.LAND_SPEED_FACTOR[0]) ** h.KNOWLEDGE_SPHERE_SPEED_EXP


class Development:
    """The origin, knowledge and spread processes of an organic history, with their cached kernel and fronts."""

    def __init__(self, ctx):
        """Prepare the interaction sphere and each culture's fronts and records."""
        world, cov = ctx.world, ctx.cov
        self.world, self.cov = world, cov
        self.land = np.flatnonzero(cov.land)
        self.kernels: dict[int, sparse.csr_matrix] = {}
        self.touch = self.kernel(0).astype(bool).astype(float)
        fuel = np.zeros(cov.land.size)
        potential = ctx.resources
        if potential is not None:
            names = [r for r in h.INDUSTRIAL_FUELS if r in potential["resource"].values]
            if names:
                fuel = potential.sel(resource=names).values.max(axis=0)
        self.fuel_in_reach = np.asarray((self.touch.multiply(fuel[None, :])).max(axis=1).todense()).ravel()
        self.arable = ctx.suit.arable if ctx.suit is not None else np.where(cov.land, 1.0, 0.0)
        self.progress = np.zeros((len(ctx.cultures), cov.land.size))
        self.credit = np.zeros((len(ctx.cultures), cov.land.size))
        self.reached = [set() for _ in ctx.cultures]
        self.lost: set[tuple[int, int]] = set()

    def add_row(self, ctx, culture) -> None:
        """Extend the per-culture arrays for a culture that arose during the history."""
        self.reached.append(set())
        self.progress = np.vstack([self.progress, np.zeros(self.progress.shape[1])])
        self.credit = np.vstack([self.credit, np.zeros(self.credit.shape[1])])

    def kernel(self, transport: int) -> sparse.csr_matrix:
        """Return the interaction sphere of a transport level, built once."""
        if transport not in self.kernels:
            self.kernels[transport] = sphere_kernel(self.world.grid.points, self.land, sphere_length_km(transport),
                                                    self.cov.radius_km, self.cov.land.size)
        return self.kernels[transport]

    def sphere_people(self, state, row: int) -> np.ndarray:
        """Return the people within each cell's interaction sphere, at the cell's own transport level."""
        people = state.population[row]
        transport = np.clip(in_force(state.development[row, 1]), 0, 5)
        out = np.zeros(people.size)
        for t in np.unique(transport[people > 0]):
            cells = transport == t
            out[cells] = (self.kernel(int(t)) @ people)[cells]
        return out

    def originate(self, ctx, state, years: float) -> None:
        """Start farming in forager cells at the surplus rate, away from existing farmers."""
        for i in np.flatnonzero(state.active):
            energy = state.development[i, 0]
            people = state.population[i]
            farming = (energy >= h.FARMING_START_ENERGY) & (people > 0)
            near = (self.touch @ farming.astype(float)) > 0
            forager = (energy < h.FARMING_START_ENERGY) & (people > 0) & ~near
            rate = h.FARMING_ORIGIN_RATE * people * self.arable * forager
            draw = ctx.rng(f"origin.{i}", state.step).random(energy.size)
            new = np.flatnonzero(draw < -np.expm1(-rate * years))
            if new.size > 1:          # one origin per sphere per step: keep the most populous
                keep = []
                for c in new[np.argsort(-people[new], kind="stable")]:
                    if not any(self.touch[c, k] for k in keep):
                        keep.append(c)
                new = np.array(sorted(keep))
            herd_rng = ctx.rng(f"livestock.{i}", state.step)
            for c in new:
                energy[c] = h.FARMING_START_ENERGY
                culture = ctx.cultures[i]
                lat, lon = ctx.world.grid.lat[c], ctx.world.grid.lon[c]
                herds = livestock(ctx).originate(state, int(c), herd_rng)
                ctx.chronicle.add(state.years_ago, "farming_origin",
                                  f"{culture.name or f'Culture {culture.id}'} begin farming at {lat:.1f}°, {lon:.1f}°"
                                  + (", with herd animals" if herds else ", without herd animals"),
                                  culture=culture.id, cell=int(c), livestock=int(herds))

    def grow_knowledge(self, ctx, state, years: float) -> None:
        """Raise the energy of farming cells with the knowledge their interaction sphere produces."""
        for i in np.flatnonzero(state.active):
            energy = state.development[i, 0]
            sphere = self.sphere_people(state, i)
            rate = energy_rate(energy, sphere, self.fuel_in_reach, ctx.history.pace)
            ceiling = np.where(self.fuel_in_reach >= h.INDUSTRIAL_FUEL_MIN, h.HISTORY_MAX_ENERGY, WITHOUT_FUEL_MAX)
            state.development[i, 0] = np.where(rate > 0, np.minimum(energy + rate * years,
                                                                    np.maximum(ceiling, energy)), energy)
            self._lose(ctx, state, i, sphere, years)

    def _lose(self, ctx, state, row: int, sphere: np.ndarray, years: float) -> None:
        """Lower the levels of cells whose interaction sphere holds too few people to keep them (Henrich 2004).

        Above farming, energy falls at KNOWLEDGE_LOSS_RATE a year per e-fold the
        sphere's people lie below KNOWLEDGE_LOSS_PEOPLE at the cell's level, to
        no lower than FARMING_START_ENERGY; colonists lose on every axis and
        colonies tied to a home power lose nothing. The first time a culture's
        highest level in force falls below a level, it enters the chronicle.
        """
        if ctx.cultures[row].id in ctx.cache.get("linked", set()):        # a tied colony learns from home
            return
        energy = state.development[row, 0]
        people = state.population[row]
        above = (energy > h.FARMING_START_ENERGY) & (people > 0)
        if not above.any():
            return
        need = np.exp(np.interp(energy, np.arange(len(h.KNOWLEDGE_LOSS_PEOPLE)), np.log(h.KNOWLEDGE_LOSS_PEOPLE)))
        deficit = np.where(above, np.log(need / np.maximum(sphere, 1.0)), 0.0)
        drop = h.KNOWLEDGE_LOSS_RATE * np.clip(deficit, 0.0, None) * years
        if not drop.any():
            return
        before = int(in_force(energy[people > 0]).max())
        axes = (0, 1, 2) if ctx.cultures[row].inhabitants == "colonist" else (0,)
        for a in axes:
            level = state.development[row, a]
            state.development[row, a] = np.where(drop > 0, np.maximum(level - drop,
                                                                      np.minimum(level, h.FARMING_START_ENERGY)), level)
        after = int(in_force(state.development[row, 0][people > 0]).max())
        if after < before and (row, before) not in self.lost:
            self.lost.add((row, before))
            c = ctx.cultures[row]
            ctx.chronicle.add(state.years_ago, "knowledge_loss",
                              f"{c.name or f'Culture {c.id}'} lose {ENERGY_LEVELS[before].replace('_', ' ')}, "
                              f"too few to keep it", culture=c.id, level=before)

    def spread_levels(self, ctx, state, years: float) -> None:
        """Pass higher levels to neighbouring occupied cells, set the companion axes, and record first arrivals."""
        pop = population(ctx)
        for i in np.flatnonzero(state.active):
            links = pop.links_for(state.culture_development(i).transport)
            energy, self.progress[i], self.credit[i], _ = advance_levels(
                state.development[i, 0], self.progress[i], self.credit[i], links, years, self.arable)
            energy = np.where(state.population[i] > 0, energy, np.minimum(energy, state.development[i, 0]))
            state.development[i, 0] = energy
            if ctx.cultures[i].inhabitants == "native":
                transport, communication = companion_levels(energy)
                state.development[i, 1] = transport
                state.development[i, 2] = communication
            self._firsts(ctx, state, i)

    def _firsts(self, ctx, state, row: int) -> None:
        """Record the first cell of a culture to reach each level in force on each axis."""
        occupied = state.population[row] > 0
        culture = ctx.cultures[row]
        name = culture.name or f"Culture {culture.id}"
        for a, (axis, levels) in enumerate((("energy", ENERGY_LEVELS), ("transport", TRANSPORT_LEVELS),
                                            ("communication", COMMUNICATION_LEVELS))):
            if not occupied.any():
                continue
            lv = in_force(state.development[row, a])
            top = int(lv[occupied].max())
            for level in range(1, top + 1):
                key = (axis, level)
                if key in self.reached[row]:
                    continue
                self.reached[row].add(key)
                if state.step == 0:
                    continue
                cells = np.flatnonzero(occupied & (lv >= level))
                c = int(cells[np.argmax(state.population[row][cells])])
                ctx.chronicle.add(state.years_ago, f"first_{axis}",
                                  f"{name} reach {levels[level].replace('_', ' ')} "
                                  f"({axis}) at {ctx.world.grid.lat[c]:.1f}°, {ctx.world.grid.lon[c]:.1f}°",
                                  culture=culture.id, cell=c, level=level)


def development(ctx) -> Development:
    """Return the history's development processes, created on first use."""
    if "development" not in ctx.cache:
        ctx.cache["development"] = Development(ctx)
    return ctx.cache["development"]


def origin_step(ctx, state, years: float) -> None:
    """Process: new farming origins."""
    development(ctx).originate(ctx, state, years)


def knowledge_step(ctx, state, years: float) -> None:
    """Process: growth of knowledge in farming cells."""
    development(ctx).grow_knowledge(ctx, state, years)


def level_spread_step(ctx, state, years: float) -> None:
    """Process: spread of levels between neighbouring cells."""
    development(ctx).spread_levels(ctx, state, years)
