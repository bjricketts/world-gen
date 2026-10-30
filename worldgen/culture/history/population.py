"""People in an organic history: carrying capacity, logistic growth and spread over the travel graph.

Capacity per cell follows the user-scoped terms at the energy level each cell
has reached, scaled so the planet's productive land would hold Earth's
population at that level (HYDE 3.2); early farming holds a fitted share of
the step from foraging to farming. Cultures in one cell grow logistically
and compete for it (Lotka–Volterra: each against all the cell's people over
its own capacity, so the culture with more capacity displaces the others).

People spread in two ways. Into empty land a front advances at the speed of
the time-delayed wave of advance (Fort & Méndez 1999), √(am/T)/(1 + aT/2),
with the growth rate, mobility and generation time of Pinhasi et al. (2005):
an empty cell is settled, with founders from the neighbour the front came
from, once the front has crossed the edge between them; edges count by their
travel time, so fronts slow in rough country and follow rivers and coasts, and
from pack_and_boat they cross the sea on routes between coastal cells. Within
settled land a share of the people moves to settled neighbours each step,
with the lattice diffusion set to Pinhasi et al.'s m/(4T). Tracking the front
keeps its speed the same on every grid; a plain diffusion front runs faster on
coarse grids, because the real front (~20 km wide) is far narrower than a cell.
A colonist culture's front runs faster by its transport level's land speed
factor. A culture first reaching a landmass enters the chronicle.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from ... import heuristics as h
from ..development import ENERGY_LEVELS
from ..model import Culture
from ..roads import TravelGraph
from ..suitability import Covariates, log_capacity, productive_area_km2


def in_habitats(culture: Culture, world=None) -> bool:
    """Return whether a culture lives in built habitats: colonists, unless the world is habitable unaided (there
    they farm like natives at their level)."""
    if culture.inhabitants != "colonist":
        return False
    occ = getattr(getattr(world, "state", None), "occupiability", None) if world is not None else None
    return not (occ is not None and occ.habitable_unaided)


def level_terms(culture: Culture, energy: int, habitats: bool | None = None) -> dict:
    """Return the capacity terms of a culture at an energy level (natives interpolate foraging to modern; cultures
    in built habitats use the colonist terms)."""
    if culture.inhabitants == "colonist" if habitats is None else habitats:
        return dict(h.DENSITY_TERMS_COLONIST)
    w = min(energy / 4.0, 1.0)
    a, b = h.DENSITY_TERMS_FORAGING, h.DENSITY_TERMS_MODERN
    return {k: (1 - w) * a[k] + w * b[k] for k in b}


def capacity_by_level(culture: Culture, cov: Covariates, world=None) -> np.ndarray:
    """Return the people each cell holds at each energy level (level × cell) for one culture.

    The planet's productive land holds Earth's population at the level per km²
    of productive land; cells share it in proportion to their relative capacity.
    Cultures in built habitats (``in_habitats``) hold Earth's population per km² of habitable land instead.
    """
    productive = float(productive_area_km2(cov).sum())
    habitats = in_habitats(culture, world) if world is not None else culture.inhabitants == "colonist"
    if habitats:
        land = float(np.where(cov.land, cov.cell_area_km2, 0.0).sum())
    out = np.zeros((len(ENERGY_LEVELS), cov.land.size))
    for level, name in enumerate(ENERGY_LEVELS):
        ln = log_capacity(cov, level_terms(culture, level, habitats))
        rel = np.where(np.isfinite(ln), np.exp(ln - np.max(np.where(np.isfinite(ln), ln, -np.inf))), 0.0)
        if rel.sum() <= 0:
            continue
        if habitats:
            total = h.POPULATION_ANCHOR[name] / h.EARTH_HABITABLE_LAND_KM2 * land
        else:
            total = h.POPULATION_ANCHOR[name] / h.EARTH_PRODUCTIVE_LAND_KM2 * productive
        out[level] = total * rel / rel.sum()
    return out


def cell_capacity(levels: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Return each cell's capacity at its real-valued energy level (levels × cell table).

    The anchor of level k applies at energy k; between two anchors capacity is
    interpolated log-linearly, so knowledge gained within an era raises it
    gradually. Early farming (FARMING_START_ENERGY) holds EARLY_FARMING_LOG_SHARE
    of the log step from foraging to farming, so foraging to farming runs through
    that point. A level without capacity in a cell leaves the lower anchor.
    """
    e = np.clip(np.asarray(energy, dtype=float), 0.0, levels.shape[0] - 1)
    lo = np.floor(e).astype(int)
    hi = np.minimum(lo + 1, levels.shape[0] - 1)
    f = e - lo
    start, share = h.FARMING_START_ENERGY, h.EARLY_FARMING_LOG_SHARE
    f = np.where(lo == 0, np.where(f < start, f / start * share, share + (f - start) / (1 - start) * (1 - share)), f)
    idx = np.arange(levels.shape[1])
    a, b = levels[lo, idx], levels[hi, idx]
    both = (a > 0) & (b > 0)
    ratio = np.where(both, b / np.where(a > 0, a, 1.0), 1.0)
    return np.where(both, a * ratio**f, a)


def grow(P: np.ndarray, K: np.ndarray, rate: float, years: float) -> np.ndarray:
    """Return the people after logistic growth with Lotka–Volterra competition (culture × cell arrays).

    Each culture follows the logistic step P e^{aΔt} / (1 + L (e^{aΔt} − 1)) with
    L = N / K, N all people in the cell and K the culture's own capacity, so where
    cultures share a cell the one with more capacity displaces the others (all
    competition coefficients 1). Exact for a culture alone; cells a culture
    cannot live in empty.
    """
    livable = K > 0
    total = P.sum(axis=0)
    load = np.where(livable, total[None, :] / np.where(livable, K, 1.0), 0.0)
    e = math.exp(rate * years)
    return np.where(livable, P * e / (1.0 + load * (e - 1.0)), 0.0)


def spread_share(years: float, spacing_km: float) -> float:
    """Return the share of a settled cell's people moving to settled neighbours in a step.

    The lattice diffusion coefficient share·Δx²/(4Δt) equals Pinhasi et al.'s m/(4T).
    """
    return h.SPREAD_MOBILITY_KM2 * years / (h.SPREAD_GENERATION_YEARS * spacing_km**2)


def front_speed_km_yr() -> float:
    """Return the speed of a population front on flat ground (Fort & Méndez 1999)."""
    a, m, t = h.SPREAD_GROWTH_RATE, h.SPREAD_MOBILITY_KM2, h.SPREAD_GENERATION_YEARS
    return math.sqrt(a * m / t) / (1.0 + a * t / 2.0)


def row_normalise(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return a matrix whose rows sum to one (empty rows stay empty)."""
    s = np.asarray(m.sum(axis=1)).ravel()
    return (sparse.diags(np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)) @ m).tocsr()


class SpreadLinks:
    """Where people can move from each cell at one transport level: move weights and each link's flat-ground length.

    ``weight`` holds exp(−hours/τ) per link (neighbours on land, and from
    pack_and_boat sea routes between coastal cells across at most the transport
    level's reach of open water); ``flat_km`` holds the link's
    travel time times the flat-ground speed, the distance a front has to cover.
    """

    def __init__(self, world, transport: int):
        """Build the links of a world at a transport level."""
        graph = TravelGraph(world, transport)
        tau = h.SPREAD_TRAVEL_TAU_HOURS
        land = ~graph.water_cell
        r, c, hours = graph.row, graph.col, graph.base_hours
        on_land = graph.usable & land[r] & land[c]
        rows, cols, link_hours = [r[on_land]], [c[on_land]], [hours[on_land]]
        reach = h.SPREAD_SEA_REACH_KM[transport]
        if reach > 0 and h.WATER_SPEED_KMH[transport] > 0:
            # open water on a voyage: sea-to-sea edges in full, the half of a coast edge that lies at sea
            sea = graph.usable & ~(land[r] & land[c])
            open_km = np.where(land[r[sea]] | land[c[sea]], 0.5, 1.0) * graph.km[sea]
            voyages = sparse.csr_matrix((open_km, (r[sea], c[sea])), shape=(graph.size, graph.size))
            coast = np.flatnonzero(land & (np.asarray(voyages.sum(axis=1)).ravel() > 0))
            if coast.size:
                i, j, sea_km = _voyages(voyages, coast, reach, land)
                # each coast keeps its SPREAD_SEA_LINKS nearest landfalls; longer voyages go by way of them
                order = np.lexsort((sea_km, i))
                i, j, sea_km = i[order], j[order], sea_km[order]
                rank = np.arange(i.size) - np.searchsorted(i, i)
                near = rank < h.SPREAD_SEA_LINKS
                hop_hours = sea_km[near] / h.WATER_SPEED_KMH[transport] + h.PORT_HOURS[transport]
                rows.append(coast[i[near]]); cols.append(j[near]); link_hours.append(hop_hours)
        rows, cols, link_hours = (np.concatenate(x) for x in (rows, cols, link_hours))
        # a pair joined by land and by sea keeps its quicker link
        order = np.lexsort((link_hours, cols, rows))
        rows, cols, link_hours = rows[order], cols[order], link_hours[order]
        first = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])]
        self.rows, self.cols, self.hours = rows[first], cols[first], link_hours[first]
        self.flat_km = self.hours * graph.flat_speed
        self.weight = sparse.csr_matrix((np.exp(-self.hours / tau), (self.rows, self.cols)),
                                        shape=(graph.size, graph.size))


def _voyages(voyages: sparse.csr_matrix, coast: np.ndarray, reach: float, land: np.ndarray):
    """Return (source index into ``coast``, landfall cell, open-water km) of every voyage within reach.

    Sources are taken in chunks, so memory stays small on fine grids.
    """
    out_i, out_j, out_d = [], [], []
    for start in range(0, coast.size, h.SPREAD_VOYAGE_CHUNK):
        chunk = coast[start:start + h.SPREAD_VOYAGE_CHUNK]
        d = dijkstra(voyages, indices=chunk, limit=reach)
        i, j = np.nonzero(np.isfinite(d) & (d > 0))
        keep = land[j] & (chunk[i] != j)
        out_i.append(i[keep] + start); out_j.append(j[keep]); out_d.append(d[i[keep], j[keep]])
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)


def diffuse(P: np.ndarray, weight: sparse.csr_matrix, settled: np.ndarray, share: float) -> np.ndarray:
    """Return the people after a share of every settled cell moves to its settled neighbours (sub-steps above the limit)."""
    to_settled = row_normalise(weight.multiply(settled[None, :]).tocsr())
    has_exit = np.asarray(to_settled.sum(axis=1)).ravel() > 0
    moving_to = to_settled.T.tocsr()
    steps = max(1, math.ceil(share / h.SPREAD_MAX_SHARE))
    for _ in range(steps):
        out = P * (share / steps) * has_exit
        P = P - out + moving_to @ out
    return P


def advance_front(P: np.ndarray, K: np.ndarray, progress: np.ndarray, credit: np.ndarray, links: SpreadLinks,
                  years: float, speed_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return people, front progress, time credit, the newly settled cells and their source cells after ``years``.

    A front leaves every established cell (at least SPREAD_ESTABLISHED_SHARE of
    capacity) and crosses each link to an empty cell the culture can live in at
    the front speed (times ``speed_factor``); the fastest incoming front counts. A cell whose front has
    crossed is settled with that share of its capacity, taken from the cell the
    front came from, and keeps the years left over in the step as credit, so its
    own front starts from the moment it was reached and not from the next step.
    """
    theta = h.SPREAD_ESTABLISHED_SHARE
    established = (K > 0) & (P >= theta * K)
    empty = (P < h.SPREAD_MIN_PEOPLE) & (K > 0)
    live = established[links.rows] & empty[links.cols]
    src, dst = links.rows[live], links.cols[live]
    speed = front_speed_km_yr() * speed_factor
    gain = speed * (years + credit[src]) / links.flat_km[live]
    best = np.zeros(P.size)
    np.maximum.at(best, dst, gain)
    progress = np.where(empty, progress + best, 0.0)
    credit = np.zeros(P.size)
    ready = empty & (progress >= 1.0)
    new = np.flatnonzero(ready)
    sources = np.empty(0, dtype=np.int64)
    if new.size:
        cand = ready[dst]
        s, d, g = src[cand], dst[cand], gain[cand]
        order = np.lexsort((-g, d))
        s, d, g = s[order], d[order], g[order]
        first = np.r_[True, d[1:] != d[:-1]]
        s, d, g = s[first], d[first], g[first]
        # years left after the crossing: the overshoot in progress, in years of this front
        credit[d] = np.minimum((progress[d] - 1.0) / np.maximum(g, 1e-12) * years, years)
        P = P.copy()
        founders = np.minimum(theta * K[d], 0.5 * P[s])
        np.subtract.at(P, s, founders)
        P[d] += founders
        progress[d] = 0.0
        new, sources = d, s
    return P, progress, credit, new, sources


def landmasses(world) -> np.ndarray:
    """Return a label per cell for each connected landmass (−1 at sea), largest first."""
    land = ~world.surface["ocean"].values.astype(bool)
    nb = world.grid.neighbours.tocsr()
    idx = np.flatnonzero(land)
    sub = nb[idx][:, idx]
    _, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    out = -np.ones(land.size, dtype=np.int64)
    out[idx] = rank[lab]
    return out


class Population:
    """The growth and spread processes of an organic history, with their cached capacities, links and fronts."""

    def __init__(self, ctx):
        """Prepare capacities per culture and level, the landmasses and empty fronts."""
        self.levels = [capacity_by_level(c, ctx.cov, ctx.world) for c in ctx.cultures]
        self.links: dict[int, SpreadLinks] = {}
        self.landmass = landmasses(ctx.world)
        self.reached = [set() for _ in ctx.cultures]
        self.progress = np.zeros((len(ctx.cultures), ctx.cov.land.size))
        self.credit = np.zeros((len(ctx.cultures), ctx.cov.land.size))
        self.spacing_km = ctx.cov.spacing_km
        self.world = ctx.world
        self.ctx_cache = ctx.cache

    def add_row(self, ctx, culture) -> None:
        """Extend the per-culture arrays for a culture that arose during the history."""
        self.levels.append(capacity_by_level(culture, ctx.cov, ctx.world))
        self.reached.append(set())
        self.progress = np.vstack([self.progress, np.zeros(self.progress.shape[1])])
        self.credit = np.vstack([self.credit, np.zeros(self.credit.shape[1])])

    def capacity(self, state) -> np.ndarray:
        """Return each culture's capacity per cell at the energy each cell has reached (culture × cell), times the
        step's stress factor where planet events press on the land."""
        K = np.array([cell_capacity(self.levels[i], state.development[i, 0]) for i in range(len(self.levels))])
        stress = self.stress
        return K * stress.factor[None, :] if stress is not None else K

    def links_for(self, transport: int) -> SpreadLinks:
        """Return the spread links for a transport level, built once."""
        if transport not in self.links:
            self.links[transport] = SpreadLinks(self.world, transport)
        return self.links[transport]

    @property
    def stress(self):
        """The stress of the current step (``events.Stress``), or None."""
        return self.ctx_cache.get("stress") if self.ctx_cache is not None else None

    def grow(self, ctx, state, years: float) -> None:
        """Grow every active culture logistically in the capacity it shares; natives do not grow while an
        epidemic runs in their cell."""
        K = self.capacity(state) * state.active[:, None]
        grown = grow(state.population, K, h.SPREAD_GROWTH_RATE, years)
        epidemics = ctx.cache.get("epidemics")
        if epidemics is not None:
            for row, sick in epidemics.running(state.years_ago, years).items():
                grown[row, sick] = np.minimum(grown[row, sick], state.population[row, sick])
        state.population[:] = grown

    def spread(self, ctx, state, years: float) -> None:
        """Move people within settled land and advance every active culture's front; record new landmasses reached."""
        share = spread_share(years, self.spacing_km)
        K = self.capacity(state)
        for i in np.flatnonzero(state.active):
            transport = state.culture_development(i).transport
            links = self.links_for(transport)
            factor = h.LAND_SPEED_FACTOR[transport] if ctx.cultures[i].inhabitants == "colonist" else 1.0
            P = state.population[i]
            P = diffuse(P, links.weight, P >= h.SPREAD_MIN_PEOPLE, share)
            P, self.progress[i], self.credit[i], new, sources = advance_front(
                P, K[i], self.progress[i], self.credit[i], links, years, factor)
            state.development[i][:, new] = state.development[i][:, sources]     # settlers bring their ways
            P[P < h.SPREAD_MIN_PEOPLE] = 0.0
            state.population[i] = P
            self._landfall(ctx, state, i)

    def _landfall(self, ctx, state, row: int) -> None:
        """Record in the chronicle each landmass a culture's people first settle in numbers."""
        people = np.bincount(self.landmass[self.landmass >= 0], weights=state.population[row][self.landmass >= 0])
        new = [int(k) for k in np.flatnonzero(people >= h.SPREAD_LANDFALL_PEOPLE) if k not in self.reached[row]]
        first = not self.reached[row]              # where a culture starts or lands is not news
        for k in new:
            self.reached[row].add(k)
            if first:
                continue
            c = ctx.cultures[row]
            ctx.chronicle.add(state.years_ago, "landfall",
                              f"{c.name or f'Culture {c.id}'} settle landmass {k + 1}", culture=c.id, landmass=k)


def population(ctx) -> Population:
    """Return the history's population processes, created on first use."""
    if "population" not in ctx.cache:
        ctx.cache["population"] = Population(ctx)
    return ctx.cache["population"]


def grow_step(ctx, state, years: float) -> None:
    """Process: logistic growth of every active culture."""
    population(ctx).grow(ctx, state, years)


def spread_step(ctx, state, years: float) -> None:
    """Process: spread of every active culture to neighbouring cells and over sea routes."""
    population(ctx).spread(ctx, state, years)
