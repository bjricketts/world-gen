"""Planet events in an organic history: eruptions, droughts, pandemics, impacts and scripted events.

- *Eruptions* (after Toohey & Sigl 2017, Sigl et al. 2015). Large
  stratospheric sulfur injections arise at VOLCANIC_EVENTS_PER_YEAR times the
  planet's volcanic activity, at the planet's volcanoes, with sulfur from a
  Pareto law. They cool a hemisphere (both from the tropics) for five to ten
  years in proportion to the square root of their sulfur.
- *Droughts* (heuristic rates). Discs of land lose a share of their rain for
  decades to centuries; land loses capacity by the rain lost times how
  water-limited it is.
- *Stress.* Each shock is counted in local standard deviations (cold over
  SUMMER_TEMPERATURE_SD_K, rain over RAIN_CV) and costs capacity. The step's
  stress (standard deviations and capacity factor per cell, weighted by the
  share of the step each shock lasts) is read by growth and spread (capacity),
  wars (onset) and states (break-up), after Hsiang et al. (2013). People above a
  stressed cell's capacity move to less stressed neighbours.
- *Pandemics* (after Marani et al. 2021). Epidemics arise at Marani's rate,
  scaled by the farming people, with intensity from their generalized Pareto
  law (up to PANDEMIC_MAX_INTENSITY); each starts in a crowded farming cell, reaches the cells within its
  years of travel and kills intensity × duration × the world's people among
  them, unevenly and at most PANDEMIC_MAX_DEATH_SHARE per cell.
- *Impacts* (after Brown et al. 2002). Impacts of 1–10 Mt strike at Brown's
  rate anywhere on the planet; land within the devastated area loses its
  people.
- *Scripted events* (``history.events``) are placed where and when the spec
  says; a sea-level rise floods land below its height for good.

Events that kill EVENT_CHRONICLE_PEOPLE or more, eruptions at or above
Tambora's sulfur, and every scripted event enter the chronicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from ... import heuristics as h
from .state import in_force


@dataclass
class Shock:
    """A climate shock: its kind, when it runs (years ago), the cells it covers, their standard deviations and
    capacity loss."""

    kind: str
    start: float
    end: float
    cells: np.ndarray
    sigma: np.ndarray
    loss: np.ndarray


@dataclass
class Stress:
    """The stress of one step per cell: standard deviations of climate anomaly and the capacity factor (0–1)."""

    sigma: np.ndarray
    factor: np.ndarray


def ramp(x, lo: float, hi: float):
    """Return 0 at ``lo`` rising linearly to 1 at ``hi`` (clipped)."""
    return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def pareto_draw(rng, lo: float, slope: float, hi: float, size=None):
    """Return draws from a Pareto law N(>x) ∝ x^−slope above ``lo``, truncated at ``hi``."""
    u = rng.random(size)
    top = (hi / lo) ** (-slope)
    return lo * (1.0 - u * (1.0 - top)) ** (-1.0 / slope)


def gpd_exceedance(x, mu: float, sigma: float, xi: float):
    """Return P(i > x | i > mu) for a generalized Pareto law."""
    return (1.0 + xi * (np.asarray(x, dtype=float) - mu) / sigma) ** (-1.0 / xi)


def gpd_draw_between(rng, floor: float, top: float, mu: float, sigma: float, xi: float) -> float:
    """Return a draw from the generalized Pareto law conditioned on lying between ``floor`` and ``top``."""
    lo, hi = gpd_exceedance(top, mu, sigma, xi), gpd_exceedance(floor, mu, sigma, xi)
    p = lo + (hi - lo) * rng.random()
    return mu + sigma / xi * (p ** (-xi) - 1.0)


def volcanic_sulfur(rng) -> float:
    """Return an eruption's stratospheric sulfur (Tg S): the Pareto law, with the tail beyond the largest on record
    set at VOLCANIC_SULFUR_MAX_TG."""
    return min(float(pareto_draw(rng, h.VOLCANIC_SULFUR_MIN_TG, h.VOLCANIC_SULFUR_SLOPE, np.inf)),
               h.VOLCANIC_SULFUR_MAX_TG)


def eruption_cooling_k(sulfur_tg: float) -> float:
    """Return the summer cooling (K) after an eruption of the given stratospheric sulfur."""
    return h.VOLCANIC_COOLING_K * math.sqrt(sulfur_tg / h.VOLCANIC_COOLING_REF_TG)


def impact_rate(min_mt: float, max_mt: float) -> float:
    """Return impacts per year with energy between the bounds (Brown et al. 2002: log N = 0.5677 − 0.90 log E, E in kt)."""
    return 3.70 * ((1e3 * min_mt) ** -0.90 - (1e3 * max_mt) ** -0.90)


class Events:
    """The planet's events over a history: shocks under way, the flooded land and the scripted events to come."""

    def __init__(self, ctx):
        """Find the volcanoes, how water-limited the land is and the hemispheres; queue the scripted events."""
        from ...surface.fields import Terrain

        world, cov = ctx.world, ctx.cov
        self.land = cov.land.astype(bool)
        self.points = world.grid.points
        self.lat = np.asarray(world.grid.lat)
        self.radius_km = cov.radius_km
        self.cell_km2 = float(np.mean(cov.cell_area_km2))
        s = world.surface
        terrain = s["terrain"].values if "terrain" in s else np.zeros(self.land.size)
        volcanoes = np.flatnonzero(terrain == Terrain.VOLCANO)
        self.volcanoes = volcanoes if volcanoes.size else np.arange(self.land.size)
        if "precipitation" in s and "potential_evaporation" in s:
            ratio = s["precipitation"].values / np.maximum(s["potential_evaporation"].values, 1e-6)
            self.water_limit = 1.0 - ramp(ratio, *h.DROUGHT_WATER_LIMIT)
        else:
            self.water_limit = np.ones(self.land.size)
        interior = getattr(getattr(world, "state", None), "interior", None)
        self.activity = float(getattr(interior, "activity_index", 1.0) or 0.0)
        self.land_km2 = float(self.land.sum() * self.cell_km2)
        self.shocks: list[Shock] = []
        self.flooded = np.zeros(self.land.size, dtype=bool)
        self.elevation = s["elevation"].values if "elevation" in s else np.zeros(self.land.size)
        self.scripted = sorted(ctx.history.events, key=lambda e: -e.years_ago)
        self.planet = ctx.history.planet_events

    # -- the step ------------------------------------------------------------------------------
    def step(self, ctx, state, years: float) -> None:
        """Start the step's events, apply the sudden ones, and set the stress the other processes read."""
        rng = ctx.rng("events", state.step)
        t0, t1 = state.years_ago, state.years_ago - years
        if self.planet:
            self._draw(ctx, state, rng, t0, t1, years)
        due = [e for e in self.scripted if e.years_ago > t1 or t1 <= 0.0]
        self.scripted = [e for e in self.scripted if e not in due]
        for e in due:
            self._scripted(ctx, state, rng, e)
        ctx.cache["stress"] = self.stress(t0, t1)
        self._migrate(ctx, state)
        self.shocks = [s for s in self.shocks if s.end < t1]

    def stress(self, t0: float, t1: float) -> Stress:
        """Return the stress of the step from ``t0`` to ``t1`` years ago."""
        n = self.land.size
        sigma, keep = np.zeros(n), np.ones(n)
        span = max(t0 - t1, 1e-9)
        for s in self.shocks:
            share = max(0.0, min(t0, s.start) - max(t1, s.end)) / span
            if share <= 0:
                continue
            sigma[s.cells] += share * s.sigma
            keep[s.cells] *= 1.0 - share * s.loss
        keep[self.flooded] = 0.0
        return Stress(sigma, keep)

    # -- drawn events --------------------------------------------------------------------------
    def _draw(self, ctx, state, rng, t0: float, t1: float, years: float) -> None:
        """Draw the planet's eruptions, droughts, impacts and pandemics in the step."""
        for _ in range(rng.poisson(h.VOLCANIC_EVENTS_PER_YEAR * self.activity * years)):
            sulfur = volcanic_sulfur(rng)
            self.eruption(ctx, rng.uniform(t1, t0), int(rng.choice(self.volcanoes)), sulfur, False)
        for _ in range(rng.poisson(h.DROUGHT_RATE_PER_KM2_YR * self.land_km2 * years)):
            cell = int(rng.choice(np.flatnonzero(self.land)))
            radius = math.exp(rng.uniform(*np.log(h.DROUGHT_RADIUS_KM)))
            length = math.exp(rng.uniform(*np.log(h.DROUGHT_YEARS)))
            self.drought(ctx, rng.uniform(t1, t0), cell, radius, length, rng.uniform(*h.DROUGHT_RAIN_LOSS), False)
        for _ in range(rng.poisson(impact_rate(h.IMPACT_MIN_MT, h.IMPACT_MAX_MT) * years)):
            energy = pareto_draw(rng, h.IMPACT_MIN_MT, 0.90, h.IMPACT_MAX_MT)
            self.impact(ctx, state, int(rng.integers(self.land.size)), energy, False)
        farmers = self._farming_people(state)
        mu, sd, xi = h.PANDEMIC_GPD
        rate = h.PANDEMIC_RATE * farmers.sum() / h.PANDEMIC_REFERENCE_PEOPLE
        rate *= gpd_exceedance(h.PANDEMIC_MIN_INTENSITY, mu, sd, xi) - gpd_exceedance(h.PANDEMIC_MAX_INTENSITY, mu, sd, xi)
        for _ in range(rng.poisson(rate * years)):
            intensity = gpd_draw_between(rng, h.PANDEMIC_MIN_INTENSITY, h.PANDEMIC_MAX_INTENSITY, mu, sd, xi)
            self.pandemic(ctx, state, rng, None, intensity, h.PANDEMIC_YEARS, False)

    def _scripted(self, ctx, state, rng, e) -> None:
        """Place one scripted event."""
        cell = self._cell(e.lat_deg, e.lon_deg) if e.lat_deg is not None else None
        when = e.years_ago
        if e.kind == "eruption":
            sulfur = e.size or volcanic_sulfur(rng)
            self.eruption(ctx, when, cell if cell is not None else int(rng.choice(self.volcanoes)), sulfur, True)
        elif e.kind == "drought":
            radius = e.radius_km or math.exp(rng.uniform(*np.log(h.DROUGHT_RADIUS_KM)))
            length = e.duration_years or math.exp(rng.uniform(*np.log(h.DROUGHT_YEARS)))
            self.drought(ctx, when, cell, radius, length, e.size or rng.uniform(*h.DROUGHT_RAIN_LOSS), True)
        elif e.kind == "impact":
            energy = e.size or pareto_draw(rng, h.IMPACT_MIN_MT, 0.90, h.IMPACT_MAX_MT)
            self.impact(ctx, state, cell if cell is not None else int(rng.integers(self.land.size)), energy, True)
        elif e.kind == "pandemic":
            mu, sd, xi = h.PANDEMIC_GPD
            intensity = 1e-3 * e.size if e.size else gpd_draw_between(rng, h.PANDEMIC_MIN_INTENSITY,
                                                                      h.PANDEMIC_MAX_INTENSITY, mu, sd, xi)
            self.pandemic(ctx, state, rng, cell, intensity, e.duration_years or h.PANDEMIC_YEARS, True)
        elif e.kind == "sea_level":
            self.sea_level(ctx, state, when, e.size)

    # -- the events ----------------------------------------------------------------------------
    def eruption(self, ctx, when: float, cell: int, sulfur: float, scripted: bool) -> None:
        """Start the volcanic cooling of an eruption of ``sulfur`` Tg S at a cell."""
        cooling = eruption_cooling_k(sulfur)
        length = h.VOLCANIC_LONG_YEARS if sulfur >= h.VOLCANIC_COOLING_REF_TG else h.VOLCANIC_YEARS
        lat = self.lat[cell]
        if abs(lat) <= h.VOLCANIC_TROPICS_DEG:
            cells = np.flatnonzero(self.land)
        else:
            cells = np.flatnonzero(self.land & (np.sign(self.lat) == np.sign(lat)))
        sigma = cooling / h.SUMMER_TEMPERATURE_SD_K
        loss = min(h.CAPACITY_LOSS_PER_SIGMA * sigma, 1.0)
        self.shocks.append(Shock("eruption", when, when - length, cells, np.full(cells.size, sigma),
                                 np.full(cells.size, loss)))
        if scripted or sulfur >= h.VOLCANIC_COOLING_REF_TG:
            where = "both hemispheres" if abs(lat) <= h.VOLCANIC_TROPICS_DEG else (
                "the northern hemisphere" if lat > 0 else "the southern hemisphere")
            ctx.chronicle.add(when, "eruption",
                              f"An eruption at {lat:.1f}°, {ctx.world.grid.lon[cell]:.1f}° puts {sulfur:.0f} Tg of "
                              f"sulfur into the stratosphere; summers in {where} cool by {cooling:.1f} K for "
                              f"{length:.0f} years", cell=int(cell), sulfur_tg=round(float(sulfur), 2),
                              cooling_k=round(cooling, 2))

    def drought(self, ctx, when: float, cell: int, radius_km: float, length: float, rain_loss: float,
                scripted: bool) -> None:
        """Start a drought of ``rain_loss`` over the land within ``radius_km`` of a cell for ``length`` years."""
        d = self.radius_km * np.arccos(np.clip(self.points @ self.points[cell], -1.0, 1.0))
        cells = np.flatnonzero(self.land & (d <= radius_km))
        if not cells.size:
            return
        sigma = np.full(cells.size, rain_loss / h.RAIN_CV)
        loss = np.clip(rain_loss * self.water_limit[cells], 0.0, 1.0)
        self.shocks.append(Shock("drought", when, when - length, cells, sigma, loss))
        if scripted or length >= 100.0:
            ctx.chronicle.add(when, "drought",
                              f"A drought takes {rain_loss:.0%} of the rain within {radius_km:,.0f} km of "
                              f"{self.lat[cell]:.1f}°, {ctx.world.grid.lon[cell]:.1f}° for {length:,.0f} years",
                              cell=int(cell), radius_km=round(radius_km), years=round(length), rain_loss=round(rain_loss, 2))

    def impact(self, ctx, state, cell: int, energy_mt: float, scripted: bool) -> None:
        """Devastate the land around an impact of ``energy_mt`` at a cell."""
        area = h.IMPACT_AREA_KM2 * (energy_mt / 10.0) ** (2.0 / 3.0)
        radius = math.sqrt(area / math.pi)
        if area < self.cell_km2:                     # less than a cell: that share of the cell's people
            hit, share = np.array([cell]), np.array([area / self.cell_km2])
        else:
            d = self.radius_km * np.arccos(np.clip(self.points @ self.points[cell], -1.0, 1.0))
            hit = np.flatnonzero(d <= radius)
            hit = hit if hit.size else np.array([cell])
            share = np.ones(hit.size)
        share = share * self.land[hit]
        killed = float((state.population[:, hit] * share).sum())
        state.population[:, hit] *= 1.0 - share
        if scripted or killed >= h.EVENT_CHRONICLE_PEOPLE:
            ctx.chronicle.add(state.years_ago, "impact",
                              f"An impact of {energy_mt:,.0f} Mt at {self.lat[cell]:.1f}°, {ctx.world.grid.lon[cell]:.1f}° "
                              f"kills {killed:,.0f}", cell=int(cell), energy_mt=round(float(energy_mt), 2),
                              killed=round(killed))

    def pandemic(self, ctx, state, rng, cell, intensity: float, length: float, scripted: bool) -> None:
        """Spread a pandemic of the given intensity (share of the world's people dying a year) from a cell."""
        from .population import population

        farmers = self._farming_people(state)
        if cell is None:
            if farmers.sum() <= 0:
                return
            weight = farmers ** 2
            cell = int(rng.choice(farmers.size, p=weight / weight.sum()))
        people = state.population[state.active].sum(axis=0)
        rows = np.flatnonzero(state.active)
        if not rows.size:
            return
        top = rows[np.argmax(state.population[rows, cell])]
        transport = int(np.clip(in_force(state.development[top, 1, cell]), 0, 5))
        links = population(ctx).links_for(transport)
        graph = sparse.csr_matrix((links.flat_km, (links.rows, links.cols)), shape=(people.size, people.size))
        reach = h.EPIDEMIC_SPEED_KM_YR * h.LAND_SPEED_FACTOR[transport] / h.LAND_SPEED_FACTOR[0] * length
        dist = csgraph.dijkstra(graph, indices=cell, limit=reach)
        reached = np.flatnonzero(np.isfinite(dist) & (people > 0))
        if not reached.size:
            return
        deaths = intensity * length * people.sum()
        uneven = np.exp(rng.normal(0.0, h.PANDEMIC_SPREAD_SD, reached.size))
        base = deaths / max((people[reached] * uneven).sum(), 1e-300)
        share = np.clip(base * uneven, 0.0, h.PANDEMIC_MAX_DEATH_SHARE)
        killed = float((people[reached] * share).sum())
        state.population[:, reached] *= 1.0 - share
        if scripted or killed >= h.PANDEMIC_CHRONICLE_PEOPLE:
            ctx.chronicle.add(state.years_ago, "pandemic",
                              f"A pandemic starting at {self.lat[cell]:.1f}°, {ctx.world.grid.lon[cell]:.1f}° kills "
                              f"{killed:,.0f} ({killed / max(people[reached].sum(), 1e-300):.0%} of the people it "
                              f"reaches) in {length:.0f} years", cell=int(cell), killed=round(killed),
                              intensity=float(intensity))

    def sea_level(self, ctx, state, when: float, rise_m: float) -> None:
        """Flood the land lower than ``rise_m`` above the sea for good."""
        new = self.land & ~self.flooded & (self.elevation < rise_m)
        self.flooded |= new
        ctx.chronicle.add(when, "sea_level",
                          f"The sea rises {rise_m:,.0f} m and floods {new.sum() * self.cell_km2:,.0f} km² of land "
                          f"holding {state.population[:, new].sum():,.0f} people", rise_m=float(rise_m),
                          cells=int(new.sum()))

    # -- helpers -------------------------------------------------------------------------------
    def _farming_people(self, state) -> np.ndarray:
        """Return the people per cell of cultures with farming in force there."""
        rows = np.flatnonzero(state.active)
        if not rows.size:
            return np.zeros(self.land.size)
        farming = in_force(state.development[rows, 0]) >= 1
        return (state.population[rows] * farming).sum(axis=0)

    def _cell(self, lat: float, lon: float) -> int:
        """Return the cell nearest a place."""
        la, lo = math.radians(lat), math.radians(lon)
        p = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
        return int(np.argmax(self.points @ p))

    def _migrate(self, ctx, state) -> None:
        """Move SHOCK_MIGRATION_SHARE of the people above a stressed cell's capacity to less stressed neighbours."""
        from .population import population, row_normalise

        stress = ctx.cache["stress"]
        stressed = stress.factor < 1.0
        if not stressed.any():
            return
        pop = population(ctx)
        K = pop.capacity(state)
        for i in np.flatnonzero(state.active):
            P = state.population[i]
            excess = np.where(stressed, np.clip(P - K[i], 0.0, None), 0.0)
            if excess.sum() <= 0:
                continue
            w = pop.links_for(state.culture_development(i).transport).weight.tocoo()
            better = stress.factor[w.col] > stress.factor[w.row]
            moves = row_normalise(sparse.csr_matrix((w.data[better], (w.row[better], w.col[better])), shape=w.shape))
            out = h.SHOCK_MIGRATION_SHARE * excess * (np.asarray(moves.sum(axis=1)).ravel() > 0)
            state.population[i] = P - out + moves.T @ out


def events(ctx) -> Events:
    """Return the history's events, created on first use."""
    if "events" not in ctx.cache:
        ctx.cache["events"] = Events(ctx)
    return ctx.cache["events"]


def events_step(ctx, state, years: float) -> None:
    """Process: the planet's events and the stress they put on the step."""
    events(ctx).step(ctx, state, years)
