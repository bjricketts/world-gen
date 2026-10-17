"""Polities and borders: capitals, territories by XTENT influence on travel time, and polity types.

Capitals are the largest settlements of a culture, kept apart from each other;
capitals named in the spec come first. Each occupied cell goes to the capital
with the highest influence, a weight growing with the capital's population
minus the travel time from it (Renfrew & Level 1979), so borders fall where
travel is slow: on ridges, wide rivers and across straits. The number of
polities is solved so that their effective number (A²/Σ A_i², Taagepera 1997)
matches Earth's at the culture's communication level, scaled by its area.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .. import heuristics as h
from .model import Polity
from .roads import TravelGraph


def effective_number(areas: np.ndarray) -> float:
    """Return the effective number of polities, (Σ A_i)² / Σ A_i²."""
    areas = areas[areas > 0]
    return float(areas.sum() ** 2 / np.sum(areas ** 2)) if areas.size else 0.0


def _arc_km(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius_km: float) -> np.ndarray:
    """Return great-circle distances (km) between point sets ``a`` (m×3) and ``b`` (n×3) as an m×n array."""
    return radius_km * np.arccos(np.clip(a @ b.T, -1.0, 1.0))


def choose_capitals(cells, population, points, count, occupied_km2, radius_km, fixed=()):
    """Return ``count`` capital settlements: the fixed ones first, then the largest, each kept apart by 0.5 √(area / count)."""
    gap = 0.5 * np.sqrt(occupied_km2 / max(count, 1))
    p = points[cells]
    nearest = np.full(cells.size, np.inf)
    chosen = []

    def take(i):
        chosen.append(int(i))
        np.minimum(nearest, radius_km * np.arccos(np.clip(p @ p[i], -1.0, 1.0)), out=nearest)

    for i in fixed:
        take(i)
    for i in np.argsort(-population, kind="stable"):
        if len(chosen) >= count:
            break
        if nearest[i] >= gap:
            take(i)
    return np.array(chosen, dtype=np.int64)


def influence_owner(hours: sparse.csr_matrix, capital_cells: np.ndarray, weight_hours: np.ndarray) -> np.ndarray:
    """Return, per cell, the index of the capital with the highest influence (weight − travel time); −1 if unreachable."""
    n = hours.shape[0]
    k = capital_cells.size
    top = weight_hours.max()
    m = hours.tocoo()
    g = sparse.csr_matrix((np.r_[m.data, top - weight_hours + 1e-9], (np.r_[m.row, np.full(k, n)],
                                                                      np.r_[m.col, capital_cells])),
                          shape=(n + 1, n + 1))
    dist, pred = dijkstra(g, indices=n, return_predecessors=True)
    owner = np.full(n + 1, -2, dtype=np.int64)
    owner[n] = -1
    # a capital shared by two polities keeps the first
    for i in range(k - 1, -1, -1):
        owner[capital_cells[i]] = i
    for v in np.argsort(dist[:n], kind="stable"):
        if owner[v] != -2:
            continue
        owner[v] = owner[pred[v]] if np.isfinite(dist[v]) else -1
    return owner[:n]


def _inside(lat: np.ndarray, lon: np.ndarray, polygon) -> np.ndarray:
    """Return which points (degrees) fall inside a polygon of (lat, lon) vertices, by ray casting in longitude–latitude."""
    poly = np.asarray(polygon, dtype=float)
    y, x = poly[:, 0], poly[:, 1]
    inside = np.zeros(lat.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cross = ((y[i] > lat) != (y[j] > lat)) & (lon < (x[j] - x[i]) * (lat - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        inside ^= cross
        j = i
    return inside


def _solve_count(partition, n_min: int, n_max: int, target: float, cell_area: float):
    """Return the capitals and owners of the polity count whose effective number is closest to ``target``.

    Counts climb geometrically until the effective number reaches the target
    (then bisect) or stops growing, so a world where it cannot be reached keeps
    the smallest count near the best it can do.
    """
    def n_eff(owner):
        return effective_number(np.bincount(owner[owner >= 0]) * cell_area)

    tried = {}

    def run(n):
        if n not in tried:
            caps, owner = partition(n)
            tried[n] = (n_eff(owner), caps, owner)
        return tried[n][0]

    n, below, stall = n_min, n_min, 0
    while True:
        value = run(n)
        if value >= target or n >= n_max:
            break
        best_so_far = max(v for v, _, _ in tried.values())
        stall = stall + 1 if value < (1 + h.POLITY_COUNT_GROWTH_MIN) * tried[below][0] else 0
        if stall >= h.POLITY_COUNT_STALL_STEPS:
            break
        below = n
        n = min(n_max, max(n + 1, int(np.ceil(n * h.POLITY_COUNT_STEP))))
    if run(n) >= target and n > below:        # bisect between the last count below and the first above
        lo, hi = below, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if run(mid) >= target:
                hi = mid
            else:
                lo = mid
    top = max(v for v, _, _ in tried.values())
    good = [k for k, (v, _, _) in tried.items() if v >= (1 - h.POLITY_COUNT_GROWTH_MIN) * min(top, target)]
    pick = min(good, key=lambda k: (abs(np.log(tried[k][0] / target)), k)) if top >= target else min(good)
    return tried[pick][1], tried[pick][2]


def polity_type(culture, area_km2: float) -> str:
    """Return a polity's type: the spec's, or one set by the culture's development and the polity's size."""
    if culture.polity_type:
        return culture.polity_type
    if culture.inhabitants == "colonist":
        return "colony"
    comm, large = culture.development.communication, area_km2 >= h.POLITY_LARGE_KM2
    if comm == 0:
        return ("band", "tribe")[min(culture.development.energy, 1)] if culture.development.energy < 2 else "chiefdom"
    if comm <= 2:
        return "empire" if large else "kingdom"
    if comm == 3:
        return "empire" if large else "republic"
    return "federation" if large else "republic"


@dataclass
class GivenStates:
    """States given by an organic history: the state of each cell (−1 where none), the polity records, the
    development in force at each state's capital (by id), and the ids of the cultures they cover."""

    owner: np.ndarray
    records: list
    development: dict
    cultures: set


def given_polities(given: GivenStates, cultures, settlements, people: np.ndarray, cell_area: float):
    """Return the polities of an organic history's states and the state of each cell per covered culture.

    Each state keeps its history id, founding date and parent; its area and
    people count every covered culture in its cells, its type follows the
    development at its capital, and its capital is the largest settlement in
    its capital cell, or else its largest settlement.
    """
    rows = [i for i, c in enumerate(cultures) if c.id in given.cultures]
    by_id = {c.id: c for c in cultures}
    occupied = people[rows].sum(axis=0) if rows else np.zeros(given.owner.size)
    owner = np.where(occupied > 0, given.owner, -1)
    covered = np.isin(settlements.culture, list(given.cultures))
    polities = []
    for r in given.records:
        cells = owner == r.id
        if not r.alive or not cells.any():
            continue
        area = float(cells.sum() * cell_area)
        inside = np.flatnonzero(covered & cells[settlements.cell])
        at_capital = inside[settlements.cell[inside] == r.capital]
        pool = at_capital if at_capital.size else inside
        capital = int(settlements.id[pool[np.argmax(settlements.population[pool])]]) if pool.size else -1
        culture = by_id.get(r.culture) or next(c for c in cultures if c.id in given.cultures)
        at = replace(culture, development=given.development[r.id]) if r.id in given.development else culture
        polities.append(Polity(id=r.id, culture=r.culture, type=polity_type(at, area), capital=capital,
                               parent=r.parent, founded_years_ago=r.founded_years_ago,
                               population=float(occupied[cells].sum() * cell_area), area_km2=area,
                               rulers=int(r.rulers), primogeniture=bool(r.primogeniture)))
    layers = {i: np.where(people[i] > 0, owner, -1) for i in rows}
    return polities, layers


def build_polities(world, cultures, settlements, people: np.ndarray, seed: int, given: GivenStates | None = None,
                   id_offset: int = 0):
    """Return the polities, each settlement's polity, and the polity of each cell per culture (−1 where none).

    ``given`` holds an organic history's states (``GivenStates``); the
    cultures it covers keep the history's states and ids, the others are
    partitioned by influence with ids from ``id_offset``.
    """
    grid = world.grid
    radius_km = float(world.surface.attrs["radius_m"]) / 1e3
    cell_area = 4 * np.pi * radius_km**2 / grid.size
    lat, lon = grid.lat, grid.lon
    owner_layer = np.full(people.shape, -1, dtype=np.int32)
    settlement_polity = np.full(len(settlements), -1, dtype=np.int32)
    polities: list[Polity] = []
    graphs: dict[int, TravelGraph] = {}
    next_id = id_offset
    if given is not None:
        found, history_layers = given_polities(given, cultures, settlements, people, cell_area)
        polities.extend(found)
    for ci, c in enumerate(cultures):
        mine = np.flatnonzero(settlements.culture == c.id)
        if given is not None and c.id in given.cultures:
            owner_layer[ci] = history_layers[ci]
            settlement_polity[mine] = history_layers[ci][settlements.cell[mine]]
            continue
        occupied = people[ci] > 0
        if mine.size == 0 or not occupied.any():
            continue
        t = c.development.transport
        if t not in graphs:
            graphs[t] = TravelGraph(world, t)
        graph = graphs[t]
        hours = graph.matrix(graph.base_hours)
        cells, pop = settlements.cell[mine], settlements.population[mine]
        area_km2 = occupied.sum() * cell_area
        fixed = []
        for place in c.capitals:
            d = _arc_km(grid.points, grid.points[cells], _unit(place["lat_deg"], place["lon_deg"])[None], radius_km)
            fixed.append(int(np.argmin(d[:, 0])))
        fixed = list(dict.fromkeys(fixed))
        n_min = max(len(fixed), len(c.territories), 1)

        def partition(count):
            caps = choose_capitals(cells, pop, grid.points, count, area_km2, radius_km, fixed)
            weight = h.POLITY_INFLUENCE_KM * (pop[caps] / pop.max()) ** h.POLITY_INFLUENCE_EXPONENT / graph.flat_speed
            owner = influence_owner(hours, cells[caps], weight)
            owner[~occupied] = -1
            for k, polygon in enumerate(c.territories):
                owner[occupied & _inside(lat, lon, polygon)] = k
            return caps, owner

        if c.polity_count is not None:
            caps, owner = partition(max(c.polity_count, n_min))
        else:
            target = max(1.0, h.POLITY_EFFECTIVE_NUMBER[c.development.communication] * area_km2 / h.EARTH_DRY_LAND_KM2)
            caps, owner = _solve_count(partition, n_min, mine.size, target, cell_area)
        base = next_id
        next_id += caps.size
        sizes = np.bincount(owner[owner >= 0], minlength=caps.size) * cell_area
        heads = np.bincount(owner[owner >= 0], weights=people[ci][owner >= 0], minlength=caps.size) * cell_area
        for k, cap in enumerate(caps):
            polities.append(Polity(id=base + k, culture=c.id, type=polity_type(c, float(sizes[k])),
                                   capital=int(settlements.id[mine][cap]), population=float(heads[k]),
                                   area_km2=float(sizes[k])))
        owner_layer[ci] = np.where(owner >= 0, owner + base, -1)
        settlement_polity[mine] = owner_layer[ci][cells]
    return polities, settlement_polity, owner_layer


def _unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the unit vector at a latitude and longitude (degrees)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
