"""Settlements: regional rank-size systems placed on the population density.

Each culture's occupied land is split into settlement regions of an area set
by its transport level. A region's whole population lives in a rank-size
(Zipf) hierarchy that runs from the smallest settlement up to a largest one
solved from the region's population, so the exponent holds within regions as
it does within countries on Earth. Settlements above a listing cut are placed
in rank order at cells drawn by their remaining population, keeping a
central-place distance from the larger settlements of their region; each
takes its people from its cell and, if needed, the nearest cells around it.
The population below the cut stays in each cell as dispersed population, which
zoomed maps turn into villages and hamlets. Foragers have no permanent
settlements: their people stay dispersed, and zoomed maps place their camps.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from .. import heuristics as h
from ..util import named_rng
from .habitats import Habitat
from .model import SETTLEMENT_FORMS, Culture, SettlementTable

EXACT_TERMS = 2000


def zipf_total(largest: float, zeta: float, smallest: float = h.SETTLEMENT_SMALLEST) -> float:
    """Return the population of a Zipf system from ``largest`` down to ``smallest``: Σ L r^(−1/ζ) for L r^(−1/ζ) ≥ smallest."""
    n = int((largest / smallest) ** zeta)
    if n < 1:
        return 0.0
    s = 1.0 / zeta
    m = min(n, EXACT_TERMS)
    total = np.sum(np.arange(1, m + 1, dtype=float) ** -s)
    if n > m:                                   # Euler–Maclaurin tail from m+1 to n
        a, b = m + 1.0, float(n)
        integral = np.log(b / a) if s == 1.0 else (b ** (1 - s) - a ** (1 - s)) / (1 - s)
        total += integral + 0.5 * (a ** -s + b ** -s)
    return largest * total


def largest_settlement(population: float, zeta: float, smallest: float = h.SETTLEMENT_SMALLEST) -> float:
    """Return the largest settlement of a Zipf system that houses ``population`` people from ``smallest`` up."""
    if population <= smallest:
        return population
    return brentq(lambda L: zipf_total(L, zeta, smallest) - population, smallest, population, xtol=1e-6 * population)


def zipf_sizes(largest: float, zeta: float, down_to: float) -> np.ndarray:
    """Return the settlement sizes L r^(−1/ζ) of a system that are at least ``down_to``."""
    n = int((largest / down_to) ** zeta)
    return largest * np.arange(1, max(n, 1) + 1, dtype=float) ** (-1.0 / zeta) if n >= 1 else np.empty(0)


def region_area_km2(transport_level: int) -> float:
    """Return the area of a settlement region for a transport level."""
    return h.REGION_AREA_KM2_LEVEL2 * h.REGION_AREA_STEP ** (transport_level - 2)


def settlement_regions(points: np.ndarray, people: np.ndarray, count: int, seed: int, name: str) -> np.ndarray:
    """Return a region label per cell (−1 where nobody lives) from population-weighted k-means with ``count`` regions."""
    occupied = np.flatnonzero(people > 0)
    labels = np.full(people.size, -1, dtype=np.int32)
    if occupied.size == 0:
        return labels
    count = int(min(max(count, 1), occupied.size))
    w = people[occupied]
    rng = named_rng(seed, name)
    centres = points[rng.choice(occupied, count, replace=False, p=w / w.sum())]
    pts = points[occupied]
    for _ in range(h.REGION_KMEANS_ITERATIONS):
        lab = np.argmax(pts @ centres.T, axis=1)
        sums = np.zeros((count, 3))
        np.add.at(sums, lab, pts * w[:, None])
        norm = np.linalg.norm(sums, axis=1)
        keep = norm > 0
        centres[keep] = sums[keep] / norm[keep, None]
    lab = np.argmax(pts @ centres.T, axis=1)
    _, labels[occupied] = np.unique(lab, return_inverse=True)       # drop empty regions, number 0..K-1
    return labels


def listing_cut(largests: np.ndarray, zeta: float, target: float) -> float:
    """Return the size down to which settlements are listed so that about ``target`` are listed in all."""
    largests = largests[largests > 0]
    if largests.size == 0 or target <= 0:
        return np.inf
    count = lambda n: np.sum(np.floor((largests / n) ** zeta)) - target
    lo = h.SETTLEMENT_SMALLEST
    if count(lo) <= 0:
        return lo
    return float(np.exp(brentq(lambda x: count(np.exp(x)), np.log(lo), np.log(largests.max()) + 1e-9)))


def settlement_form(population: np.ndarray, culture: Culture, habitat: np.ndarray | None = None) -> np.ndarray:
    """Return the form code of settlements of a culture by their population and, for colonists, their cell's habitat."""
    names = np.full(population.shape, "metropolis", dtype=object)
    if culture.development.energy == 0:
        names[:] = "camp"
    else:
        for form, limit in reversed(list(h.FORM_LIMITS.items())):
            names[population < limit] = form
    if culture.inhabitants == "colonist":
        if habitat is not None:
            names[habitat == Habitat.DOME] = "dome"
            names[habitat == Habitat.UNDERGROUND] = "underground"
        names[population < h.OUTPOST_MAX_POPULATION] = "outpost"
    return np.array([SETTLEMENT_FORMS.index(n) for n in names], dtype=np.int8)


def _place_region(points, cells, remaining, sizes, area_km2, radius_km, rng):
    """Place a region's settlements in rank order; return their cells and update ``remaining`` (region cells)."""
    placed = np.empty(sizes.size, dtype=np.int64)
    nearest_placed = np.full(cells.size, np.inf)
    pts = points[cells]
    for r, n in enumerate(sizes, start=1):
        alive = remaining > 0
        if not alive.any():
            return placed[:r - 1]
        spaced = alive & (nearest_placed >= h.SETTLEMENT_EXCLUSION * np.sqrt(area_km2 / r))
        pool = spaced if spaced.any() else alive
        weight = np.where(pool, remaining, 0.0) ** 2
        k = rng.choice(cells.size, p=weight / weight.sum())
        dist = radius_km * np.arccos(np.clip(pts @ pts[k], -1.0, 1.0))
        need = n
        for j in np.argsort(dist, kind="stable"):
            take = min(remaining[j], need)
            remaining[j] -= take
            need -= take
            if need <= 1e-9:
                break
        placed[r - 1] = k
        nearest_placed = np.minimum(nearest_placed, dist)
    return placed


def _jitter(points: np.ndarray, cells: np.ndarray, spacing_km: float, radius_km: float, rng) -> tuple:
    """Return latitude and longitude (degrees) of points scattered around their cells' centres."""
    p = points[cells]
    east = np.cross([0.0, 0.0, 1.0], p)
    bad = np.linalg.norm(east, axis=1) < 1e-9
    east[bad] = [1.0, 0.0, 0.0]
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(p, east)
    angle = rng.uniform(0, 2 * np.pi, cells.size)
    reach = h.SETTLEMENT_JITTER * spacing_km / radius_km * np.sqrt(rng.uniform(0, 1, cells.size))
    q = p + reach[:, None] * (np.cos(angle)[:, None] * east + np.sin(angle)[:, None] * north)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1))), np.degrees(np.arctan2(q[:, 1], q[:, 0]))


def _regions_for(points, people, culture, occupied_km2, seed, key):
    """Return region labels and each region's largest settlement; with a given largest settlement, solve the region count."""
    zeta = culture.zipf_exponent or h.ZIPF_DEFAULT

    def build(count):
        labels = settlement_regions(points, people, count, seed, key)
        totals = np.bincount(labels[labels >= 0], weights=people[labels >= 0])
        return labels, np.array([largest_settlement(p, zeta) for p in totals])

    count = max(1, int(round(occupied_km2 / region_area_km2(culture.development.transport))))
    if culture.largest_population is None:
        return build(count)
    lo, hi = 1, int(np.count_nonzero(people > 0))
    best = None
    while lo <= hi:                               # fewer regions give larger settlements
        mid = (lo + hi) // 2
        labels, largests = build(mid)
        if best is None or abs(np.log(largests.max() / culture.largest_population)) < best[0]:
            best = (abs(np.log(largests.max() / culture.largest_population)), labels, largests)
        if largests.max() > culture.largest_population:
            lo = mid + 1
        else:
            hi = mid - 1
    return best[1], best[2]


def place_settlements(points: np.ndarray, density: np.ndarray, cultures: list[Culture], cell_area_km2: float,
                      spacing_km: float, radius_km: float, seed: int, habitat: np.ndarray | None = None):
    """Return the listed settlements, dispersed population and region labels (culture × cell), and filled cultures.

    ``density`` is people/km² per culture and cell. Cultures get their Zipf
    exponent and largest settlement filled in when the spec left them open.
    """
    people = density * cell_area_km2
    dispersed = people.copy()
    regions = np.full(people.shape, -1, dtype=np.int32)
    columns = {k: [] for k in ("culture", "cell", "population", "region", "rank", "lat_deg", "lon_deg", "form")}
    filled = []
    region_offset = 0
    for i, c in enumerate(cultures):
        zeta = c.zipf_exponent or h.ZIPF_DEFAULT
        if c.development.energy == 0:
            filled.append(replace(c, zipf_exponent=zeta))
            continue
        occupied_km2 = np.count_nonzero(people[i] > 0) * cell_area_km2
        labels, largests = _regions_for(points, people[i], c, occupied_km2, seed, f"culture.{i}.regions")
        target = h.SETTLEMENTS_PER_CELL * np.count_nonzero(people[i] > 0)
        cut = listing_cut(largests, zeta, target)
        rng = named_rng(seed, f"culture.{i}.settlements")
        for k, largest in enumerate(largests):
            cells = np.flatnonzero(labels == k)
            sizes = zipf_sizes(largest, zeta, cut)
            if sizes.size == 0:
                continue
            area = cells.size * cell_area_km2
            remaining = dispersed[i, cells]
            where = _place_region(points, cells, remaining, sizes, area, radius_km, rng)
            dispersed[i, cells] = remaining
            n = where.size
            columns["culture"].append(np.full(n, c.id))
            columns["cell"].append(cells[where])
            columns["population"].append(sizes[:n])
            columns["region"].append(np.full(n, region_offset + k))
            columns["rank"].append(np.arange(1, n + 1))
            columns["form"].append(settlement_form(sizes[:n], c, None if habitat is None else habitat[cells[where]]))
        regions[i] = np.where(labels >= 0, labels + region_offset, -1)
        region_offset += largests.size
        filled.append(replace(c, zipf_exponent=zeta,
                              largest_population=float(largests.max()) if largests.size else c.largest_population))
    if not columns["cell"]:
        return SettlementTable.empty(), dispersed / cell_area_km2, regions, filled
    data = {k: np.concatenate(v) for k, v in columns.items() if v}
    order = np.lexsort((data["rank"], data["region"], data["culture"]))
    data = {k: v[order] for k, v in data.items()}
    lat, lon = _jitter(points, data["cell"], spacing_km, radius_km, named_rng(seed, "settlements.jitter"))
    data["lat_deg"], data["lon_deg"] = lat, lon
    table = SettlementTable(id=np.arange(data["cell"].size), **data)
    return table, dispersed / cell_area_km2, regions, filled
