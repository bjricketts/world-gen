"""Population density of each culture: where it lives, how many people, and how they gather.

Each culture gets a range (its homeland, the reach of its colony, or a share of
the land left to cultures without one), a carrying capacity from
``suitability``, and a total population (given, or from Earth's population at
the culture's energy level scaled by the productive land in its range). People
then gather where others already live: density follows capacity times market
access, D = K A^c, where A sums the density nearby with a distance decay; the
iteration converges in a few tens of steps.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .. import heuristics as h
from ..util import named_rng
from .model import Culture
from .suitability import Covariates, density_terms, log_capacity, productive_area_km2


def _unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the unit vector at a latitude and longitude (degrees)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _arc_km(points: np.ndarray, centre: np.ndarray, radius_km: float) -> np.ndarray:
    """Return the great-circle distance (km) from each point to a centre (unit vectors)."""
    return radius_km * np.arccos(np.clip(points @ centre, -1.0, 1.0))


def _soft_disc(points: np.ndarray, centre: np.ndarray, reach_km: float, radius_km: float) -> np.ndarray:
    """Return 1 within ``reach_km`` of a centre, fading outside over a share of the reach, cut where it falls below HOMELAND_TAIL."""
    d = _arc_km(points, centre, radius_km)
    share = np.exp(-np.maximum(d - reach_km, 0.0) / (h.HOMELAND_EDGE * reach_km))
    return np.where(share >= h.HOMELAND_TAIL, share, 0.0)


def _draw_cell(capacity: np.ndarray, seed: int, name: str) -> int:
    """Return a cell drawn with probability proportional to capacity, on its own random stream."""
    p = np.where(np.isfinite(capacity), capacity, 0.0)
    if p.sum() <= 0:
        raise ValueError("no land this culture can live on (for colonists: no land cell has a survivable temperature)")
    return int(named_rng(seed, name).choice(p.size, p=p / p.sum()))


def landing_weight(cov: Covariates) -> np.ndarray:
    """Return the weight of each cell as a landing site: low latitude and plain ground preferred (Golombek et al. 2012)."""
    w = np.ones(cov.land.size)
    if cov.lat_deg is not None:
        w = np.where(np.abs(cov.lat_deg) <= h.LANDING_MAX_LATITUDE_DEG, w, h.LANDING_OFF_LATITUDE_WEIGHT)
    if cov.rough is not None:
        w = np.where(cov.rough, w * h.LANDING_ROUGH_WEIGHT, w)
    return w


def colony_age(culture: Culture) -> float:
    """Return the years since a colony was founded, with the default when the spec left it open."""
    return h.COLONY_DEFAULT_AGE_YEARS if culture.founded_years_ago is None else culture.founded_years_ago


def culture_ranges(points: np.ndarray, cov: Covariates, cultures: list[Culture], capacity: np.ndarray,
                   seed: int) -> tuple[np.ndarray, list[list[dict]]]:
    """Return each culture's share of every cell (0–1, summing to at most 1) and the colonies' landing sites.

    Homelands and colonies claim their discs; natives without a homeland
    divide the land left over, each taking the cells nearest a seed drawn
    where the land suits it.
    """
    n, size = len(cultures), points.shape[0]
    share = np.zeros((n, size))
    sites_used: list[list[dict]] = [[] for _ in cultures]
    homeless = []
    for i, c in enumerate(cultures):
        if c.homeland:
            centre = _unit(c.homeland["lat_deg"], c.homeland["lon_deg"])
            share[i] = _soft_disc(points, centre, c.homeland["radius_km"], cov.radius_km)
        elif c.inhabitants == "colonist":
            sites = list(c.landing_sites)
            if not sites:
                cell = _draw_cell(capacity[i] * landing_weight(cov), seed, f"culture.{i}.landing_site")
                lat = float(np.degrees(np.arcsin(np.clip(points[cell, 2], -1, 1))))
                lon = float(np.degrees(np.arctan2(points[cell, 1], points[cell, 0])))
                sites = [{"lat_deg": lat, "lon_deg": lon}]
            sites_used[i] = sites
            reach = h.COLONY_REACH_KM + h.COLONY_SPREAD_KM_PER_YEAR * colony_age(c)
            for site in sites:
                disc = _soft_disc(points, _unit(site["lat_deg"], site["lon_deg"]), reach, cov.radius_km)
                share[i] = np.maximum(share[i], disc)
        else:
            homeless.append(i)
    if homeless:
        free = np.clip(1.0 - share.sum(axis=0), 0.0, 1.0)
        seeds = np.array([points[_draw_cell(capacity[i], seed, f"culture.{i}.homeland")] for i in homeless])
        nearest = np.argmax(points @ seeds.T, axis=1)
        for k, i in enumerate(homeless):
            share[i] = free * (nearest == k)
    total = share.sum(axis=0)
    return share / np.maximum(total, 1.0), sites_used


def market_kernel(points: np.ndarray, cells: np.ndarray, radius_km: float) -> sparse.csr_matrix:
    """Return the distance-decay weights exp(−d / length) between the given cells, within the kernel reach."""
    length = h.DENSITY_CLUSTER_LENGTH_KM
    reach = h.DENSITY_CLUSTER_REACH * length
    tree = cKDTree(points[cells])
    chord = 2 * np.sin(reach / radius_km / 2)
    pairs = tree.sparse_distance_matrix(tree, chord, output_type="coo_matrix")
    arc = 2 * np.arcsin(np.clip(pairs.data / 2, 0, 1)) * radius_km
    k = sparse.coo_matrix((np.exp(-arc / length), (pairs.row, pairs.col)), shape=(cells.size, cells.size)).tocsr()
    k.setdiag(1.0)
    return k


def cluster(capacity: np.ndarray, kernel: sparse.csr_matrix, strength: float) -> np.ndarray:
    """Return the relative density that capacity and market access settle to, D = K A^strength (sum as K)."""
    total = capacity.sum()
    if strength <= 0 or total <= 0:
        return capacity.copy()
    d = capacity.copy()
    for _ in range(h.DENSITY_CLUSTER_ITERATIONS):
        access = kernel @ d
        new = capacity * np.power(np.maximum(access, 1e-300), strength)
        new *= total / new.sum()
        done = np.abs(new - d).max() < h.DENSITY_CLUSTER_TOLERANCE * new.max()
        d = 0.5 * d + 0.5 * new
        if done:
            break
    return d * (total / d.sum())


def default_population(culture: Culture, cov: Covariates, share: np.ndarray) -> float:
    """Return the population of a culture whose spec leaves it open.

    Natives: Earth's population at their energy level per km² of productive
    land, times the productive land in their range. Colonists: founders grown
    exponentially over the colony's age, up to the same kind of capacity per
    km² of land in their range.
    """
    anchor = h.POPULATION_ANCHOR[culture.development.name("energy")]
    if culture.inhabitants == "colonist":
        land = np.where(cov.land, cov.cell_area_km2, 0.0)
        capacity = anchor / h.EARTH_HABITABLE_LAND_KM2 * float((share * land).sum())
        grown = h.COLONY_FOUNDERS * np.exp(min(h.COLONY_GROWTH_PER_YEAR * colony_age(culture), 700.0))
        return float(min(grown, capacity))
    return anchor / h.EARTH_PRODUCTIVE_LAND_KM2 * float((share * productive_area_km2(cov)).sum())


def population_density(points: np.ndarray, cov: Covariates, cultures: list[Culture], seed: int):
    """Return per-culture density (people/km², shape culture × cell), relative capacity, populations and sites."""
    n, size = len(cultures), points.shape[0]
    for c in cultures:
        if c.inhabitants == "native" and not cov.land_life:
            raise ValueError(f"culture {c.id}: natives need land life; this planet has none (use colonists)")
    terms = [density_terms(c) for c in cultures]
    ln_cap = np.array([log_capacity(cov, t) for t in terms])
    top = np.max(np.where(np.isfinite(ln_cap), ln_cap, -1e300), axis=1, keepdims=True)
    capacity = np.where(np.isfinite(ln_cap), np.exp(ln_cap - top), 0.0)
    share, sites = culture_ranges(points, cov, cultures, capacity, seed)
    livable = np.flatnonzero(np.isfinite(ln_cap).any(axis=0))
    kernel = None
    density = np.zeros((n, size))
    populations = []
    for i, c in enumerate(cultures):
        k = share[i] * capacity[i]
        if k.sum() <= 0:
            raise ValueError(f"culture {c.id}: no land it can live on within its range"
                             + (" (colonists need a cell with a survivable temperature)" if c.inhabitants == "colonist" else ""))
        population = c.population if c.population is not None else default_population(c, cov, share[i])
        populations.append(float(population))
        if terms[i]["clustering"] > 0 and kernel is None:
            kernel = market_kernel(points, livable, cov.radius_km)
        rel = np.zeros(size)
        rel[livable] = cluster(k[livable], kernel, terms[i]["clustering"]) if kernel is not None else k[livable]
        density[i] = population * rel / (rel.sum() * cov.cell_area_km2)
    return density, capacity, populations, sites
