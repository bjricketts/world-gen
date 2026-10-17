"""Compare the population a culture spec places on the Earth example with Earth's population distribution.

Usage: python scripts/validate_culture.py [--seeds 6] [--first-seed 0] [--resolution standard] [--worlds DIR ...]
       [--zoom 2] [--hostile DIR ...]

Places one modern (electric) culture and one foraging culture on each Earth world (``--worlds`` reuses
saved ones) and checks the population, settlement, road and polity measures against Earth; checks the
sound systems of 2000 generated cultures against WALS and PHOIBLE; zooms into ``--zoom`` regions around
towns on polity borders of each world; and, with ``--hostile``, places a colony on the given saved worlds.
Each row passes, fails, is a known gap (shown, not counted) or is reported without a verdict. The exit
code is 1 if any counted row fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from worldgen import load_world
from worldgen.culture.build import apply_culture, build_civilisation
from worldgen.culture.spec import CultureSpec
from worldgen.culture.suitability import covariates
from worldgen.generate import generate_world
from worldgen.spec import load_spec

EARTH_SPEC = Path(__file__).resolve().parent.parent / "examples" / "earth.yaml"
EARTH_MEAN_DENSITY = 5.6e9 / 1.297e8          # Cohen & Small (1998): 1994 population over occupied land

# Each row: name, Earth value, format, test on the six-world mean, and kind ("check" counts towards the
# exit code, "gap" is a known shortfall shown but not counted, "report" has no verdict). Elevation: Cohen &
# Small (1998); coast: Small & Cohen (2004); niche: Xu et al. (2020); density quantiles relative to the
# mean: Cohen & Small (1998); world population at each epoch: HYDE 3.2 (Klein Goldewijk et al. 2017).
# Tallavaara et al. (2018) observe forager densities up to ln ~5 per 100 km² (their Fig. 2A).


def near(tol: float) -> Callable:
    """Return a test that the model is within ``tol`` of the Earth value."""
    return lambda m, e: abs(m - e) <= tol


def factor(f: float) -> Callable:
    """Return a test that the model is within a factor ``f`` of the Earth value."""
    return lambda m, e: e / f <= m <= e * f


def between(lo: float, hi: float) -> Callable:
    """Return a test that the model lies in [lo, hi]."""
    return lambda m, e: lo <= m <= hi


# Tolerances: shares ±0.05, or ±0.1 where the distance is at or below the 124 km grid spacing or the
# six worlds' coastlines spread the value by more than 0.15; densities within a factor 1.5 (Cohen &
# Small's units average ~7000 km², the grid's cells 15 000 km²); calibrated values within 5-10%.
COLUMNS = [
    ("population below 100 m", 0.335, "{:.3f}", near(0.05), "check"),
    ("within 100 km of coast", 0.38, "{:.3f}", near(0.10), "check"),
    ("within 200 km of coast", 0.50, "{:.3f}", near(0.05), "check"),
    ("within 400 km of coast", 0.67, "{:.3f}", near(0.10), "check"),
    ("temperature mode (C)", 13, "{:.0f}", between(11, 15), "check"),
    ("people above 29 C", None, "{:.3f}", None, "report"),
    ("person-median density / mean", 262 / EARTH_MEAN_DENSITY, "{:.2f}", factor(1.5), "check"),
    ("person-75th density / mean", 648 / EARTH_MEAN_DENSITY, "{:.2f}", factor(1.5), "check"),
    ("land-median density / mean", 4.3 / EARTH_MEAN_DENSITY, "{:.3f}", factor(1.5), "check"),
    ("land-75th density / mean", 24.5 / EARTH_MEAN_DENSITY, "{:.2f}", factor(1.5), "check"),
    ("modern default population (G)", 7.26, "{:.2f}", factor(1.05), "check"),
    ("forager default population (M)", 4.4, "{:.2f}", factor(1.05), "check"),
    ("forager 99th pct ln(/100 km2)", 5.0, "{:.2f}", between(1.0, 5.0), "check"),
    ("culture step (s)", None, "{:.2f}", None, "report"),
    ("largest settlement, water_wind (M)", 1.0, "{:.2f}", factor(1.5), "check"),
    ("largest settlement, electric (M)", None, "{:.1f}", None, "report"),
    ("electric share in places >= 12k", 0.63, "{:.2f}", between(0.53, 0.83), "check"),
    ("electric global zeta above 100k", 1.0, "{:.2f}", between(0.94, 1.10), "check"),
    ("electric regions", None, "{:.0f}", None, "report"),
    ("spacing of top 200 (Clark-Evans)", 1.0, "{:.2f}", between(1.0, 2.15), "check"),
    ("farming grid roads, top-quarter regions (km/km2)", 0.026, "{:.4f}", factor(1.3), "check"),
    ("electric grid roads, top-quarter regions (km/km2)", 0.025, "{:.4f}", factor(1.3), "check"),
    ("electric route factor (300-1500 km)", None, "{:.2f}", None, "report"),
    ("electric mean degree at grid road nodes", None, "{:.2f}", None, "report"),
    ("electric main roads ~ area^a", 1.01, "{:.2f}", near(0.2), "gap"),
    ("electric main roads ~ density^b", 0.80, "{:.2f}", near(0.2), "gap"),
    ("largest polity / Taagepera, writing", 1.0, "{:.2f}", factor(1.5), "gap"),
    ("largest polity / Taagepera, print", 1.0, "{:.2f}", factor(1.5), "gap"),
    ("largest polity / Taagepera, telegraph", 1.0, "{:.2f}", factor(1.5), "gap"),
    ("largest polity / Taagepera, digital", 1.0, "{:.2f}", factor(1.5), "gap"),
    ("digital effective number (per 133 Mm2)", 24, "{:.0f}", factor(1.1), "check"),
    ("digital border edges at large rivers", None, "{:.2f}", None, "report"),
    ("  all edges at large rivers", None, "{:.2f}", None, "report"),
]


def verdict(test, kind: str, model: float, earth) -> str:
    """Return the verdict of one row: pass, FAIL, gap (a known shortfall), ok (a known gap now met) or blank."""
    if kind == "report" or test is None:
        return ""
    good = bool(test(model, earth))
    if kind == "gap":
        return "ok" if good else "gap"
    return "pass" if good else "FAIL"


def _within(coast_km: np.ndarray, spacing_km: float, km: float) -> np.ndarray:
    """Return the share of each land cell within ``km`` of the coast; a cell spans half a spacing either side of its centre."""
    return np.clip((km - coast_km + spacing_km / 2) / spacing_km, 0.0, 1.0)


def _clark_evans(points: np.ndarray, cells: np.ndarray, occupied_km2: float, radius_km: float) -> float:
    """Return the mean nearest-neighbour distance over its value for random points (1 random, above 1 regular)."""
    from scipy.spatial import cKDTree
    p = points[cells]
    d, _ = cKDTree(p).query(p, k=2)
    nn = 2 * np.arcsin(np.minimum(d[:, 1] / 2, 1)) * radius_km
    return nn.mean() / (0.5 * np.sqrt(occupied_km2 / len(p)))


def measure(world) -> list[float]:
    """Return the COLUMNS measures for one world."""
    modern = CultureSpec(cultures=[{"development": {"energy": "electric"}}])
    forager = CultureSpec(cultures=[{"development": {"energy": "foraging"}}])
    t = time.perf_counter()
    civ = build_civilisation(world, modern)
    step = time.perf_counter() - t
    wild = build_civilisation(world, forager)
    cov = covariates(world)
    s = world.surface
    d = civ.layers["population_density"].values[0]
    land = d > 0
    pop = d * cov.cell_area_km2
    share = pop / pop.sum()
    temp = s["air_temperature"].values - 273.15
    hist, edges = np.histogram(temp[land], bins=np.arange(-20, 36, 2), weights=share[land])
    rel = d[land] / d[land].mean()
    order = np.argsort(rel)
    cum = np.cumsum(share[land][order])
    q = lambda f: rel[order][np.searchsorted(cum, f)]
    wild_d = wild.layers["population_density"].values[0]
    return [
        share[s["elevation"].values < 100].sum(),
        *[(share * _within(cov.coast_km, cov.spacing_km, km)).sum() for km in (100, 200, 400)],
        edges[np.argmax(hist)] + 1,
        share[temp > 29].sum(),
        q(0.5), q(0.75), np.median(rel), np.percentile(rel, 75),
        civ.cultures[0].population / 1e9,
        wild.cultures[0].population / 1e6,
        np.log(np.percentile(wild_d[wild_d > 0], 99) * 100),
        step,
        *settlement_measures(world, civ, cov),
        road_density(world, farm := build_civilisation(world, CultureSpec(cultures=[{"development": {"energy": "farming"}}])), cov),
        *road_measures(world, civ, cov),
        *polity_measures(world, {"farming": farm, "electric": civ}),
    ]


# Largest polity by Earth epoch (Taagepera 1978, Mm², of 133 Mm² of dry land): 1 CE Han, 1700 Russia,
# 1900 British, 1975 USSR; matched to the default development of each energy level.
LARGEST_POLITY_MM2 = {"farming": 6.2, "water_wind": 12.0, "fossil": 30.0, "electric": 22.5}


def polity_measures(world, built: dict) -> list[float]:
    """Return the largest polity over Taagepera's by level, the effective number and river borders (digital)."""
    from worldgen.culture.polities import effective_number
    ratios = []
    for level, largest in LARGEST_POLITY_MM2.items():
        civ = built.get(level) or build_civilisation(world, CultureSpec(cultures=[{"development": {"energy": level}}]))
        areas = np.array([p.area_km2 for p in civ.polities])
        ratios.append(areas.max() / areas.sum() * 133.0 / largest)
    civ = built["electric"]
    areas = np.array([p.area_km2 for p in civ.polities])
    owner = civ.layers["polity"].values[0]
    s = world.surface
    adj = world.grid.neighbours.tocoo()
    both = (owner[adj.row] >= 0) & (owner[adj.col] >= 0) & (adj.row < adj.col)
    border = both & (owner[adj.row] != owner[adj.col])
    river = (s["discharge"].values > 1000) & ~s["ocean"].values
    at_river = river[adj.row] | river[adj.col]
    return [*ratios, effective_number(areas) * 1.33e8 / areas.sum(), at_river[border].mean(), at_river[both].mean()]


def _road_edges(civ, kinds=("track", "road", "trunk", "highway")):
    """Return the grid edges (cell pairs) of the land roads of the given kinds."""
    from worldgen.culture.model import ROAD_KINDS
    r = civ.roads
    a, b = [], []
    for i in np.flatnonzero(np.isin(r.kind, [ROAD_KINDS.index(k) for k in kinds])):
        p = r.path(i)
        a.append(p[:-1]); b.append(p[1:])
    return (np.concatenate(a), np.concatenate(b)) if a else (np.empty(0, int), np.empty(0, int))


def _per_region_roads(world, civ, cov, kinds=("track", "road", "trunk", "highway")):
    """Return per settlement region: area (km2), population density and land-road length (km) of the given kinds."""
    a, b = _road_edges(civ, kinds)
    p = world.grid.points
    km = cov.radius_km * np.arccos(np.clip(np.einsum("ij,ij->i", p[a], p[b]), -1, 1))
    region = civ.layers["region"].values[0]
    people = civ.layers["population_density"].values[0] * cov.cell_area_km2
    n = region.max() + 1
    roads = np.bincount(region[a].clip(0), weights=km / 2, minlength=n) + np.bincount(region[b].clip(0), weights=km / 2, minlength=n)
    area = np.bincount(region[region >= 0], minlength=n) * cov.cell_area_km2
    pop = np.bincount(region[region >= 0], weights=people[region >= 0], minlength=n)
    return area, pop / np.maximum(area, 1), roads


def road_density(world, civ, cov) -> float:
    """Return the land-road length per km2 over the most densely populated quarter of settlement regions."""
    area, density, roads = _per_region_roads(world, civ, cov)
    top = density >= np.percentile(density, 75)
    return roads[top].sum() / area[top].sum()


def road_measures(world, civ, cov) -> list[float]:
    """Return road density, route factor, mean degree and the elasticities of road length across regions."""
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    a, b = _road_edges(civ)
    p = world.grid.points
    km = cov.radius_km * np.arccos(np.clip(np.einsum("ij,ij->i", p[a], p[b]), -1, 1))
    n = world.grid.size
    net = sparse.csr_matrix((np.r_[km, km], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    nodes = np.unique(civ.settlements.cell)
    rng = np.random.default_rng(0)
    src = rng.choice(nodes, min(80, nodes.size), replace=False)
    dist = dijkstra(net, indices=src)[:, nodes]
    straight = cov.radius_km * np.arccos(np.clip(p[src] @ p[nodes].T, -1, 1))
    ok = np.isfinite(dist) & (straight > 300) & (straight < 1500)
    edges = np.unique(np.sort(np.stack([a, b], 1), axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=n)
    area, density, roads = _per_region_roads(world, civ, cov, ("trunk", "highway"))
    good = (roads > 0) & (density > 0)
    X = np.column_stack([np.ones(good.sum()), np.log10(area[good]), np.log10(density[good])])
    coef = np.linalg.lstsq(X, np.log10(roads[good]), rcond=None)[0]
    return [road_density(world, civ, cov), np.median(dist[ok] / straight[ok]), degree[degree > 0].mean(),
            coef[1], coef[2]]


def settlement_measures(world, civ, cov) -> list[float]:
    """Return the settlement rows: largest settlements, the share above 12 000, the global exponent, spacing."""
    from worldgen.culture.settlements import zipf_sizes
    mid = build_civilisation(world, CultureSpec(cultures=[{"development": {"energy": "water_wind"}}]))
    st = civ.settlements
    c = civ.cultures[0]
    # every region's full hierarchy, including settlements below the listing cut
    full = np.concatenate([zipf_sizes(st.population[(st.region == k)].max(), c.zipf_exponent, 50.0)
                           for k in np.unique(st.region)])
    big = full[full >= 1e5]
    zeta = len(big) / np.log(big / 1e5).sum()
    top = np.argsort(st.population)[::-1][:200]
    occupied = np.count_nonzero(civ.layers["population_density"].values[0] > 0) * cov.cell_area_km2
    return [mid.settlements.population.max() / 1e6, st.population.max() / 1e6,
            full[full >= 12e3].sum() / full.sum(), zeta, len(np.unique(st.region)),
            _clark_evans(world.grid.points, st.cell[top], occupied, cov.radius_km)]


def phonology_rows(count: int = 2000) -> list:
    """Return verdict rows for the sound systems of generated cultures against WALS and PHOIBLE."""
    from worldgen import heuristics as h
    from worldgen.culture.names import make_phonology
    phons = [make_phonology(0, f"check.{i}") for i in range(count)]
    nc = np.array([len(p["consonants"]) for p in phons])
    nv = np.array([len(p["vowels"]) for p in phons])
    syl = [p["syllables"] for p in phons]
    segs = list(h.CONSONANTS)
    got = np.array([np.mean([s in p["consonants"] for p in phons]) for s in segs])
    ref = np.array([h.CONSONANTS[s][0] for s in segs])
    rows = [("consonants, median (WALS)", 21, np.median(nc), "{:.0f}", near(1)),
            ("consonants, mean (WALS)", 22.7, nc.mean(), "{:.1f}", near(2))]
    for name, earth, share in [("vowel qualities 2-4", 0.16, np.mean(nv <= 4)),
                               ("vowel qualities 5-6", 0.51, np.mean((nv >= 5) & (nv <= 6))),
                               ("vowel qualities 7-14", 0.33, np.mean(nv >= 7))]:
        rows.append((f"{name} (WALS)", earth, share, "{:.2f}", near(0.05)))
    for kind, earth in (("simple", 0.13), ("moderate", 0.56), ("complex", 0.31)):
        rows.append((f"syllables {kind} (WALS)", earth, syl.count(kind) / count, "{:.2f}", near(0.05)))
    rows.append(("consonant presence vs PHOIBLE share (r)", 1.0, np.corrcoef(got, ref)[0, 1], "{:.2f}",
                 between(0.9, 1.0)))
    return [(label, fmt.format(e), fmt.format(m), verdict(test, "check", m, e)) for label, e, m, fmt, test in rows]


def hostile_rows(paths) -> list:
    """Return rows of habitat shares, elevator taper and colony forms on the given saved worlds, with a verdict
    that every land cell's habitat follows its air and temperature, and that a colony exists where any can."""
    from worldgen.culture.habitats import Habitat, elevator_taper, habitat_classes
    from worldgen.culture.model import SETTLEMENT_FORMS
    rows = []
    for path in paths:
        world = load_world(path)
        land = ~world.surface["ocean"].values
        hab = habitat_classes(world)
        shares = " / ".join(f"{np.mean(hab[land] == k):.2f}" for k in Habitat)
        try:
            civ = build_civilisation(world, CultureSpec(cultures=[{"inhabitants": "colonist",
                                                                   "colony": {"founded_years_ago": 300}}]))
            forms = ", ".join(np.unique([SETTLEMENT_FORMS[f] for f in civ.settlements.form]).tolist())
            settled = True
        except ValueError:
            forms, settled = "no colony", False
        livable = bool(np.any(hab[land] != Habitat.NONE))
        rows.append((f"{Path(path).name}: none / open / dome / underground", "",
                     f"{shares}; taper {elevator_taper(world.state):.3g}; {forms}",
                     "pass" if settled == livable else "FAIL"))
    return rows


# GRIP (Meijer et al. 2018): 21 million km of roads; over 133 Mm² of land that is 0.16 km/km² on average.
GRIP_MEAN_ROAD_DENSITY = 21e6 / 1.33e8
# Popelka & Smith (2020): share of border length along rivers, national and first-order subnational.
RIVER_BORDER_SHARE = (0.23, 0.17)


def zoom_measures(world, count: int, seed: int = 0) -> list[tuple]:
    """Return per region, around towns on polity borders: people/km², villages per 100 km², built-up share,
    road km/km², border and all-edge shares at rivers, mean degree of road junctions, and seconds."""
    from worldgen.culture.local import border_river_share, zoom_culture
    from worldgen.zoom import zoom_region
    world = apply_culture(world, CultureSpec(cultures=[{"development": {"energy": "electric"}}]))
    civ = world.culture
    pol = civ.layers["polity"].values[0]
    adj = world.grid.neighbours.tocoo()
    edge = np.zeros(world.grid.size, dtype=bool)
    edge[adj.row[(pol[adj.row] != pol[adj.col]) & (pol[adj.row] >= 0) & (pol[adj.col] >= 0)]] = True
    st = civ.settlements
    towns = np.flatnonzero(edge[st.cell] & (st.population > 1e4) & (st.population < 5e5))
    out = []
    for k in np.random.default_rng(seed).choice(towns, min(count, towns.size), replace=False):
        t = time.perf_counter()
        region = zoom_region(world, float(st.lat_deg[k]), float(st.lon_deg[k]))
        local = zoom_culture(world, region)
        dt = time.perf_counter() - t
        land = ~region.ocean & ~region.lake
        node_km2 = (float(np.median(region.grid.neighbours.data)) * region.radius_m / 1e3) ** 2
        land_km2 = land.sum() * node_km2
        pairs = set()
        km = 0.0
        for nodes, _, _ in local.roads:
            p = region.grid.points[nodes]
            km += region.radius_m / 1e3 * np.arccos(np.clip(np.einsum("ij,ij->i", p[:-1], p[1:]), -1, 1)).sum()
            pairs.update(zip(np.minimum(nodes[:-1], nodes[1:]).tolist(), np.maximum(nodes[:-1], nodes[1:]).tolist()))
        degree = np.bincount(np.array(sorted(pairs)).ravel(), minlength=region.grid.size) if pairs else np.zeros(1)
        junction = (degree > 0) & (degree != 2)
        border, every = border_river_share(region, local)
        out.append((local.population.sum() / land_km2, 100 * np.sum(local.listed < 0) / land_km2,
                    np.sum(np.pi * local.radius_km**2) / land_km2, km / land_km2, border, every,
                    degree[junction].mean() if junction.any() else np.nan, dt))
    return out


def zoom_rows(measures: list[tuple]) -> list:
    """Return verdict rows for the zoomed regions of all worlds."""
    from scipy.stats import spearmanr
    z = np.array(measures, dtype=float)
    density, villages, built, roads, border, every, degree, dt = z.T
    rho = spearmanr(density, roads).statistic
    med = float(np.nanmedian(border))
    lo, hi = RIVER_BORDER_SHARE[0] - 0.1, RIVER_BORDER_SHARE[0] + 0.1
    rng = lambda v, f: f"{f.format(np.nanmin(v))}–{f.format(np.nanmax(v))}"
    return [
        ("regions (people/km2)", "", f"{len(z)} ({rng(density, '{:.0f}')})", ""),
        ("villages per 100 km2", "", rng(villages, "{:.1f}"), ""),
        ("built-up share of land", "", rng(built, "{:.3f}"), ""),
        ("local road km/km2", f"{GRIP_MEAN_ROAD_DENSITY:.2f} (GRIP, all land)", rng(roads, "{:.2f}"), ""),
        ("road density rises with people (Spearman)", "> 0.7", f"{rho:.2f}", "pass" if rho > 0.7 else "FAIL"),
        ("border edges at order-5 rivers, median", f"{RIVER_BORDER_SHARE[0]:.2f}", f"{med:.2f}",
         "pass" if lo <= med <= hi else "FAIL"),
        ("  all land edges at order-5 rivers", "", rng(every, "{:.2f}"), ""),
        ("mean degree of road junctions and ends", "", f"{np.nanmean(degree):.2f}", ""),
        ("zoom and culture (s)", "", rng(dt, "{:.0f}"), ""),
    ]


def print_rows(title: str, rows: list) -> None:
    """Print a titled block of (label, Earth, model, verdict) rows."""
    width = max(len(r[0]) for r in rows)
    print(f"\n{title}")
    for label, earth, model, v in rows:
        print(f"{label:{width}}  {earth:>24}  {model:>28}  {v}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--resolution", default="standard")
    parser.add_argument("--worlds", nargs="*", default=None)
    parser.add_argument("--zoom", type=int, default=2, help="regions per world (0: none)")
    parser.add_argument("--hostile", nargs="*", default=None)
    args = parser.parse_args()
    if args.worlds:
        worlds = [(Path(p).name, lambda p=p: load_world(p)) for p in args.worlds]
    else:
        spec = load_spec(EARTH_SPEC)
        worlds = [(f"seed {i}", lambda i=i: generate_world(spec.model_copy(update={"seed": i}), args.resolution))
                  for i in range(args.first_seed, args.first_seed + args.seeds)]
    rows, zoomed, taper = [], [], None
    from worldgen.culture.habitats import elevator_taper
    for name, make in worlds:
        world = make()
        rows.append(measure(world))
        if args.zoom:
            zoomed += zoom_measures(world, args.zoom)
        taper = elevator_taper(world.state)
        print(f"{name} done", flush=True)
    rows = np.array(rows)
    table = []
    for (label, earth, fmt, test, kind), col in zip(COLUMNS, rows.T):
        e = "" if earth is None else fmt.format(earth)
        spread = f"{fmt.format(col.mean())} ({fmt.format(col.min())}–{fmt.format(col.max())})"
        table.append((label, e, spread, verdict(test, kind, col.mean(), earth)))
    table.append(("elevator taper on the Earth example", "10 (Pearson 1975)", f"{taper:.1f}",
                  verdict(factor(1.1), "check", taper, 10.0)))
    blocks = [(f"Earth example, {len(worlds)} worlds: mean (range)", table),
              ("Sound systems of 2000 generated cultures (WALS: Maddieson 2013; PHOIBLE 2.0)", phonology_rows())]
    if zoomed:
        blocks.append((f"Zoomed regions around towns on polity borders ({args.zoom} per world, electric culture)",
                       zoom_rows(zoomed)))
    if args.hostile:
        blocks.append(("Colonists (300-year colony): habitat shares of land; elevator taper; forms",
                       hostile_rows(args.hostile)))
    for title, block in blocks:
        print_rows(title, block)
    verdicts = [r[3] for _, block in blocks for r in block]
    failed = verdicts.count("FAIL")
    print(f"\n{verdicts.count('pass')} passed, {failed} failed, {verdicts.count('gap')} known gaps"
          f"{', ' + str(verdicts.count('ok')) + ' gaps now met' if 'ok' in verdicts else ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
