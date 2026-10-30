"""Compare organic histories on the Earth example with Earth's Holocene.

Usage: python scripts/validate_organic.py [--seeds 6] [--first-seed 0] [--resolution standard] [--worlds DIR ...]
       [--colony-worlds 2] [--workers 2]

Runs one organic native over 12 015 years (the present taken as 2015; the k-th world's history has seed k, so
planet events differ between worlds) on each Earth world (``--worlds`` reuses
saved ones) and checks population, farming, development, states, wars and planet events against Earth. On the
first ``--colony-worlds`` worlds it runs the native again with an industrial colony from off the planet, carrying
crowd diseases, landing 500 years ago on another landmass, and reports the natives' losses after contact (off-planet
colonies are reported, not validated). Each row passes, fails, is a known gap (shown, not counted) or is reported
without a verdict; values are means over the worlds (medians for dates, and for the largest state, where one large
state in one world would move a mean; its mean is shown too). The exit code is 1 if any counted row fails.

A world enters a largest-state row only once some occupied cell holds Earth's leading energy level at that date
(and, at 2800 BCE, once it has a state);
the Mongol row counts only worlds whose largest steppe region spans STEPPE_BELT_KM.

Rows marked as known gaps failed when the script was written. Since then (tasks 9.12–9.13) the
population from 5000 BCE to 1800, the farming front, the largest state from 500 BCE to 1700 and the steppe
empire are met; populations stay below HYDE from 1900, half the worlds never industrialise, the largest state is
0.46 × Taagepera's at 1900, and most worlds have no state yet at 2800 BCE (see OPEN_POINTS.md).

References: HYDE 3.2 (Klein Goldewijk et al. 2017) for world population; Larson et al. (2014) for farming
origins; Pinhasi et al. (2005) for the farming front; Zabel et al. (2014) for arable land; Taagepera (1978) for
the largest state and Taagepera (1997) for the effective number of polities; Arbesman (2011) for the lifetimes
of large states; Clauset (2018) for the war-size exponent; Toohey & Sigl (2017) for large eruptions; Marani et
al. (2021) for pandemics; Koch et al. (2019) for native losses after contact.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_culture import between, factor, near, print_rows, verdict  # noqa: E402

EARTH_SPEC = Path(__file__).resolve().parent.parent / "examples" / "earth.yaml"
START = 12015.0
# HYDE 3.2 world population (years before 2015, people)
HYDE = [(7015, 19e6), (2015, 232e6), (1015, 323e6), (315, 592e6), (215, 943e6), (115, 1.64e9), (65, 2.53e9),
        (0, 7.26e9)]
HYDE_LABELS = ["5000 BCE", "1 CE", "1000", "1700", "1800", "1900", "1950", "2015"]
# Taagepera (1978): the largest state (Mm² of 133 Mm² of dry land) at a date (years before 2015)
TAAGEPERA = [(4815, 0.25, "2800 BCE"), (2515, 5.5, "500 BCE"), (2015, 6.2, "1 CE"), (715, 24.0, "1300"),
             (315, 12.0, "1700"), (115, 30.0, "1900")]
# Earth's leading energy level in force at each Taagepera date: farming (the 1 CE population anchor) until 1 CE,
# water_wind (the 1700 anchor) in 1300 and 1700, fossil (the 1900 anchor) in 1900. A world enters a date's
# largest-state row only once some occupied cell holds that level (the date's leading level, not a count of states).
TAAGEPERA_LEVEL = {4815: 1, 2515: 1, 2015: 1, 715: 2, 315: 2, 115: 3}
# At 2800 BCE only worlds with a state (writing) by then count (an earlier state is not required of worlds that have none).
TAAGEPERA_NEEDS_STATE = {4815}
EARTH_LAND_MM2 = 133.0
# The Eurasian steppe runs ~8000 km from Hungary to Manchuria (Britannica, "The Steppe"); a world's largest connected
# steppe region (Köppen BS) must span STEPPE_BELT_KM, three quarters of that, for the Mongol row. Taagepera (1978):
# the Mongol empire at 24 Mm² (his 1300 row).
STEPPE_BELT_KM = 6000.0
MONGOL_MM2 = 24.0
EPOCHS = sorted({float(y) for y, _ in HYDE[:-1]} | {float(y) for y, _, _ in TAAGEPERA}, reverse=True)
TAMBORA_TG = 28.1
SPANISH_FLU = 5.7e-3          # deaths a year as a share of the world's people (Marani et al. 2021)
COLONY_YEARS = 500.0
FRONT_BEFORE = 5000.0


def _history(world, cultures_spec: dict, seed: int):
    """Run an organic history on a world and return the result, the covariates and the suitability."""
    from worldgen.culture import CultureSpec, resolve_cultures
    from worldgen.culture.build import geology_layers
    from worldgen.culture.history.run import default_processes, run_history
    from worldgen.culture.history.suitability import build_suitability
    from worldgen.culture.spec import HistorySpec
    from worldgen.culture.suitability import covariates

    geo = geology_layers(world)
    cov = covariates(world, geo["resource_potential"])
    suit = build_suitability(world, cov, geo["resource_potential"])
    spec = CultureSpec.model_validate(cultures_spec)
    history = HistorySpec(start_years_ago=START, epochs=EPOCHS)
    result = run_history(world, resolve_cultures(spec, seed), history, seed, cov, default_processes(), suit,
                         geo["resource_potential"])
    return result, cov, suit


def _distance_km(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius_km: float) -> np.ndarray:
    """Return great-circle distances between rows of cells ``a`` and columns of cells ``b``."""
    return radius_km * np.arccos(np.clip(points[a] @ points[b].T, -1.0, 1.0))


def front_speed(world, result, cov) -> float:
    """Return the median speed (km/yr) at which farming reached cells 500 km or more from its nearest earlier
    origin, over cells farmed at least FRONT_BEFORE years ago (the Neolithic spread Pinhasi et al. measure ran
    9000–5000 years ago)."""
    settled = result.cells.settled_since
    origins = [(e.refs["cell"], e.years_ago) for e in result.chronicle.entries if e.kind == "farming_origin"]
    cells = np.flatnonzero(np.isfinite(settled) & (np.nan_to_num(settled) >= FRONT_BEFORE))
    if not origins or not cells.size:
        return float("nan")
    oc = np.array([o[0] for o in origins])
    ot = np.array([o[1] for o in origins])
    d = _distance_km(world.grid.points, cells, oc, cov.radius_km)
    d = np.where(ot[None, :] > settled[cells, None], d, np.inf)
    k = np.argmin(d, axis=1)
    dist = d[np.arange(cells.size), k]
    years = ot[k] - settled[cells]
    keep = np.isfinite(dist) & (dist >= 500.0) & (years > 0)
    return float(np.median(dist[keep] / years[keep])) if keep.any() else float("nan")


def expected_flu(result) -> float:
    """Return the Spanish-flu-intensity pandemics expected in 1600–1945 at the model's rate, for its farmers."""
    from worldgen import heuristics as h
    from worldgen.culture.history.events import gpd_exceedance

    s = result.series.table() if result.series is not None else None
    if s is None:
        return float("nan")
    mu, sd, xi = h.PANDEMIC_GPD
    rate = h.PANDEMIC_RATE * (gpd_exceedance(SPANISH_FLU, mu, sd, xi) - gpd_exceedance(h.PANDEMIC_MAX_INTENSITY, mu,
                                                                                      sd, xi))
    years = np.unique(s.years_ago)
    years = years[(years >= 70) & (years <= 415)]
    farmers = np.array([s.people[(s.years_ago == y) & (s.top_energy >= 1)].sum() for y in years])
    return float(rate * np.trapezoid(farmers, years) / h.PANDEMIC_REFERENCE_PEOPLE) if years.size else 0.0


def state_measures(snap, cell_km2: float, land_km2: float) -> tuple[float, float]:
    """Return the largest state's share of the land and the effective number of states by area."""
    owned = snap.owner[snap.owner >= 0]
    if not owned.size:
        return 0.0, 0.0
    area = np.bincount(owned).astype(float) * cell_km2
    area = area[area > 0]
    return float(area.max() / land_km2), float(area.sum() ** 2 / (area ** 2).sum())


def steppe_belt(world) -> tuple[float, float]:
    """Return the area (Mm²) and the longest span (km) of the largest connected region of steppe climate (Köppen BS)."""
    from scipy.sparse.csgraph import connected_components

    from worldgen.biosphere import KOPPEN_CODES

    s, grid = world.surface, world.grid
    steppe = np.isin(s["koppen"].values, [KOPPEN_CODES.index("BSk"), KOPPEN_CODES.index("BSh")])
    steppe &= ~s["ocean"].values.astype(bool)
    cells = np.flatnonzero(steppe)
    if not cells.size:
        return 0.0, 0.0
    _, label = connected_components(grid.neighbours.tocsr()[cells][:, cells], directed=False)
    belt = cells[label == np.argmax(np.bincount(label))]
    radius_km = float(s.attrs["radius_m"]) / 1e3
    p = grid.points[belt]
    span = 0.0
    for start in range(0, len(p), 2000):
        span = max(span, float(np.arccos(np.clip(p[start:start + 2000] @ p.T, -1.0, 1.0)).max() * radius_km))
    return belt.size * 4 * np.pi * radius_km**2 / grid.size / 1e6, span


def steppe_empire(world, final, first_water_wind: float, belt_km: float, land_km2: float) -> float:
    """Return the peak area (share of the land) of the largest state with its capital on the steppe founded after
    the world's first water_wind, on worlds whose steppe belt spans STEPPE_BELT_KM (NaN elsewhere)."""
    from worldgen.culture.history.polities import steppe_cells

    if belt_km < STEPPE_BELT_KM or not np.isfinite(first_water_wind):
        return float("nan")
    steppe = steppe_cells(world)
    peaks = [p.peak_km2 for p in final.polities if steppe[p.capital] and p.founded_years_ago <= first_water_wind]
    return float(max(peaks, default=0.0) / land_km2)


def colonists_on_struck(world, result, final) -> float:
    """Return the colonists' share of the people now on the landmasses a contact epidemic struck (NaN if none)."""
    from worldgen.culture.history.population import landmasses

    struck = {e.refs["landmass"] for e in result.chronicle.entries if e.kind == "epidemic" and "landmass" in e.refs}
    if not struck:
        return float("nan")
    mass = landmasses(world)
    on = np.isin(mass, list(struck))
    colonist = np.array([c.inhabitants == "colonist" for c in result.cultures]) if hasattr(result, "cultures") else None
    if colonist is None or colonist.size != final.population.shape[0]:
        return float("nan")
    total = final.population[:, on].sum()
    return float(final.population[colonist][:, on].sum() / total) if total > 0 else float("nan")


def leading_level(snap) -> int:
    """Return the highest energy level in force in any occupied cell."""
    from worldgen.culture.history.state import in_force

    occupied = snap.population > 0
    return int(in_force(snap.development[:, 0][occupied]).max()) if occupied.any() else 0


def war_exponent(conflicts) -> float:
    """Return the maximum-likelihood power-law exponent of conflict severity above its median (Clauset et al. 2009;
    Clauset 2018 fits the upper 54% of wars)."""
    x = np.array([c.severity for c in conflicts if c.severity > 0])
    if x.size < 20:
        return float("nan")
    xmin = np.median(x)
    tail = x[x >= xmin]
    return float(1.0 + tail.size / np.log(tail / xmin).sum())


def measure_world(args) -> dict:
    """Run the native history on one world and return its measures."""
    name, make, colony, seed = args
    from worldgen import load_world

    world = make() if callable(make) else load_world(make)
    t = time.perf_counter()
    result, cov, suit = _history(world, {"cultures": [{"mode": "organic"}]}, seed)
    elapsed = time.perf_counter() - t
    snaps = {s.years_ago: s for s in result.snapshots}
    cell_km2 = float(np.mean(cov.cell_area_km2))
    land = cov.land.astype(bool)
    land_km2 = float(land.sum() * cell_km2)
    ice_free = land & (world.surface["ice_thickness"].values <= 0)
    entries = result.chronicle.entries
    firsts = {(e.kind, e.refs.get("level")): e.years_ago for e in entries if e.kind == "first_energy"}
    out = {"name": name, "seconds": elapsed}
    out["hyde"] = [snaps[y].population.sum() / p for y, p in HYDE]
    out["origins"] = sum(e.kind == "farming_origin" for e in entries)
    origins = [e for e in entries if e.kind == "farming_origin"]
    out["herd_origins"] = float(np.mean([e.refs.get("livestock", 0) for e in origins])) if origins else np.nan
    out["contact_struck"] = float(sum(e.refs.get("people", 0) for e in entries if e.kind == "epidemic"))
    out["colonists_on_struck"] = colonists_on_struck(world, result, snaps[0.0])
    out["front_km_yr"] = front_speed(world, result, cov)
    out["arable"] = float(np.mean(suit.arable[ice_free] > 0))
    out["first_water_wind"] = firsts.get(("first_energy", 2), np.nan)
    out["first_fossil"] = firsts.get(("first_energy", 3), np.nan)
    out["industrial_now"] = float(np.isfinite(out["first_fossil"]))
    largest, effective = zip(*[state_measures(snaps[y], cell_km2, land_km2) for y, _, _ in TAAGEPERA])
    out["largest"] = [v if leading_level(snaps[y]) >= TAAGEPERA_LEVEL[y]
                      and (y not in TAAGEPERA_NEEDS_STATE or (snaps[y].owner >= 0).any()) else np.nan
                      for v, (y, _, _) in zip(largest, TAAGEPERA)]
    out["effective"] = list(effective) + [state_measures(snaps[0.0], cell_km2, land_km2)[1]]
    final = snaps[0.0]
    out["steppe_belt_km"] = steppe_belt(world)[1]
    out["steppe_empire"] = steppe_empire(world, final, out["first_water_wind"], out["steppe_belt_km"], land_km2)
    big = [p for p in final.polities if p.peak_km2 >= 1e6 and not p.alive]
    out["lifetime_1e6"] = float(np.mean([p.founded_years_ago - p.ended_years_ago for p in big])) if big else np.nan
    out["war_alpha"] = war_exponent(result.conflicts)
    out["wars_per_century"] = 100.0 * len(result.conflicts) / START
    activity = float(world.state.interior.activity_index)
    eruptions = [e for e in result.events if e[0] == "eruption" and e[2] >= TAMBORA_TG]
    out["tambora_per_century"] = 100.0 * len(eruptions) / START / max(activity, 1e-9)
    flu = [e for e in result.events if e[0] == "pandemic" and e[2] >= SPANISH_FLU and 70 <= e[1] <= 415]
    out["flu_1600_1945"] = len(flu)
    out["flu_expected"] = expected_flu(result)
    out["farmers_1800"] = float(snaps[215.0].population.sum())
    if colony:
        out.update(colony_measures(world, seed))
    return out


def _landing(world) -> tuple[float, float]:
    """Return a temperate landing site on the second-largest landmass (an Americas for the colony)."""
    from worldgen.culture.history.population import landmasses

    mass = landmasses(world)
    sizes = np.bincount(mass[mass >= 0])
    second = int(np.argsort(sizes)[-2]) if sizes.size > 1 else int(np.argmax(sizes))
    lat = np.asarray(world.grid.lat)
    temp = world.surface["air_temperature"].values
    temp = temp.mean(axis=0) if temp.ndim > 1 else temp
    temp = temp - 273.15 if np.nanmean(temp) > 150 else temp
    cand = np.flatnonzero((mass == second) & (temp > 5) & (temp < 18))
    if not cand.size:
        cand = np.flatnonzero(mass == second)
    cell = int(cand[np.argmin(np.abs(np.abs(lat[cand]) - 40.0))])
    return float(lat[cell]), float(np.asarray(world.grid.lon)[cell])


def colony_measures(world, seed: int) -> dict:
    """Run the native with an industrial colony landing COLONY_YEARS ago and return the contact measures."""
    from worldgen.culture.history.population import landmasses

    lat, lon = _landing(world)
    spec = {"cultures": [{"mode": "organic"}, {
        "mode": "organic", "inhabitants": "colonist",
        "development": {"energy": "fossil", "transport": "rail_and_steam", "communication": "telegraph"},
        "colony": {"founded_years_ago": COLONY_YEARS, "landing_sites": [{"lat_deg": lat, "lon_deg": lon}],
                   "crowd_diseases": True}}]}
    result, cov, _ = _history_with_epochs(world, spec, [COLONY_YEARS, COLONY_YEARS - 100.0], seed)
    snaps = {s.years_ago: s for s in result.snapshots}
    at, after, now = snaps[COLONY_YEARS], snaps[COLONY_YEARS - 100.0], snaps[0.0]
    mass = landmasses(world)
    points = world.grid.points
    site = int(np.argmax(points @ _unit(lat, lon)))
    near_site = (mass == mass[site]) & (cov.radius_km * np.arccos(np.clip(points @ points[site], -1, 1)) < 1000.0)
    native_at = at.population[0][near_site].sum()
    on_mass = mass == mass[site]
    colonists = now.population[1][on_mass].sum()                   # the given colony, not the natives' own colonies
    independence = [e.years_ago for e in result.chronicle.entries
                    if e.kind == "independence" and e.refs.get("culture") == 1]
    return {"native_kept_100y": after.population[0][near_site].sum() / native_at if native_at > 0 else np.nan,
            "colonist_share_now": float(colonists / max(now.population[:, on_mass].sum(), 1e-9)),
            "colony_people_now": float(now.population[1].sum()),
            "independence_after": COLONY_YEARS - independence[0] if independence else np.nan}


def _unit(lat: float, lon: float) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _history_with_epochs(world, cultures_spec: dict, epochs: list[float], seed: int):
    """Run an organic history keeping the given epochs."""
    global EPOCHS
    saved, EPOCHS = EPOCHS, sorted(set(EPOCHS) | set(epochs), reverse=True)
    try:
        return _history(world, cultures_spec, seed)
    finally:
        EPOCHS = saved


def _summary(values, dates: bool = False) -> tuple[float, str, str]:
    """Return the aggregate (mean, or median for dates) and its text with the range."""
    v = np.array(values, dtype=float)
    ok = v[np.isfinite(v)]
    if not ok.size:
        return np.nan, "-", ""
    agg = float(np.median(ok)) if dates else float(ok.mean())
    return agg, f"{len(ok)}/{v.size}", f"{ok.min():.3g}–{ok.max():.3g}"


def rows(measures: list[dict]) -> list:
    """Return the table rows (label, Earth, model, verdict) from the per-world measures."""
    def row(label, earth, values, fmt, test, kind="check", dates=False, earth_text=None, median=False):
        agg, count, spread = _summary(values, dates or median)
        model = "-" if not np.isfinite(agg) else f"{fmt.format(agg)} ({spread})"
        if median and np.isfinite(agg):
            model = f"{fmt.format(agg)} (mean {fmt.format(_summary(values)[0])}; {spread})"
        if "/" in count and count.split("/")[0] != count.split("/")[1]:
            model += f" [{count}]"
        e = earth_text if earth_text is not None else ("" if earth is None else fmt.format(earth))
        return (label, e, model, verdict(test, kind, agg, earth) if np.isfinite(agg) else "")

    out = []
    for k, label in enumerate(HYDE_LABELS):
        out.append(row(f"population / HYDE, {label}", 1.0, [m["hyde"][k] for m in measures], "{:.2f}", factor(2.0),
                       "gap"))
    out += [
        row("farming origins", 11, [m["origins"] for m in measures], "{:.1f}", between(11, 24),
            earth_text=">= 11 (Larson)"),
        row("farming front (km/yr)", 0.95, [m["front_km_yr"] for m in measures], "{:.2f}", between(0.6, 1.3),
            "gap", earth_text="0.6–1.3 (Pinhasi)"),
        row("arable share of ice-free land", 0.61, [m["arable"] for m in measures], "{:.2f}", near(0.05), "gap"),
        row("first water_wind (years ago)", None, [m["first_water_wind"] for m in measures], "{:.0f}", None,
            "report", dates=True),
        row("first fossil (years ago)", 215, [m["first_fossil"] for m in measures], "{:.0f}", between(100, 400),
            "gap", dates=True),
        row("worlds industrial at present", 1.0, [m["industrial_now"] for m in measures], "{:.2f}", between(0.99, 1), "gap"),
    ]
    for k, (_, mm2, label) in enumerate(TAAGEPERA):
        out.append(row(f"largest state / land, {label}", mm2 / EARTH_LAND_MM2, [m["largest"][k] for m in measures],
                       "{:.3f}", factor(2.0), "gap", median=True))
    out += [
        row("steppe belt span (km)", 8000, [m["steppe_belt_km"] for m in measures], "{:.0f}", None, "report",
            earth_text="~8000 (Britannica)"),
        row(f"largest steppe empire / land (belt >= {STEPPE_BELT_KM:.0f} km)", MONGOL_MM2 / EARTH_LAND_MM2,
            [m["steppe_empire"] for m in measures], "{:.3f}", factor(2.0), "gap", median=True,
            earth_text=f"{MONGOL_MM2 / EARTH_LAND_MM2:.3f} (Mongols)"),
        row("farming origins with herd animals (share)", None, [m["herd_origins"] for m in measures], "{:.2f}",
            None, "report"),
        row("people struck by contact epidemics (natives' own colonists)", None,
            [m["contact_struck"] for m in measures], "{:.3g}", None, "report"),
        row("colonists' share of struck landmasses now", None, [m["colonists_on_struck"] for m in measures], "{:.2f}",
            None, "report"),
        row("effective number of states, 1 CE", None, [m["effective"][2] for m in measures], "{:.0f}", None,
            "report"),
        row("effective number of states, 1300", 30, [m["effective"][3] for m in measures], "{:.0f}", factor(2.0)),
        row("effective number of states, present", 21, [m["effective"][-1] for m in measures], "{:.0f}",
            factor(2.0), earth_text="19–24 (1990–95)"),
        row("lifetime of states >= 1 Mm2 (yr)", 220, [m["lifetime_1e6"] for m in measures], "{:.0f}", factor(1.5)),
        row("war-size exponent (upper half)", 1.53, [m["war_alpha"] for m in measures], "{:.2f}", near(0.3)),
        row("conflicts per century", None, [m["wars_per_century"] for m in measures], "{:.1f}", None, "report"),
        row("eruptions >= Tambora per century", 0.25, [m["tambora_per_century"] for m in measures], "{:.3f}",
            between(0.2, 0.3)),
        row("Spanish-flu pandemics, 1600–1945", None, [m["flu_1600_1945"] for m in measures], "{:.2f}", None,
            "report", earth_text="1.4 at Marani's mean rate"),
        row("  expected at the model's rate for its farmers", None, [m["flu_expected"] for m in measures], "{:.2f}",
            None, "report"),
        row("history time (s)", None, [m["seconds"] for m in measures], "{:.0f}", None, "report"),
    ]
    colonies = [m for m in measures if "native_kept_100y" in m]
    if colonies:
        out += [
            row("natives kept 100 yr after contact", 0.10, [m["native_kept_100y"] for m in colonies], "{:.2f}",
                None, "report", earth_text="0.10 (Koch)"),
            row("the colony's share of its landmass now", None, [m["colonist_share_now"] for m in colonies],
                "{:.3f}", None, "report"),
            row("the colony's people now", None, [m["colony_people_now"] for m in colonies], "{:.3g}", None,
                "report"),
            row("independence (years after landing)", None, [m["independence_after"] for m in colonies], "{:.0f}",
                None, "report", dates=True),
        ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--resolution", default="standard")
    parser.add_argument("--worlds", nargs="*", default=None)
    parser.add_argument("--colony-worlds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.worlds:
        jobs = [(Path(p).name, p) for p in args.worlds]
    else:
        from worldgen.spec import load_spec

        spec = load_spec(EARTH_SPEC)
        jobs = [(f"seed {i}", _Generate(spec.model_copy(update={"seed": i}), args.resolution))
                for i in range(args.first_seed, args.first_seed + args.seeds)]
    jobs = [(name, make, k < args.colony_worlds, k) for k, (name, make) in enumerate(jobs)]   # history seed k
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        measures = []
        for m in pool.map(measure_world, jobs):
            print(f"{m['name']} done ({m['seconds']:.0f} s history)", flush=True)
            measures.append(m)
    table = rows(measures)
    print_rows(f"Organic histories on the Earth example, {len(measures)} worlds ({args.colony_worlds} with a colony): "
               "mean (range), medians for dates", table)
    verdicts = [r[3] for r in table]
    failed = verdicts.count("FAIL")
    print(f"\n{verdicts.count('pass')} passed, {failed} failed, {verdicts.count('gap')} known gaps"
          f"{', ' + str(verdicts.count('ok')) + ' gaps now met' if 'ok' in verdicts else ''}"
          f" in {time.perf_counter() - start:.0f} s")
    sys.exit(1 if failed else 0)


class _Generate:
    """A picklable world maker for the worker processes."""

    def __init__(self, spec, resolution):
        self.spec, self.resolution = spec, resolution

    def __call__(self):
        from worldgen.generate import generate_world

        return generate_world(self.spec, self.resolution)


if __name__ == "__main__":
    main()
