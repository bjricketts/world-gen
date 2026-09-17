"""Compare generated Earth-example surfaces with Earth over several seeds.

Usage: python scripts/validate_earth.py [--seeds 6] [--resolution standard] [--tectonics simulated]
       [--set NAME=VALUE ...]

``--set`` overrides a constant in ``worldgen.heuristics`` for this run, for calibration sweeps.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

from worldgen import heuristics as h
from worldgen.biosphere.biomes import KOPPEN_CODES
from worldgen.generate import generate_world
from worldgen.spec import load_spec

EARTH_SPEC = Path(__file__).resolve().parent.parent / "examples" / "earth.yaml"

# Name, reference value on Earth (or None), and how to print it. Köppen main-class shares of
# land (Antarctica included) are from Peel et al. (2007). Antarctica alone holds 58 m of sea-level
# equivalent (Fretwell et al. 2013). The ocean carries 22% of the poleward transport at 35° N
# (Trenberth & Caron 2001).
COLUMNS = [
    ("land fraction", 0.29, "{:.2f}"),
    ("continental crust fraction", 0.41, "{:.2f}"),
    ("mean land elevation (m)", 840, "{:.0f}"),
    ("median land elevation (m)", None, "{:.0f}"),
    ("land above 2 km", 0.11, "{:.2f}"),
    ("mean ocean depth (m)", 3700, "{:.0f}"),
    ("highest point (km)", 8.8, "{:.1f}"),
    ("shelf 0 to -200 m", 0.07, "{:.3f}"),
    ("median sea-floor age (Myr)", 60, "{:.0f}"),
    ("plates", 15, "{:.0f}"),
    ("land under lakes", 0.03, "{:.3f}"),
    ("land draining inland", 0.18, "{:.2f}"),
    ("largest basin (million km2)", 6.9, "{:.1f}"),
    ("largest river (1000 m3/s)", 209, "{:.0f}"),
    ("outflow (1000 km3/yr)", 40, "{:.1f}"),
    ("ocean precipitation (m/yr)", 1.1, "{:.2f}"),
    ("land precipitation (m/yr)", 0.75, "{:.2f}"),
    ("land runoff (m/yr)", 0.27, "{:.2f}"),
    ("land temperature (K)", 282, "{:.0f}"),
    ("tropical temperature (K)", 299, "{:.0f}"),
    ("ocean share of transport, 35N", 0.22, "{:.2f}"),
    ("polar temperature, >70° (K)", 246, "{:.0f}"),
    ("  north of 70N (K)", None, "{:.0f}"),
    ("  south of 70S (K)", None, "{:.0f}"),
    ("  >70°, reduced to sea level (K)", None, "{:.0f}"),
    ("annual range, land 50-70N (K)", 35, "{:.0f}"),
    ("sea ice, share of ocean", 0.07, "{:.2f}"),
    ("land under ice sheets", 0.10, "{:.3f}"),
    ("  ice, sea-level equivalent (m)", None, "{:.0f}"),
    ("  land share poleward of 70°", None, "{:.3f}"),
    ("  ice cover of land >60°", None, "{:.2f}"),
    ("Koppen A, share of land", 0.190, "{:.2f}"),
    ("Koppen B", 0.302, "{:.2f}"),
    ("Koppen C", 0.134, "{:.2f}"),
    ("Koppen D", 0.246, "{:.2f}"),
    ("Koppen E", 0.128, "{:.2f}"),
    ("  Df", None, "{:.2f}"),
    ("  Dw", None, "{:.2f}"),
    ("vegetated share of land", None, "{:.2f}"),
    ("coupling passes", None, "{:.0f}"),
    ("run time (s)", None, "{:.1f}"),
]


def _ocean_share(features: dict) -> float:
    """Return the ocean's share of the poleward heat transport at 35° N."""
    ocean = features.get("ocean_heat_transport_35n_pw", np.nan)
    return ocean / (ocean + features.get("atmosphere_heat_transport_35n_pw", np.nan))


def surface_statistics(world, seconds: float) -> list[float]:
    """Return the COLUMNS statistics for one generated world."""
    s = world.surface
    e = s.elevation.values.astype(float)
    ocean = s.ocean.values
    land = ~ocean
    lat = s.lat.values
    f = world.state.surface.features
    polar = np.abs(lat) > 70
    # Temperature reduced to sea level over the land and ice surface.
    top = np.where(ocean, 0.0, np.maximum(e, 0.0) + s.ice_surface.values)
    sea_level_k = s.air_temperature.values + h.LAPSE_RATE_K_PER_M * top
    codes = [KOPPEN_CODES[i] for i in s.koppen.values[land]]
    code = np.array([c[:1] for c in codes])
    sub = np.array([c[:2] for c in codes])
    return [
        land.mean(),
        f.get("continental_crust_fraction", np.nan),
        e[land].mean(),
        np.median(e[land]),
        np.mean(e[land] > 2000),
        -e[ocean].mean(),
        e.max() / 1e3,
        np.mean((e < 0) & (e > -200)),
        f.get("median_ocean_age_myr", np.nan),
        f.get("plates", np.nan),
        f.get("lake_fraction_of_land", np.nan),
        f.get("endorheic_fraction_of_land", np.nan),
        f.get("largest_basin_km2", np.nan) / 1e6,
        f.get("largest_river_m3_s", np.nan) / 1e3,
        f.get("river_outflow_km3_yr", np.nan) / 1e3,
        s.precipitation.values[ocean].mean(),
        s.precipitation.values[land].mean(),
        s.runoff.values[land].mean(),
        s.air_temperature.values[land].mean(),
        s.air_temperature.values[np.abs(lat) < 10].mean(),
        _ocean_share(f),
        s.air_temperature.values[polar].mean(),
        s.air_temperature.values[lat > 70].mean(),
        s.air_temperature.values[lat < -70].mean(),
        sea_level_k[polar].mean(),
        np.ptp(s.monthly_temperature.values, axis=0)[land & (lat > 50) & (lat < 70)].mean(),
        s.sea_ice.values[ocean].mean(),
        (s.ice_thickness.values[land] > 0).mean(),
        f.get("ice_sea_level_equivalent_m", 0.0),
        (land & polar).sum() / land.sum(),
        (s.ice_thickness.values[land & (np.abs(lat) > 60)] > 0).mean() if (land & (np.abs(lat) > 60)).any() else np.nan,
        *(np.mean(code == k) for k in "ABCDE"),
        np.mean(sub == "Df"),
        np.mean(sub == "Dw"),
        s.vegetation_cover.values[land].mean(),
        f.get("climate_coupling_passes", np.nan),
        seconds,
    ]


def main() -> None:
    """Generate the Earth example for several seeds and print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--resolution", default="standard")
    parser.add_argument("--tectonics", choices=["simulated", "heuristic"], default="simulated")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # empty or all-NaN columns
    for item in args.set:
        name, value = item.split("=")
        setattr(h, name, float(value))

    spec = load_spec(EARTH_SPEC)
    spec = spec.model_copy(update={"surface": spec.surface.model_copy(update={"tectonics": args.tectonics})})
    rows = []
    for seed in range(args.seeds):
        start = time.time()
        world = generate_world(spec.model_copy(update={"seed": seed}), args.resolution)
        rows.append(surface_statistics(world, time.time() - start))
    table = np.array(rows, dtype=float)

    print(f"{'quantity':<30}{'min':>9}{'mean':>9}{'max':>9}{'Earth':>9}   per seed")
    for (name, earth, fmt), column in zip(COLUMNS, table.T):
        cells = [fmt.format(v) for v in (np.nanmin(column), np.nanmean(column), np.nanmax(column))]
        ref = "" if earth is None else fmt.format(earth)
        seeds = " ".join(fmt.format(v) for v in column)
        print(f"{name:<30}{cells[0]:>9}{cells[1]:>9}{cells[2]:>9}{ref:>9}   {seeds}")


if __name__ == "__main__":
    main()
