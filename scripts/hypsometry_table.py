"""Measure the reference hypsometries used to estimate land fraction from ocean volume (``worldgen/water.py``).

Usage:
  python scripts/hypsometry_table.py plates [--depths 6000 5000 ...] [--seeds 3] [--resolution standard]
  python scripts/hypsometry_table.py regimes [--seeds 3]

``plates``: generates the Earth example with the total water that gives each
ocean depth (ocean volume per unit planet area, m) and reports the land
fraction reached. The continental crust area itself depends on the current
table, so rerun after changing it until the values settle.

``regimes``: builds each non-plate surface at Earth gravity and reports the
ocean depth needed for a range of land fractions.
"""

from __future__ import annotations

import argparse

import numpy as np

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.generate import generate_world
from worldgen.grid import build_grid
from worldgen.spec import load_spec
from worldgen.surface import regimes
from worldgen.surface.fields import SurfaceFields
from worldgen.surface.sealevel import ocean_fill
from worldgen.water import total_water_for_surface

from validate_earth import EARTH_SPEC

DEFAULT_DEPTHS = [6000, 5000, 4300, 3800, 3400, 3000, 2600, 2200, 1800, 1400, 1000, 700, 400, 150]
LID_LAND = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def plate_table(depths: list[float], seeds: int, resolution: str) -> None:
    """Print land fraction against ocean depth for simulated Earth-like planets."""
    spec = load_spec(EARTH_SPEC)
    area = 4 * np.pi * c.R_EARTH**2
    print("ocean depth (m)  total water  land fraction per seed  mean")
    for depth in depths:
        surface = depth * area * h.SEAWATER_DENSITY / c.M_EARTH
        total = total_water_for_surface(surface, 1.0, True)
        lands = []
        for seed in range(seeds):
            body = spec.body.model_copy(update={"water_mass_fraction": total})
            world = generate_world(spec.model_copy(update={"seed": seed, "body": body}), resolution)
            lands.append(world.state.surface.land_fraction)
        print(f"{depth:>15.0f}  {total:>11.3g}  {np.round(lands, 3)}  {np.mean(lands):.3f}", flush=True)


def regime_tables(seeds: int) -> None:
    """Print ocean depth against land fraction for each non-plate surface generator."""
    grid = build_grid(40_000)
    builders = {
        "stagnant_lid": lambda f, s: regimes.build_stagnant_lid(f, 0.5, 1.0, 4.5, h.CRATER_WATER_EROSION, s),
        "episodic": lambda f, s: regimes.build_episodic(f, 1.0, 1.0, h.CRATER_WATER_EROSION, s),
        "heat_pipe": lambda f, s: regimes.build_heat_pipe(f, 1.0, s),
        "inactive": lambda f, s: regimes.build_inactive(f, 1.0, 4.5, h.CRATER_WATER_EROSION, s),
    }
    print("land fractions:", LID_LAND)
    for name, build in builders.items():
        depths = []
        for seed in range(seeds):
            fields = SurfaceFields(grid=grid, radius_m=c.R_EARTH)
            build(fields, seed)
            fill = ocean_fill(grid, fields.elevation, c.R_EARTH)
            depths.append([fill.volume(fill.level_for_land_fraction(land)) / (4 * np.pi * c.R_EARTH**2)
                           for land in LID_LAND])
        print(f"{name}: {np.round(np.mean(depths, axis=0)).tolist()}")


def main() -> None:
    """Run the requested measurement."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["plates", "regimes"])
    parser.add_argument("--depths", type=float, nargs="*", default=DEFAULT_DEPTHS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--resolution", default="standard")
    args = parser.parse_args()
    if args.kind == "plates":
        plate_table(args.depths, args.seeds, args.resolution)
    else:
        regime_tables(args.seeds)


if __name__ == "__main__":
    main()
