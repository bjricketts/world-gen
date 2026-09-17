"""Generate a few planets and save their reports, maps and figures for viewing.

Usage: python scripts/planet_gallery.py [--out gallery] [--resolution standard] [--seed 1]
       [--archetypes temperate arid ocean ice tidally_locked super_earth] [--spec examples/earth.yaml ...]
       [--snapshots 40]

Each planet gets its own folder with report.txt, the saved world, an overview and climate figure,
every map field that applies to it, two globe views, a polar ice view and, for simulated plate
tectonics, the tectonic history as a panel figure and a GIF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from worldgen import PlanetSpec, format_report, generate_world, save_world  # noqa: E402
from worldgen.render import (FIELDS, animate_history, has_history, plot_climate, plot_history,  # noqa: E402
                             plot_map, plot_overview, save_figure)
from worldgen.spec import load_spec  # noqa: E402


def render_planet(world, folder: Path) -> list[str]:
    """Write every figure that applies to a world and return the file names."""
    written = []

    def keep(fig, name):
        save_figure(fig, folder / name)
        written.append(name)

    keep(plot_overview(world), "overview.png")
    keep(plot_climate(world), "climate.png")
    for field in FIELDS:
        try:
            keep(plot_map(world, field, "robinson").figure, f"map_{field}.png")
        except ValueError:
            pass          # the field does not apply (e.g. rivers on a frozen world)
    for lon, lat in ((0.0, 25.0), (180.0, -25.0)):
        keep(plot_map(world, "elevation", "orthographic", central_longitude=lon, central_latitude=lat).figure,
             f"globe_{lon:.0f}.png")
    for pole in ("north_polar", "south_polar"):
        keep(plot_map(world, "ice", pole).figure, f"ice_{pole}.png")
    if has_history(world):
        keep(plot_history(world), "history.png")
        animate_history(world, folder / "history.gif")
        written.append("history.gif")
    return written


def main() -> None:
    """Generate the requested planets and write their figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("gallery"))
    parser.add_argument("--resolution", default="standard")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--archetypes", nargs="*",
                        default=["temperate", "arid", "ocean", "ice", "tidally_locked", "super_earth"])
    parser.add_argument("--spec", nargs="*", type=Path, default=[], help="Spec files to add")
    parser.add_argument("--snapshots", type=float, default=40.0, help="Tectonic history interval (Myr); 0 for none")
    args = parser.parse_args()

    specs = [(f"{name}_{args.seed}", PlanetSpec(name=f"{name} {args.seed}", seed=args.seed,
                                                 priors={"archetype": name}))
             for name in args.archetypes]
    specs += [(path.stem, load_spec(path)) for path in args.spec]
    for label, spec in specs:
        folder = args.out / label
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{label}: generating ...", flush=True)
        world = generate_world(spec, args.resolution, snapshot_interval_myr=args.snapshots or None)
        (folder / "report.txt").write_text(format_report(world.state), encoding="utf-8")
        save_world(world, folder / "world")
        if world.surface is None:
            print(f"{label}: no solid surface, report only")
            continue
        files = render_planet(world, folder)
        print(f"{label}: {len(files)} figures in {folder}", flush=True)


if __name__ == "__main__":
    main()
