"""Render the example figures kept in "Claude outputs" for the Earth example.

Usage: python scripts/render_examples.py [--out "Claude outputs"] [--seed 0] [--snapshots 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from worldgen.generate import generate_world  # noqa: E402
from worldgen.render import (animate_history, plot_climate, plot_history, plot_map, plot_overview,  # noqa: E402
                             save_figure)
from worldgen.spec import load_spec  # noqa: E402

from validate_earth import EARTH_SPEC  # noqa: E402


def main() -> None:
    """Generate the Earth example once and write its overview, maps and history figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "Claude outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshots", type=float, default=20.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = load_spec(EARTH_SPEC).model_copy(update={"seed": args.seed})
    world = generate_world(spec, "standard", snapshot_interval_myr=args.snapshots)
    save_figure(plot_overview(world), args.out / "earth_rivers_overview.png")
    save_figure(plot_climate(world), args.out / "earth_climate.png")
    for field in ("rainfall", "basins", "biomes", "koppen"):
        save_figure(plot_map(world, field, "robinson").figure, args.out / f"earth_{field}.png")
    save_figure(plot_map(world, "orogeny_age").figure, args.out / "earth_orogeny_age.png")
    globe = plot_map(world, "elevation", "orthographic", central_longitude=30, central_latitude=20)
    save_figure(globe.figure, args.out / "earth_rivers_globe.png")
    save_figure(plot_history(world), args.out / "earth_tectonic_history.png")
    animate_history(world, args.out / "earth_tectonic_history.gif")
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
