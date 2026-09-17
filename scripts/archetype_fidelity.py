"""Summarise archetype outcomes over many draws (snapshot mode, no surfaces).

Usage: python scripts/archetype_fidelity.py [--draws 300]

Prints a markdown table for DESIGN.md section 5.2: runaway and locked shares,
surface water phase, temperature range, the land fraction estimated before the
surface is built, and redraw attempts.
"""

from __future__ import annotations

import argparse

import numpy as np

from worldgen import PlanetSpec
from worldgen.evolve.snapshot import SnapshotEvolver
from worldgen.priors import ARCHETYPES


def summarise(name: str, draws: int) -> str:
    """Return one table row for an archetype."""
    evolver = SnapshotEvolver()
    states = [evolver.evolve(PlanetSpec(seed=s, priors={"archetype": name}))[0] for s in range(draws)]
    runaway = np.mean([s.atmosphere.runaway_greenhouse for s in states])
    locked = np.mean([s.orbit.spin_state == "synchronous" for s in states])
    phases = [s.atmosphere.surface_water for s in states]
    water = " / ".join(str(round(100 * np.mean([p == k for p in phases]))) for k in ("liquid", "ice", "none"))
    t5, t95 = np.percentile([s.atmosphere.surface_temperature_k for s in states], [5, 95])
    land = np.array([s.water.land_fraction_estimate if s.water and s.atmosphere.surface_water in ("liquid", "ice")
                     else np.nan for s in states])
    if np.isfinite(land).any():
        lo, mid, hi = np.nanpercentile(land, [5, 50, 95])
        land_text = f"{lo:.0%}–{hi:.0%} ({mid:.0%})"
    else:
        land_text = "–"
    attempts = np.mean([s.attempts for s in states])
    met = np.mean([s.constraints_met for s in states])
    open_ocean = [s.climate.open_ocean_fraction for s in states
                  if s.climate is not None and s.atmosphere.surface_water in ("liquid", "ice")]
    open_text = f"{np.median(open_ocean):.0%}" if open_ocean else "–"
    lives = [s.biosphere.life if s.biosphere is not None else "none" for s in states]
    life_text = " / ".join(str(round(100 * np.mean([x == k for x in lives]))) for k in ("surface", "ocean", "subsurface"))
    return (f"| {name} | {runaway:.0%} | {locked:.0%} | {water} | {t5:.0f}–{t95:.0f} K | {open_text} | {land_text} | "
            f"{life_text} | {attempts:.1f} | {met:.0%} |")


def main() -> None:
    """Print the archetype fidelity table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=300)
    args = parser.parse_args()
    print("| Archetype | Runaway | Locked | Water (liquid / ice / none) | T_s 5–95% | Open ocean (median) | Land | "
          "Life (surface / ocean / subsurface) | Mean attempts | Met |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in ARCHETYPES:
        print(summarise(name, args.draws), flush=True)


if __name__ == "__main__":
    main()
