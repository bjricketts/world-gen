"""Rank the poorly constrained history parameters by how much they move the outcome.

Usage: python scripts/history_sensitivity.py [--planets earth,mars,venus]
       [--quantiles 0.1,0.5,0.9] [--parameters NAME,NAME] [--csv out.csv]

Each parameter is set in turn to the given quantiles of its prior while the
others keep the example spec's values, and the planet is integrated from
formation. The spread of each outcome over those runs measures the parameter's
influence; discrete outcomes (dynamo, tectonic regime, climate branch, life)
are reported as flips. One parameter at a time is cheap and reads directly, but
it cannot see interactions between parameters.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from worldgen import constants as c
from worldgen.evolve import evolve
from worldgen.priors.archetypes import BASE_PRIORS
from worldgen.priors.distributions import LogUniform
from worldgen.spec import PlanetSpec, load_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SPECS = {"earth": "earth_history.yaml", "mars": "mars_history.yaml", "venus": "venus_history.yaml"}

# Inventories have no prior of their own (they default to Earth's), so the sweep gives them the
# same spread as the volatile richness prior; the water fraction follows the body prior.
EXTRA_PRIORS = {
    "history.initial_water_mass_fraction": BASE_PRIORS["body.water_mass_fraction"],
    "history.carbon_inventory": LogUniform(0.3, 3.0),
    "history.nitrogen_inventory": LogUniform(0.3, 3.0),
}
PARAMETERS = ([name for name in BASE_PRIORS if name.startswith("history.")]
              + list(EXTRA_PRIORS) + ["body.radiogenic_abundance"])


def prior_of(name: str):
    """Return the distribution a parameter is drawn from."""
    return EXTRA_PRIORS.get(name) or BASE_PRIORS[name]


@dataclass
class Outcome:
    """One number read off a finished planet, with the spread that counts as a large move."""

    label: str
    read: Callable
    scale: float                 # a spread this large is a strong influence
    logarithmic: bool = False
    floor: float = 0.0           # values below this are treated as nothing left, so the spread stays meaningful
    form: str = "{:8.2f}"

    def value(self, state) -> float:
        """Return the outcome for one run, in the units it is compared in."""
        raw = float(self.read(state))
        return math.log10(max(raw, self.floor)) if self.logarithmic else raw


OUTCOMES = [
    Outcome("T [K]", lambda s: s.atmosphere.surface_temperature_k, 10.0),
    Outcome("P [dex]", lambda s: s.atmosphere.surface_pressure_pa / c.BAR, 0.3, logarithmic=True, floor=1e-6),
    Outcome("CO₂ [dex]", lambda s: s.inputs["history.co2_bar"], 0.3, logarithmic=True, floor=1e-9),
    Outcome("O₂", lambda s: s.biosphere.oxygen_fraction if s.biosphere else 0.0, 0.05, form="{:8.3f}"),
    Outcome("water [dex]", lambda s: s.water.surface_mass_kg / c.EARTH_OCEAN_MASS, 0.3, logarithmic=True,
            floor=1e-4),
]
FLAGS = {
    "dynamo": lambda s: s.interior.magnetic_field,
    "regime": lambda s: s.interior.tectonic_regime,
    "climate": lambda s: s.inputs.get("history.climate_mode"),
    "life": lambda s: s.biosphere.life if s.biosphere else "none",
}


@dataclass
class Row:
    """The influence of one parameter on one planet."""

    planet: str
    parameter: str
    values: list[float]
    spreads: dict[str, float] = field(default_factory=dict)
    flips: list[str] = field(default_factory=list)
    failed: int = 0

    @property
    def influence(self) -> float:
        """Return the largest spread relative to what counts as a large move for that outcome."""
        return max((self.spreads[o.label] / o.scale for o in OUTCOMES if o.label in self.spreads), default=0.0)


def with_parameter(spec: PlanetSpec, name: str, value) -> PlanetSpec:
    """Return the spec with one dotted parameter set."""
    section, field_name = name.split(".", 1)
    return spec.model_copy(update={section: getattr(spec, section).model_copy(update={field_name: value})})


def sweep(planet: str, parameters: list[str], quantiles: list[float], verbose: bool = True) -> list[Row]:
    """Return the influence of each parameter on one planet."""
    spec = load_spec(EXAMPLES / SPECS[planet])
    rows = []
    for name in parameters:
        prior = prior_of(name)
        settings = [prior.quantile(q) for q in quantiles]
        states = []
        row = Row(planet=planet, parameter=name, values=settings)
        start = time.time()
        for value in settings:
            try:
                state, _ = evolve(with_parameter(spec, name, value))
                states.append(state)
            except Exception as error:                       # a parameter can push a run out of the model's range
                row.failed += 1
                if verbose:
                    print(f"    {name}={value:.3g} failed: {error}")
        if len(states) > 1:
            for outcome in OUTCOMES:
                readings = [outcome.value(s) for s in states]
                row.spreads[outcome.label] = max(readings) - min(readings)
            row.flips = [flag for flag, read in FLAGS.items() if len({read(s) for s in states}) > 1]
        rows.append(row)
        if verbose:
            print(f"    {name:<42}{time.time() - start:5.1f} s   influence {row.influence:5.2f}", flush=True)
    return sorted(rows, key=lambda r: r.influence, reverse=True)


def print_table(planet: str, rows: list[Row], quantiles: list[float]) -> None:
    """Print one planet's ranking, strongest first."""
    span = ", ".join(f"{q:g}" for q in quantiles)
    print(f"\n=== {planet}: spread over the {span} quantiles of each prior ===")
    header = "".join(f"{o.label:>12}" for o in OUTCOMES)
    print(f"  {'parameter':<42}{'rank':>6}{header}   flips")
    for row in rows:
        cells = "".join(f"{row.spreads.get(o.label, float('nan')):12.3f}" for o in OUTCOMES)
        note = ", ".join(row.flips) + (f"  ({row.failed} runs failed)" if row.failed else "")
        print(f"  {row.parameter:<42}{row.influence:6.2f}{cells}   {note}")


def print_summary(rows: list[Row], top: int = 3) -> None:
    """Print which parameters move each outcome the most, over all the planets."""
    print("\n=== biggest movers ===")
    for outcome in OUTCOMES:
        ranked = sorted((r for r in rows if outcome.label in r.spreads),
                        key=lambda r: r.spreads[outcome.label], reverse=True)[:top]
        items = ", ".join(f"{r.parameter.split('.', 1)[1]} ({r.planet}, {r.spreads[outcome.label]:.2f})"
                          for r in ranked)
        print(f"  {outcome.label:<12}{items}")
    flipped = [r for r in rows if r.flips]
    print("\n=== parameters that change the planet's state ===")
    for row in sorted(flipped, key=lambda r: len(r.flips), reverse=True):
        print(f"  {row.parameter:<42}{row.planet:<8}{', '.join(row.flips)}")
    if not flipped:
        print("  none: every parameter leaves the dynamo, regime, climate branch and life unchanged")
    print("\nThe ranking is of the planet at the target epoch. A parameter can set when something happens "
          "without\nchanging where the planet ends up: the reductant decay time moves the first oxygenation "
          "by ~0.8 Gyr\nand barely shows here.")


def main() -> None:
    """Sweep each parameter over its prior for each planet and print the rankings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planets", default="earth,mars,venus")
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--parameters", default=None, help="Comma-separated subset of the parameters to sweep.")
    parser.add_argument("--csv", type=Path, default=None, help="Also write the table to this CSV file.")
    args = parser.parse_args()
    quantiles = [float(q) for q in args.quantiles.split(",")]
    parameters = args.parameters.split(",") if args.parameters else PARAMETERS

    everything: list[Row] = []
    for planet in args.planets.split(","):
        print(f"\nsweeping {planet}:")
        rows = sweep(planet.strip(), parameters, quantiles)
        print_table(planet.strip(), rows, quantiles)
        everything += rows
    print_summary(everything)
    if args.csv is not None:
        write_csv(everything, args.csv)
        print(f"\ntable written to {args.csv}")


def write_csv(rows: list[Row], path: Path) -> None:
    """Write the sweep as a CSV file."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["planet", "parameter", "influence"] + [o.label for o in OUTCOMES] + ["flips", "failed"])
        for row in rows:
            writer.writerow([row.planet, row.parameter, f"{row.influence:.4f}"]
                            + [f"{row.spreads.get(o.label, float('nan')):.4f}" for o in OUTCOMES]
                            + [" ".join(row.flips), row.failed])


if __name__ == "__main__":
    main()
