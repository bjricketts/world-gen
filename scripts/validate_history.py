"""Compare history-mode Earth, Mars and Venus with the real planets.

Usage: python scripts/validate_history.py [--planets earth,mars,venus] [--set NAME=VALUE ...]

Each planet is integrated from formation with its example spec and checked
against measured values: present-day conditions, the inventories, the interior
and the dated events (the Great Oxidation at 2.4 Ga, Lyons et al. 2014; Mars's
early dynamo, from the crustal magnetisation of Acuña et al. 1999). Earth is
also compared with the calibrated snapshot Earth of examples/earth.yaml, which
the history should reproduce.

``--set`` overrides a constant in ``worldgen.heuristics`` for this run, for
calibration sweeps. The exit code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from worldgen import constants as c
from worldgen import heuristics as h
from worldgen.evolve import evolve
from worldgen.spec import load_spec
from worldgen.state import PlanetState, Timeline

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@dataclass
class Check:
    """One measured quantity with the value the real planet has."""

    label: str
    measure: Callable[[PlanetState, Timeline], object]
    observed: object
    test: Callable[[object, object], bool]
    form: str = "{}"

    def run(self, state: PlanetState, timeline: Timeline) -> tuple[str, str, bool]:
        """Return the modelled value, the observed one and whether they agree."""
        value = self.measure(state, timeline)
        return self.text(value), self.text(self.observed), bool(self.test(value, self.observed))

    def text(self, value) -> str:
        """Return a value formatted for the table."""
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return self.form.format(value)
        return str(value)


def within(fraction: float) -> Callable:
    """Return a test that the value is within a relative fraction of the observed one."""
    return lambda v, o: o * (1 - fraction) <= v <= o * (1 + fraction)


def near(amount: float) -> Callable:
    """Return a test that the value is within an absolute amount of the observed one."""
    return lambda v, o: abs(v - o) <= amount


def between(low: float, high: float) -> Callable:
    """Return a test that the value lies in a range (the observed value is only printed)."""
    return lambda v, o: low <= v <= high


SAME = (lambda v, o: v == o)

# --- measurements -------------------------------------------------------------------------------


def temperature(state, timeline):
    """Return the global mean surface temperature (K)."""
    return state.atmosphere.surface_temperature_k


def pressure(state, timeline):
    """Return the surface pressure (bar)."""
    return state.atmosphere.surface_pressure_pa / c.BAR


def co2_ppm(state, timeline):
    """Return the CO₂ mixing ratio in parts per million."""
    return state.inputs["history.co2_bar"] / max(pressure(state, timeline), 1e-12) * 1e6


def gas(name: str) -> Callable:
    """Return a measurement of a biosphere gas fraction."""
    return lambda state, timeline: getattr(state.biosphere, name) if state.biosphere else 0.0


def surface_water(state, timeline):
    """Return the surface water in Earth oceans."""
    return state.water.surface_mass_kg / c.EARTH_OCEAN_MASS


def mantle_water(state, timeline):
    """Return the water held in the mantle, in Earth oceans."""
    return state.water.mantle_mass_fraction * state.bulk.mass_kg / c.EARTH_OCEAN_MASS


def heat_flux(state, timeline):
    """Return the surface heat flux in mW/m²."""
    return state.interior.surface_heat_flux_w_m2 * 1e3


def event_age(kind: str, first: bool = True) -> Callable:
    """Return the age at which an event happened (None if it never did)."""
    def age(state, timeline):
        """Return the age in Gyr."""
        times = [e.time_s / c.SECONDS_PER_GYR for e in timeline.events if e.kind == kind]
        return (times[0] if first else times[-1]) if times else None
    return age


def happened(kind: str) -> Callable:
    """Return whether an event of this kind is in the log."""
    return lambda state, timeline: any(e.kind == kind for e in timeline.events)


def before(age_gyr: float) -> Callable:
    """Return a test that an event happened, and before a given age."""
    return lambda v, o: v is not None and v <= age_gyr


# Present-day values: Earth's pre-industrial air (280 ppm CO₂, 0.7 ppm CH₄), its 87 mW/m² mean heat
# flow and its 1–2 ocean mantle inventory; Mars 6.4 mbar and 210 K; Venus 92 bar and 737 K.
PLANETS = {
    "earth": ("earth_history.yaml", [
        Check("surface temperature (K)", temperature, 288.0, near(6.0), "{:.0f}"),
        Check("surface pressure (bar)", pressure, 1.013, within(0.25), "{:.3g}"),
        Check("CO₂ (ppm)", co2_ppm, 280.0, between(100.0, 1500.0), "{:.3g}"),
        Check("O₂ fraction", gas("oxygen_fraction"), 0.209, near(0.04), "{:.3f}"),
        Check("CH₄ (ppm)", lambda s, t: gas("methane_fraction")(s, t) * 1e6, 0.7, between(0.1, 5.0), "{:.2g}"),
        Check("surface water (oceans)", surface_water, 1.0, within(0.3), "{:.2f}"),
        Check("mantle water (oceans)", mantle_water, 1.5, between(0.3, 4.0), "{:.2f}"),
        Check("surface heat flux (mW/m²)", heat_flux, 87.0, within(0.25), "{:.0f}"),
        Check("surface water phase", lambda s, t: s.atmosphere.surface_water, "liquid", SAME),
        Check("tectonic regime", lambda s, t: s.interior.tectonic_regime, "mobile_lid", SAME),
        Check("magnetic field", lambda s, t: s.interior.magnetic_field, True, SAME),
        Check("life", lambda s, t: s.biosphere.life if s.biosphere else "none", "surface", SAME),
        Check("origin of life (Gyr)", event_age("origin_of_life"), 1.0, before(1.5), "{:.2f}"),
        Check("first oxygenation (Gyr)", event_age("oxygenation"), 2.17, between(1.5, 3.2), "{:.2f}"),
    ]),
    "mars": ("mars_history.yaml", [
        Check("surface temperature (K)", temperature, 210.0, near(25.0), "{:.0f}"),
        Check("surface pressure (bar)", pressure, 0.0064, between(1e-4, 0.1), "{:.3g}"),
        Check("surface water phase", lambda s, t: s.atmosphere.surface_water, "ice", SAME),
        Check("tectonic regime", lambda s, t: s.interior.tectonic_regime, "stagnant_lid", SAME),
        Check("magnetic field", lambda s, t: s.interior.magnetic_field, False, SAME),
        Check("dynamo shutdown (Gyr)", event_age("dynamo_shutdown"), 0.5, before(1.5), "{:.2f}"),
        Check("surface heat flux (mW/m²)", heat_flux, 20.0, between(10.0, 35.0), "{:.0f}"),
        Check("life", lambda s, t: s.biosphere.life if s.biosphere else "none", "none", SAME),
    ]),
    "venus": ("venus_history.yaml", [
        Check("surface temperature (K)", temperature, 737.0, near(80.0), "{:.0f}"),
        Check("surface pressure (bar)", pressure, 92.0, between(30.0, 200.0), "{:.3g}"),
        Check("CO₂ (ppm)", co2_ppm, 965000.0, between(5e5, 1e6), "{:.3g}"),
        Check("surface water phase", lambda s, t: s.atmosphere.surface_water, "none", SAME),
        Check("surface water (oceans)", surface_water, 0.0, lambda v, o: v < 0.01, "{:.2g}"),
        Check("O₂ fraction", gas("oxygen_fraction"), 0.0, lambda v, o: v < 0.01, "{:.2g}"),
        Check("magnetic field", lambda s, t: s.interior.magnetic_field, False, SAME),
        Check("runaway greenhouse", happened("runaway_onset"), True, SAME),
        Check("ocean lost", happened("ocean_loss"), True, SAME),
        Check("ocean loss (Gyr)", event_age("ocean_loss"), 0.5, before(1.0), "{:.2f}"),
    ]),
}

# Earth's history should land close to the calibrated snapshot Earth of examples/earth.yaml.
SNAPSHOT_FIELDS = [
    ("surface temperature (K)", temperature, 8.0, "{:.0f}"),
    ("surface pressure (bar)", pressure, 0.3, "{:.3g}"),
    ("O₂ fraction", gas("oxygen_fraction"), 0.05, "{:.3f}"),
    ("land fraction", lambda s, t: s.water.land_fraction_estimate, 0.1, "{:.2f}"),
    ("surface water (oceans)", surface_water, 0.4, "{:.2f}"),
]


def compare_with_snapshot(state: PlanetState) -> list[tuple[str, str, str, bool]]:
    """Return the history and snapshot values of Earth side by side."""
    snapshot, _ = evolve(load_spec(EXAMPLES / "earth.yaml"))
    rows = []
    for label, measure, tolerance, form in SNAPSHOT_FIELDS:
        a, b = measure(state, None), measure(snapshot, None)
        rows.append((label, form.format(a), form.format(b), abs(a - b) <= tolerance))
    return rows


def main() -> None:
    """Integrate the reference planets and print how they compare with the real ones."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planets", default="earth,mars,venus")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    for item in args.set:
        name, value = item.split("=")
        setattr(h, name, float(value))

    failures = 0
    earth: Optional[PlanetState] = None
    for name in args.planets.split(","):
        spec_name, checks = PLANETS[name.strip()]
        start = time.time()
        state, timeline = evolve(load_spec(EXAMPLES / spec_name))
        seconds = time.time() - start
        if name.strip() == "earth":
            earth = state
        print(f"\n=== {state.name} ({seconds:.1f} s, {len(timeline.events)} events) ===")
        print(f"  {'quantity':<28}{'model':>13}{'observed':>14}")
        for check in checks:
            value, observed, ok = check.run(state, timeline)
            failures += not ok
            print(f"  {check.label:<28}{value:>13}{observed:>14}   {'ok' if ok else 'MISS'}")

    if earth is not None:
        print("\n=== Earth: history against its snapshot build ===")
        print(f"  {'quantity':<28}{'history':>13}{'snapshot':>14}")
        for label, a, b, ok in compare_with_snapshot(earth):
            failures += not ok
            print(f"  {label:<28}{a:>13}{b:>14}   {'ok' if ok else 'MISS'}")

    print(f"\n{failures} checks missed" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
