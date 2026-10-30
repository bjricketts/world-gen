"""Compare the giants, their rings and moons, the Earth–Moon system and the big moons with the Solar System.

Usage: python scripts/validate_solar_system.py [--seeds N]

The four giants are built from their masses, radii, orbits and envelope
fractions; their moons and rings are drawn, and drawn systems are compared
over ``--seeds`` draws with the architectures they should reproduce. The big
moons are then built as worlds from their real masses, radii and orbits,
listed on their planets. Each check names its source. The exit
code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import types
from dataclasses import dataclass

import numpy as np

from worldgen import constants as c
from worldgen import generate
from worldgen import heuristics as h
from worldgen.giants import moon_spec
from worldgen.giants.moons import draw_moons, impact_moon
from worldgen.spec import PlanetSpec
from worldgen.util import named_rng

GIANTS = {   # a (AU), e, obliquity (deg), rotation (h), mass (M⊕), radius (R⊕), envelope fraction
    "Jupiter": (5.203, 0.048, 3.13, 9.925, 317.8, 11.21, 0.9),
    "Saturn": (9.537, 0.054, 26.73, 10.656, 95.16, 9.45, 0.85),
    "Uranus": (19.19, 0.047, 97.77, 17.24, 14.54, 4.007, 0.15),
    "Neptune": (30.07, 0.009, 28.32, 16.11, 17.15, 3.883, 0.15),
}
# Moons listed on their planets: masses from JPL GM, radii and orbits (planet radii) from the NSSDC fact sheets.
MOONS = {
    "Jupiter": [
        {"name": "Io", "mass_mearth": 0.01496, "radius_rearth": 0.286, "semi_major_axis_rp": 5.90,
         "eccentricity": 0.0041},
        {"name": "Europa", "mass_mearth": 0.00804, "radius_rearth": 0.245, "water_mass_fraction": 0.08,
         "semi_major_axis_rp": 9.40, "eccentricity": 0.0094},
        {"name": "Ganymede", "mass_mearth": 0.0248, "radius_rearth": 0.413, "water_mass_fraction": 0.45,
         "semi_major_axis_rp": 14.97, "eccentricity": 0.0013},
        {"name": "Callisto", "mass_mearth": 0.0180, "radius_rearth": 0.378, "water_mass_fraction": 0.5,
         "semi_major_axis_rp": 26.33, "eccentricity": 0.0074}],
    "Saturn": [
        {"name": "Enceladus", "mass_mearth": 1.81e-5, "radius_rearth": 0.0396, "water_mass_fraction": 0.4,
         "semi_major_axis_rp": 3.95, "eccentricity": 0.0047},
        {"name": "Titan", "mass_mearth": 0.0225, "radius_rearth": 0.404, "water_mass_fraction": 0.45,
         "semi_major_axis_rp": 20.27, "eccentricity": 0.0288}],
    "Neptune": [
        {"name": "Triton", "mass_mearth": 0.00359, "radius_rearth": 0.2122, "water_mass_fraction": 0.3,
         "semi_major_axis_rp": 14.4, "eccentricity": 2e-5, "inclination_deg": 156.9, "retrograde": True}],
}
EARTH = {"name": "Earth", "seed": 1, "star": {"mass_msun": 1.0, "age_gyr": 4.57},
         "orbit": {"semi_major_axis_au": 1.0, "eccentricity": 0.0167, "obliquity_deg": 23.44,
                   "rotation_period_h": 23.934},
         "body": {"mass_mearth": 1.0, "radius_rearth": 1.0},
         "system": {"moons": [{"name": "Moon", "mass_mearth": 0.0123, "radius_rearth": 0.2727,
                               "semi_major_axis_rp": 60.3, "eccentricity": 0.0549, "inclination_deg": 5.1}]}}


def giant_spec(name: str, seed: int = 1, moons=None) -> PlanetSpec:
    """Return a Solar System giant's spec, with its listed moons or with moons left to draw."""
    a, e, obliquity, rotation, mass, radius, envelope = GIANTS[name]
    return PlanetSpec.model_validate({
        "name": name, "seed": seed, "star": {"mass_msun": 1.0, "age_gyr": 4.57},
        "orbit": {"semi_major_axis_au": a, "eccentricity": e, "obliquity_deg": obliquity, "rotation_period_h": rotation},
        "body": {"mass_mearth": mass, "radius_rearth": radius, "envelope_mass_fraction": envelope},
        "system": {"moons": moons} if moons is not None else {}})


@dataclass
class Check:
    """One modelled quantity against the observed value and its source."""

    group: str
    label: str
    value: object
    observed: str
    passed: bool
    source: str


def within(value: float, target: float, tolerance: float) -> bool:
    """Return whether a value lies within an absolute tolerance of a target."""
    return abs(value - target) <= tolerance


def factor(value: float, target: float, times: float) -> bool:
    """Return whether a positive value lies within a factor of a target."""
    return value > 0 and target / times <= value <= target * times


def giant_checks(states: dict, seeds: int) -> list[Check]:
    """Return the checks on the giants' appearance and rings."""
    out = []
    ap = {n: s.appearance for n, s in states.items()}
    targets = {"Jupiter": (124.7, 3.0, 0.503, 0.06, "Li et al. 2018 (A 0.503, 7.485 W/m² internal)"),
               "Saturn": (96.3, 3.0, 0.41, 0.06, "Wang et al. 2024 (A 0.41, 4.88 W/m² emitted)")}
    for name, (t, dt, a, da, src) in targets.items():
        out.append(Check(name, "effective temperature (K)", round(ap[name].effective_temperature_k, 1),
                         f"{t} ± {dt}", within(ap[name].effective_temperature_k, t, dt), src))
        out.append(Check(name, "Bond albedo", round(ap[name].bond_albedo, 3), f"{a} ± {da}",
                         within(ap[name].bond_albedo, a, da), src))
    # Uranus and Neptune differ 10× in internal heat for reasons not tied to mass; the model draws it
    t_eff = []
    for seed in range(1, seeds + 1):
        t_eff += [generate(giant_spec(n, seed))[0].appearance.effective_temperature_k for n in ("Uranus", "Neptune")]
    lo, hi = min(t_eff), max(t_eff)
    out.append(Check("Uranus, Neptune", f"effective temperature range over {seeds} seeds (K)",
                     f"{lo:.1f}–{hi:.1f}", "59.1 and 59.3 inside", lo <= 59.1 <= hi and lo <= 59.3 <= hi,
                     "Pearl et al. 1990 via Irwin et al. 2025; Guillot & Gautier 2015 via Scheibe et al. 2021"))
    for name in ("Uranus", "Neptune"):
        out.append(Check(name, "Bond albedo", round(ap[name].bond_albedo, 3), "0.29–0.35",
                         0.27 <= ap[name].bond_albedo <= 0.37, "Pearl & Conrath 1991; Irwin et al. 2025"))
    for name, state in states.items():
        out.append(Check(name, "cloud class", ap[name].sudarsky_class, "I", ap[name].sudarsky_class == "I",
                         "Sudarsky et al. 2000"))
        jets = ap[name].jet_count
        many = name in ("Jupiter", "Saturn")
        out.append(Check(name, "jets", jets, "~20" if many else "3", 15 <= jets <= 30 if many else jets == 3,
                         "Lian & Showman 2010"))
        direction = "prograde" if many else "retrograde"
        out.append(Check(name, "equatorial jet", ap[name].equatorial_jet, direction,
                         ap[name].equatorial_jet == direction, "Liu & Schneider 2010"))
    regimes = {"Jupiter": "dusty warm", "Saturn": "broad", "Uranus": "narrow", "Neptune": "dusty cold"}
    for name, regime in regimes.items():
        got = states[name].system.ring_regime
        out.append(Check(name, "ring regime", got, regime, got == regime, "Sicardy et al. 2024; Hedman 2015"))
    rings = states["Saturn"].system.rings
    if rings:
        inner = min(r.inner_radius_m for r in rings) / 1e3
        outer = max(r.outer_radius_m for r in rings) / 1e3
        mass = sum(r.mass_kg for r in rings)
        out += [Check("Saturn", "C ring inner edge (km)", round(inner), "74,490 ± 3%", factor(inner, 74490, 1.03),
                      "Miller et al. 2023"),
                Check("Saturn", "A ring outer edge (km)", round(outer), "136,770 ± 3%", factor(outer, 136770, 1.03),
                      "Miller et al. 2023"),
                Check("Saturn", "ring mass (kg)", f"{mass:.2g}", "1.5e19 within 2×", factor(mass, 1.5e19, 2),
                      "Sicardy et al. 2024")]
    rings = states["Uranus"].system.rings
    if rings:
        outer = max(r.outer_radius_m for r in rings) / 1e3
        out.append(Check("Uranus", "outermost (ε) ring (km)", round(outer), "51,150 ± 3%", factor(outer, 51150, 1.03),
                         "Hedman 2015"))
    return out


# Disc colours of the Payne et al. (2025) geometric albedo spectra under a 5772 K star.
OBSERVED_COLOURS = {"Jupiter": "#c1c1b2", "Saturn": "#c4b99b", "Uranus": "#a0c3ca", "Neptune": "#8db7ca"}


def colour_checks(states: dict) -> list[Check]:
    """Return checks of the layered giant colours at each giant's mass-trend metallicity, the
    ice giants with their observed deep methane and internal heat, the seeded haze scatter left out."""
    from worldgen.giants import colours as gc

    def rgb(hex_colour: str) -> np.ndarray:
        return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], float)

    ch4_solar = gc.mole_fractions(1.0)["CH4"]
    ice = {"Uranus": (0.03, 0.04), "Neptune": (0.07, 0.43)}    # CH4 (Irwin et al. 2022), W/m² (Liu & Schneider 2010)
    out, discs = [], {}
    for name, state in states.items():
        ap = state.appearance
        if name in ice:
            z, internal = ice[name][0] / ch4_solar, ice[name][1]
        else:
            z = 10 ** (h.GIANT_METALLICITY_LOG_EPS_MJ + h.GIANT_METALLICITY_SLOPE
                       * math.log10(state.bulk.mass_kg / c.M_JUPITER) - gc.PROTOSOLAR_LOG_EPS["C"])
            internal = ap.internal_flux_w_m2
        absorbed = (1 - ap.bond_albedo) * state.orbit.instellation_earth * c.S_EARTH / 4
        column, t_of_p = gc.cloud_column(state.bulk.surface_gravity_m_s2, absorbed, internal, z, 1, scatter=False)
        discs[name] = gc.layered_colours(column, t_of_p, 5772.0).disc
        miss = float(np.linalg.norm(rgb(discs[name]) - rgb(OBSERVED_COLOURS[name])))
        out.append(Check(name, "disc colour (distance in sRGB 0–255)", f"{discs[name]} ({miss:.0f})",
                         f"{OBSERVED_COLOURS[name]}, within 12", miss < 12, "Payne et al. 2025 spectra"))
        out.append(Check(name, "1-bar temperature (K)", round(column.t_1bar_k, 1),
                         {"Jupiter": "165", "Saturn": "135", "Uranus": "76", "Neptune": "72"}[name] + " ± 7",
                         abs(column.t_1bar_k - {"Jupiter": 165, "Saturn": 135, "Uranus": 76, "Neptune": 72}[name]) <= 7,
                         "Helled & Howard compilation"))
    for a, b in (("Jupiter", "Saturn"), ("Uranus", "Neptune")):
        gap = float(np.linalg.norm(rgb(discs[a]) - rgb(discs[b])))
        out.append(Check(f"{a}, {b}", "colours differ (sRGB distance)", round(gap, 1), "> 6", gap > 6,
                         "Payne et al. 2025 spectra"))
    return out


def system_checks(seeds: int) -> list[Check]:
    """Return the checks on drawn moon systems over many draws."""
    out = []
    draws = {}
    for name in GIANTS:
        state = generate(giant_spec(name))[0]
        sy = state.system
        draws[name] = (state, [draw_moons(name, state.star, state.orbit, state.bulk, sy.limits, sy.ring_regime,
                                          0.3, s) for s in range(seeds)])
    state, out_j = draws["Jupiter"]
    regular = [[m for m in moons if m.origin == "regular"] for moons, _, _ in out_j]
    share = np.mean([4 <= len(r) <= 5 for r in regular])
    total = np.median([sum(m.mass_kg for m in r) / state.bulk.mass_kg for r in regular if r])
    out += [Check("Jupiter system", "draws with 4–5 regular moons", f"{share:.0%}", "≥ 70% (80% in Sasaki)",
                  share >= 0.7, "Sasaki et al. 2010"),
            Check("Jupiter system", "median regular-moon mass (M_p)", f"{total:.2g}", "2.1e-4 within 2×",
                  factor(total, 2.1e-4, 2), "Galilean moons, JPL GM")]
    state, out_s = draws["Saturn"]
    single = np.mean([sum(m.origin == "regular" for m in moons) == 1 for moons, _, _ in out_s])
    out.append(Check("Saturn system", "draws with one large regular moon", f"{single:.0%}", "≥ 55% (67% in Sasaki)",
                     single >= 0.55, "Sasaki et al. 2010"))
    for name in ("Uranus", "Neptune"):
        state, outs = draws[name]
        uranian = np.mean([regime == "uranian" for _, regime, _ in outs])
        captured = np.mean([any(m.origin == "captured" for m in moons) for moons, _, _ in outs])
        out += [Check(f"{name} system", "draws with an impact-born Uranian system", f"{uranian:.0%}", "40–60%",
                      0.4 <= uranian <= 0.6, "Woo et al. 2021 (heuristic share)"),
                Check(f"{name} system", "draws keeping a captured moon", f"{captured:.0%}", "10–40%",
                      0.1 <= captured <= 0.4, "Triton: one of two ice giants (Agnor & Hamilton 2006)")]
    giants_captured = np.mean([any(m.origin == "captured" for m in moons)
                               for name in ("Jupiter", "Saturn") for moons, _, _ in draws[name][1]])
    out.append(Check("Jupiter, Saturn systems", "draws keeping a captured moon", f"{giants_captured:.0%}", "none",
                     giants_captured == 0.0, "Rufu & Canup 2017 (none survives the regular moons)"))
    earth = generate(PlanetSpec.model_validate({**EARTH, "system": {}}))[0]
    moons = [impact_moon(earth.bulk, earth.orbit.tidal_q, 0.3, earth.star.age_s, named_rng(s, "moons.impact"))
             for s in range(seeds * 10)]
    any_moon = np.mean([m is not None for m in moons])
    large = np.mean([m is not None and m.mass_kg > 0.5 * 7.342e22 for m in moons])
    out += [Check("Earth", "draws with an impact moon", f"{any_moon:.0%}", "~40%", 0.3 <= any_moon <= 0.55,
                  "Elser et al. 2011"),
            Check("Earth", "draws with a moon above half the Moon's mass", f"{large:.1%}", "1 in 4 to 1 in 45",
                  1 / 45 <= large <= 1 / 4, "Elser et al. 2011")]
    return out


def moon_checks() -> list[Check]:
    """Return the checks on the big moons built as worlds from their real masses and orbits."""
    worlds = {}
    for planet, listed in MOONS.items():
        spec = giant_spec(planet, moons=listed)
        state = generate(spec)[0]
        world = types.SimpleNamespace(state=state, spec=spec)
        for k, m in enumerate(state.system.moons):
            worlds[m.name] = generate(moon_spec(world, k))[0]
    spec = PlanetSpec.model_validate(EARTH)
    earth = generate(spec)[0]
    worlds["Moon"] = generate(moon_spec(types.SimpleNamespace(state=earth, spec=spec), 0))[0]
    out = []
    io = worlds["Io"]
    flux = io.parent.tidal_heating_w / (4 * math.pi * io.bulk.radius_m**2)
    out += [Check("Io", "tidal heat flux (W/m²)", f"{flux:.2g}", "~2 within 2×", factor(flux, 2.0, 2),
                  "Heller & Zuluaga 2013"),
            Check("Io", "partially molten", io.parent.partially_molten, "yes", io.parent.partially_molten,
                  "Paris et al. 2026 (melt below ~20%)")]
    enc = worlds["Enceladus"]
    out += [Check("Enceladus", "tidal heat (GW)", f"{enc.parent.tidal_heating_w / 1e9:.3g}", "15.8 within 2×",
                  factor(enc.parent.tidal_heating_w, 15.8e9, 2), "Howett et al. 2011 via Lainey et al. 2012"),
            Check("Enceladus", "subsurface ocean (km)", f"{enc.ice_shell.ocean_depth_m / 1e3:.0f}", "present",
                  enc.ice_shell.ocean_depth_m > 0, "Lainey et al. 2012; Kimura 2025")]
    eu = worlds["Europa"].ice_shell
    eu_flux = (worlds["Europa"].parent.tidal_heating_w + worlds["Europa"].interior.radiogenic_power_w) / (
        4 * math.pi * worlds["Europa"].bulk.radius_m**2) * 1e3
    out += [Check("Europa", "subsurface ocean (km)", f"{eu.ocean_depth_m / 1e3:.0f}", "present",
                  eu.ocean_depth_m > 0, "Kimura 2025"),
            Check("Europa", "heat flux (mW/m²)", f"{eu_flux:.3g}", "15–35", 15 <= eu_flux <= 35,
                  "Bagheri et al. 2026 (Extended Data Table 1)"),
            Check("Europa", "ice shell (km)", f"{eu.shell_thickness_m / 1e3:.0f}", "19–39 (conductive)",
                  19e3 <= eu.shell_thickness_m <= 39e3, "Levin et al. 2026 via Bagheri et al. 2026"),
            Check("Europa", "hydrosphere (km)", f"{eu.water_layer_m / 1e3:.0f}", "≤ 200", eu.water_layer_m <= 200e3,
                  "Kimura 2025")]
    ga = worlds["Ganymede"].ice_shell
    out += [Check("Ganymede", "subsurface ocean (km)", f"{ga.ocean_depth_m / 1e3:.0f}", "present (≳ 100)",
                  ga.ocean_depth_m > 0, "Saur et al. 2015; Jia et al. 2025; via Cabanes et al. 2026"),
            Check("Callisto", "subsurface ocean (km)", f"{worlds['Callisto'].ice_shell.ocean_depth_m / 1e3:.0f}",
                  "present (induction)", worlds["Callisto"].ice_shell.ocean_depth_m > 0,
                  "Khurana et al. 1998 via Daniel et al. 2025"),
            Check("Enceladus", "ice shell (km)", f"{enc.ice_shell.shell_thickness_m / 1e3:.0f}", "> 10 (15–40)",
                  enc.ice_shell.shell_thickness_m > 10e3, "Bagheri et al. 2025"),
            Check("Ganymede", "ice Ih shell (km)", f"{ga.shell_thickness_m / 1e3:.0f}", "≤ 200 (HP ice below ~150)",
                  ga.shell_thickness_m <= 200e3, "Kimura 2025"),
            Check("Ganymede", "high-pressure ice (km)", f"{ga.high_pressure_ice_m / 1e3:.0f}", "present",
                  ga.high_pressure_ice_m > 0, "Kimura 2025")]
    ti = worlds["Titan"]
    p_bar = ti.atmosphere.surface_pressure_pa / c.BAR
    out += [Check("Titan", "surface pressure (bar)", f"{p_bar:.3g}", "1.467 within 1.5×", factor(p_bar, 1.467, 1.5),
                  "Fulchignoni et al. 2005 via Kimura 2025"),
            Check("Titan", "surface temperature (K)", f"{ti.atmosphere.surface_temperature_k:.1f}", "93.7 ± 5",
                  within(ti.atmosphere.surface_temperature_k, 93.7, 5.0), "Fulchignoni et al. 2005 via Kimura 2025"),
            Check("Titan", "Bond albedo", f"{ti.atmosphere.bond_albedo:.3f}", "0.25–0.27",
                  0.24 <= ti.atmosphere.bond_albedo <= 0.28, "Creecy et al. 2021"),
            Check("Titan", "ocean and shell (km)", f"{ti.ice_shell.ocean_depth_m / 1e3:.0f} under "
                  f"{ti.ice_shell.shell_thickness_m / 1e3:.0f}", "≥ 50 under ≥ 50",
                  ti.ice_shell.ocean_depth_m >= 50e3 and ti.ice_shell.shell_thickness_m >= 50e3,
                  "Goossens et al. 2024 via Kimura 2025 (disputed by Petricca et al. 2025)")]
    tr = worlds["Triton"]
    p_ubar = tr.atmosphere.surface_pressure_pa / c.BAR * 1e6
    out.append(Check("Triton", "surface pressure (µbar)", f"{p_ubar:.3g}", "14.5 within 10×", factor(p_ubar, 14.5, 10),
                     "Sicardy et al. 2024"))
    for name in ("Ganymede", "Callisto", "Io", "Europa", "Moon"):
        present = worlds[name].atmosphere.present
        out.append(Check(name, "substantial atmosphere", present, "no", not present, "Zahnle & Catling 2017"))
    moon = worlds["Moon"]
    out += [Check("Moon", "radius (km)", f"{moon.bulk.radius_m / 1e3:.0f}", "1737 ± 1%",
                  factor(moon.bulk.radius_m / 1e3, 1737.4, 1.01), "NSSDC"),
            Check("Moon", "spin", moon.orbit.spin_state, "synchronous", "synchronous" in moon.orbit.spin_state,
                  "observed")]
    return out


def main(argv=None) -> int:
    """Run the checks, print the table and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=200, help="draws for the system statistics")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    states = {name: generate(giant_spec(name))[0] for name in GIANTS}
    checks = giant_checks(states, min(args.seeds, 20)) + colour_checks(states) + system_checks(args.seeds) + moon_checks()
    width = max(len(f"{k.group}: {k.label}") for k in checks)
    for k in checks:
        mark = "ok  " if k.passed else "MISS"
        print(f"{mark} {f'{k.group}: {k.label}':<{width}}  {str(k.value):>18}  vs {k.observed:<28} [{k.source}]")
    misses = sum(not k.passed for k in checks)
    print(f"\n{len(checks) - misses} of {len(checks)} checks pass ({time.perf_counter() - start:.0f} s)")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
