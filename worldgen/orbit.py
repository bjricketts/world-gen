"""Orbit, instellation, habitable zone, tidal locking and spin."""

from __future__ import annotations

import math
from typing import Optional

from . import constants as c
from . import heuristics as h
from .state import BulkState, HabitableZone, Issue, OrbitState, StarState

# Kopparapu et al. (2014) coefficients, from the authors' hzcalc.py:
# S_eff = S_sun + a T + b T^2 + c T^3 + d T^4, with T = T_eff - 5780 K.
HZ_COEFFICIENTS = {
    "recent_venus": (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
    "runaway_greenhouse": (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
    "maximum_greenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
    "early_mars": (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16),
    "runaway_greenhouse_5me": (1.188, 1.433e-4, 1.707e-8, -8.968e-12, -2.084e-15),
    "runaway_greenhouse_0.1me": (0.99, 1.209e-4, 1.404e-8, -7.418e-12, -1.713e-15),
}
HZ_TEFF_RANGE_K = (2600.0, 7200.0)


def orbital_period(a_m: float, star_mass_kg: float, planet_mass_kg: float) -> float:
    """Return the orbital period in seconds (Kepler's third law)."""
    return 2 * math.pi * math.sqrt(a_m**3 / (c.G * (star_mass_kg + planet_mass_kg)))


def instellation(luminosity_w: float, a_m: float, eccentricity: float = 0.0) -> float:
    """Return the orbit-averaged instellation in units of Earth's."""
    return (luminosity_w / c.L_SUN) / (a_m / c.AU) ** 2 / math.sqrt(1 - eccentricity**2)


def semi_major_axis_for_instellation(luminosity_w: float, s_earth: float, eccentricity: float = 0.0) -> float:
    """Return the semi-major axis (m) that gives a target orbit-averaged instellation."""
    return c.AU * math.sqrt((luminosity_w / c.L_SUN) / (s_earth * math.sqrt(1 - eccentricity**2)))


def equilibrium_temperature(s_earth: float, bond_albedo: float) -> float:
    """Return the planet's equilibrium temperature (K), assuming full heat redistribution."""
    return (s_earth * c.S_EARTH * (1 - bond_albedo) / (4 * c.SIGMA_SB)) ** 0.25


def _seff(coeffs: tuple[float, ...], teff_k: float) -> float:
    """Return a habitable-zone instellation limit from its polynomial coefficients."""
    s_sun, a, b, cc, d = coeffs
    t = teff_k - 5780.0
    return s_sun + a * t + b * t**2 + cc * t**3 + d * t**4


def runaway_greenhouse_limit(teff_k: float, planet_mass_mearth: float) -> float:
    """Return the runaway-greenhouse instellation limit for a planet mass (Earth units)."""
    points = [(0.1, "runaway_greenhouse_0.1me"), (1.0, "runaway_greenhouse"), (5.0, "runaway_greenhouse_5me")]
    m = min(max(planet_mass_mearth, 0.1), 5.0)
    logm = math.log10(m)
    for (m0, k0), (m1, k1) in zip(points, points[1:]):
        if m <= m1:
            w = (logm - math.log10(m0)) / (math.log10(m1) - math.log10(m0))
            return (1 - w) * _seff(HZ_COEFFICIENTS[k0], teff_k) + w * _seff(HZ_COEFFICIENTS[k1], teff_k)
    return _seff(HZ_COEFFICIENTS["runaway_greenhouse_5me"], teff_k)


def habitable_zone(luminosity_w: float, teff_k: float, planet_mass_mearth: float,
                   s_planet: float) -> tuple[HabitableZone, list[Issue]]:
    """Return the habitable-zone limits for this star and planet mass, and where the planet sits."""
    issues: list[Issue] = []
    lo, hi = HZ_TEFF_RANGE_K
    if not lo <= teff_k <= hi:
        issues.append(Issue("warning", "validity", "orbit",
                            f"habitable-zone fits are valid for {lo:.0f}–{hi:.0f} K; star is {teff_k:.0f} K (clamped)"))
    t = min(max(teff_k, lo), hi)
    s = {
        "recent_venus": _seff(HZ_COEFFICIENTS["recent_venus"], t),
        "runaway_greenhouse": runaway_greenhouse_limit(t, planet_mass_mearth),
        "maximum_greenhouse": _seff(HZ_COEFFICIENTS["maximum_greenhouse"], t),
        "early_mars": _seff(HZ_COEFFICIENTS["early_mars"], t),
    }
    l_sun = luminosity_w / c.L_SUN
    d = {k: c.AU * math.sqrt(l_sun / v) for k, v in s.items()}

    d_planet = c.AU * math.sqrt(l_sun / s_planet)
    fraction = (d_planet - d["runaway_greenhouse"]) / (d["maximum_greenhouse"] - d["runaway_greenhouse"])

    if s_planet > s["recent_venus"]:
        position = "hotter than the habitable zone"
    elif s_planet > s["runaway_greenhouse"]:
        position = "optimistic habitable zone (inner edge)"
    elif s_planet >= s["maximum_greenhouse"]:
        position = "conservative habitable zone"
    elif s_planet >= s["early_mars"]:
        position = "optimistic habitable zone (outer edge)"
    else:
        position = "colder than the habitable zone"

    hz = HabitableZone(
        recent_venus_s=s["recent_venus"], runaway_greenhouse_s=s["runaway_greenhouse"],
        maximum_greenhouse_s=s["maximum_greenhouse"], early_mars_s=s["early_mars"],
        recent_venus_m=d["recent_venus"], runaway_greenhouse_m=d["runaway_greenhouse"],
        maximum_greenhouse_m=d["maximum_greenhouse"], early_mars_m=d["early_mars"],
        position=position,
        conservative_fraction=fraction,
    )
    return hz, issues


def instellation_at_hz_fraction(luminosity_w: float, teff_k: float, planet_mass_mearth: float,
                                fraction: float) -> float:
    """Return the instellation (Earth units) at a position across the conservative habitable zone.

    ``fraction`` is 0 at the runaway-greenhouse limit and 1 at the
    maximum-greenhouse limit, measured in orbital distance; values outside
    0–1 lie inside or beyond the zone.
    """
    hz, _ = habitable_zone(luminosity_w, teff_k, planet_mass_mearth, 1.0)
    d_in, d_out = hz.runaway_greenhouse_m, hz.maximum_greenhouse_m
    d = d_in + fraction * (d_out - d_in)
    if d <= 0:
        raise ValueError(f"habitable-zone fraction {fraction} places the planet inside the star")
    return (luminosity_w / c.L_SUN) / (d / c.AU) ** 2


def default_tidal_q(bulk: BulkState) -> float:
    """Return the tidal quality factor assumed when none is supplied."""
    return h.TIDAL_Q_GIANT if _is_giant(bulk) else h.TIDAL_Q_ROCKY


def _is_giant(bulk: BulkState) -> bool:
    """Return whether the planet is a giant without a solid surface."""
    return bulk.planet_class in ("ice giant", "gas giant", "brown dwarf")


def tidal_lock_time(a_m: float, star_mass_kg: float, bulk: BulkState, tidal_q: Optional[float] = None) -> float:
    """Return the time (s) for stellar tides to despin the planet (Gladman et al. 1996)."""
    q = default_tidal_q(bulk) if tidal_q is None else tidal_q
    k2 = h.TIDAL_K2_GIANT if _is_giant(bulk) else h.TIDAL_K2_ROCKY
    omega = 2 * math.pi / (h.INITIAL_SPIN_PERIOD_H * c.SECONDS_PER_HOUR)
    inertia = bulk.moment_of_inertia_factor * bulk.mass_kg * bulk.radius_m**2
    return omega * a_m**6 * inertia * q / (3 * c.G * star_mass_kg**2 * k2 * bulk.radius_m**5)


def hill_radius(a_m: float, eccentricity: float, star_mass_kg: float, planet_mass_kg: float) -> float:
    """Return the Hill radius (m), the rough limit for stable moon orbits."""
    return a_m * (1 - eccentricity) * (planet_mass_kg / (3 * star_mass_kg)) ** (1 / 3)


def roche_limit(planet_radius_m: float, planet_density: float, satellite_density: float = 1000.0) -> float:
    """Return the fluid Roche limit (m) inside which rings rather than moons form."""
    return 2.44 * planet_radius_m * (planet_density / satellite_density) ** (1 / 3)


def build_orbit(
    star: StarState,
    bulk: BulkState,
    semi_major_axis_m: Optional[float],
    instellation_earth: Optional[float],
    eccentricity: float,
    obliquity_rad: float,
    rotation_period_s: Optional[float],
    rotation_is_user: bool,
    obliquity_is_user: bool,
    tidal_q: Optional[float] = None,
    periapsis_longitude_rad: float = 0.0,
) -> tuple[OrbitState, list[Issue], dict[str, str]]:
    """Return the orbit and spin state, notes, and provenance of any values set by physics."""
    issues: list[Issue] = []
    provenance: dict[str, str] = {}

    if semi_major_axis_m is None:
        if instellation_earth is None:
            raise ValueError("orbit needs a semi-major axis or an instellation")
        semi_major_axis_m = semi_major_axis_for_instellation(star.luminosity_w, instellation_earth, eccentricity)
        provenance["orbit.semi_major_axis_au"] = "derived"
    s = instellation(star.luminosity_w, semi_major_axis_m, eccentricity)
    if instellation_earth is None:
        provenance["orbit.instellation_earth"] = "derived"

    period = orbital_period(semi_major_axis_m, star.mass_kg, bulk.mass_kg)
    if tidal_q is None:
        tidal_q = default_tidal_q(bulk)
        provenance["body.tidal_q"] = "default"
    t_lock = tidal_lock_time(semi_major_axis_m, star.mass_kg, bulk, tidal_q)
    issues.append(Issue("info", "heuristic", "orbit",
                        "tidal locking uses an assumed initial spin and Love number k2"))

    spin_state = "free"
    if t_lock < star.age_s:
        spin_state = "3:2 resonance" if eccentricity > h.RESONANCE_ECCENTRICITY else "synchronous"
        locked_period = period * (2 / 3 if spin_state == "3:2 resonance" else 1.0)
        if rotation_period_s is not None and rotation_is_user:
            if abs(rotation_period_s / locked_period - 1) > 0.05:
                issues.append(Issue("warning", "conflict", "orbit",
                                    f"planet should be tidally locked ({spin_state}) but the rotation period is user-set"))
                spin_state = "free (user override)"
        else:
            rotation_period_s = locked_period
            provenance["orbit.rotation_period_h"] = "derived"
        if spin_state == "synchronous":
            if obliquity_is_user and obliquity_rad > math.radians(5):
                issues.append(Issue("warning", "conflict", "orbit",
                                    "tidally locked planets are expected to have near-zero obliquity"))
            elif not obliquity_is_user:
                obliquity_rad = 0.0
                provenance["orbit.obliquity_deg"] = "derived"
    if rotation_period_s is None:
        raise ValueError("rotation period must be supplied for a planet that is not tidally locked")

    hz, hz_issues = habitable_zone(star.luminosity_w, star.effective_temperature_k,
                                   bulk.mass_kg / c.M_EARTH, s)
    issues.extend(hz_issues)

    state = OrbitState(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        obliquity_rad=obliquity_rad,
        orbital_period_s=period,
        rotation_period_s=rotation_period_s,
        instellation_earth=s,
        tidal_lock_time_s=t_lock,
        tidal_q=tidal_q,
        spin_state=spin_state,
        hill_radius_m=hill_radius(semi_major_axis_m, eccentricity, star.mass_kg, bulk.mass_kg),
        habitable_zone=hz,
        periapsis_longitude_rad=periapsis_longitude_rad,
    )
    return state, issues, provenance
