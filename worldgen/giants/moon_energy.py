"""What a moon receives from its parent: eclipses, planetshine, tidal heat and magnetic shelter.

- Eclipses (Heller 2012): on a circular orbit in the plane of the star, the
  moon spends R_p / (π a) of each orbit in the parent's shadow. When the moon's
  orbit is tilted to the parent's orbit by ψ, the shadow reaches it only while
  the star lies within arcsin(R_p / (a sin ψ)) of the line of nodes, so over a
  year the lost share is R_p / (π a) × (2/π) arcsin(min(1, R_p / (a sin ψ)))
  (the geometry of Heller's eq. 9, averaged over the year; shadow chords are
  taken as full).
- Planetshine (Heller & Barnes 2013, eq. 22): the parent reflects
  π R_p² α_p / (2a²) of the direct starlight onto the moon, and its thermal
  emission R_p² σ T_p⁴ / a² is absorbed over the moon's sphere as
  (1 − α_IR)/4, with T_p⁴ = T_eq⁴ + T_int⁴ (Heller & Zuluaga 2013).
- Tidal heat (Tjoa et al. 2020, eq. 1): Ė = (21/2) G^1.5 (k₂/Q) M_p^2.5 R⁵ e² / a^7.5,
  with the moon's own k₂ and Q. Moons with an ice shell over an ocean dissipate
  as their shell does: k₂/Q between Europa's ~10⁻³ (Bagheri et al. 2026) and
  Enceladus's ~0.015 (Bagheri et al. 2025), interpolated in the shell's
  stiffness against gravity (heuristic; moon calibration, 2026-10-25).
- Eccentricity damping (Tokadjian & Piro 2023, eq. 20): a free eccentricity
  decays as e₀ exp(−t / t_damp), 1/t_damp = (21/2)(M_p/m)(R/a)⁵ (k₂/Q) n.
  Moons in a resonance keep a forced eccentricity.
- Magnetic shelter (Heller & Zuluaga 2013, eq. 1): the standoff distance
  R_S = (μ₀ f₀² / 8π²)^(1/6) 𝓜^(1/3) P_sw^(-1/6), and the share of the moon's
  orbit inside a magnetopause drawn as a half sphere of radius 1.2 R_S with a
  tail of that radius.
"""

from __future__ import annotations

import math

import numpy as np

from .. import constants as c
from .. import heuristics as h

MU_0 = 4e-7 * math.pi


def eclipse_fraction(parent_radius_m: float, a_m: float, tilt_rad: float) -> float:
    """Return x_s, the share of starlight a moon receives after eclipses by its parent, averaged over a year."""
    per_orbit = parent_radius_m / (math.pi * a_m)
    s = abs(math.sin(tilt_rad))
    seasons = 1.0 if s * a_m <= parent_radius_m else (2.0 / math.pi) * math.asin(parent_radius_m / (a_m * s))
    return 1.0 - per_orbit * seasons


def reflected_fraction(parent_radius_m: float, a_m: float, parent_albedo: float) -> float:
    """Return the starlight the parent reflects onto the moon, over the direct starlight (orbit average)."""
    return math.pi * parent_radius_m**2 * parent_albedo / (2.0 * a_m**2)


def parent_heat_flux(parent_radius_m: float, a_m: float, parent_temperature_k: float) -> float:
    """Return the parent's thermal emission absorbed by the moon (W per m² of the moon's surface)."""
    emitted = parent_radius_m**2 * c.SIGMA_SB * parent_temperature_k**4 / a_m**2
    return emitted * (1.0 - h.MOON_INFRARED_ALBEDO) / 4.0


def tidal_heating(parent_mass_kg: float, moon_radius_m: float, a_m: float, eccentricity: float,
                  love_number_k2: float, tidal_q: float) -> float:
    """Return the tidal heat (W) raised in a synchronous moon on an eccentric orbit (Tjoa et al. 2020, eq. 1)."""
    phi = love_number_k2 / tidal_q
    return 10.5 * c.G**1.5 * phi * parent_mass_kg**2.5 * moon_radius_m**5 * eccentricity**2 / a_m**7.5


def dipole_moment(parent_mass_kg: float, giant: bool, dynamo: bool = True) -> float:
    """Return a parent's dipole moment (A m²): log-log in mass through the Solar System giants, or Earth's scaled.

    Uranus and Neptune enter as their geometric mean; beyond the table the end
    segments are extended as power laws.
    """
    table = np.array(h.PLANET_DIPOLE_MOMENTS)
    mass = parent_mass_kg / c.M_EARTH
    if not giant:
        return float(table[0, 1] * mass) if dynamo else 0.0
    ice = np.exp(np.log(table[1:3]).mean(axis=0))
    points = np.log(np.vstack([ice, table[3:]]))
    x = math.log(mass)
    k = int(np.clip(np.searchsorted(points[:, 0], x) - 1, 0, len(points) - 2))
    (x0, y0), (x1, y1) = points[k], points[k + 1]
    return float(math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0)))


def standoff_distance(moment_am2: float, distance_m: float) -> float:
    """Return the magnetopause standoff distance (m) against the stellar wind (Heller & Zuluaga 2013, eq. 1)."""
    if moment_am2 <= 0.0:
        return 0.0
    pressure = h.SOLAR_WIND_PRESSURE_1AU_PA * (c.AU / distance_m) ** 2
    return (MU_0 * h.MAGNETOPAUSE_FORM_FACTOR**2 / (8.0 * math.pi**2)) ** (1 / 6) * moment_am2 ** (1 / 3) * \
        pressure ** (-1 / 6)


def magnetosphere_fraction(standoff_m: float, a_m: float, samples: int = 720) -> float:
    """Return the share of a circular orbit inside a magnetopause of standoff ``standoff_m`` (sphere and tail)."""
    if standoff_m <= 0.0:
        return 0.0
    big = h.MAGNETOPAUSE_SPHERE_RATIO * standoff_m
    centre = big - standoff_m                     # behind the planet, away from the star
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    x, y = a_m * np.cos(theta), a_m * np.sin(theta)   # x towards the star
    sphere = (x + centre) ** 2 + y**2 < big**2
    tail = (x < -centre) & (np.abs(y) < big)
    return float(np.mean(sphere | tail))


def ice_shell_world(water_mass_fraction: float, instellation_earth: float, extra_flux_w_m2: float = 0.0) -> bool:
    """Return whether a moon is a frozen water world (an ice shell over any ocean), from its water and its
    equilibrium temperature at the thin-air albedo."""
    if water_mass_fraction < h.ICE_SHELL_WORLD_MIN_WATER:
        return False
    absorbed = instellation_earth * c.S_EARTH * (1.0 - h.ALBEDO_THIN) / 4.0 + extra_flux_w_m2
    return (absorbed / c.SIGMA_SB) ** 0.25 < h.ICE_SHELL_WORLD_BELOW_K


def ice_shell_dissipation(mass_kg: float, radius_m: float) -> float:
    """Return k₂/Q of a moon whose tides work an ice shell over an ocean.

    Interpolated log-log in x = μ_ice / (ρ_ice g R), the shell's stiffness
    against the moon's gravity, between Europa (~10⁻³) and Enceladus (~0.015).
    """
    g = c.G * mass_kg / radius_m**2
    x = h.ICE_SHELL_RIGIDITY_PA / (h.ICE_SHELL_DENSITY * g * radius_m)
    (x0, q0), (x1, q1) = h.ICE_SHELL_DISSIPATION_ANCHORS
    slope = math.log(q1 / q0) / math.log(x1 / x0)
    return float(np.clip(q0 * (x / x0) ** slope, h.ICE_SHELL_DISSIPATION_MIN, q1))


def eccentricity_damping_time(parent_mass_kg: float, moon_mass_kg: float, moon_radius_m: float, a_m: float,
                              k2_over_q: float) -> float:
    """Return the time (s) in which a free eccentricity decays by e, from tides in the moon
    (Tokadjian & Piro 2023, eq. 20)."""
    if k2_over_q <= 0.0:
        return math.inf
    n = math.sqrt(c.G * (parent_mass_kg + moon_mass_kg) / a_m**3)
    rate = 10.5 * (parent_mass_kg / moon_mass_kg) * (moon_radius_m / a_m) ** 5 * k2_over_q * n
    return 1.0 / rate


def damped_eccentricity(e0: float, age_s: float, damping_time_s: float) -> float:
    """Return a free eccentricity after ``age_s`` of tidal damping."""
    if not math.isfinite(damping_time_s):
        return e0
    return float(e0 * math.exp(-age_s / damping_time_s))

