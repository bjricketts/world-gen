"""Where moons and rings can orbit a planet: Roche, Hill, synchronous and tidal-survival limits."""

from __future__ import annotations

import math

from .. import constants as c
from .. import heuristics as h


def roche_limit_rubble(planet_mass_kg: float, density_kg_m3: float, gamma: float = h.ROCHE_GAMMA_RUBBLE) -> float:
    """Return the ring–moon boundary a_R = (3M / γρ)^(1/3) (m) for strengthless material of a given density."""
    return (3.0 * planet_mass_kg / (gamma * density_kg_m3)) ** (1.0 / 3.0)


def roche_limit_fluid(planet_radius_m: float, planet_density: float, satellite_density: float) -> float:
    """Return the fluid Roche limit 2.44 R (ρ_p/ρ_s)^(1/3) (m), inside which a large moon is torn apart."""
    return 2.44 * planet_radius_m * (planet_density / satellite_density) ** (1.0 / 3.0)


def stable_moon_limit(hill_radius_m: float, retrograde: bool = False) -> float:
    """Return the outermost stable moon orbit (m) as a fraction of the Hill radius (Barnes & O'Brien 2002)."""
    f = h.MOON_STABLE_FRACTION_RETROGRADE if retrograde else h.MOON_STABLE_FRACTION_PROGRADE
    return f * hill_radius_m


def moon_hill_radius(a_m: float, eccentricity: float, primary_mass_kg: float, secondary_mass_kg: float) -> float:
    """Return the Hill radius (m) of a body of ``secondary_mass_kg`` orbiting ``primary_mass_kg``."""
    return a_m * (1.0 - eccentricity) * (secondary_mass_kg / (3.0 * primary_mass_kg)) ** (1.0 / 3.0)


def synchronous_orbit_radius(planet_mass_kg: float, rotation_period_s: float) -> float:
    """Return the radius (m) of the orbit whose period equals the planet's rotation."""
    return (c.G * planet_mass_kg * rotation_period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)


def max_surviving_moon_mass(planet_mass_kg: float, planet_radius_m: float, a_m: float, star_mass_kg: float,
                            age_s: float, tidal_q: float, love_number_k2: float,
                            fraction: float = h.MOON_STABLE_FRACTION_PROGRADE) -> float:
    """Return the largest moon (kg) whose tidal migration across the stable region takes the planet's age.

    Barnes & O'Brien (2002, eq. 8): M ≤ (2/13) [(f a)³/3M★]^(13/6) M_p^(8/3) Q / (3 k₂ T R⁵ √G).
    Their eq. 9, for planets despun midway through their life, allows twice this.
    """
    hill_term = ((fraction * a_m) ** 3 / (3.0 * star_mass_kg)) ** (13.0 / 6.0)
    return (2.0 / 13.0) * hill_term * planet_mass_kg ** (8.0 / 3.0) * tidal_q / (
        3.0 * love_number_k2 * age_s * planet_radius_m**5 * math.sqrt(c.G))


def orbital_period(a_m: float, primary_mass_kg: float, secondary_mass_kg: float = 0.0) -> float:
    """Return the period (s) of an orbit of semi-major axis ``a_m`` about ``primary_mass_kg``."""
    return 2.0 * math.pi * math.sqrt(a_m**3 / (c.G * (primary_mass_kg + secondary_mass_kg)))
