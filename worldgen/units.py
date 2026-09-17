"""Conversions between the user-facing units in specs and the SI units used internally.

Specs use astronomer-friendly units (solar masses, AU, Earth masses, bar, hours).
Every internal quantity is SI, and internal names carry a unit suffix where it
is not obvious (e.g. ``mass_kg``, ``period_s``).
"""

import math

from . import constants as c


def msun_to_kg(m: float) -> float:
    """Convert solar masses to kilograms."""
    return m * c.M_SUN


def kg_to_msun(m: float) -> float:
    """Convert kilograms to solar masses."""
    return m / c.M_SUN


def mearth_to_kg(m: float) -> float:
    """Convert Earth masses to kilograms."""
    return m * c.M_EARTH


def kg_to_mearth(m: float) -> float:
    """Convert kilograms to Earth masses."""
    return m / c.M_EARTH


def rearth_to_m(r: float) -> float:
    """Convert Earth radii to metres."""
    return r * c.R_EARTH


def m_to_rearth(r: float) -> float:
    """Convert metres to Earth radii."""
    return r / c.R_EARTH


def au_to_m(d: float) -> float:
    """Convert astronomical units to metres."""
    return d * c.AU


def m_to_au(d: float) -> float:
    """Convert metres to astronomical units."""
    return d / c.AU


def gyr_to_s(t: float) -> float:
    """Convert gigayears to seconds."""
    return t * c.SECONDS_PER_GYR


def s_to_gyr(t: float) -> float:
    """Convert seconds to gigayears."""
    return t / c.SECONDS_PER_GYR


def hours_to_s(t: float) -> float:
    """Convert hours to seconds."""
    return t * c.SECONDS_PER_HOUR


def s_to_hours(t: float) -> float:
    """Convert seconds to hours."""
    return t / c.SECONDS_PER_HOUR


def s_to_days(t: float) -> float:
    """Convert seconds to days."""
    return t / c.SECONDS_PER_DAY


def bar_to_pa(p: float) -> float:
    """Convert bar to pascals."""
    return p * c.BAR


def pa_to_bar(p: float) -> float:
    """Convert pascals to bar."""
    return p / c.BAR


def deg_to_rad(x: float) -> float:
    """Convert degrees to radians."""
    return math.radians(x)
