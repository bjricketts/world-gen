"""Host star properties from mass and age, using analytic main-sequence relations.

The relations are textbook approximations valid for roughly 0.08–2 solar
masses. They sit behind ``build_star`` so tabulated stellar tracks (e.g. MIST)
can replace them later without changing callers.
"""

from __future__ import annotations

from . import constants as c
from . import heuristics as h
from .state import Issue, StarState

MIN_MASS_MSUN = 0.08
MAX_VALID_MASS_MSUN = 2.0

_SPECTRAL_TYPES = [   # (lower T_eff bound in K, class)
    (30000, "O"), (10000, "B"), (7500, "A"), (6000, "F"),
    (5200, "G"), (3700, "K"), (2400, "M"), (0, "L"),
]


def main_sequence_luminosity(mass_msun: float) -> float:
    """Return the representative main-sequence luminosity in solar units for a stellar mass."""
    m = mass_msun
    if m < 0.43:
        return 0.23 * m**2.3
    if m < 2.0:
        return m**4.0
    if m < 55.0:
        return 1.4 * m**3.5
    return 32000.0 * m


def main_sequence_radius(mass_msun: float) -> float:
    """Return the representative main-sequence radius in solar units for a stellar mass."""
    return mass_msun**0.8 if mass_msun < 1.0 else mass_msun**0.57


def main_sequence_lifetime_gyr(mass_msun: float) -> float:
    """Return the main-sequence lifetime in Gyr, scaled from the Sun's 10 Gyr."""
    return 10.0 * mass_msun / main_sequence_luminosity(mass_msun)


def brightening_factor(age_gyr: float, lifetime_gyr: float) -> float:
    """Return luminosity at a given age relative to the representative main-sequence value."""
    x = age_gyr / lifetime_gyr
    return 1.0 / (1.0 + h.BRIGHTENING_AMPLITUDE * (1.0 - x / h.SUN_MS_FRACTION))


def xuv_fraction(mass_msun: float, age_gyr: float) -> float:
    """Return the ratio of X-ray + EUV luminosity to bolometric luminosity."""
    t_sat = h.XUV_SAT_TIME_SUN_GYR * mass_msun**h.XUV_SAT_MASS_EXPONENT
    if age_gyr <= t_sat:
        return h.XUV_SATURATED
    return h.XUV_SATURATED * (age_gyr / t_sat) ** h.XUV_DECAY_EXPONENT


def spectral_type(teff_k: float) -> str:
    """Return the spectral class letter and subclass for an effective temperature."""
    bounds = [b for b, _ in _SPECTRAL_TYPES]
    for i, (lower, letter) in enumerate(_SPECTRAL_TYPES):
        if teff_k >= lower:
            upper = bounds[i - 1] if i > 0 else lower * 1.5
            sub = int(10 * (upper - teff_k) / (upper - lower)) if upper > lower else 0
            return f"{letter}{min(max(sub, 0), 9)}V"
    return "?"


def build_star(mass_msun: float, age_gyr: float, metallicity_feh: float) -> tuple[StarState, list[Issue]]:
    """Return the star's state at the given age, plus any validity notes."""
    issues: list[Issue] = []
    if mass_msun < MIN_MASS_MSUN:
        issues.append(Issue("error", "validity", "star",
                            f"{mass_msun:.3f} M☉ is below the hydrogen-burning limit; this is a brown dwarf"))
    elif mass_msun > MAX_VALID_MASS_MSUN:
        issues.append(Issue("warning", "validity", "star",
                            f"analytic stellar relations are unreliable above {MAX_VALID_MASS_MSUN} M☉"))

    lifetime = main_sequence_lifetime_gyr(mass_msun)
    if age_gyr > lifetime:
        issues.append(Issue("error", "validity", "star",
                            f"age {age_gyr:.2f} Gyr exceeds the main-sequence lifetime ({lifetime:.2f} Gyr); "
                            "post-main-sequence evolution is not modelled"))

    boost = brightening_factor(min(age_gyr, lifetime), lifetime)
    luminosity = main_sequence_luminosity(mass_msun) * boost
    radius = main_sequence_radius(mass_msun) * boost**h.RADIUS_LUMINOSITY_EXPONENT
    teff = c.T_SUN * (luminosity / radius**2) ** 0.25

    frac = xuv_fraction(mass_msun, age_gyr)
    frac_sun = xuv_fraction(1.0, c.AGE_SUN_GYR)
    issues.append(Issue("info", "heuristic", "star",
                        "XUV activity and main-sequence brightening use calibrated heuristics"))

    state = StarState(
        mass_kg=mass_msun * c.M_SUN,
        age_s=age_gyr * c.SECONDS_PER_GYR,
        metallicity_feh=metallicity_feh,
        luminosity_w=luminosity * c.L_SUN,
        radius_m=radius * c.R_SUN,
        effective_temperature_k=teff,
        main_sequence_lifetime_s=lifetime * c.SECONDS_PER_GYR,
        spectral_type=spectral_type(teff),
        xuv_fraction=frac,
        xuv_fraction_relative_sun=frac / frac_sun,
    )
    return state, issues
