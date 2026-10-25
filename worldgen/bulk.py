"""Bulk properties: mass, radius and composition, and the planet's class.

Rocky planets use the Zeng et al. (2016) relation
    R / R_E = (1.07 - 0.21 CMF) (M / M_E)^(1/3.7),
valid for 1–8 Earth masses and CMF 0–0.4 (extrapolated outside that).
Planets with volatile envelopes use the mean Chen & Kipping (2017) relation.
"""

from __future__ import annotations

import math
from typing import Optional

from scipy.optimize import brentq

from . import constants as c
from . import heuristics as h
from .state import BulkState, Issue

# Chen & Kipping (2017): log10(R/R_E) = C + S log10(M/M_E), piecewise in mass.
# Values as tabulated by the NASA Exoplanet Archive.
CHEN_KIPPING = [   # (upper mass bound in M_E, C, S)
    (2.04, 0.00346, 0.2790),
    (132.0, -0.0925, 0.589),
    (26600.0, 1.25, -0.044),
    (math.inf, -2.85, 0.881),
]
JOVIAN_MIN_MEARTH = 132.0
ICE_GIANT_MIN_MEARTH = 10.0
TERRAN_MAX_MEARTH = 2.04
BROWN_DWARF_MIN_MEARTH = 13 * c.M_JUPITER / c.M_EARTH
ROCKY_RADIUS_LIMIT_REARTH = 1.6   # above this, a planet of unknown mass is assumed volatile-rich
ZENG_MASS_RANGE = (1.0, 8.0)
ZENG_CMF_RANGE = (0.0, 0.4)
DEFAULT_ENVELOPE = {"sub-Neptune": 0.02, "ice giant": 0.15, "gas giant": 0.9}


def chen_kipping_radius(mass_mearth: float) -> float:
    """Return the mean radius (Earth radii) for a mass, from Chen & Kipping (2017)."""
    logm = math.log10(mass_mearth)
    for upper, cc, s in CHEN_KIPPING:
        if mass_mearth < upper:
            return 10 ** (cc + s * logm)
    raise AssertionError("unreachable")


def chen_kipping_mass_neptunian(radius_rearth: float) -> float:
    """Return the mass (Earth masses) for a radius on the Neptunian branch of Chen & Kipping."""
    _, cc, s = CHEN_KIPPING[1]
    return 10 ** ((math.log10(radius_rearth) - cc) / s)


def _zeng_radius(mass_mearth: float, cmf: float, wmf: float) -> float:
    """Return the Zeng-type radius (Earth radii) of a compressed rocky planet, with water inflation."""
    return (1.07 - 0.21 * cmf) * mass_mearth ** (1 / 3.7) * (1 + h.WATER_RADIUS_INFLATION * wmf)


def _uncompressed_radius(mass_mearth: float, cmf: float, wmf: float) -> float:
    """Return the radius (Earth radii) of a small body from uncompressed iron, rock and ice densities."""
    volume = mass_mearth * c.M_EARTH * (cmf / h.SMALL_BODY_IRON_DENSITY
                                        + (1.0 - cmf - wmf) / h.SMALL_BODY_ROCK_DENSITY
                                        + wmf / h.SMALL_BODY_ICE_DENSITY)
    return (3.0 * max(volume, 1e-30) / (4.0 * math.pi)) ** (1 / 3) / c.R_EARTH


def rocky_radius(mass_mearth: float, cmf: float, wmf: float = 0.0) -> float:
    """Return the radius (Earth radii) of a rocky body with a given core and water fraction.

    Above 1 M⊕ the Zeng-type relation; below 0.1 M⊕, where self-compression is
    small, uncompressed densities; in between a log-log blend of the two.
    """
    lo, hi = h.SMALL_BODY_BLEND_MEARTH
    if mass_mearth >= hi:
        return _zeng_radius(mass_mearth, cmf, wmf)
    small = _uncompressed_radius(mass_mearth, cmf, wmf)
    if mass_mearth <= lo:
        return small
    w = math.log(mass_mearth / lo) / math.log(hi / lo)
    return small ** (1 - w) * _zeng_radius(mass_mearth, cmf, wmf) ** w


def rocky_mass(radius_rearth: float, cmf: float, wmf: float = 0.0) -> float:
    """Return the mass (Earth masses) of a rocky body with a given radius and composition."""
    zeng = (radius_rearth / ((1.07 - 0.21 * cmf) * (1 + h.WATER_RADIUS_INFLATION * wmf))) ** 3.7
    if zeng >= h.SMALL_BODY_BLEND_MEARTH[1]:
        return zeng
    return math.exp(brentq(lambda lm: math.log(rocky_radius(math.exp(lm), cmf, wmf) / radius_rearth),
                           math.log(1e-12), math.log(h.SMALL_BODY_BLEND_MEARTH[1]) + 1e-9))


def rocky_cmf(mass_mearth: float, radius_rearth: float, wmf: float = 0.0) -> float:
    """Return the core mass fraction implied by a rocky body's mass and radius (unclipped, may leave 0–1)."""
    if mass_mearth >= h.SMALL_BODY_BLEND_MEARTH[1]:
        return (1.07 - radius_rearth / ((1 + h.WATER_RADIUS_INFLATION * wmf) * mass_mearth ** (1 / 3.7))) / 0.21
    f = lambda x: rocky_radius(mass_mearth, x, wmf) - radius_rearth   # noqa: E731
    lo, hi = -3.0, 3.0
    if f(lo) * f(hi) > 0:
        return lo if f(lo) < 0 else hi
    return brentq(f, lo, hi)


def rocky_wmf(mass_mearth: float, radius_rearth: float, cmf: float) -> float:
    """Return the water mass fraction implied by a rocky body's mass, radius and core fraction."""
    if mass_mearth >= h.SMALL_BODY_BLEND_MEARTH[1]:
        return (radius_rearth / rocky_radius(mass_mearth, cmf) - 1) / h.WATER_RADIUS_INFLATION
    f = lambda x: rocky_radius(mass_mearth, cmf, x) - radius_rearth   # noqa: E731
    lo, hi = -1.0, 1.0 - cmf
    if f(lo) * f(hi) > 0:
        return hi if f(hi) < 0 else lo
    return brentq(f, lo, hi)


def classify(mass_mearth: float, envelope: float, wmf: float) -> str:
    """Return the planet class name."""
    if mass_mearth >= BROWN_DWARF_MIN_MEARTH:
        return "brown dwarf"
    if envelope > 1e-3:
        if mass_mearth >= JOVIAN_MIN_MEARTH:
            return "gas giant"
        if mass_mearth >= ICE_GIANT_MIN_MEARTH:
            return "ice giant"
        return "sub-Neptune"
    if wmf >= 0.01:
        return "water world"
    if mass_mearth > TERRAN_MAX_MEARTH:
        return "super-Earth"
    if mass_mearth < 0.1:
        return "dwarf rocky world"
    return "rocky"


def _envelope_class(mass_mearth: float) -> str:
    """Return the class name of an envelope-bearing planet from its mass."""
    if mass_mearth >= JOVIAN_MIN_MEARTH:
        return "gas giant"
    if mass_mearth >= ICE_GIANT_MIN_MEARTH:
        return "ice giant"
    return "sub-Neptune"


def build_bulk(
    mass_mearth: Optional[float],
    radius_rearth: Optional[float],
    cmf: Optional[float],
    wmf: float,
    envelope: Optional[float],
    wmf_is_user: bool,
) -> tuple[BulkState, list[Issue], dict[str, str]]:
    """Return bulk properties from any two of mass, radius and composition.

    Unset values among mass, radius, core fraction and envelope fraction are
    derived; the returned provenance dict names them.
    """
    issues: list[Issue] = []
    prov: dict[str, str] = {}
    if mass_mearth is None and radius_rearth is None:
        raise ValueError("bulk properties need at least a mass or a radius")

    # Decide whether the planet has a volatile envelope.
    if envelope is None:
        if mass_mearth is not None:
            has_envelope = mass_mearth > h.ENVELOPE_ASSUMED_ABOVE_MEARTH
        else:
            has_envelope = radius_rearth > ROCKY_RADIUS_LIMIT_REARTH
        if (mass_mearth is not None and radius_rearth is not None and not has_envelope
                and mass_mearth > h.SMALL_BODY_BLEND_MEARTH[0]):
            # Too large even for a half-water planet: must hold an envelope (small bodies are taken as icy instead).
            has_envelope = radius_rearth > rocky_radius(mass_mearth, 0.0, 0.5)
    else:
        has_envelope = envelope > 1e-3

    if has_envelope:
        if mass_mearth is None:
            if radius_rearth > chen_kipping_radius(JOVIAN_MIN_MEARTH):
                mass_mearth = c.M_JUPITER / c.M_EARTH
                issues.append(Issue("warning", "validity", "bulk",
                                    "radius does not constrain the mass of a Jovian planet; assumed 1 Jupiter mass"))
            else:
                mass_mearth = chen_kipping_mass_neptunian(radius_rearth)
            prov["body.mass_mearth"] = "derived"
        expected = max(chen_kipping_radius(mass_mearth), rocky_radius(mass_mearth, 0.3))
        if radius_rearth is None:
            radius_rearth = expected
            prov["body.radius_rearth"] = "derived"
        elif abs(radius_rearth / expected - 1) > 0.3:
            issues.append(Issue("info", "note", "bulk",
                                f"radius differs from the Chen & Kipping mean ({expected:.2f} R⊕) by more than 30%"))
        if envelope is None:
            envelope = DEFAULT_ENVELOPE[_envelope_class(mass_mearth)]
            prov["body.envelope_mass_fraction"] = "heuristic"
        if cmf is None:
            cmf = default_cmf()
            prov["body.core_mass_fraction"] = "default"
    else:
        if envelope is None:
            envelope = 0.0
            prov["body.envelope_mass_fraction"] = "derived"
        if mass_mearth is not None and radius_rearth is not None:
            implied = rocky_cmf(mass_mearth, radius_rearth, wmf)
            if cmf is None:
                if implied < 0 and not wmf_is_user:
                    cmf = 0.0
                    wmf = rocky_wmf(mass_mearth, radius_rearth, cmf)
                    prov["body.water_mass_fraction"] = "derived"
                    prov["body.core_mass_fraction"] = "derived"
                else:
                    cmf = implied
                    prov["body.core_mass_fraction"] = "derived"
                tol = h.CMF_TOLERANCE
                if not -tol <= cmf <= 1.0 + tol:
                    issues.append(Issue("warning", "conflict", "bulk",
                                        f"mass and radius imply a core mass fraction of {implied:.2f}, "
                                        "outside the physical range 0–1"))
                cmf = min(max(cmf, 0.0), 1.0)
            elif abs(implied - cmf) > 0.05:
                issues.append(Issue("warning", "conflict", "bulk",
                                    f"mass, radius and core fraction are over-determined; "
                                    f"mass and radius imply CMF = {implied:.2f}"))
        elif radius_rearth is None:
            if cmf is None:
                cmf = default_cmf()
                prov["body.core_mass_fraction"] = "default"
            radius_rearth = rocky_radius(mass_mearth, cmf, wmf)
            prov["body.radius_rearth"] = "derived"
        else:
            if cmf is None:
                cmf = default_cmf()
                prov["body.core_mass_fraction"] = "default"
            mass_mearth = rocky_mass(radius_rearth, cmf, wmf)
            prov["body.mass_mearth"] = "derived"

        lo, hi = ZENG_MASS_RANGE
        if not lo <= mass_mearth <= hi:
            level = "warning" if mass_mearth > hi else "info"
            issues.append(Issue(level, "validity", "bulk",
                                f"rocky mass–radius relation extrapolated outside {lo:g}–{hi:g} M⊕"))
        if mass_mearth > TERRAN_MAX_MEARTH:
            issues.append(Issue("info", "note", "bulk",
                                "most observed planets above ~2 M⊕ hold volatile envelopes; this one is rocky"))
        if wmf > 0:
            issues.append(Issue("info", "heuristic", "bulk", "radius inflation by water uses a heuristic"))

    mass_kg = mass_mearth * c.M_EARTH
    radius_m = radius_rearth * c.R_EARTH
    planet_class = classify(mass_mearth, envelope, wmf)
    if planet_class == "brown dwarf":
        issues.append(Issue("error", "validity", "bulk",
                            "mass exceeds ~13 Jupiter masses: this is a brown dwarf, not a planet"))
    giant = planet_class in ("ice giant", "gas giant", "brown dwarf")

    state = BulkState(
        mass_kg=mass_kg,
        radius_m=radius_m,
        core_mass_fraction=cmf,
        water_mass_fraction=wmf,
        envelope_mass_fraction=envelope,
        planet_class=planet_class,
        density_kg_m3=mass_kg / (4 / 3 * math.pi * radius_m**3),
        surface_gravity_m_s2=c.G * mass_kg / radius_m**2,
        escape_velocity_m_s=math.sqrt(2 * c.G * mass_kg / radius_m),
        moment_of_inertia_factor=h.MOI_FACTOR_GIANT if giant else h.MOI_FACTOR_ROCKY,
    )
    return state, issues, prov


def default_cmf() -> float:
    """Return the core mass fraction assumed when nothing else constrains it (Earth's)."""
    return c.EARTH_CMF
