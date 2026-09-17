"""Bulk properties: mass, radius and composition, and the planet's class.

Rocky planets use the Zeng et al. (2016) relation
    R / R_E = (1.07 - 0.21 CMF) (M / M_E)^(1/3.7),
valid for 1–8 Earth masses and CMF 0–0.4 (extrapolated outside that).
Planets with volatile envelopes use the mean Chen & Kipping (2017) relation.
"""

from __future__ import annotations

import math
from typing import Optional

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


def rocky_radius(mass_mearth: float, cmf: float, wmf: float = 0.0) -> float:
    """Return the radius (Earth radii) of a rocky planet with a given core and water fraction."""
    return (1.07 - 0.21 * cmf) * mass_mearth ** (1 / 3.7) * (1 + h.WATER_RADIUS_INFLATION * wmf)


def rocky_mass(radius_rearth: float, cmf: float, wmf: float = 0.0) -> float:
    """Return the mass (Earth masses) of a rocky planet with a given radius and composition."""
    return (radius_rearth / ((1.07 - 0.21 * cmf) * (1 + h.WATER_RADIUS_INFLATION * wmf))) ** 3.7


def rocky_cmf(mass_mearth: float, radius_rearth: float, wmf: float = 0.0) -> float:
    """Return the core mass fraction implied by a rocky planet's mass and radius (unclipped)."""
    return (1.07 - radius_rearth / ((1 + h.WATER_RADIUS_INFLATION * wmf) * mass_mearth ** (1 / 3.7))) / 0.21


def rocky_wmf(mass_mearth: float, radius_rearth: float, cmf: float) -> float:
    """Return the water mass fraction implied by a rocky planet's mass, radius and core fraction."""
    return (radius_rearth / rocky_radius(mass_mearth, cmf) - 1) / h.WATER_RADIUS_INFLATION


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
        if mass_mearth is not None and radius_rearth is not None and not has_envelope:
            # Too large even for a half-water planet: must hold an envelope.
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
                if not 0.0 <= cmf <= 1.0:
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
