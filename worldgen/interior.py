"""Interior heat budget, tectonic regime and magnetic dynamo for rocky planets.

Radiogenic heating uses measured isotope data. The tectonic regime and dynamo
have no standard predictive formula, so they are drawn from calibrated
heuristic probabilities (see ``heuristics.py``) unless the user fixes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constants as c
from . import heuristics as h
from .state import BulkState, InteriorState, Issue
from .util import choose, log_threshold

GIANT_CLASSES = ("ice giant", "gas giant", "brown dwarf")
REGIMES = ("mobile_lid", "stagnant_lid", "episodic", "heat_pipe", "inactive")


@dataclass(frozen=True)
class Isotope:
    """A heat-producing isotope."""

    name: str
    heat_w_per_kg: float        # heat production per kg of the isotope
    half_life_gyr: float
    element_fraction: float     # mass fraction of the isotope in its element
    bse_element_ppm: float      # element concentration in Earth's bulk silicate [ppm by mass]


# Heat production and half-lives: Turcotte & Schubert (2002), Table 4.2.
# Bulk-silicate-Earth concentrations: McDonough & Sun (1995).
ISOTOPES = (
    Isotope("U238", 9.46e-5, 4.47, 0.9928, 0.0203),
    Isotope("U235", 5.69e-4, 0.704, 0.0071, 0.0203),
    Isotope("Th232", 2.64e-5, 14.0, 1.0, 0.0795),
    Isotope("K40", 2.92e-5, 1.25, 1.19e-4, 240.0),
)


def radiogenic_heat_per_kg(age_gyr: float, abundance: float = 1.0) -> float:
    """Return silicate heat production (W/kg) at a planet age, for Earth-like abundances scaled by ``abundance``."""
    total = 0.0
    for iso in ISOTOPES:
        present = iso.heat_w_per_kg * iso.element_fraction * iso.bse_element_ppm * 1e-6
        total += present * math.exp(math.log(2) * (c.AGE_SUN_GYR - age_gyr) / iso.half_life_gyr)
    return abundance * total


def silicate_mass(bulk: BulkState) -> float:
    """Return the mass (kg) of the rocky mantle and crust."""
    frac = 1.0 - bulk.core_mass_fraction - bulk.water_mass_fraction - bulk.envelope_mass_fraction
    return max(frac, 0.0) * bulk.mass_kg


def surface_heat_flux(radiogenic_w: float, tidal_w: float, radius_m: float) -> float:
    """Return the mean surface heat flux (W/m²) from radiogenic and tidal power."""
    return (radiogenic_w * h.INVERSE_UREY_RATIO + tidal_w) / (4 * math.pi * radius_m**2)


def _earth_heat_flux() -> float:
    """Return the model's present-day surface heat flux for Earth (W/m²)."""
    mantle = c.M_EARTH * (1 - c.EARTH_CMF)
    return surface_heat_flux(mantle * radiogenic_heat_per_kg(c.AGE_SUN_GYR), 0.0, c.R_EARTH)


def activity_index(heat_flux: float, mass_kg: float) -> float:
    """Return the interior activity index, 1 for present-day Earth."""
    return (heat_flux / _earth_heat_flux()) * (mass_kg / c.M_EARTH) ** h.ACTIVITY_MASS_EXPONENT


def regime_probabilities(activity: float, flux_ratio: float, mass_kg: float, surface_temperature_k: float,
                         surface_water: bool) -> dict[str, float]:
    """Return heuristic probabilities for each tectonic regime.

    ``flux_ratio`` is the surface heat flux relative to Earth's; ``surface_water``
    is whether liquid water or ice is present at the surface.
    """
    p_dead = 1.0 - log_threshold(activity, h.ACTIVITY_DEAD_BELOW, h.ACTIVITY_LOG_WIDTH)
    p_heatpipe = log_threshold(flux_ratio, h.HEATPIPE_FLUX_RATIO_ABOVE, h.ACTIVITY_LOG_WIDTH)
    p_active = max(1.0 - p_dead - p_heatpipe, 0.0)

    vigour = log_threshold(activity, h.ACTIVITY_PLATES_ABOVE, h.ACTIVITY_PLATES_LOG_WIDTH)
    hot = surface_temperature_k > h.HOT_SURFACE_K
    w_mobile = (vigour
                * (1.0 if surface_water else h.PLATES_DRY_FACTOR)
                * (h.PLATES_HOT_FACTOR if hot else 1.0)
                * (mass_kg / c.M_EARTH) ** h.PLATES_MASS_EXPONENT)
    w_episodic = h.EPISODIC_BASE * vigour * (1.0 if hot else h.EPISODIC_COOL_FACTOR)
    w_stagnant = (1.0 - vigour) + h.STAGNANT_BASE
    total = w_mobile + w_episodic + w_stagnant

    probs = {
        "mobile_lid": p_active * w_mobile / total,
        "stagnant_lid": p_active * w_stagnant / total,
        "episodic": p_active * w_episodic / total,
        "heat_pipe": p_heatpipe,
        "inactive": p_dead,
    }
    norm = sum(probs.values())
    return {k: v / norm for k, v in probs.items()}


def dynamo_probability(regime: str, activity: float, cmf: float) -> float:
    """Return the heuristic probability of an active core dynamo."""
    if cmf < h.DYNAMO_MIN_CMF or activity < h.DYNAMO_MIN_ACTIVITY:
        return h.DYNAMO_P_INACTIVE
    if regime == "mobile_lid":
        return h.DYNAMO_P_PLATES
    if regime in ("stagnant_lid", "episodic", "heat_pipe"):
        return h.DYNAMO_P_OTHER_ACTIVE
    return h.DYNAMO_P_INACTIVE


def build_interior(
    bulk: BulkState,
    age_gyr: float,
    radiogenic_abundance: float,
    tidal_power_w: float,
    surface_temperature_k: float,
    surface_water: bool,
    regime_override: Optional[str],
    dynamo_override: Optional[bool],
    regime_rng: np.random.Generator,
    dynamo_rng: np.random.Generator,
) -> tuple[InteriorState, list[Issue], dict[str, str]]:
    """Return the interior state, notes, and provenance of drawn outcomes."""
    issues: list[Issue] = []
    prov: dict[str, str] = {}

    if bulk.planet_class in GIANT_CLASSES:
        state = InteriorState(
            radiogenic_power_w=0.0, tidal_power_w=tidal_power_w, surface_heat_flux_w_m2=0.0,
            activity_index=0.0, tectonic_regime="fluid", regime_probabilities={"fluid": 1.0},
            magnetic_field=True if dynamo_override is None else dynamo_override,
            dynamo_probability=1.0,
        )
        prov["interior.tectonic_regime"] = "derived"
        if dynamo_override is None:
            prov["interior.magnetic_field"] = "derived"
        issues.append(Issue("info", "note", "interior", "giant planet: no solid surface or tectonics"))
        return state, issues, prov

    radiogenic = silicate_mass(bulk) * radiogenic_heat_per_kg(age_gyr, radiogenic_abundance)
    flux = surface_heat_flux(radiogenic, tidal_power_w, bulk.radius_m)
    activity = activity_index(flux, bulk.mass_kg)
    probs = regime_probabilities(activity, flux / _earth_heat_flux(), bulk.mass_kg,
                                 surface_temperature_k, surface_water)

    if regime_override is not None:
        regime = regime_override
        if probs.get(regime, 0.0) < 0.05:
            issues.append(Issue("warning", "conflict", "interior",
                                f"tectonic regime '{regime}' is unlikely here (p = {probs.get(regime, 0.0):.2f})"))
    else:
        regime = choose(regime_rng, probs)
        prov["interior.tectonic_regime"] = "heuristic"
        issues.append(Issue("info", "heuristic", "interior",
                            "tectonic regime drawn from heuristic probabilities"))

    p_dynamo = dynamo_probability(regime, activity, bulk.core_mass_fraction)
    if dynamo_override is not None:
        field = dynamo_override
    else:
        field = bool(dynamo_rng.random() < p_dynamo)
        prov["interior.magnetic_field"] = "heuristic"
        issues.append(Issue("info", "heuristic", "interior",
                            "magnetic dynamo drawn from heuristic probability"))

    state = InteriorState(
        radiogenic_power_w=radiogenic,
        tidal_power_w=tidal_power_w,
        surface_heat_flux_w_m2=flux,
        activity_index=activity,
        tectonic_regime=regime,
        regime_probabilities=probs,
        magnetic_field=field,
        dynamo_probability=p_dynamo,
    )
    return state, issues, prov
