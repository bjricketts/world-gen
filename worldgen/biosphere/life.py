"""Whether a planet has life, what kind, and what it does to the air.

Unless the spec says otherwise, planets older than
``LIFE_MIN_PLANET_AGE_GYR`` with liquid surface water have surface life
(ocean-only life if there is almost no land), and frozen water worlds have
subsurface life. Oxygenic biospheres raise O₂ in two steps with biosphere
age, as Earth's did; methanogenic ones add methane. The colour of
vegetation follows the star's light (``pigment.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..state import AtmosphereState, BiosphereState, Issue, StarState, WaterState
from ..util import logistic
from .pigment import absorption_peak, pigment_colour
from .vegetation import light_factor, miami_productivity


@dataclass
class LifeInputs:
    """Spec values for the biosphere; ``None`` means use the default rules."""

    life: Optional[str] = None
    biochemistry: Optional[str] = None
    alien_product: Optional[str] = None
    age_gyr: Optional[float] = None
    optimum_temperature_k: Optional[float] = None
    temperature_tolerance_k: Optional[float] = None
    pigment_absorption_nm: Optional[float] = None
    oxygen_fraction: Optional[float] = None
    weathering_target_is_user: bool = False


def default_life(planet_age_gyr: float, surface_water: str, land_fraction: float) -> str:
    """Return the life a planet has by default: 'surface', 'ocean', 'subsurface' or 'none'."""
    if planet_age_gyr < h.LIFE_MIN_PLANET_AGE_GYR:
        return "none"
    if surface_water == "liquid":
        return "surface" if land_fraction >= h.LIFE_MIN_LAND_FRACTION else "ocean"
    if surface_water == "ice":
        return "subsurface"
    return "none"


def oxygen_level(biosphere_age_gyr: float) -> float:
    """Return the O₂ mole fraction made by a mature oxygenic surface biosphere of the given age."""
    first_age, first_width, first_share = h.OXYGEN_FIRST_RISE
    second_age, second_width = h.OXYGEN_SECOND_RISE
    first = first_share * logistic((biosphere_age_gyr - first_age) / first_width)
    second = (1.0 - first_share) * logistic((biosphere_age_gyr - second_age) / second_width)
    return h.OXYGEN_EARTH * (first + second)


def product_of(biochemistry: str, alien_product: Optional[str]) -> str:
    """Return the gas a biochemistry releases."""
    if biochemistry == "oxygenic":
        return "o2"
    if biochemistry == "methanogenic":
        return "ch4"
    return alien_product or "none"


def expected_productivity(land_temperatures_k: np.ndarray, frozen: np.ndarray, mean_k: float,
                          optimum_k: float, tolerance_k: float, instellation_earth: float) -> float:
    """Return land productivity relative to Earth's, from zonal land temperatures (equal-area bands)."""
    rain = h.EVAPORATION_EARTH_M * h.LAND_RAIN_SHARE * math.exp(h.HYDROLOGY_SENSITIVITY * (mean_k - c.T_SURFACE_EARTH))
    npp = miami_productivity(land_temperatures_k, np.full(land_temperatures_k.shape, rain), optimum_k, tolerance_k)
    npp = np.where(frozen, 0.0, npp) * light_factor(instellation_earth)
    return float(npp.mean() / h.EARTH_ZONAL_NPP)


def build_biosphere(inputs: LifeInputs, star: StarState, instellation_earth: float, atmosphere: AtmosphereState,
                    water: WaterState, zonal, planet_age_gyr: float) -> tuple[Optional[BiosphereState], list[Issue],
                                                                              dict[str, str]]:
    """Return the biosphere of a planet (``None`` without life), with notes and provenance.

    ``zonal`` is the Tier 0 climate (or ``None`` for planets without a surface climate).
    """
    issues: list[Issue] = []
    prov: dict[str, str] = {}
    land = water.land_fraction_estimate if water is not None else 1.0
    life = inputs.life
    if life is None:
        life = default_life(planet_age_gyr, atmosphere.surface_water, land)
        prov["biosphere.life"] = "heuristic"
    elif life == "surface" and atmosphere.surface_water != "liquid":
        issues.append(Issue("warning", "conflict", "biosphere",
                            f"surface life requested but the surface water is '{atmosphere.surface_water}'"))
    if life == "none":
        return None, issues, prov

    biochemistry = inputs.biochemistry or "oxygenic"
    product = product_of(biochemistry, inputs.alien_product)
    age = inputs.age_gyr if inputs.age_gyr is not None else max(planet_age_gyr - h.LIFE_ORIGIN_DELAY_GYR, 0.0)
    if inputs.age_gyr is None:
        prov["biosphere.age_gyr"] = "heuristic"
    optimum = inputs.optimum_temperature_k or h.EARTH_OPTIMUM_K
    tolerance = inputs.temperature_tolerance_k or h.EARTH_TOLERANCE_K

    photosynthetic = life in ("surface", "ocean")
    oxygen = 0.0
    methane = 0.0
    if photosynthetic and product == "o2":
        oxygen = oxygen_level(age) * (1.0 if life == "surface" else h.OCEAN_ONLY_OXYGEN_SHARE)
    if product == "ch4" and life != "none":
        methane = h.METHANE_BIOTIC * logistic((age - 0.5) / 0.2)
    if inputs.oxygen_fraction is not None:
        oxygen = inputs.oxygen_fraction
    else:
        prov["biosphere.oxygen_fraction"] = "heuristic"

    peak = inputs.pigment_absorption_nm or absorption_peak(star.effective_temperature_k)
    if inputs.pigment_absorption_nm is None:
        prov["biosphere.pigment_absorption_nm"] = "derived"
    colour, brightness = pigment_colour(peak, star.effective_temperature_k)
    earth_brightness = pigment_colour(absorption_peak(c.T_SUN), c.T_SUN)[1]
    canopy = h.CANOPY_ALBEDO_EARTH * brightness / earth_brightness

    productivity = 0.0
    if life == "surface" and zonal is not None:
        land_t = zonal.land_k.mean(axis=0)
        frozen = zonal.land_k.max(axis=0) < 273.15
        productivity = expected_productivity(land_t, frozen, zonal.mean_k, optimum, tolerance, instellation_earth)
    cooling = h.BIOTIC_WEATHERING_COOLING_K if life == "surface" and not inputs.weathering_target_is_user else 0.0

    issues.append(Issue("info", "heuristic", "biosphere",
                        f"{life} life ({biochemistry}) from rules on water, temperature and age; "
                        "O₂ follows an Earth-like oxygenation history"))
    state = BiosphereState(
        life=life, biochemistry=biochemistry, product=product, age_gyr=age, oxygen_fraction=oxygen,
        methane_fraction=methane, optimum_temperature_k=optimum, temperature_tolerance_k=tolerance,
        pigment_absorption_nm=peak, pigment_colour=colour, vegetation_albedo=canopy, productivity=productivity,
        weathering_cooling_k=cooling)
    return state, issues, prov


def methane_optical_depth(methane_fraction: float) -> float:
    """Return the extra greenhouse optical depth from biotic methane."""
    if methane_fraction <= 0.0:
        return 0.0
    return h.METHANE_OPTICAL_DEPTH * math.sqrt(methane_fraction / h.METHANE_BIOTIC)


def tier0_land_albedo(biosphere: Optional[BiosphereState]) -> float:
    """Return the snow-free land albedo the zonal climate model uses."""
    if biosphere is None or biosphere.life != "surface":
        return h.SURFACE_ALBEDO_LAND
    cover = min(biosphere.productivity, 1.0) * h.TIER0_COVER_PER_PRODUCTIVITY
    return h.SURFACE_ALBEDO_LAND + (biosphere.vegetation_albedo - h.SURFACE_ALBEDO_LAND) * cover
