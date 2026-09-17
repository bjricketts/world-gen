"""Occupiability score: how suitable a planet is for people, unaided or with technology.

Each factor is in [0, 1]. The score is a weighted geometric mean, so one
very hostile factor pulls the score down strongly.
"""

from __future__ import annotations

import math
from typing import Optional

from .. import constants as c
from .. import heuristics as h
from ..state import AtmosphereState, BiosphereState, BulkState, InteriorState, Occupiability, StarState

NO_SURFACE_SCORE = 0.02


def _band(x: float, ideal: tuple[float, float], tolerable: tuple[float, float],
          tolerable_score: float, hostile_score: float) -> float:
    """Return a factor that is 1 inside the ideal band, lower in the tolerable band, lowest outside."""
    if ideal[0] <= x <= ideal[1]:
        return 1.0
    if tolerable[0] <= x <= tolerable[1]:
        return tolerable_score
    return hostile_score


def temperature_factor(t_surface: float) -> float:
    """Return the temperature factor: shirt-sleeve, survivable with suits/habitats, or extreme."""
    return _band(t_surface, (263.0, 310.0), (150.0, 400.0), 0.4, 0.05)


def gravity_factor(g: float) -> float:
    """Return the gravity factor relative to Earth."""
    return _band(g / c.G_EARTH, (0.5, 1.5), (0.15, 3.0), 0.5, 0.1)


def pressure_factor(atmosphere: AtmosphereState) -> float:
    """Return the pressure factor: breathable range, suit/dome range, or crushing."""
    if not atmosphere.present:
        return 0.3
    return _band(atmosphere.surface_pressure_pa / c.BAR, (0.5, 3.0), (0.006, 30.0), 0.5, 0.05)


def water_factor(atmosphere: AtmosphereState) -> float:
    """Return the water availability factor."""
    return {"liquid": 1.0, "ice": 0.6, "vapour": 0.3, "none": 0.2}[atmosphere.surface_water]


def radiation_factor(star: StarState, atmosphere: AtmosphereState, interior: InteriorState) -> float:
    """Return the radiation-shielding factor from magnetic field, atmosphere and stellar activity."""
    shields = int(interior.magnetic_field) + int(atmosphere.present and atmosphere.surface_pressure_pa > 0.1 * c.BAR)
    base = {2: 1.0, 1: 0.6, 0: 0.3}[shields]
    if star.xuv_fraction_relative_sun > h.XUV_HARSH_RATIO:
        base *= 0.7
    return base


def breathable_factor(atmosphere: AtmosphereState, biosphere: Optional[BiosphereState]) -> float:
    """Return the air factor: breathable O₂ partial pressure, tolerable with care, or needing breathing gear."""
    oxygen = biosphere.oxygen_fraction if biosphere is not None else 0.0
    if not atmosphere.present or oxygen <= 0.0:
        return h.BREATHING_GEAR_SCORE
    return _band(oxygen * atmosphere.surface_pressure_pa / c.BAR, h.OXYGEN_BREATHABLE_BAR,
                 h.OXYGEN_TOLERABLE_BAR, 0.6, h.BREATHING_GEAR_SCORE)


def food_factor(productivity: float) -> float:
    """Return the food factor from productivity relative to Earth's (1 at half Earth's or more)."""
    return float(min(max(productivity / 0.5, h.NO_FOOD_SCORE), 1.0))


def biosphere_factor(atmosphere: AtmosphereState, biosphere: Optional[BiosphereState],
                     productivity: Optional[float] = None) -> tuple[float, float]:
    """Return the biosphere factor and its air part.

    ``productivity`` (relative to Earth) replaces the global-state estimate
    once the surface has been built.
    """
    air = breathable_factor(atmosphere, biosphere)
    if productivity is None:
        productivity = biosphere.productivity if biosphere is not None and biosphere.life == "surface" else 0.0
    return math.sqrt(air * food_factor(productivity)), air


def score_planet(star: StarState, bulk: BulkState, interior: InteriorState, atmosphere: AtmosphereState,
                 weights: Optional[dict[str, float]] = None, biosphere: Optional[BiosphereState] = None,
                 productivity: Optional[float] = None) -> Occupiability:
    """Return the occupiability score and its factors.

    ``productivity`` is the land productivity relative to Earth measured on
    a built surface; without it the global-state estimate is used.
    """
    w = {**h.DEFAULT_OCCUPIABILITY_WEIGHTS, **(weights or {})}
    unknown = set(w) - set(h.DEFAULT_OCCUPIABILITY_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown occupiability weights: {', '.join(sorted(unknown))}")

    if interior.tectonic_regime == "fluid":
        return Occupiability(score=NO_SURFACE_SCORE, factors={"surface": 0.0}, habitable_unaided=False)

    factors = {
        "temperature": temperature_factor(atmosphere.surface_temperature_k),
        "gravity": gravity_factor(bulk.surface_gravity_m_s2),
        "pressure": pressure_factor(atmosphere),
        "water": water_factor(atmosphere),
        "radiation": radiation_factor(star, atmosphere, interior),
    }
    factors["biosphere"], air = biosphere_factor(atmosphere, biosphere, productivity)
    total_w = sum(w.values())
    score = math.exp(sum(w[k] * math.log(v) for k, v in factors.items()) / total_w) if total_w > 0 else 0.0
    unaided = all(factors[k] == 1.0 for k in ("temperature", "gravity", "pressure", "water")) and air == 1.0
    return Occupiability(score=score, factors=factors, habitable_unaided=unaided, weights=weights)
