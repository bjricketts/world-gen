"""Suitability of the land for a culture: per-cell carrying capacity from the physical surface.

Capacity is log-linear in the covariates, ln K = Σ terms, with terms set by
the culture's energy level (see ``heuristics``): climatic productivity, fresh
water (river discharge), closeness to the coast, elevation and heat stress,
and for colonists resources and ice. Foragers follow the productivity response
of Tallavaara et al. (2018); the modern set is fitted to Earth's population
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import heuristics as h
from ..surface.distance import distance_from
from ..surface.fields import Terrain
from .habitats import habitat_capacity, habitat_classes
from .model import Culture

COAST_LIMIT_KM = 3000.0


@dataclass
class Covariates:
    """Per-cell inputs of the carrying capacity."""

    land: np.ndarray                  # not ocean
    ice: np.ndarray                   # covered by an ice sheet
    coast_km: np.ndarray              # from the cell centre to the coastline, half a spacing inside the nearest sea cell
    discharge: np.ndarray             # m³/s
    elevation: np.ndarray             # m
    temperature_c: np.ndarray         # annual mean
    productivity: np.ndarray          # g m⁻² yr⁻¹
    resources: np.ndarray             # mean potential of the colonist resource classes, 0–1
    cell_area_km2: float
    spacing_km: float
    radius_km: float
    land_life: bool
    habitat_share: np.ndarray = None     # capacity share a colonist habitat keeps (0: not habitable)
    lat_deg: np.ndarray = None
    rough: np.ndarray = None             # landform other than plain


def covariates(world, resource_potential=None) -> Covariates:
    """Return the covariates of every cell of a world's surface."""
    s, grid = world.surface, world.grid
    radius_km = float(s.attrs["radius_m"]) / 1e3
    ocean = s["ocean"].values.astype(bool)
    spacing = grid.mean_spacing(radius_km)
    dist, _ = distance_from(grid, np.flatnonzero(ocean), radius_km, COAST_LIMIT_KM)
    coast = np.maximum(np.where(np.isfinite(dist), dist, COAST_LIMIT_KM) - spacing / 2, 0.0)
    if resource_potential is not None:
        names = [r for r in h.COLONIST_RESOURCES if r in resource_potential["resource"].values]
        resources = resource_potential.sel(resource=names).values.mean(axis=0)
    else:
        resources = np.zeros(grid.size)
    bio = world.state.biosphere if world.state is not None else None
    return Covariates(
        land=~ocean,
        ice=s["ice_thickness"].values > 0 if "ice_thickness" in s else np.zeros(grid.size, dtype=bool),
        coast_km=coast,
        discharge=s["discharge"].values.astype(float) if "discharge" in s else np.zeros(grid.size),
        elevation=s["elevation"].values.astype(float),
        temperature_c=s["air_temperature"].values.astype(float) - 273.15,
        productivity=s["productivity"].values.astype(float) if "productivity" in s else np.zeros(grid.size),
        resources=resources,
        cell_area_km2=4 * np.pi * radius_km**2 / grid.size,
        spacing_km=spacing,
        radius_km=radius_km,
        land_life=bio is not None and bio.life == "surface",
        habitat_share=habitat_capacity(world, habitat_classes(world)) if world.state is not None
        else np.where(~ocean, 1.0, 0.0),
        lat_deg=grid.lat,
        rough=s["terrain"].values != Terrain.PLAIN if "terrain" in s else np.zeros(grid.size, dtype=bool),
    )


def density_terms(culture: Culture) -> dict:
    """Return the capacity terms for a culture's inhabitants and energy level."""
    if culture.inhabitants == "colonist":
        return dict(h.DENSITY_TERMS_COLONIST)
    w = min(culture.development.energy / 4.0, 1.0)
    a, b = h.DENSITY_TERMS_FORAGING, h.DENSITY_TERMS_MODERN
    return {k: (1 - w) * a[k] + w * b[k] for k in b}


def log_capacity(cov: Covariates, terms: dict) -> np.ndarray:
    """Return ln of the relative carrying capacity per cell; −inf where the culture cannot live."""
    npp = np.clip(cov.productivity, 1.0, h.DENSITY_NPP_PLATEAU)
    ln = terms["log_npp"] * np.log(npp / h.DENSITY_NPP_PLATEAU)
    ln += terms["linear_npp"] * (npp - h.DENSITY_NPP_PLATEAU)
    ln += terms["discharge"] * np.log1p(np.maximum(cov.discharge, 0.0) / h.DENSITY_DISCHARGE_SCALE_M3S)
    ln += terms["coast"] * np.exp(-cov.coast_km / terms["coast_km"])
    ln -= np.maximum(cov.elevation, 0.0) / terms["elevation_m"]
    ln -= terms["heat"] * np.maximum(cov.temperature_c - terms["heat_from_c"], 0.0)
    ln += terms.get("resources", 0.0) * cov.resources
    livable = cov.land.copy()
    if "ice" in terms:
        ln += terms["ice"] * cov.ice
    else:
        livable &= ~cov.ice
    if terms.get("habitat") and cov.habitat_share is not None:
        livable &= cov.habitat_share > 0
        ln += np.log(np.where(cov.habitat_share > 0, cov.habitat_share, 1.0))
    return np.where(livable, ln, -np.inf)


def productive_area_km2(cov: Covariates) -> np.ndarray:
    """Return each cell's productive area: its area scaled by productivity up to the plateau (0 on ice and sea)."""
    share = np.clip(cov.productivity, 0.0, h.DENSITY_NPP_PLATEAU) / h.DENSITY_NPP_PLATEAU
    return np.where(cov.land & ~cov.ice, cov.cell_area_km2 * share, 0.0)
