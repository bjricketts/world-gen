"""Tier 0 climate: a zonal energy-balance model with seasons.

Each latitude band (equal area) has a land column and an ocean column that
share heat with each other and with neighbouring bands (moist-static-energy
diffusion). Synchronously rotating planets use bands of substellar angle
instead of latitude and have no seasons. The model is solved directly for
its periodic state as an annual mean plus two harmonics (North et al. 1983),
iterating on the temperature-dependent albedo. It runs in milliseconds and
gives the global state its mean temperature, planetary albedo, ice line and
open-water fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .. import heuristics as h
from .insolation import harmonics, seasonal_insolation, substellar_insolation, synthesise
from .physics import (
    ClimateSetting,
    freeze_fraction,
    moist_slope,
    substellar_albedo_pattern,
    surface_albedo,
    zonal_albedo_pattern,
)

WARM_START_K = 320.0
COLD_START_K = 200.0


@dataclass
class ZonalClimate:
    """Result of the zonal model."""

    coordinate: np.ndarray           # band centres: sine of latitude, or cosine of substellar angle
    ocean_k: np.ndarray              # (samples, bands) ocean-column temperature
    land_k: np.ndarray               # (samples, bands) land-column temperature
    mean_k: float                    # global annual mean surface temperature
    albedo: float                    # planetary albedo, insolation weighted
    ocean_ice: np.ndarray            # (bands,) annual mean sea-ice cover
    open_ocean_fraction: float       # share of the ocean free of ice (annual mean)
    open_water_fraction: float       # share of the whole surface that is open water
    ice_line_deg: float              # latitude (or substellar angle) where permanent sea ice begins; 90 or 180 if none
    dayside_k: Optional[float]       # synchronous planets: day-side and night-side mean temperatures
    nightside_k: Optional[float]
    land_fraction: np.ndarray        # land share of each band
    outgoing: tuple[float, float]    # (a, b): outgoing longwave flux a + b·T used in the solution
    converged: bool
    atmosphere_transport: np.ndarray = None   # (bands − 1,) annual poleward heat flux across band edges, W per R²
    ocean_transport: np.ndarray = None        # (bands − 1,) same for the ocean


def band_centres(bands: int) -> np.ndarray:
    """Return equal-area band centres in the sine of latitude."""
    return -1.0 + (np.arange(bands) + 0.5) * 2.0 / bands


def _diffusion_matrix(x: np.ndarray, conductance: np.ndarray) -> np.ndarray:
    """Return the matrix of d/dx[(1 − x²) K dT/dx] on equal bands; ``conductance`` is K at the inner interfaces."""
    n = x.size
    dx = 2.0 / n
    edges = -1.0 + dx * np.arange(1, n)
    k = conductance * (1.0 - edges**2) / dx**2
    m = np.zeros((n, n))
    idx = np.arange(n - 1)
    m[idx, idx + 1] += k
    m[idx + 1, idx] += k
    m[idx, idx] -= k
    m[idx + 1, idx + 1] -= k
    return m


def solve_zonal(setting: ClimateSetting, land_fraction: float | np.ndarray, start_k: float = WARM_START_K,
                bands: int | None = None, land_albedo: float | None = None,
                initial: Optional["ZonalClimate"] = None, target_mean_k: Optional[float] = None) -> ZonalClimate:
    """Return the periodic climate of a planet with the given land fraction, one value or one per band.

    ``initial`` is a nearby solution to start from instead of a uniform
    ``start_k``. With ``target_mean_k`` the constant term of the outgoing
    radiation is adjusted until the global mean temperature equals the
    target; the result's ``outgoing`` gives the adjusted coefficients.
    """
    n = bands or h.ZONAL_BANDS
    x = band_centres(n)
    f_land = np.clip(np.broadcast_to(np.asarray(land_fraction if setting.has_water else 1.0, dtype=float), (n,)),
                     0.0, 1.0)
    f_ocean = 1.0 - f_land
    land_alb = h.SURFACE_ALBEDO_LAND if land_albedo is None else land_albedo

    if setting.synchronous:
        samples, count = 1, 1
        insolation = substellar_insolation(x, setting.flux_w_m2)[None, :]
        pattern = substellar_albedo_pattern(x)
    else:
        samples, count = h.CLIMATE_TIME_SAMPLES, h.CLIMATE_HARMONICS
        insolation = seasonal_insolation(x, setting.flux_w_m2, setting.obliquity_rad, setting.eccentricity,
                                         setting.periapsis_longitude_rad, samples)
        pattern = zonal_albedo_pattern(x)
    q_mean = float(insolation.mean())

    c_land = float(setting.heat_capacity(np.array(1.0)))
    c_ocean = float(setting.heat_capacity(np.array(0.0)))
    omega = 2.0 * np.pi / setting.year_s
    diffusivity = setting.diffusivity()
    exchange = h.LAND_OCEAN_EXCHANGE * np.sqrt(diffusivity / h.MOIST_DIFFUSIVITY_EARTH)
    mask = setting.cloud_masking()

    reference = setting.reference_k if target_mean_k is None else target_mean_k
    if initial is not None and initial.ocean_k.shape == (samples, n):
        ocean, land = initial.ocean_k.copy(), initial.land_k.copy()
        reference = initial.mean_k
    else:
        ocean = np.full((samples, n), float(start_k))
        land = ocean.copy()
    offset = 0.0
    converged = False

    def albedos(ocean_t, land_t):
        """Return the planetary albedo of the ocean and land columns."""
        free = setting.albedo + offset + pattern
        ao = free + mask * (surface_albedo(0.0, ocean_t, setting.has_water, land_alb) - h.SURFACE_ALBEDO_OCEAN)
        al = free + mask * (surface_albedo(1.0, land_t, setting.has_water, land_alb) - h.SURFACE_ALBEDO_OCEAN)
        return np.clip(ao, 0.0, 0.95), np.clip(al, 0.0, 0.95)

    for relinearise in range(3 if target_mean_k is None else 1):
        a, b = setting.outgoing(reference)
        previous_change = np.inf
        relax = 1.0
        for it in range(h.CLIMATE_MAX_ITERATIONS):
            alb_ocean, alb_land = albedos(ocean, land)
            if setting.albedo_fixed:
                current = float((insolation * (f_ocean * alb_ocean + f_land * alb_land)).mean()) / q_mean
                offset += setting.albedo - current
                alb_ocean, alb_land = albedos(ocean, land)
            if target_mean_k is not None:
                absorbed = float((insolation * (f_ocean * (1.0 - alb_ocean) + f_land * (1.0 - alb_land))).mean())
                a = absorbed - b * target_mean_k
            # Warming reduces ice albedo; including this gain in the solve speeds up convergence.
            up_ocean, up_land = albedos(ocean + 0.5, land + 0.5)
            gain = np.concatenate([(insolation * (alb_ocean - up_ocean)).mean(axis=0) / 0.5,
                                   (insolation * (alb_land - up_land)).mean(axis=0) / 0.5])
            gain = np.clip(gain, 0.0, h.CLIMATE_GAIN_LIMIT * b)
            ocean_flux = ocean_heat_flux(x, setting, f_ocean, ocean, _edge_flux(
                x, _atmosphere_conductance(setting, diffusivity, ocean, land, f_land),
                (f_ocean * ocean + f_land * land).mean(axis=0)))
            factors = _factor(x, ocean, land, f_land, setting, diffusivity, b, exchange, c_land, c_ocean,
                              omega, count, gain)
            state = np.concatenate([ocean, land], axis=1)
            forcing = np.concatenate([insolation * (1.0 - alb_ocean) - a, insolation * (1.0 - alb_land) - a],
                                     axis=1) - gain * state
            if ocean_flux.any():
                forcing[:, :n] += convergence(ocean_flux, n) / np.maximum(f_ocean, 1e-6)
            coeffs = harmonics(forcing, count)
            solved = np.stack([lu_solve(factors[k], coeffs[k]) for k in range(count)])
            new = synthesise(solved, samples)
            change = float(np.abs(new - state).max())
            if change > previous_change:
                relax = max(0.5 * relax, 0.25)
            previous_change = change
            ocean = ocean + relax * (new[:, :n] - ocean)
            land = land + relax * (new[:, n:] - land)
            if change < h.CLIMATE_TOLERANCE_K:
                converged = True
                break
        mean = float(np.mean(f_ocean * ocean + f_land * land))
        if abs(mean - reference) < 3.0:
            break
        reference = mean
    alb_ocean, alb_land = albedos(ocean, land)

    column_mean = f_ocean * ocean + f_land * land
    mean = float(column_mean.mean())
    albedo_total = f_ocean * alb_ocean + f_land * alb_land
    albedo = float((insolation * albedo_total).mean() / q_mean) if q_mean > 0 else float(albedo_total.mean())
    ice = freeze_fraction(ocean, h.SEA_ICE_AIR_THRESHOLD_K).mean(axis=0) if setting.has_water else np.ones(n)
    ocean_share = float(f_ocean.mean())
    open_ocean = float((f_ocean * (1.0 - ice)).mean() / ocean_share) if ocean_share > 0 else 0.0
    dayside = nightside = None
    if setting.synchronous:
        angle = np.degrees(np.arccos(x))
        frozen = (ice > 0.5) | (f_ocean == 0.0)
        ice_line = float(angle[~frozen].max()) if (~frozen).any() else 0.0
        if not frozen.any():
            ice_line = 180.0
        annual = column_mean.mean(axis=0)
        dayside, nightside = float(annual[x > 0].mean()), float(annual[x < 0].mean())
    else:
        lat = np.degrees(np.arcsin(x))
        frozen = (ice > 0.5) & (f_ocean > 0.0)
        ice_line = float(np.abs(lat[frozen]).min()) if frozen.any() else 90.0
    return ZonalClimate(coordinate=x, ocean_k=ocean, land_k=land, mean_k=mean, albedo=albedo, ocean_ice=ice,
                        open_ocean_fraction=open_ocean, open_water_fraction=ocean_share * open_ocean,
                        ice_line_deg=ice_line, dayside_k=dayside, nightside_k=nightside, land_fraction=f_land,
                        outgoing=(a, b), converged=converged,
                        atmosphere_transport=_edge_flux(
                            x, _atmosphere_conductance(setting, diffusivity, ocean, land, f_land),
                            column_mean.mean(axis=0)),
                        ocean_transport=ocean_flux)


def _atmosphere_conductance(setting, diffusivity, ocean, land, f_land) -> np.ndarray:
    """Return the moist diffusivity at the band edges for the current temperatures."""
    annual = ((1.0 - f_land) * ocean + f_land * land).mean(axis=0)
    edge_t = 0.5 * (annual[1:] + annual[:-1])
    return diffusivity * moist_slope(edge_t, setting.pressure_pa, setting.cp, setting.humidity)


def convergence(flux: np.ndarray, n: int) -> np.ndarray:
    """Return the heating (W/m²) of each band from the flux across its edges (W per R²)."""
    padded = np.concatenate([[0.0], flux, [0.0]])
    return -(padded[1:] - padded[:-1]) / (2.0 * np.pi * 2.0 / n)


def ocean_heat_flux(x, setting, f_ocean, ocean, atmosphere_flux) -> np.ndarray:
    """Return the ocean's poleward heat flux at band edges (W per R²).

    The flux has a fixed wind-driven-like shape s(1 − s)³ in latitude (or
    substellar angle) scaled to the sea-ice edge, s = angle / (reach × edge
    angle), so it peaks in the subtropics and carries some heat under the
    ice margin. Its peak is a fixed share of the atmospheric transport,
    reduced where the ocean is narrower than Earth's.
    """
    if not setting.has_water or not f_ocean.any():
        return np.zeros(x.size - 1)
    n = x.size
    edges = -1.0 + 2.0 / n * np.arange(1, n)
    ice = freeze_fraction(ocean, h.SEA_ICE_AIR_THRESHOLD_K).mean(axis=0)
    frozen = (ice > 0.5) & (f_ocean > 0.0)
    peak = h.OCEAN_TRANSPORT_RATIO * float(np.abs(atmosphere_flux).max())
    connected = np.minimum(np.minimum(f_ocean[1:], f_ocean[:-1]) / h.OCEAN_FRACTION_EARTH, 1.0)
    if setting.synchronous:
        angle = np.degrees(np.arccos(edges))                  # from the substellar point
        band_angle = np.degrees(np.arccos(x))
        open_water = ~frozen & (f_ocean > 0.0)
        limit = band_angle[open_water].max() if open_water.any() else 0.0
        if not frozen.any():
            limit = 180.0
        limit = min(limit * h.OCEAN_ICE_REACH, 180.0)
        s = np.clip(angle / max(limit, 1e-6), 0.0, 1.0)
        sign = -1.0                                            # away from the substellar point (toward x = −1)
    else:
        lat = np.degrees(np.arcsin(edges))
        band_lat = np.degrees(np.arcsin(x))
        north = band_lat[(band_lat > 0) & frozen]
        south = band_lat[(band_lat < 0) & frozen]
        limit = np.where(lat >= 0, north.min() if north.size else 90.0, -south.max() if south.size else 90.0)
        limit = np.minimum(limit * h.OCEAN_ICE_REACH, 90.0)
        s = np.clip(np.abs(lat) / np.maximum(limit, 1e-6), 0.0, 1.0)
        sign = np.sign(lat)
    shape = s * (1.0 - s) ** 3 * 256.0 / 27.0
    return sign * peak * shape * connected


def _edge_flux(x, conductance, temperature) -> np.ndarray:
    """Return the poleward heat flux across band edges (W per R², positive northward, or away from the substellar point)."""
    n = x.size
    dx = 2.0 / n
    edges = -1.0 + dx * np.arange(1, n)
    gradient = (temperature[1:] - temperature[:-1]) / dx
    return -2.0 * np.pi * conductance * (1.0 - edges**2) * gradient


def _factor(x, ocean, land, f_land, setting, diffusivity, b, exchange, c_land, c_ocean, omega, count, gain):
    """Return LU factors of the harmonic systems for the current temperature profile and albedo gain."""
    n = x.size
    f_ocean = 1.0 - f_land
    annual = (f_ocean * ocean + f_land * land).mean(axis=0)
    edge_t = 0.5 * (annual[1:] + annual[:-1])
    conductance = diffusivity * moist_slope(edge_t, setting.pressure_pa, setting.cp, setting.humidity)
    lap = _diffusion_matrix(x, conductance)
    factors = []
    for k in range(count):
        m = np.zeros((2 * n, 2 * n), dtype=complex)
        eye = np.eye(n)
        m[:n, :n] = eye * (1j * k * omega * c_ocean + b + exchange * f_land) - lap * f_ocean
        m[:n, n:] = -eye * exchange * f_land - lap * f_land
        m[n:, n:] = eye * (1j * k * omega * c_land + b + exchange * f_ocean) - lap * f_land
        m[n:, :n] = -eye * exchange * f_ocean - lap * f_ocean
        m[np.arange(2 * n), np.arange(2 * n)] -= gain
        factors.append(lu_factor(m))
    return factors
