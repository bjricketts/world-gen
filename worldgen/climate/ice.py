"""Ice sheets: extent from surface mass balance, thickness from a plastic profile, and snowline erosion.

Snow accumulates in months below freezing; melt follows the positive
degree-day method (Braithwaite 1995), with monthly temperatures spread by
a normal distribution to count the warm days of a cool month. Land with a
positive annual balance is glaciated. Ice thickness follows the perfectly
plastic profile h = √(2 τ₀ L / (ρ_i g)) with L the distance to the ice
margin (Nye 1952); the ice surface stands above the bedrock by the
thickness less its isostatic sinking. The equilibrium-line altitude (ELA),
where the balance is zero, caps mountains: land far above it is worn down
toward it ("glacial buzzsaw"; Egholm et al. 2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from .. import heuristics as h
from ..grid import SphereGrid
from .physics import FREEZING_K

ICE_DENSITY = 917.0
DAYS_PER_MONTH = 365.25 / 12.0
ELA_LEVELS_M = np.arange(0.0, 12_001.0, 500.0)


@dataclass
class IceSheets:
    """Land ice on the surface grid."""

    thickness_m: np.ndarray          # 0 outside ice sheets
    surface_rise_m: np.ndarray       # ice surface above the bedrock, after isostatic sinking
    balance_m: np.ndarray            # annual surface mass balance at the ice surface (m water per year)
    ela_m: np.ndarray                # equilibrium-line altitude above sea level (inf where ice cannot form)

    @property
    def glaciated(self) -> np.ndarray:
        """Return the cells covered by ice sheets."""
        return self.thickness_m > 0.0


def positive_degree_days(monthly_k: np.ndarray) -> np.ndarray:
    """Return the expected positive degree-days per year (K·day) from monthly mean temperatures (axis 0)."""
    sigma = h.PDD_TEMPERATURE_SPREAD_K
    mu = monthly_k - FREEZING_K
    expected = sigma / np.sqrt(2 * np.pi) * np.exp(-0.5 * (mu / sigma) ** 2) + 0.5 * mu * erfc(-mu / (np.sqrt(2) * sigma))
    return expected.sum(axis=0) * DAYS_PER_MONTH


def snow_share(monthly_k: np.ndarray) -> np.ndarray:
    """Return the share of precipitation falling as snow at each monthly temperature."""
    return 1.0 / (1.0 + np.exp((monthly_k - FREEZING_K - h.SNOW_RAIN_THRESHOLD_C) / 1.0))


def mass_balance(monthly_k: np.ndarray, monthly_p_m: np.ndarray, lowering_k: np.ndarray | float = 0.0) -> np.ndarray:
    """Return the annual surface mass balance (m water per year).

    ``monthly_p_m`` holds precipitation as annual rates; ``lowering_k`` is
    subtracted from the temperatures (height above the reference surface).
    """
    t = monthly_k - lowering_k
    accumulation = (monthly_p_m * snow_share(t)).mean(axis=0)
    melt = h.DEGREE_DAY_FACTOR_M * positive_degree_days(t)
    return accumulation - melt


def equilibrium_line(monthly_k: np.ndarray, monthly_p_m: np.ndarray, ground_m: np.ndarray,
                     cells: np.ndarray | None = None) -> np.ndarray:
    """Return the height above sea level where the mass balance turns positive (inf where it never does).

    ``monthly_k`` is the temperature at ``ground_m``. Precipitation is taken
    as independent of height. Only ``cells`` (a mask) are evaluated; the
    rest get inf.
    """
    idx = np.arange(ground_m.size) if cells is None else np.flatnonzero(cells)
    sea_level_k = monthly_k[:, idx] + h.LAPSE_RATE_K_PER_M * ground_m[idx]
    p = monthly_p_m[:, idx]
    out = np.full(idx.size, np.inf)
    previous = None
    for z in ELA_LEVELS_M:
        b = mass_balance(sea_level_k, p, h.LAPSE_RATE_K_PER_M * z)
        if previous is None:
            out[b > 0.0] = 0.0
        else:
            cross = (b > 0.0) & ~np.isfinite(out)
            # The balance rises from previous ≤ 0 to b > 0 across this step; the snowline is where it is zero.
            frac = -previous[cross] / np.maximum(b[cross] - previous[cross], 1e-12)
            out[cross] = z - (ELA_LEVELS_M[1] - ELA_LEVELS_M[0]) * (1.0 - np.clip(frac, 0.0, 1.0))
        previous = b
    ela = np.full(ground_m.size, np.inf)
    ela[idx] = out
    return ela


def build_ice_sheets(grid: SphereGrid, radius_m: float, ground_m: np.ndarray, ocean: np.ndarray,
                     monthly_k: np.ndarray, monthly_p_m: np.ndarray, gravity_m_s2: float,
                     previous: IceSheets | None = None) -> IceSheets:
    """Return the ice sheets of a surface from its monthly climate.

    ``ground_m`` is the bedrock height above sea level (0 over the ocean).
    ``monthly_k`` is the temperature at the surface the climate was solved
    for: the bedrock raised by the ``previous`` ice, if any.
    """
    from ..surface.distance import distance_from   # imported here: the surface package builds on this one

    land = ~ocean
    rise_before = previous.surface_rise_m if previous is not None else np.zeros(grid.size)
    rise = rise_before.copy()
    thickness = np.zeros(grid.size)
    ela = equilibrium_line(monthly_k, monthly_p_m, ground_m + rise_before, land)
    for _ in range(h.ICE_HEIGHT_ITERATIONS):
        # Temperatures follow the ice surface: height above the surface the climate saw lowers them.
        balance = mass_balance(monthly_k, monthly_p_m, h.LAPSE_RATE_K_PER_M * (rise - rise_before))
        ice = land & (balance > 0.0)
        if not ice.any():
            thickness[:] = 0.0
            rise[:] = 0.0
            break
        margin = np.flatnonzero(~ice)
        dist_km, _ = distance_from(grid, margin, radius_m / 1e3)
        spacing_km = grid.mean_spacing(radius_m / 1e3)
        length = (np.where(ice, dist_km, 0.0) + 0.5 * spacing_km) * 1e3
        thickness = np.where(ice, np.sqrt(2.0 * h.ICE_YIELD_STRESS_PA * length / (ICE_DENSITY * gravity_m_s2)), 0.0)
        rise = thickness * (1.0 - h.ICE_ISOSTATIC_SINKING)
    balance = mass_balance(monthly_k, monthly_p_m, h.LAPSE_RATE_K_PER_M * (rise - rise_before))
    return IceSheets(thickness_m=thickness, surface_rise_m=rise, balance_m=balance, ela_m=ela)


def buzzsaw(height_m: np.ndarray, ocean: np.ndarray, ela_m: np.ndarray, duration_myr: float) -> np.ndarray:
    """Return the land height after glacial erosion toward the snowline over ``duration_myr``.

    Land above ``ela_m + GLACIAL_PEAK_OFFSET_M`` erodes at a rate that grows
    with its excess height x: dx/dt = −x² / (L·τ). Peaks are worn down
    without a fixed ceiling, so under fast uplift they settle higher,
    at x ≈ √(U·L·τ).
    """
    base = ela_m + h.GLACIAL_PEAK_OFFSET_M
    excess = np.where(~ocean & np.isfinite(base), np.maximum(height_m - base, 0.0), 0.0)
    rate = duration_myr / (h.GLACIAL_EROSION_SCALE_M * h.GLACIAL_EROSION_TIMESCALE_MYR)
    return height_m - excess + excess / (1.0 + excess * rate)


def seasonal_temperatures(annual_k: np.ndarray, amplitude_k: np.ndarray, northern: np.ndarray,
                          months: int = 12) -> np.ndarray:
    """Return monthly temperatures from annual means and a cosine seasonal cycle (coldest in month 1 in the north)."""
    phase = np.cos(2 * np.pi * (np.arange(months) + 0.5) / months)[:, None]
    sign = np.where(northern, -1.0, 1.0)[None, :]
    return annual_k[None, :] + sign * phase * amplitude_k[None, :]


__all__ = ["IceSheets", "build_ice_sheets", "buzzsaw", "equilibrium_line", "mass_balance",
           "positive_degree_days", "seasonal_temperatures"]
