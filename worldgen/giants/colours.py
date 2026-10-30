"""Colours of cold giants from the composition of their atmospheres.

Classes I and II (Sudarsky et al. 2000) are coloured by a layered reflectance
model calibrated on the four Solar System giants:

1. Heavy elements. The carbon enrichment follows the Solar System mass trend
   log ε(C) = 8.84 − 1.13 log10(M/M_J) (Lothringer et al. 2025), shifted by the
   star's [Fe/H] and a seeded scatter; N and O scale with C from the protosolar
   abundances (Asplund et al. 2009, Table 5). ``body.atmospheric_metallicity``
   overrides it.
2. Temperature. The analytic radiative–convective profile of Robinson &
   Catling (2012) with their Jupiter parameters, the optical depth scaled as
   p²/g, and the adiabat's α lowered for metal-rich envelopes (heuristic).
3. Clouds. NH3, H2O and CH4 condense where their partial pressure reaches the
   saturation pressure (NH3: Ackerman & Marley 2001 eq. A1; H2O: Buck 1981; CH4:
   the CH4 I fit of Grundy et al. 2023, Table 2, valid to the triple point
   90.7 K). The uppermost deck is the visible cloud.
4. Reflectance. Delta-Eddington two-stream layers with H2
   Rayleigh scattering (Lecavelier des Etangs et al. 2008), CH4 absorption
   (Karkoschka 1994), an extended upper haze and a haze at the methane level
   (Irwin et al. 2022), Saturn's tropospheric haze above the ammonia cloud
   (Barstow et al. 2016) and Jupiter's chromophore layer (Braude et al. 2020),
   over a bright cloud. The optical tables are built by
   ``scripts/build_giant_optics.py``.

The haze amounts are laws through two planets each (heuristic): the
tropospheric haze grows with the depth of the ammonia cloud, from Jupiter to
Saturn, and the methane-level haze falls with internal heat flux, from Uranus
to Neptune. Classes III–V keep the Cahoy et al. (2010) shapes, darkened with
metallicity at the rate their 1×/3× and 10×/30× models give.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, gammaincc

from .. import constants as c
from .. import heuristics as h
from ..biosphere.pigment import WAVELENGTHS_NM, reflected_colour
from ..util import named_rng

# Protosolar abundances log ε (Asplund et al. 2009, Table 5).
PROTOSOLAR_LOG_EPS = {"C": 8.47, "N": 7.87, "O": 8.73, "He": 10.98}
# Robinson & Catling (2012) Jupiter fit: reference pressure (bar), gray optical depth there, pressure
# exponent, ratio of specific heats, two shortwave channels (absorption relative to the gray opacity,
# share of the absorbed flux), diffusivity factor.
RC_P0_BAR = 1.1
RC_TAU0 = 6.0
RC_GRAVITY_M_S2 = 24.79
RC_N = 2.0
RC_GAMMA = 1.4
RC_K = (100.0, 0.06)
RC_SHARE_1 = 1.3 / 8.3
RC_D = 1.66
H2_RAYLEIGH_M2 = 2.3e-31            # at 430 nm, ∝ λ⁻⁴ (Lecavelier des Etangs et al. 2008)
CH4_TRIPLE_K = 90.7


@lru_cache(maxsize=1)
def optics() -> dict[str, np.ndarray]:
    """Return the optical tables (wavelength grid, CH4 cross-sections, aerosol Mie properties)."""
    with np.load(Path(__file__).parents[1] / "data" / "giant_optics.npz") as data:
        return {k: data[k] for k in data.files}


# --- composition --------------------------------------------------------------------------------

def atmospheric_metallicity(mass_kg: float, metallicity_feh: float, seed: int) -> float:
    """Return the heavy-element enrichment of the envelope relative to protosolar (C/H)."""
    log_eps = h.GIANT_METALLICITY_LOG_EPS_MJ + h.GIANT_METALLICITY_SLOPE * math.log10(mass_kg / c.M_JUPITER)
    log_eps += metallicity_feh + h.GIANT_METALLICITY_SCATTER_DEX * float(named_rng(seed, "giant.metallicity").normal())
    return 10 ** (log_eps - PROTOSOLAR_LOG_EPS["C"])


def mole_fractions(metallicity: float) -> dict[str, float]:
    """Return the mole fractions of CH4, NH3, H2O, H2 and He in an H2–He envelope."""
    he = 10 ** (PROTOSOLAR_LOG_EPS["He"] - 12)
    total = 0.5 + he                                        # molecules per H atom
    x = {gas: 10 ** (PROTOSOLAR_LOG_EPS[el] - 12) * metallicity / total
         for gas, el in (("CH4", "C"), ("NH3", "N"), ("H2O", "O"))}
    x["H2"], x["He"] = 0.5 / total, he / total
    return x


# --- temperature profile ------------------------------------------------------------------------

def adiabat_alpha(metallicity: float) -> float:
    """Return the empirical adiabat factor α of Robinson & Catling (2012), lowered for metal-rich envelopes."""
    lo, hi = h.GIANT_ALPHA_METALLICITY
    f = float(np.clip(math.log10(metallicity / lo) / math.log10(hi / lo), 0.0, 1.0))
    return h.GIANT_ALPHA_SOLAR + f * (h.GIANT_ALPHA_RICH - h.GIANT_ALPHA_SOLAR)


def temperature_profile(absorbed_w_m2: float, internal_w_m2: float, gravity_m_s2: float,
                        alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return T(p) (K, p in bar) of the Robinson & Catling (2012) radiative–convective model."""
    tau0 = RC_TAU0 * RC_GRAVITY_M_S2 / gravity_m_s2
    f1, f2 = RC_SHARE_1 * absorbed_w_m2, (1 - RC_SHARE_1) * absorbed_w_m2
    (k1, k2), d, fi = RC_K, RC_D, internal_w_m2
    beta = alpha * (RC_GAMMA - 1) / RC_GAMMA
    e = 4 * beta / RC_N

    def sigma_t4(t):
        return (f1 / 2 * (1 + d / k1 + (k1 / d - d / k1) * np.exp(-k1 * t))
                + f2 / 2 * (1 + d / k2 + (k2 / d - d / k2) * np.exp(-k2 * t)) + fi / 2 * (1 + d * t))

    def upgamma(x):
        return gammaincc(1 + e, x) * gamma(1 + e)

    def residual(t):
        t04 = sigma_t4(t) / (t / tau0) ** e                 # temperature continuity at the boundary
        up_conv = t04 * np.exp(d * t) * (np.exp(-d * tau0) + (d * tau0) ** (-e) * (upgamma(d * t) - upgamma(d * tau0)))
        up_rad = (f1 / 2 * (1 + d / k1 + (1 - d / k1) * np.exp(-k1 * t))
                  + f2 / 2 * (1 + d / k2 + (1 - d / k2) * np.exp(-k2 * t)) + fi / 2 * (2 + d * t))
        return up_conv - up_rad

    ts = np.logspace(-4, math.log10(tau0 * 0.999), 400)
    r = residual(ts)
    flips = np.nonzero(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    t_rc = brentq(residual, ts[flips[0]], ts[flips[0] + 1]) if len(flips) else tau0
    t0 = (sigma_t4(t_rc) / (t_rc / tau0) ** e / c.SIGMA_SB) ** 0.25

    def t_of_p(p):
        t = tau0 * (np.asarray(p, float) / RC_P0_BAR) ** RC_N
        return np.where(t < t_rc, (sigma_t4(t) / c.SIGMA_SB) ** 0.25, t0 * (t / tau0) ** (beta / RC_N))

    return t_of_p


# --- condensation -------------------------------------------------------------------------------

def saturation_bar(gas: str, temperature_k: np.ndarray) -> np.ndarray:
    """Return the saturation vapour pressure (bar) of NH3, H2O or CH4 over its ice (H2O: over liquid above 273 K)."""
    t = np.asarray(temperature_k, float)
    if gas == "NH3":
        return np.exp(10.53 - 2161.0 / t - 86596.0 / t**2)                 # Ackerman & Marley (2001) eq. A1
    if gas == "H2O":
        tc = t - 273.15                                                     # Buck (1981), hPa
        ice = 6.1115 * np.exp((23.036 * tc - tc**2 / 333.7) / (tc + 279.82))
        liquid = 6.1121 * np.exp((18.729 * tc - tc**2 / 227.3) / (tc + 257.87))
        return np.where(t < 273.16, ice, liquid) * 1e-3
    if gas == "CH4":                                                        # Grundy et al. (2023) Table 2, CH4 I
        tt = np.minimum(t, CH4_TRIPLE_K)
        p = np.exp(-2.510 - 1139.3 / tt + 3.85295 * np.log(tt) - 0.06385 * tt + 2.281e-4 * tt**2 - 5.42e-7 * tt**3)
        return np.where(t > CH4_TRIPLE_K, np.inf, p)
    raise ValueError(gas)


def condensation_base(t_of_p: Callable, mole_fraction: float, gas: str, p_max_bar: float) -> Optional[float]:
    """Return the deepest pressure (bar) at which the gas is saturated, or None if it never is."""
    p = np.logspace(-1.5, math.log10(p_max_bar), 3000)
    saturated = np.nonzero(mole_fraction * p >= saturation_bar(gas, t_of_p(p)))[0]
    return float(p[saturated.max()]) if len(saturated) else None


# --- the column ---------------------------------------------------------------------------------

@dataclass
class CloudColumn:
    """Composition, temperature and aerosols of a cold giant's upper atmosphere."""

    metallicity: float             # C/H relative to protosolar
    mole_fractions: dict
    alpha: float
    t_1bar_k: float
    deck: str                      # uppermost cloud: "NH3", "H2O", "CH4" or "none"
    deck_bar: float                # its base
    upper_haze_tau: float          # optical depths at 800 nm
    haze_tau: float                # tropospheric haze (gas giants) or methane-level haze (ice giants)
    chromophore_tau: float
    gravity_m_s2: float

    @property
    def methane_deck(self) -> bool:
        """Return whether methane condenses (the ice-giant structure)."""
        return self.deck == "CH4"


def cloud_column(gravity_m_s2: float, absorbed_w_m2: float, internal_w_m2: float, metallicity: float,
                 seed: int, scatter: bool = True) -> tuple[CloudColumn, Callable[[np.ndarray], np.ndarray]]:
    """Return the cloud structure of a giant and its T(p), from gravity, absorbed and internal fluxes and
    metallicity; ``scatter=False`` leaves out the seeded scatter of the hazes."""
    x = mole_fractions(metallicity)
    alpha = adiabat_alpha(metallicity)
    t_of_p = temperature_profile(absorbed_w_m2, internal_w_m2, gravity_m_s2, alpha)
    bases = {gas: condensation_base(t_of_p, x[gas], gas, p_max) for gas, p_max in
             (("CH4", 100.0), ("NH3", 100.0), ("H2O", 3000.0))}
    if bases["CH4"] is not None:
        deck = "CH4"
    else:
        found = [(p, gas) for gas, p in bases.items() if p is not None]
        deck = min(found)[1] if found else "none"
    deck_bar = bases[deck] if deck != "none" else h.GIANT_CLEAR_DECK_BAR
    rng = named_rng(seed, "giant.haze")
    upper_dex, haze_dex = h.GIANT_HAZE_SCATTER_DEX * rng.normal(size=2) * scatter
    upper = h.GIANT_UPPER_HAZE_TAU * 10 ** float(upper_dex)
    scatter = 10 ** float(haze_dex)
    chromophore = 0.0
    if deck == "CH4":
        lo, hi = h.GIANT_METHANE_HAZE_FLUX_W_M2
        f = np.clip(math.log10(max(internal_w_m2, 1e-9) / lo) / math.log10(hi / lo), 0.0, 1.0)
        haze = h.GIANT_METHANE_HAZE_TAU * float(1 - f) * scatter
    elif deck == "NH3":
        (p_lo, tau_lo), (p_hi, tau_hi) = h.GIANT_TROPOSPHERIC_HAZE
        haze = float(np.clip(tau_lo + (deck_bar - p_lo) * (tau_hi - tau_lo) / (p_hi - p_lo),
                             0.0, h.GIANT_TROPOSPHERIC_HAZE_MAX)) * scatter
        chromophore = h.GIANT_CHROMOPHORE_TAU
    else:
        haze = 0.0
    return CloudColumn(metallicity=metallicity, mole_fractions=x, alpha=alpha, t_1bar_k=float(t_of_p(1.0)),
                       deck=deck, deck_bar=deck_bar, upper_haze_tau=upper, haze_tau=haze,
                       chromophore_tau=chromophore, gravity_m_s2=gravity_m_s2), t_of_p


def _two_stream(tau: np.ndarray, omega: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the diffuse reflectance and transmittance of homogeneous layers (delta-Eddington two-stream)."""
    f = g * g
    tau_s, omega_s, g_s = (1 - omega * f) * tau, (1 - f) * omega / (1 - omega * f), (g - f) / (1 - f)
    g1, g2 = 2 - omega_s * (1 + g_s), omega_s * (1 - g_s)
    k = np.sqrt(np.maximum(g1 * g1 - g2 * g2, 1e-12))
    e = np.exp(-k * tau_s)
    den = (k + g1) - (g1 - k) * e * e
    return g2 * (1 - e * e) / den, 2 * k * e / den


def geometric_albedo(column: CloudColumn, t_of_p: Callable, cloud_scale: float = 1.0) -> np.ndarray:
    """Return the geometric albedo spectrum on the optics grid (380–1000 nm)."""
    o = optics()
    x = column.mole_fractions
    p_deck = column.deck_bar
    floor = h.GIANT_ICE_CLOUD_FLOOR_BAR if column.methane_deck else 2.0 * p_deck
    edges = np.logspace(-3, math.log10(floor), h.GIANT_RT_LAYERS + 1)
    pm, dp = np.sqrt(edges[1:] * edges[:-1]), np.diff(edges)
    mu = 2.0 * x["H2"] + 4.0 * x["He"] + 16.0 * x["CH4"]
    column_density = dp * 1e5 / (mu * c.M_H * column.gravity_m_s2)
    x_ch4 = np.minimum(x["CH4"], saturation_bar("CH4", t_of_p(pm)) / pm)
    wl = o["wavelength_nm"]
    tau = np.outer(column_density * x["H2"], H2_RAYLEIGH_M2 * (430.0 / wl) ** 4)
    scat = tau.copy()
    gsum = np.zeros_like(tau)
    tau += np.outer(column_density * x_ch4, o["ch4_cross_section_m2"])

    def add(profile: np.ndarray, total: float, kind: str, omega_scale: float = 1.0) -> None:
        nonlocal tau, scat, gsum
        if total <= 0 or profile.sum() <= 0:
            return
        t = np.outer(profile / profile.sum() * total, o[f"{kind}_extinction"])
        w = o[f"{kind}_albedo"] * omega_scale
        tau, scat, gsum = tau + t, scat + t * w, gsum + t * w * o[f"{kind}_asymmetry"]

    log_p = np.log(pm)
    add(np.where(pm <= p_deck, (pm / p_deck) ** 0.5, 0.0) * dp, column.upper_haze_tau, "haze_upper")
    if column.methane_deck:
        add(np.exp(-0.5 * ((log_p - math.log(p_deck)) / 0.1) ** 2), column.haze_tau, "haze_methane")
        base = floor
    else:
        add(np.where((pm >= 0.1) & (pm <= min(0.6, p_deck)), dp, 0.0), column.haze_tau, "haze_troposphere")
        add(np.exp(-0.5 * ((log_p - math.log(0.7 * p_deck)) / 0.25) ** 2), column.chromophore_tau, "chromophore")
        base = p_deck
    add(np.where((pm <= base) & (pm >= 0.6 * base), dp, 0.0), h.GIANT_CLOUD_TAU * cloud_scale, "cloud",
        h.GIANT_CLOUD_ALBEDO)
    omega = np.clip(scat / np.maximum(tau, 1e-30), 0.0, 0.999999)
    g = np.where(scat > 0, gsum / np.maximum(scat, 1e-30), 0.0)
    r_layer, t_layer = _two_stream(tau, omega, g)
    total = np.full(len(wl), h.GIANT_CLOUD_ALBEDO * h.GIANT_FLOOR_REFLECTANCE)
    for r, t in zip(r_layer[::-1], t_layer[::-1]):          # add layers from the bottom up
        total = r + t * t * total / (1 - r * total)
    return h.GIANT_GEOMETRIC_FACTOR * total


@dataclass
class LayeredColours:
    """Colours (hex sRGB) of a cold giant under its star."""

    disc: str
    zone: str
    belt: str
    vortex: str                    # chromophore-rich storms ("" without chromophores)
    brightness: float


def _hex(spectrum_nm: np.ndarray, star_temperature_k: float) -> tuple[str, float]:
    """Return the colour and brightness of an albedo spectrum on the optics grid."""
    wl = optics()["wavelength_nm"]
    return reflected_colour(np.interp(WAVELENGTHS_NM, wl, spectrum_nm), star_temperature_k)


def layered_colours(column: CloudColumn, t_of_p: Callable, star_temperature_k: float) -> LayeredColours:
    """Return the disc, zone, belt and vortex colours of a cold giant.

    Belts carry the chromophore tint of the cloud class, scaled with the chromophore
    optical depth and hidden by the tropospheric haze; zones and belts average
    to the modelled disc.
    """
    disc = geometric_albedo(column, t_of_p)
    wl = optics()["wavelength_nm"]
    tint = np.interp(wl, h.GIANT_BANDS_NM, h.CHROMOPHORE_TRANSMISSION)
    strength = column.chromophore_tau / h.GIANT_CHROMOPHORE_TAU / (1 + column.haze_tau / h.GIANT_BELT_HAZE_TAU)
    if column.methane_deck or strength == 0:
        zone, belt = geometric_albedo(column, t_of_p, h.GIANT_ZONE_CLOUD_SCALE), disc
        zone = (zone + disc) / 2
        belt = np.maximum(2 * disc - zone, 0.0)
        vortex = ""
    else:
        t = tint**strength
        clear = disc / ((1 - h.BELT_SHARE) * t**h.ZONE_CHROMOPHORE_SHARE + h.BELT_SHARE * t)
        zone, belt = clear * t**h.ZONE_CHROMOPHORE_SHARE, clear * t
        vortex = _hex(clear * t**h.VORTEX_CHROMOPHORE_POWER, star_temperature_k)[0]
    hex_disc, brightness = _hex(disc, star_temperature_k)
    return LayeredColours(disc=hex_disc, zone=_hex(zone, star_temperature_k)[0], belt=_hex(belt, star_temperature_k)[0],
                          vortex=vortex, brightness=brightness)


def hot_darkening(metallicity: float) -> np.ndarray:
    """Return the B, V, R, I factors darkening the class III–V spectra with metallicity.

    Albedo ∝ Z^−b per band, b from the clear 0.8 AU models of Cahoy et al.
    (2010) Table 4 (1× to 3× and 10× to 30×, averaged).
    """
    z = float(np.clip(metallicity, *h.GIANT_HOT_METALLICITY_RANGE))
    return z ** -np.array(h.GIANT_HOT_METALLICITY_EXPONENT)
