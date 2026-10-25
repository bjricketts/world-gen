"""Appearance of giant planets: cloud class, albedo, colour and jets (task 11.2).

Internal heat sets the effective temperature with the starlight,
T_eff⁴ = T_eq⁴ + T_int⁴, from intrinsic luminosities measured for Saturn
and Jupiter, the ice giants' mean, and the Burrows et al. (1997) cooling
tracks, scaled as L ∝ t⁻¹ (Linder et al. 2019). T_eff picks the cloud class
of Sudarsky, Burrows & Pinto (2000), and the class the Bond albedo; since
T_eq depends on the albedo, the effective temperature is solved for.

Classes I and II are coloured by the layered model of ``colours`` (task
11.11) from the atmosphere's metallicity; classes III–V by the Cahoy, Marley &
Fortney (2010) model albedo spectra, darkened with metallicity, under the
star's light. Jets are spaced by the Rhines length π√(2U/β) (Rhines 1975; Lian &
Showman 2010), and the equatorial jet is prograde on gas giants with enough
intrinsic heat flux (Liu & Schneider 2010). Tidally locked giants have one
broad superrotating jet with a hot spot east of the substellar point
(Showman & Polvani 2011) and polar vortices (Skinner & Wei 2025).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..biosphere.pigment import WAVELENGTHS_NM, reflected_colour
from .colours import atmospheric_metallicity, cloud_column, hot_darkening, layered_colours
from ..state import AtmosphereState, BulkState, GiantAppearance, OrbitState, StarState
from ..util import named_rng

GIANT_CLASSES = ("ice giant", "gas giant", "brown dwarf")
CLASSES = ("I", "II", "III", "IV", "V")


def is_giant(bulk: BulkState) -> bool:
    """Return whether the planet is a giant that gets an appearance."""
    return bulk.planet_class in GIANT_CLASSES


def is_ice_like(bulk: BulkState) -> bool:
    """Return whether a giant is ice-giant-like (Uranus, Neptune) by its envelope fraction, whatever its mass class."""
    return bulk.envelope_mass_fraction < h.GIANT_ICE_ENVELOPE_MAX


def internal_luminosity(mass_kg: float, age_s: float, ice_giant: bool, seed: int) -> float:
    """Return the intrinsic luminosity (W) from the mass and age, with seeded scatter for ice giants."""
    masses = np.log10([m for m, _ in h.GIANT_LUMINOSITY_TABLE])
    lums = np.log10([lum for _, lum in h.GIANT_LUMINOSITY_TABLE])
    x = math.log10(mass_kg / c.M_EARTH)
    if x <= masses[0]:
        log_l = lums[0] + (x - masses[0])          # L ∝ M below the ice giants
    elif x >= masses[-1]:
        slope = (lums[-1] - lums[-2]) / (masses[-1] - masses[-2])
        log_l = lums[-1] + slope * (x - masses[-1])
    else:
        log_l = float(np.interp(x, masses, lums))
    if ice_giant:
        log_l += h.ICE_GIANT_LUMINOSITY_SCATTER_DEX * float(named_rng(seed, "giant.luminosity").uniform(-1, 1))
    age_gyr = max(age_s / c.SECONDS_PER_GYR, h.GIANT_LUMINOSITY_MIN_AGE_GYR)
    return 10**log_l * h.L_JUPITER_W * (h.GIANT_LUMINOSITY_AGE_GYR / age_gyr)


def sudarsky_class(t_eff_k: float, gravity_m_s2: float) -> str:
    """Return the Sudarsky class (I–V) for an effective temperature and surface gravity."""
    b1, b2, b3, b4 = h.SUDARSKY_BOUNDARIES_K
    if t_eff_k >= b4 or (t_eff_k >= b3 and gravity_m_s2 < h.SUDARSKY_LOW_GRAVITY_M_S2):
        return "V"
    if t_eff_k >= b3:
        return "IV"
    if t_eff_k >= b2:
        return "III"
    if t_eff_k >= b1:
        return "II"
    return "I"


def _class_key(cls: str, ice_giant: bool) -> str:
    """Return the table key of a class (class I splits into gas and ice giants)."""
    return ("I ice" if ice_giant else "I gas") if cls == "I" else cls


def _boundaries(gravity_m_s2: float) -> list[tuple[float, str, str]]:
    """Return the class boundaries in T_eff with the classes below and above each."""
    b1, b2, b3, b4 = h.SUDARSKY_BOUNDARIES_K
    if gravity_m_s2 < h.SUDARSKY_LOW_GRAVITY_M_S2:
        return [(b1, "I", "II"), (b2, "II", "III"), (b3, "III", "V")]
    return [(b1, "I", "II"), (b2, "II", "III"), (b3, "III", "IV"), (b4, "IV", "V")]


def class_albedo(t_eff_k: float, gravity_m_s2: float, ice_giant: bool) -> float:
    """Return the Bond albedo at an effective temperature, blended smoothly across class boundaries."""
    albedo = h.GIANT_ALBEDO[_class_key(sudarsky_class(t_eff_k, gravity_m_s2), ice_giant)]
    x = math.log10(t_eff_k)
    for t_b, below, above in _boundaries(gravity_m_s2):
        d = (x - math.log10(t_b)) / h.GIANT_ALBEDO_BLEND_DEX
        if abs(d) < 1:
            w = 0.5 * (1 + d)
            a_lo = h.GIANT_ALBEDO[_class_key(below, ice_giant)]
            a_hi = h.GIANT_ALBEDO[_class_key(above, ice_giant)]
            albedo = (1 - w) * a_lo + w * a_hi
    return albedo


def solve_effective_temperature(instellation_earth: float, internal_temperature_k: float, gravity_m_s2: float,
                                ice_giant: bool, bond_albedo: float | None = None) -> tuple[float, float]:
    """Return the effective temperature (K) and Bond albedo consistent with each other.

    With a given albedo the temperature follows directly; otherwise the
    coldest self-consistent temperature on a fine logarithmic grid is used.
    """
    flux = instellation_earth * c.S_EARTH / (4 * c.SIGMA_SB)

    def t_eff(albedo: float) -> float:
        """Return T_eff for a Bond albedo."""
        return (flux * (1 - albedo) + internal_temperature_k**4) ** 0.25

    if bond_albedo is not None:
        return t_eff(bond_albedo), bond_albedo
    grid = np.logspace(0.5, 4.0, 1401)
    resid = np.array([t_eff(class_albedo(t, gravity_m_s2, ice_giant)) - t for t in grid])
    crossing = np.nonzero((resid[:-1] > 0) & (resid[1:] <= 0))[0]
    i = int(crossing[0]) if crossing.size else int(np.argmin(np.abs(resid)))
    t0, t1, r0, r1 = grid[i], grid[i + 1], resid[i], resid[i + 1]
    t = t0 if r0 == r1 else t0 + (t1 - t0) * r0 / (r0 - r1)
    albedo = class_albedo(t, gravity_m_s2, ice_giant)
    return t_eff(albedo), albedo


def giant_bond_albedo(star: StarState, orbit: OrbitState, bulk: BulkState, seed: int) -> float:
    """Return the default Bond albedo of a giant from its cloud class."""
    ice = is_ice_like(bulk)
    t_int = internal_temperature(bulk, star, seed)
    return solve_effective_temperature(orbit.instellation_earth, t_int, bulk.surface_gravity_m_s2, ice)[1]


def internal_temperature(bulk: BulkState, star: StarState, seed: int) -> float:
    """Return the internal temperature T_int (K) of a giant."""
    lum = internal_luminosity(bulk.mass_kg, star.age_s, is_ice_like(bulk), seed)
    return (lum / (4 * math.pi * bulk.radius_m**2 * c.SIGMA_SB)) ** 0.25


def _spectrum(cls: str, ice_giant: bool, metallicity: float = 1.0) -> np.ndarray:
    """Return the geometric albedo in the B, V, R, I bands of a class III–V giant, darkened with metallicity."""
    if cls == "IV":
        base = np.array(h.GIANT_GEOMETRIC_ALBEDO["III"]) * h.GIANT_ALBEDO["IV"] / h.GIANT_ALBEDO["III"]
    elif cls == "V":
        base = np.array(h.GIANT_GEOMETRIC_ALBEDO["II"]) * h.GIANT_ALBEDO["V"] / h.GIANT_ALBEDO["II"]
    else:
        base = np.array(h.GIANT_GEOMETRIC_ALBEDO[_class_key(cls, ice_giant)])
    return base * hot_darkening(metallicity)


def _band_colour(bands: np.ndarray, star_temperature_k: float) -> tuple[str, float]:
    """Return the colour and brightness of a four-band albedo spectrum."""
    return reflected_colour(np.interp(WAVELENGTHS_NM, h.GIANT_BANDS_NM, bands), star_temperature_k)


def _rgb(hex_colour: str) -> np.ndarray:
    """Return sRGB components (0–1) of a hex colour."""
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def hot_rgb(cls: str, ice_giant: bool, star_temperature_k: float, metallicity: float = 1.0) -> np.ndarray:
    """Return the sRGB colour (0–1) of a class III–V giant under the star."""
    return _rgb(_band_colour(_spectrum(cls, ice_giant, metallicity), star_temperature_k)[0])


def class_at_temperature_rgb(temperatures_k: np.ndarray, gravity_m_s2: float, ice_giant: bool,
                             star_temperature_k: float, metallicity: float = 1.0,
                             cold_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the colour (sRGB, 0–1) of the cloud class at each temperature, for colouring locked giants.

    Classes III–V take the Cahoy shapes; classes I and II take ``cold_rgb`` (the
    planet's layered colour), or class II's shape without it.
    """
    table = {cls: hot_rgb(cls, ice_giant, star_temperature_k, metallicity) for cls in ("II", "III", "IV", "V")}
    table["I"] = table["II"]
    if cold_rgb is not None:
        table["I"] = table["II"] = np.asarray(cold_rgb, float)
    return np.array([table[sudarsky_class(float(t), gravity_m_s2)] for t in temperatures_k])


def jet_speed(envelope_fraction: float, seed: int) -> float:
    """Return the Rhines speed scale U (m/s), from ice-giant to gas-giant values by envelope fraction."""
    f = (np.clip(envelope_fraction, h.JET_ENVELOPE_ICE, h.JET_ENVELOPE_GAS) - h.JET_ENVELOPE_ICE) / (
        h.JET_ENVELOPE_GAS - h.JET_ENVELOPE_ICE)
    log_u = (1 - f) * math.log10(h.JET_SPEED_ICE_M_S) + f * math.log10(h.JET_SPEED_GAS_M_S)
    log_u += h.JET_SPEED_SCATTER_DEX * float(named_rng(seed, "giant.jet_speed").normal())
    return 10**log_u


def jet_latitudes(radius_m: float, rotation_period_s: float, speed_m_s: float) -> np.ndarray:
    """Return the jet-core latitudes (radians) from the equator poleward, the first at the equator.

    Jets are stacked from the equator poleward, each band as wide as the local
    Rhines length π√(2U/β) with β = 2Ω cos φ / a.
    """
    omega = 2 * math.pi / abs(rotation_period_s)
    lat, cores = 0.0, [0.0]
    limit = math.radians(h.JET_MAX_LATITUDE_DEG)
    while True:
        beta = 2 * omega * max(math.cos(lat), 0.05) / radius_m
        lat += math.pi * math.sqrt(2 * speed_m_s / beta) / radius_m
        if lat > limit:
            return np.array(cores)
        cores.append(lat)


def jet_count(radius_m: float, rotation_period_s: float, speed_m_s: float) -> tuple[int, float]:
    """Return the number of alternating jets pole to pole and the equatorial band half-width (degrees)."""
    cores = jet_latitudes(radius_m, rotation_period_s, speed_m_s)
    omega = 2 * math.pi / abs(rotation_period_s)
    first = math.pi * math.sqrt(2 * speed_m_s / (2 * omega / radius_m)) / radius_m
    return 2 * len(cores) - 1, math.degrees(first) / 2


def rhines_length(radius_m: float, rotation_period_s: float, speed_m_s: float) -> float:
    """Return the Rhines length π√(2U/β) at the equator (m)."""
    beta = 4 * math.pi / abs(rotation_period_s) / radius_m
    return math.pi * math.sqrt(2 * speed_m_s / beta)


def deformation_half_width(radius_m: float, rotation_period_s: float, t_eff_k: float,
                           molecular_weight: float) -> float:
    """Return the half-width (degrees) of the equatorial jet of a locked giant: the equatorial deformation radius.

    L = (√(gH)/β)^½ with the gravity-wave speed √(gH) ≈ N H = R_s √(T/c_p) of
    an isothermal layer (Showman & Polvani 2011).
    """
    r_specific = c.R_GAS / (molecular_weight * 1e-3)
    wave_speed = r_specific * math.sqrt(t_eff_k / (h.GIANT_CP_OVER_R * r_specific))
    beta = 4 * math.pi / abs(rotation_period_s) / radius_m
    width = math.degrees(math.sqrt(wave_speed / beta) / radius_m)
    return min(width, h.LOCKED_JET_MAX_LATITUDE_DEG)


def build_appearance(star: StarState, orbit: OrbitState, bulk: BulkState, atmosphere: AtmosphereState,
                     seed: int, metallicity: Optional[float] = None) -> GiantAppearance:
    """Return the appearance of a giant from its star, orbit, bulk and the Bond albedo in use.

    ``metallicity`` (C/H relative to protosolar) overrides the mass trend.
    """
    ice = is_ice_like(bulk)
    t_int = internal_temperature(bulk, star, seed)
    t_eff, albedo = solve_effective_temperature(orbit.instellation_earth, t_int, bulk.surface_gravity_m_s2, ice,
                                                atmosphere.bond_albedo)
    cls = sudarsky_class(t_eff, bulk.surface_gravity_m_s2)
    if metallicity is None:
        metallicity = atmospheric_metallicity(bulk.mass_kg, star.metallicity_feh, seed)
    deck, deck_bar, haze, vortex = "", 0.0, 0.0, ""
    if cls in ("I", "II"):
        absorbed = (1 - albedo) * orbit.instellation_earth * c.S_EARTH / 4
        column, t_of_p = cloud_column(bulk.surface_gravity_m_s2, absorbed, c.SIGMA_SB * t_int**4, metallicity, seed)
        look = layered_colours(column, t_of_p, star.effective_temperature_k)
        disc, zone, belt, brightness, vortex = look.disc, look.zone, look.belt, look.brightness, look.vortex
        deck, deck_bar, haze = column.deck, column.deck_bar, column.haze_tau
    else:
        disc, brightness = _band_colour(_spectrum(cls, ice, metallicity), star.effective_temperature_k)
        zone = belt = disc
    flux = c.SIGMA_SB * t_int**4
    speed = jet_speed(bulk.envelope_mass_fraction, seed)
    locked = orbit.spin_state == "synchronous"
    if locked:
        count, half_width = 1, deformation_half_width(bulk.radius_m, orbit.rotation_period_s, t_eff,
                                                      atmosphere.mean_molecular_weight)
        direction, offset = "prograde", h.HOT_SPOT_OFFSET_DEG
    else:
        count, half_width = jet_count(bulk.radius_m, orbit.rotation_period_s, speed)
        prograde = not ice and flux > h.PROGRADE_JET_MIN_FLUX_W_M2
        direction, offset = ("prograde" if prograde else "retrograde"), 0.0
    return GiantAppearance(
        sudarsky_class=cls,
        effective_temperature_k=t_eff,
        internal_temperature_k=t_int,
        internal_flux_w_m2=flux,
        bond_albedo=albedo,
        colour=disc,
        zone_colour=zone,
        belt_colour=belt,
        visible_albedo=brightness,
        regime="locked" if locked else "banded",
        jet_speed_m_s=speed,
        rhines_length_m=rhines_length(bulk.radius_m, orbit.rotation_period_s, speed),
        jet_count=count,
        equatorial_jet=direction,
        jet_half_width_deg=half_width,
        hot_spot_offset_deg=offset,
        polar_vortices=locked,
        atmospheric_metallicity=metallicity,
        cloud_deck=deck,
        cloud_deck_bar=deck_bar,
        haze_optical_depth=haze,
        vortex_colour=vortex,
    )
