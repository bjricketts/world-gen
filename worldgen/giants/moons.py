"""Drawn moon systems (task 11.5): regular moons of giants, ring-born moons, and impact moons of rocky planets.

Three sources, each from its literature:

- Gas-dominated giants keep regular moons from their circumplanetary disc. A
  planet that opens a gap in its star's disc (Crida et al. 2006 criterion)
  keeps a Jovian architecture of 4–5 similar moons with the inner ones in 2:1
  resonances; one that does not keeps one dominant icy moon (Sasaki, Stewart &
  Ida 2010). The outcomes are drawn from their published distributions, scaled
  by the planet's mass and radius. Ice is available only where the star's
  nebula is below 160 K.
- Giants with rings spread moons out of the Roche limit (Crida & Charnoz
  2012): a chain in which each moon is twice the mass of the next one in, with
  q ∝ Δ^(9/5) near the rings and ∝ (1+Δ)^3.9 beyond, the outermost carried out
  by tides over the planet's age. The ring disc's mass follows the ring regime.
- Rocky planets have a moon-forming giant impact about half the time (Elser et
  al. 2011). The disc mass follows Canup (2008), the moon takes 10–55% of it
  (Kokubo et al. 2000), forms at 1.3 Roche radii and recedes by tides.
- Half of ice giants have a giant impact that leaves a Uranian system (Woo et
  al. 2021): ~1e-4 of the planet's mass in five moons, heavier outward.
- Every giant meets one binary planetesimal close enough to split it (task
  11.7b). A member is captured when lighter than Williams & Zugger's (2024)
  eq. 1 limit, survives its loose orbit as often as in Porter & Grundy (2011),
  and survives the regular moons as often as Triton did in Rufu & Canup (2017),
  clearing them. Its orbit circularises at a = q(1 + e) and then migrates by
  the planet's tides, inward if retrograde.

Moons inside their Roche limit, beyond the stable limit, or heavier than the
largest moon that survives tidal migration (Barnes & O'Brien 2002) are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..state import BulkState, Issue, MoonState, OrbitState, StarState, SystemLimits
from ..util import named_rng
from .appearance import is_ice_like
from .limits import orbital_period, roche_limit_fluid
from .moon_energy import damped_eccentricity, eccentricity_damping_time, ice_shell_dissipation, ice_shell_world
from .rings import GIANT_CLASSES, zero_albedo_temperature


@dataclass
class DrawnMoon:
    """A moon before naming: orbit in metres, mass in kg, and its resonance partner (index), if any."""

    a_m: float
    mass_kg: float
    water: float
    origin: str
    density: float
    eccentricity: float = 0.0
    inclination_rad: float = 0.0
    resonance_with: Optional["DrawnMoon"] = None
    resonance: str = ""
    retrograde: bool = False


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
         "XVII", "XVIII", "XIX", "XX"]


def name_moons(moons: list[MoonState], planet_name: str) -> None:
    """Name unnamed moons (ordered outward) after the planet with Roman numerals."""
    for k, moon in enumerate(moons):
        if not moon.name:
            moon.name = f"{planet_name} {ROMAN[k] if k < len(ROMAN) else k + 1}"


# --- Gap opening and ice ----------------------------------------------------------------------------------

def nebula_aspect_ratio(a_m: float) -> float:
    """Return the aspect ratio H/r of the star's nebula at a distance (heuristic power law)."""
    return h.NEBULA_ASPECT_RATIO_5AU * (a_m / (5.2 * c.AU)) ** h.NEBULA_ASPECT_SLOPE


def gap_criterion(star: StarState, orbit: OrbitState, bulk: BulkState) -> float:
    """Return Crida et al. (2006) 3H/(4R_H) + 50/(qℛ); a planet opens a gap in the nebula when it is ≤ 1."""
    q = bulk.mass_kg / star.mass_kg
    aspect = nebula_aspect_ratio(orbit.semi_major_axis_m)
    hill = (q / 3.0) ** (1.0 / 3.0)
    reynolds = 1.0 / (h.NEBULA_ALPHA * aspect**2)
    return 0.75 * aspect / hill + 50.0 / (q * reynolds)


def ice_available(star: StarState, orbit: OrbitState) -> bool:
    """Return whether the nebula at the planet was cold enough (< 160 K) for water ice."""
    return zero_albedo_temperature(star, orbit) < h.ICE_CONDENSATION_K


def moon_density(water: float) -> float:
    """Return the bulk density (kg/m³) of a moon of a given water fraction, mixing ice and rock."""
    return 1.0 / (water / h.MOON_ICE_DENSITY + (1.0 - water) / h.MOON_ROCK_DENSITY)


def _count(rng: np.random.Generator, table: dict) -> int:
    """Return a draw from a {value: probability} table."""
    keys = list(table)
    p = np.array([table[k] for k in keys], float)
    return int(keys[rng.choice(len(keys), p=p / p.sum())])


def _regular_orbit(rng) -> tuple[float, float]:
    """Return the eccentricity and inclination (rad) of a regular moon."""
    return float(rng.uniform(*h.REGULAR_MOON_ECCENTRICITY)), math.radians(rng.uniform(*h.REGULAR_MOON_INCLINATION_DEG))


# --- Regular moons of gas giants (Sasaki et al. 2010) -------------------------------------------------------

def jovian_moons(bulk: BulkState, icy_ok: bool, rng: np.random.Generator) -> list[DrawnMoon]:
    """Return a Jovian architecture: 3–6 similar moons, the inner three (and often the fourth) in 2:1 resonances."""
    n = _count(rng, h.JOVIAN_MOON_COUNTS)
    n_rocky = n if not icy_ok else min(_count(rng, h.JOVIAN_ROCKY_COUNTS), n)
    a = h.JOVIAN_INNER_ORBIT_RP * 10 ** rng.normal(0.0, h.JOVIAN_ORBIT_DEX)
    resonant = 2 ** (2 / 3)
    moons = []
    for k in range(n):
        if k > 0:
            trapped = k < 3 or (k == 3 and rng.uniform() < h.JOVIAN_OUTER_TRAPPED)
            if trapped:
                a *= resonant
            elif k == 3:
                a *= rng.uniform(*h.JOVIAN_FREE_PERIOD_RATIO) ** (2 / 3)
            else:
                a *= rng.uniform(*h.JOVIAN_EXTRA_SPACING)
        else:
            trapped = False
        mass = h.JOVIAN_MOON_MASS * 10 ** rng.normal(0.0, h.JOVIAN_MOON_MASS_DEX) * bulk.mass_kg
        water = rng.uniform(*(h.ROCKY_MOON_WATER_FRACTION if k < n_rocky else h.ICY_MOON_WATER_FRACTION))
        e, i = _regular_orbit(rng)
        moons.append(DrawnMoon(a * bulk.radius_m, mass, float(water), "regular", moon_density(water), e, i,
                               resonance_with=moons[-1] if (k > 0 and trapped) else None,
                               resonance="2:1" if (k > 0 and trapped) else ""))
    return moons


def saturnian_moons(bulk: BulkState, icy_ok: bool, rng: np.random.Generator) -> list[DrawnMoon]:
    """Return a Saturnian architecture: one dominant moon about Titan's place, sometimes smaller large moons inward."""
    n = _count(rng, h.SATURNIAN_MOON_COUNTS)
    water_range = h.ICY_MOON_WATER_FRACTION if icy_ok else h.ROCKY_MOON_WATER_FRACTION
    a = min(h.SATURNIAN_MOON_ORBIT_RP * 10 ** rng.normal(0.0, h.SATURNIAN_ORBIT_DEX),
            h.DISC_OUTER_FRACTION * h.CIRCUMPLANETARY_DISC_RP)
    mass = h.SATURNIAN_MOON_MASS * 10 ** rng.normal(0.0, h.SATURNIAN_MASS_DEX)
    water = float(rng.uniform(*water_range))
    moons = [DrawnMoon(a * bulk.radius_m, mass * bulk.mass_kg, water, "regular", moon_density(water),
                       *_regular_orbit(rng))]
    lo, hi = np.log10(h.SATURNIAN_EXTRA_MASS)
    for _ in range(n - 1):
        a /= rng.uniform(*h.SATURNIAN_EXTRA_SPACING)
        water = float(rng.uniform(*water_range))
        moons.append(DrawnMoon(a * bulk.radius_m, 10 ** rng.uniform(lo, hi) * bulk.mass_kg, water, "regular",
                               moon_density(water), *_regular_orbit(rng)))
    return sorted(moons, key=lambda m: m.a_m)


# --- Ring-born moons (Crida & Charnoz 2012) -----------------------------------------------------------------

def chain_shape(delta: float) -> float:
    """Return Crida & Charnoz's Q(Δ): mass ∝ Δ^(9/5) inside the 2:1 point and ∝ (1+Δ)^3.9 beyond."""
    d21 = h.RING_CHAIN_DELTA_21
    if delta < d21:
        return (delta / d21) ** 1.8
    return ((delta + 1.0) / (d21 + 1.0)) ** 3.9


def chain_delta(shape: float) -> float:
    """Return the Δ at which Q(Δ) takes a value (the inverse of ``chain_shape``)."""
    d21 = h.RING_CHAIN_DELTA_21
    if shape < 1.0:
        return d21 * shape ** (5.0 / 9.0)
    return (d21 + 1.0) * shape ** (1.0 / 3.9) - 1.0


def tidal_reach(mass_ratio: float, bulk: BulkState, tidal_q: float, k2: float, age_s: float,
                start_m: float = 0.0) -> float:
    """Return the orbit (m) a moon reaches by planetary tides over a time, a^(13/2) = a0^(13/2) + K q t."""
    k = 39.0 * k2 * math.sqrt(c.G * bulk.mass_kg) * bulk.radius_m**5 / (2.0 * tidal_q)
    return (start_m**6.5 + k * mass_ratio * age_s) ** (2.0 / 13.0)


def ring_moons(regime: str, bulk: BulkState, roche_m: float, outer_limit_m: float, tidal_q: float, k2: float,
               age_s: float, rng: np.random.Generator) -> list[DrawnMoon]:
    """Return the chain of moons spread out of the rings, outermost first in the drawing, ordered outward."""
    if regime not in h.RING_DISC_MASS or outer_limit_m <= roche_m * (1 + h.RING_CHAIN_MIN_DELTA):
        return []
    disc = h.RING_DISC_MASS[regime] * 10 ** rng.normal(0.0, h.RING_DISC_MASS_DEX)
    q_out = disc / h.RING_DISC_MOON_RATIO / 2.0
    r_out = max(tidal_reach(q_out, bulk, tidal_q, k2, age_s), roche_m * (1 + h.RING_CHAIN_DELTA_21))
    r_out = min(r_out, outer_limit_m)
    shape_out = chain_shape(r_out / roche_m - 1.0)
    water_range = h.RING_MOON_WATER_FRACTION[regime]
    moons = []
    k = 0
    while True:
        q = q_out * 2.0**-k
        delta = chain_delta(shape_out * 2.0**-k)
        mass = q * bulk.mass_kg * 10 ** rng.normal(0.0, h.RING_CHAIN_MASS_DEX)
        if delta < h.RING_CHAIN_MIN_DELTA or q * bulk.mass_kg < h.MOON_MIN_MASS_KG:
            break
        water = float(rng.uniform(*water_range))
        moons.append(DrawnMoon(roche_m * (1 + delta), mass, water, "ring", moon_density(water), *_regular_orbit(rng)))
        k += 1
    return sorted(moons, key=lambda m: m.a_m)


# --- Impact moons of rocky planets (Elser et al. 2011; Canup 2008) -------------------------------------------

_LATTICE = None


def _unit_ball_lattice(n: int = 24) -> np.ndarray:
    """Return a regular lattice of points filling the unit ball (cached)."""
    global _LATTICE
    if _LATTICE is None:
        g = (np.arange(n) + 0.5) / n * 2 - 1
        pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        _LATTICE = pts[(pts**2).sum(1) < 1.0]
    return _LATTICE


def passing_mass_fraction(gamma: float, b: float) -> float:
    """Return m_pass/m_tot: the impactor mass outside the target's sphere at contact (Canup 2008, appendix).

    The impactor has a core of half its radius at twice the mantle density (Elser et al. 2011).
    """
    rt, ri = (1.0 - gamma) ** (1 / 3), gamma ** (1 / 3)
    d = b * (rt + ri)
    pts = _unit_ball_lattice() * ri
    hit = ((pts - np.array([d, 0.0, 0.0])) ** 2).sum(1) < rt**2
    weight = np.where((pts**2).sum(1) < (h.IMPACT_CORE_RADIUS * ri) ** 2, 2.0, 1.0)
    return gamma * (1.0 - float((weight * hit).sum() / weight.sum()))


def impact_disc_fraction(gamma: float, b: float) -> float:
    """Return m_disk/m_tot after a giant impact, 2.8 (0.1/γ)^1.25 (m_pass/m_tot)² (Canup 2008, eq. B)."""
    return 2.8 * (0.1 / gamma) ** 1.25 * passing_mass_fraction(gamma, b) ** 2


def impact_moon(bulk: BulkState, tidal_q: float, k2: float, age_s: float,
                rng: np.random.Generator) -> Optional[DrawnMoon]:
    """Return the moon left by a giant impact, receded by tides over the planet's age, or None."""
    if rng.uniform() >= h.IMPACT_MOON_PROBABILITY:
        return None
    gamma = 10 ** rng.uniform(*np.log10(h.IMPACT_GAMMA_RANGE))
    b = rng.uniform(*h.IMPACT_B_RANGE)
    mass = rng.uniform(*h.IMPACT_EFFICIENCY) * impact_disc_fraction(gamma, b) * bulk.mass_kg
    if mass < h.MOON_MIN_MASS_KG:
        return None
    start = h.IMPACT_MOON_START * roche_limit_fluid(bulk.radius_m, bulk.density_kg_m3, h.IMPACT_MOON_DENSITY)
    a = tidal_reach(mass / bulk.mass_kg, bulk, tidal_q, k2, age_s, start)
    return DrawnMoon(a, mass, 0.0, "impact", h.IMPACT_MOON_DENSITY, float(rng.uniform(*h.IMPACT_MOON_ECCENTRICITY)),
                     math.radians(rng.uniform(*h.IMPACT_MOON_INCLINATION_DEG)))


# --- Impact systems of ice giants (Woo et al. 2021) ----------------------------------------------------------

def uranian_moons(bulk: BulkState, rng: np.random.Generator) -> list[DrawnMoon]:
    """Return an impact-born Uranian system: five moons about Uranus's, scaled to the planet, with scatter."""
    moons = []
    for a_rp, ratio in h.URANIAN_MOONS:
        a = a_rp * 10 ** rng.normal(0.0, h.URANIAN_ORBIT_DEX) * bulk.radius_m
        mass = ratio * 10 ** rng.normal(0.0, h.URANIAN_MASS_DEX) * bulk.mass_kg
        water = float(rng.uniform(*h.URANIAN_WATER_FRACTION))
        moons.append(DrawnMoon(a, mass, water, "impact", moon_density(water), *_regular_orbit(rng)))
    return sorted(moons, key=lambda m: m.a_m)


# --- Captured moons (binary exchange) -------------------------------------------------------------------------

def capture_mass_limit(planet_mass_kg: float, q_m: float, v_enc: float, v_peri: float, m2_kg: float) -> float:
    """Return the largest mass captured when a binary splits (Williams & Zugger 2024, eq. 1; inf if any)."""
    if v_enc <= v_peri:
        return math.inf
    return 3.0 * planet_mass_kg * (c.G * math.pi * m2_kg / (2.0 * q_m * v_enc * (v_enc - v_peri))) ** 1.5 - m2_kg


def capture_survival(star_mass_kg: float) -> float:
    """Return the share of loose captures that circularise rather than escape or fall (Porter & Grundy 2011)."""
    stars, shares = zip(*h.CAPTURE_SURVIVAL_BY_STAR)
    return float(np.interp(math.log(star_mass_kg / c.M_SUN), np.log(stars), shares))


def survival_among_moons(mass_ratio: float) -> float:
    """Return the chance a captured moon survives a regular system of 1/``mass_ratio`` its mass (Rufu & Canup 2017)."""
    x, y = np.log([r for r, _ in h.CAPTURE_AMONG_MOONS]), np.array([s for _, s in h.CAPTURE_AMONG_MOONS])
    t = math.log(mass_ratio)
    k = 0 if t < x[1] else 1
    return float(min(max(y[k] + (y[k + 1] - y[k]) * (t - x[k]) / (x[k + 1] - x[k]), 0.0), 1.0))


def tidal_decay(mass_ratio: float, bulk: BulkState, tidal_q: float, k2: float, age_s: float, start_m: float) -> float:
    """Return the orbit (m) a retrograde moon decays to by planetary tides, a^(13/2) = a0^(13/2) − K q t (0 if gone)."""
    k = 39.0 * k2 * math.sqrt(c.G * bulk.mass_kg) * bulk.radius_m**5 / (2.0 * tidal_q)
    left = start_m**6.5 - k * mass_ratio * age_s
    return left ** (2.0 / 13.0) if left > 0.0 else 0.0


@dataclass
class Capture:
    """The outcome of a giant's encounter with a binary: the moon, if kept, and what happened."""

    moon: Optional[DrawnMoon]
    fate: str
    heat_j_kg: float = 0.0          # orbital energy released per kilogram as the orbit circularised


def captured_moon(star: StarState, orbit: OrbitState, bulk: BulkState, limits: SystemLimits, k2: float,
                  regular_mass_kg: float, rng: np.random.Generator) -> Capture:
    """Return the outcome of one binary encounter: capture, survival, circularisation and tidal migration."""
    gm = c.G * bulk.mass_kg
    q = bulk.radius_m * 10 ** rng.uniform(*np.log10(h.CAPTURE_PERIAPSIS_RP))
    v_inf = rng.uniform(*h.CAPTURE_SPEED_FRACTION) * math.sqrt(c.G * star.mass_kg / orbit.semi_major_axis_m)
    retrograde = bool(rng.uniform() < h.CAPTURE_RETROGRADE_SHARE)
    icy = ice_available(star, orbit)
    lo, hi = np.log10(h.CAPTURE_BODY_MEARTH_ICY if icy else h.CAPTURE_BODY_MEARTH_ROCKY)
    m1, m2 = (float(m) for m in 10 ** rng.uniform(lo, hi, 2) * c.M_EARTH)
    apo = h.CAPTURE_APOAPSIS_HILL["retrograde" if retrograde else "prograde"] * limits.hill_radius_m
    if apo <= q:
        return Capture(None, "a binary passed, but too close to the edge of the Hill sphere to leave a bound member")
    a0, e0 = 0.5 * (q + apo), (apo - q) / (apo + q)
    v_enc, v_peri = math.sqrt(v_inf**2 + 2.0 * gm / q), math.sqrt(gm * (2.0 / q - 1.0 / a0))
    if m1 > capture_mass_limit(bulk.mass_kg, q, v_enc, v_peri, m2):
        return Capture(None, "a binary passed without leaving a captured member (Williams & Zugger 2024)")
    if rng.uniform() > capture_survival(star.mass_kg):
        return Capture(None, "a captured body escaped or hit the planet before its orbit circularised "
                             "(Porter & Grundy 2011)")
    if regular_mass_kg > 0.0 and rng.uniform() > survival_among_moons(m1 / regular_mass_kg):
        return Capture(None, "a captured body was lost among the regular moons (Rufu & Canup 2017)")
    a_f = q * (1.0 + e0)
    heat = 0.5 * gm * (1.0 / a_f - 1.0 / a0)
    ratio = m1 / bulk.mass_kg
    if retrograde:
        a_now = tidal_decay(ratio, bulk, orbit.tidal_q, k2, star.age_s, a_f)
    else:
        a_now = tidal_reach(ratio, bulk, orbit.tidal_q, k2, star.age_s, a_f)
    water = float(rng.uniform(*(h.CAPTURED_ICY_WATER_FRACTION if icy else h.ROCKY_MOON_WATER_FRACTION)))
    density = moon_density(water)
    if a_now < roche_limit_fluid(bulk.radius_m, bulk.density_kg_m3, density):
        return Capture(None, "a captured retrograde moon has since spiralled into the planet by tides", heat)
    tilt = rng.uniform(*h.CAPTURED_TILT_DEG)
    moon = DrawnMoon(a_now, m1, water, "captured", density, float(rng.uniform(*h.CAPTURED_FINAL_ECCENTRICITY)),
                     math.radians(180.0 - tilt if retrograde else tilt), retrograde=retrograde)
    return Capture(moon, "captured " + ("retrograde" if retrograde else "prograde"), heat)


# --- The system -------------------------------------------------------------------------------------------

def moon_regime(star: StarState, orbit: OrbitState, bulk: BulkState) -> str:
    """Return how a planet's moons form: 'jovian', 'saturnian', 'rings only' (ice giants, which may draw a
    Uranian impact system) or 'impact' (rocky planets)."""
    if bulk.planet_class not in GIANT_CLASSES:
        return "impact"
    if is_ice_like(bulk):
        return "rings only"
    return "jovian" if gap_criterion(star, orbit, bulk) <= 1.0 else "saturnian"


def _survives(m: DrawnMoon, bulk: BulkState, limits: SystemLimits) -> Optional[str]:
    """Return why a drawn moon is lost (inside its Roche limit, beyond the stable limit, too heavy), or None."""
    if m.a_m < roche_limit_fluid(bulk.radius_m, bulk.density_kg_m3, m.density):
        return "inside its Roche limit"
    if m.a_m > (limits.stable_retrograde_m if m.retrograde else limits.stable_prograde_m):
        return "beyond the stable limit"
    if not m.retrograde and m.mass_kg > limits.max_surviving_moon_mass_kg:
        return "too heavy to survive tidal migration (Barnes & O'Brien 2002)"
    return None



def _damp_free_eccentricities(moons: list[DrawnMoon], bulk: BulkState, orbit: OrbitState, age_s: float) -> None:
    """Damp the eccentricity of every moon not held by a resonance, by its own tides over the system's age
    (Tokadjian & Piro 2023, eq. 20). Both partners of a resonance keep their forced eccentricity."""
    held = {id(m) for m in moons if m.resonance_with is not None}
    held |= {id(m.resonance_with) for m in moons if m.resonance_with is not None}
    for m in moons:
        if id(m) in held or m.eccentricity <= 0.0:
            continue
        radius = (3.0 * m.mass_kg / (4.0 * math.pi * m.density)) ** (1 / 3)
        if ice_shell_world(m.water, orbit.instellation_earth):
            phi = ice_shell_dissipation(m.mass_kg, radius)
        else:
            phi = h.TIDAL_K2_ROCKY / h.TIDAL_Q_ROCKY
        m.eccentricity = damped_eccentricity(
            m.eccentricity, age_s, eccentricity_damping_time(bulk.mass_kg, m.mass_kg, radius, m.a_m, phi))

def draw_moons(planet_name: str, star: StarState, orbit: OrbitState, bulk: BulkState, limits: SystemLimits,
               ring_regime: str, k2: float, seed: int) -> tuple[list[MoonState], str, list[Issue]]:
    """Return a planet's drawn moons (named, ordered outward), how they formed, and notes."""
    regime = moon_regime(star, orbit, bulk)
    drawn: list[DrawnMoon] = []
    if regime in ("jovian", "saturnian"):
        rng = named_rng(seed, "moons.regular")
        icy = ice_available(star, orbit)
        regular = (jovian_moons if regime == "jovian" else saturnian_moons)(bulk, icy, rng)
        # around dense planets the template moves out so the innermost moon stays well outside its Roche limit
        floor = h.REGULAR_ROCHE_MARGIN * roche_limit_fluid(bulk.radius_m, bulk.density_kg_m3, regular[0].density)
        stretch = max(floor / regular[0].a_m, 1.0)
        for m in regular:
            m.a_m *= stretch
        drawn = regular
    elif regime == "rings only":
        rng = named_rng(seed, "moons.uranian")
        if rng.uniform() < h.URANIAN_IMPACT_PROBABILITY:
            regime = "uranian"
            drawn = uranian_moons(bulk, rng)
    if regime in ("jovian", "saturnian", "rings only"):
        rocky = ring_regime == "dusty warm"
        roche = limits.roche_rocky_m if rocky else limits.roche_icy_m
        inner = min([m.a_m for m in drawn], default=limits.stable_prograde_m * h.RING_CHAIN_CLEARANCE)
        drawn = ring_moons(ring_regime, bulk, roche, inner / h.RING_CHAIN_CLEARANCE, orbit.tidal_q, k2, star.age_s,
                           named_rng(seed, "moons.rings")) + drawn
    elif regime == "impact":
        moon = impact_moon(bulk, orbit.tidal_q, k2, star.age_s, named_rng(seed, "moons.impact"))
        drawn = [moon] if moon is not None else []
    issues: list[Issue] = []
    if regime != "impact":
        # every giant meets one binary; a surviving capture clears the disc-born and impact-born moons
        system_mass = sum(m.mass_kg for m in drawn if m.origin in ("regular", "impact"))
        capture = captured_moon(star, orbit, bulk, limits, k2, system_mass, named_rng(seed, "moons.capture"))
        text = capture.fate
        if capture.moon is not None:
            cleared = sum(m.origin in ("regular", "impact") for m in drawn)
            drawn = [m for m in drawn if m.origin not in ("regular", "impact")] + [capture.moon]
            text += (f" ({capture.moon.mass_kg / c.M_EARTH:.2g} M⊕); circularising its orbit released "
                     f"{capture.heat_j_kg / 1e6:.3g} MJ/kg as tidal heat"
                     + (f", and it cleared {cleared} regular moon(s)" if cleared else ""))
        issues.append(Issue("info", "heuristic", "system", f"capture: {text} (DESIGN 4.12)"))
    kept, lost = [], []
    for m in drawn:
        why = _survives(m, bulk, limits)
        (lost if why else kept).append(why or m)
    if lost:
        issues.append(Issue("info", "note", "system", f"{len(lost)} drawn moon(s) lost: " + "; ".join(sorted(set(lost)))))
    kept.sort(key=lambda m: m.a_m)
    _damp_free_eccentricities(kept, bulk, orbit, star.age_s)
    moons = [MoonState(
        name="", mass_kg=m.mass_kg, radius_m=(3.0 * m.mass_kg / (4.0 * math.pi * m.density)) ** (1 / 3),
        density_kg_m3=m.density, semi_major_axis_m=m.a_m, eccentricity=m.eccentricity,
        inclination_rad=m.inclination_rad, orbital_period_s=orbital_period(m.a_m, bulk.mass_kg, m.mass_kg),
        water_mass_fraction=m.water, origin=m.origin, retrograde=m.retrograde) for m in kept]
    name_moons(moons, planet_name)
    # a trapped moon is in 2:1 resonance with the regular moon inside it, if both survived
    where = {id(m): k for k, m in enumerate(kept)}
    for m in kept:
        if m.resonance_with is not None and id(m.resonance_with) in where:
            moons[where[id(m)]].resonance = f"{m.resonance} with {moons[where[id(m.resonance_with)]].name}"
    if moons:
        how = {"jovian": "a Jovian architecture (the planet opened a gap in its nebula)",
               "saturnian": "a Saturnian architecture (no gap in the nebula)",
               "rings only": "moons spread from the rings", "uranian": "a giant impact (Uranian architecture)",
               "impact": "a giant impact"}[regime]
        issues.append(Issue("info", "heuristic", "system", f"moons drawn from {how} (DESIGN 4.12)"))
    return moons, regime, issues
