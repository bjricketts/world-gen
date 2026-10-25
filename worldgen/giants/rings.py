"""Rings of a planet: which kind it has, and the rings themselves (task 11.4).

The kind follows Hedman (2015): dense rings form where weak icy material sits
near 70 K. A planet whose ring material can reach that temperature with an
albedo between the darkest and brightest ring surfaces may have dense rings:
bright broad icy rings if the albedo needed is high (Saturn), dark narrow
ringlets if it is low (Uranus). Elsewhere giants have tenuous dusty rings fed
by small inner moons: Jupiter-like when warm, Neptune-like (narrow, with arcs)
when cold. Poynting–Robertson drag (Hyodo & Ida 2025) clears rings around
planets close to their star. Rocky planets have rings only when listed.

The rings are laid out from templates scaled to the Roche limits: Saturn's
C, B, Cassini Division and A regions as fractions of the icy limit, with the
division moved to a moon's 2:1 resonance where one falls there (Ohtsuki
2025); Uranus-like ringlets inside the limit for 1.2 g/cm³ (Hedman 2015);
dusty rings in planet radii after Jupiter's and Neptune's (Sicardy et al.
2024). Colours are the ring material's reflectance under the star.
"""

from __future__ import annotations


import numpy as np

from .. import constants as c
from .. import heuristics as h
from ..biosphere.pigment import WAVELENGTHS_NM, reflected_colour
from ..state import BulkState, Issue, MoonState, OrbitState, RingState, StarState
from ..util import named_rng
from .limits import roche_limit_rubble

GIANT_CLASSES = ("ice giant", "gas giant", "brown dwarf")


def zero_albedo_temperature(star: StarState, orbit: OrbitState) -> float:
    """Return the temperature (K) of a black ring particle at the planet's distance (Hedman 2015, eq. 4)."""
    lum = star.luminosity_w / c.L_SUN
    return 283.0 * lum**0.25 * (c.AU / orbit.semi_major_axis_m) ** 0.5


def ring_window(t_zero_k: float) -> tuple[bool, float]:
    """Return whether dense rings can reach the 70 K window, and the albedo that brings them to 70 K."""
    a_lo, a_hi = h.RING_ALBEDO_RANGE
    t_min, t_max = t_zero_k * (1 - a_hi) ** 0.25, t_zero_k * (1 - a_lo) ** 0.25
    lo, hi = h.RING_TEMPERATURE_WINDOW_K
    needed = float(np.clip(1 - (h.RING_TEMPERATURE_K / t_zero_k) ** 4, a_lo, a_hi))
    return bool(t_max >= lo and t_min <= hi), needed


def pr_lifetime_gyr(star: StarState, orbit: OrbitState, radius_m: float = 0.01, density: float = 3000.0) -> float:
    """Return the Poynting–Robertson lifetime (Gyr) of a ring particle (Hyodo & Ida 2025, eq. 4)."""
    lum = star.luminosity_w / c.L_SUN
    return h.PR_DECAY_GYR * (density / 3000.0) * radius_m * (orbit.semi_major_axis_m / c.AU) ** 2 / lum


def ring_regime(star: StarState, orbit: OrbitState, bulk: BulkState, seed: int) -> str:
    """Return the kind of ring system a planet has: 'broad', 'narrow', 'dusty warm', 'dusty cold' or 'none'."""
    if bulk.planet_class not in GIANT_CLASSES:
        return "none"
    if pr_lifetime_gyr(star, orbit) < h.RING_MIN_PR_LIFETIME_GYR:
        return "none"
    rng = named_rng(seed, "rings.regime")
    t_zero = zero_albedo_temperature(star, orbit)
    dense, needed = ring_window(t_zero)
    if dense and rng.uniform() < h.DENSE_RING_PROBABILITY:
        return "broad" if needed >= h.RING_BROAD_ALBEDO_FROM else "narrow"
    if rng.uniform() < h.DUSTY_RING_PROBABILITY:
        return "dusty cold" if t_zero < h.DUSTY_COLD_FROM_T0_K else "dusty warm"
    return "none"


def ring_colour(kind: str, albedo: float, star: StarState) -> str:
    """Return the colour (hex sRGB) of a ring material of a Bond albedo under the star."""
    shape = np.asarray(h.RING_SPECTRA[kind])
    bands = shape / shape[1] * albedo
    return reflected_colour(np.interp(WAVELENGTHS_NM, h.GIANT_BANDS_NM, bands), star.effective_temperature_k)[0]


def _ring(name, inner, outer, tau, albedo, kind, star, area_weight) -> tuple[RingState, float]:
    """Return a ring (mass filled in later) and its weight for sharing out the system's mass."""
    ring = RingState(inner_radius_m=inner, outer_radius_m=outer, mass_kg=0.0, optical_depth=tau, kind=kind,
                     colour=ring_colour(kind, albedo, star), name=name, bond_albedo=albedo)
    return ring, area_weight * tau * (outer**2 - inner**2)


def _share_mass(rings: list[tuple[RingState, float]], total: float) -> list[RingState]:
    """Return the rings with the system's mass shared out by optical depth times area."""
    weight = sum(w for _, w in rings) or 1.0
    for ring, w in rings:
        ring.mass_kg = total * w / weight
    return [r for r, _ in rings]


def broad_rings(star: StarState, bulk: BulkState, moons: list[MoonState]) -> list[RingState]:
    """Return Saturn-like rings scaled to the icy Roche limit, the Cassini Division at a moon's 2:1 resonance."""
    a_r = roche_limit_rubble(bulk.mass_kg, h.RING_DENSITY_ICY)
    edges = [(name, lo, hi, tau, alb, kind) for name, lo, hi, tau, alb, kind in h.BROAD_RING_TEMPLATE]
    # move the division to the 2:1 resonance of the innermost moon outside the rings that falls in range
    lo_r, hi_r = h.CASSINI_RESONANCE_RANGE
    for moon in sorted(moons, key=lambda m: m.semi_major_axis_m):
        if moon.semi_major_axis_m <= a_r:
            continue
        frac = moon.semi_major_axis_m * 2 ** (-2 / 3) / a_r
        if lo_r <= frac <= hi_r:
            width = edges[3][2] - edges[3][1]
            shift = frac - edges[3][1]
            edges[2] = (*edges[2][:2], edges[2][2] + shift, *edges[2][3:])
            edges[3] = (edges[3][0], frac, frac + width, *edges[3][3:])
            edges[4] = (edges[4][0], frac + width, max(edges[4][2], frac + width + 0.02), *edges[4][3:])
        break
    floor = 1.1 * bulk.radius_m
    out = []
    for name, lo, hi, tau, alb, kind in edges:
        inner, outer = max(lo * a_r, floor), hi * a_r
        if outer > inner:
            out.append(_ring(name, inner, outer, tau, alb, kind, star, 1.0))
    return _share_mass(out, h.BROAD_RING_MASS_RATIO * bulk.mass_kg)


def narrow_rings(star: StarState, bulk: BulkState, seed: int) -> list[RingState]:
    """Return Uranus-like dark ringlets inside the Roche limit for 1.2 g/cm³, the outermost at it."""
    rng = named_rng(seed, "rings.narrow")
    a_r = roche_limit_rubble(bulk.mass_kg, h.NARROW_RING_DENSITY)
    n = int(rng.integers(h.NARROW_RING_COUNT[0], h.NARROW_RING_COUNT[1] + 1))
    lo, hi = h.NARROW_RING_SPAN
    radii = np.sort(rng.uniform(lo, hi, n - 1)) * a_r
    radii = np.append(radii, hi * a_r)
    out = []
    for k, r in enumerate(radii):
        outermost = k == len(radii) - 1
        width = 1e3 * rng.uniform(*(h.NARROW_OUTER_WIDTH_KM if outermost else h.NARROW_RING_WIDTH_KM))
        inner = max(r - width / 2, 1.1 * bulk.radius_m)
        tau = float(rng.uniform(*h.NARROW_RING_TAU))
        out.append(_ring(f"ring {k + 1}", inner, inner + width, tau, h.NARROW_RING_ALBEDO, "dark", star, 1.0))
    return _share_mass(out, h.NARROW_RING_MASS_RATIO * bulk.mass_kg)


def dusty_rings(star: StarState, bulk: BulkState, cold: bool, seed: int) -> list[RingState]:
    """Return tenuous dusty rings in planet radii after Jupiter's (warm) or Neptune's (cold, with arcs)."""
    template = h.DUSTY_COLD_TEMPLATE if cold else h.DUSTY_WARM_TEMPLATE
    r_p = bulk.radius_m
    out = [_ring(name, lo * r_p, hi * r_p, tau, h.DUSTY_RING_ALBEDO, "dusty", star, 1.0)
           for name, lo, hi, tau in template]
    rings = _share_mass(out, h.DUSTY_RING_MASS_RATIO * bulk.mass_kg)
    if cold:
        rng = named_rng(seed, "rings.arcs")
        start = rng.uniform(0, 360)
        for _ in range(int(rng.integers(h.RING_ARC_COUNT[0], h.RING_ARC_COUNT[1] + 1))):
            length = rng.uniform(*h.RING_ARC_LENGTH_DEG)
            rings[-1].arcs.append([float(start % 360), float((start + length) % 360)])
            start += length + rng.uniform(1.0, 8.0)
    return rings


def user_rings(entries, star: StarState, bulk: BulkState) -> list[RingState]:
    """Return the rings the user listed, ordered outward."""
    out = []
    for k, e in enumerate(sorted(entries, key=lambda e: e.inner_radius_rp)):
        out.append(_ring(e.name or f"ring {k + 1}", e.inner_radius_rp * bulk.radius_m, e.outer_radius_rp * bulk.radius_m,
                         e.optical_depth, e.bond_albedo, e.kind, star, 1.0))
    total = (h.BROAD_RING_MASS_RATIO if any(r.optical_depth > 0.3 for r, _ in out) else h.DUSTY_RING_MASS_RATIO)
    return _share_mass(out, total * bulk.mass_kg)


def build_rings(spec_rings, star: StarState, orbit: OrbitState, bulk: BulkState, moons: list[MoonState],
                seed: int) -> tuple[list[RingState], str, list[Issue]]:
    """Return a planet's rings, the regime they follow and notes."""
    issues: list[Issue] = []
    if isinstance(spec_rings, list):
        rings = user_rings(spec_rings, star, bulk)
        a_r = roche_limit_rubble(bulk.mass_kg, h.RING_DENSITY_ICY)
        if any(r.outer_radius_m > 1.2 * a_r and r.optical_depth > 0.3 for r in rings):
            issues.append(Issue("info", "note", "system", "a dense ring lies well outside the ring–moon boundary; "
                                "its material would gather into moons unless something confines it"))
        return rings, "user", issues
    regime = ring_regime(star, orbit, bulk, seed) if spec_rings == "auto" else spec_rings
    if spec_rings not in ("auto", "none") and bulk.planet_class not in GIANT_CLASSES:
        issues.append(Issue("info", "note", "system", f"'{spec_rings}' rings are given to a {bulk.planet_class}"))
    if regime == "broad":
        rings = broad_rings(star, bulk, moons)
    elif regime == "narrow":
        rings = narrow_rings(star, bulk, seed)
    elif regime in ("dusty warm", "dusty cold"):
        rings = dusty_rings(star, bulk, regime == "dusty cold", seed)
    else:
        rings = []
    if rings and spec_rings == "auto":
        issues.append(Issue("info", "heuristic", "system", f"{regime} rings from the ring-temperature window "
                            "(Hedman 2015) and templates of the Solar System rings"))
    return rings, regime, issues
