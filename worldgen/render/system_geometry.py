"""Where a planet's moons are, and how a scene compresses distances.

Positions are in the planet frame: z along the spin axis, x towards the
ascending node of the planet's orbit on its equator. Each moon follows a
Kepler ellipse with its semi-major axis, eccentricity and inclination to the
equator; its node, argument of periapsis and mean anomaly at time zero are
drawn once from the planet's seed and the moon's name, so a system always
starts from the same configuration. Retrograde moons (inclination above 90°)
run backwards because their inclination flips the orbit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import heuristics as h
from ..util import named_rng


@dataclass(frozen=True)
class MoonOrbit:
    """A moon's orbit elements about its planet."""

    name: str
    a_m: float
    e: float
    inclination_rad: float
    node_rad: float
    periapsis_rad: float
    mean_anomaly_rad: float      # at time zero
    period_s: float
    radius_m: float

    def position(self, t_s: float) -> np.ndarray:
        """Return the moon's position (m, planet frame) at a time (s)."""
        m = self.mean_anomaly_rad + 2.0 * math.pi * t_s / self.period_s
        ecc = eccentric_anomaly(m, self.e)
        x = self.a_m * (math.cos(ecc) - self.e)
        y = self.a_m * math.sqrt(1.0 - self.e**2) * math.sin(ecc)
        return orbit_rotation(self.node_rad, self.inclination_rad, self.periapsis_rad) @ np.array([x, y, 0.0])

    def path(self, samples: int = 360) -> np.ndarray:
        """Return points (m, planet frame) around the whole orbit."""
        ecc = np.linspace(0.0, 2.0 * math.pi, samples)
        pts = np.column_stack([self.a_m * (np.cos(ecc) - self.e),
                               self.a_m * math.sqrt(1.0 - self.e**2) * np.sin(ecc), np.zeros(samples)])
        return pts @ orbit_rotation(self.node_rad, self.inclination_rad, self.periapsis_rad).T


def eccentric_anomaly(mean_anomaly: float, e: float) -> float:
    """Return the eccentric anomaly solving Kepler's equation M = E − e sin E (Newton's method)."""
    m = math.remainder(mean_anomaly, 2.0 * math.pi)
    ecc = m if e < 0.8 else math.pi
    for _ in range(50):
        step = (ecc - e * math.sin(ecc) - m) / (1.0 - e * math.cos(ecc))
        ecc -= step
        if abs(step) < 1e-12:
            break
    return ecc


def rotation_z(angle: float) -> np.ndarray:
    """Return the rotation about z by an angle (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(angle: float) -> np.ndarray:
    """Return the rotation about x by an angle (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def orbit_rotation(node: float, inclination: float, periapsis: float) -> np.ndarray:
    """Return the rotation from an orbit's plane (x to periapsis) to the planet frame."""
    return rotation_z(node) @ rotation_x(inclination) @ rotation_z(periapsis)


def moon_orbits(state) -> list[MoonOrbit]:
    """Return the orbits of a planet's moons, with seeded angles."""
    if state.system is None:
        return []
    out = []
    for m in state.system.moons:
        rng = named_rng(state.seed, f"system.orbit.{m.name}")
        inc = m.inclination_rad
        if m.retrograde and inc <= math.pi / 2:
            inc = math.pi - inc
        out.append(MoonOrbit(m.name, m.semi_major_axis_m, m.eccentricity, inc, float(rng.uniform(0, 2 * math.pi)),
                             float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0, 2 * math.pi)),
                             m.orbital_period_s, m.radius_m))
    return out


def planet_to_scene(obliquity_rad: float) -> np.ndarray:
    """Return the rotation from the planet frame to the scene frame (z normal to the planet's orbit)."""
    return rotation_x(obliquity_rad)


class Compression:
    """The scene's distance scale: true, or true out past the rings and logarithmic beyond."""

    def __init__(self, planet_radius_m: float, outer_ring_m: float = 0.0, mode: str = "compressed"):
        if mode not in ("compressed", "true"):
            raise ValueError("scale must be 'compressed' or 'true'")
        self.mode = mode
        self.planet_radius_m = planet_radius_m
        self.r0 = max(h.SCENE_COMPRESS_FROM_RP * planet_radius_m, 1.05 * outer_ring_m)

    def distance(self, r: np.ndarray) -> np.ndarray:
        """Return drawn distances for true distances from the planet's centre."""
        r = np.asarray(r, dtype=float)
        if self.mode == "true":
            return r
        far = self.r0 * (1.0 + h.SCENE_COMPRESS_GAIN * np.log(np.maximum(r, self.r0) / self.r0))
        return np.where(r <= self.r0, r, far)

    def point(self, p: np.ndarray) -> np.ndarray:
        """Return drawn positions for true positions (radial compression)."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        r = np.linalg.norm(p, axis=1)
        scale = np.where(r > 0, self.distance(r) / np.where(r > 0, r, 1.0), 1.0)
        return p * scale[:, None]

    def moon_radius(self, radius_m: float) -> float:
        """Return a moon's drawn radius: true, or inflated to stay visible."""
        if self.mode == "true":
            return radius_m
        inflated = h.SCENE_MOON_RADIUS_RP * self.planet_radius_m * (radius_m / 1e6) ** (1.0 / 3.0)
        return max(radius_m, inflated)
