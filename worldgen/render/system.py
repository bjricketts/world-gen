"""A planet's system as a diagram (task 11.9): its moons' orbits, rings and limits, seen from above or edge-on.

Distances are in planet radii on a logarithmic radial axis by default (the
planet's surface at 1, each further unit ten times farther), so close ring
systems and far moons share one figure; ``radial='linear'`` keeps true
proportions. The planet is drawn in its appearance colour, rings as annuli
as opaque as 1 − exp(−τ), the limits as labelled circles, and the moons at
their seeded positions (``system_geometry``), sized by radius on a log scale
and coloured by type: rocky, icy, with a subsurface ocean or with air (the
last two only for moons whose worlds are given). Retrograde orbits are
dashed and resonant pairs joined and labelled with their ratio. The edge-on
view looks along the planet's orbit, with the spin axis tilted by the
obliquity. A side table lists the moons. Plots carry no title.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .. import constants as c
from .. import heuristics as h
from .system_geometry import moon_orbits, planet_to_scene

ROCKY_PLANET_COLOUR = "#8a7f74"


def radial_map(r_rp: np.ndarray, radial: str = "log") -> np.ndarray:
    """Return drawn radii for distances in planet radii: 1 + log10(r) beyond the surface, or r itself."""
    r_rp = np.asarray(r_rp, dtype=float)
    if radial == "linear":
        return r_rp
    return np.where(r_rp <= 1.0, r_rp, 1.0 + np.log10(np.maximum(r_rp, 1.0)))


def _project(points_m: np.ndarray, radius_m: float, view: str, obliquity: float, radial: str) -> np.ndarray:
    """Return 2D diagram coordinates of planet-frame points for a view.

    The radial map acts on each point's distance from the planet's centre in
    3D, so a sphere of any radius draws as a circle of the mapped radius and
    an orbit seen edge-on as a line.
    """
    scene = points_m if view == "top" else points_m @ planet_to_scene(obliquity).T
    xy = scene[:, :2] if view == "top" else scene[:, [1, 2]]     # edge-on: looking along x, so the tilt shows
    r = np.linalg.norm(points_m, axis=1) / radius_m
    drawn = radial_map(r, radial)
    scale = np.where(r > 0, drawn / np.where(r > 0, r, 1.0), 1.0)
    return xy / radius_m * scale[:, None]


def moon_kind(moon, world=None) -> str:
    """Return 'air', 'ocean', 'icy' or 'rocky' for a moon (the first two need its built world)."""
    if world is not None:
        state = world.state if hasattr(world, "state") else world
        if state.atmosphere.present and state.atmosphere.surface_pressure_pa > 1e3:
            return "air"
        if state.ice_shell is not None and state.ice_shell.ocean_depth_m > 0:
            return "ocean"
    return "icy" if moon.water_mass_fraction > h.SYSTEM_ICY_WATER_FRACTION else "rocky"


def _marker_size(radius_m: float) -> float:
    """Return a marker size (points) growing with the log of a moon's radius."""
    lo, hi = h.SYSTEM_MOON_MARKER_PT
    x = (math.log10(max(radius_m, 1e4)) - 4.0) / (math.log10(3e6) - 4.0)
    return lo + (hi - lo) * min(max(x, 0.0), 1.0)


def _limits(state) -> list[tuple[str, float, str]]:
    """Return the system's limits as (label, radius m, line style), skipping any inside the planet."""
    lim = state.system.limits
    out = [("icy Roche", lim.roche_icy_m, ":"), ("rocky Roche", lim.roche_rocky_m, ":"),
           ("synchronous", lim.synchronous_orbit_m, "--"), ("stable prograde", lim.stable_prograde_m, "-."),
           ("stable retrograde", lim.stable_retrograde_m, "-."), ("Hill radius", lim.hill_radius_m, "-")]
    return [(name, r, ls) for name, r, ls in out if r > state.bulk.radius_m]


def extent_hint(state, radial: str) -> float:
    """Return the diagram's half-width (drawn units): the Hill radius or the farthest moon, whichever is larger."""
    radius = state.bulk.radius_m
    far = max([state.system.limits.hill_radius_m] + [m.semi_major_axis_m * (1 + m.eccentricity)
                                                     for m in state.system.moons])
    return float(radial_map(far / radius, radial))


def plot_system(world, view: str = "top", radial: str = "log", highlight: Optional[str] = None,
                moon_worlds: Optional[dict] = None, table: bool = True, path: str | Path | None = None,
                time_s: float = 0.0):
    """Return a figure of a planet's system (and write it to ``path``, PNG or SVG, if given).

    ``world`` is a World or a PlanetState with a system; ``highlight`` names a
    moon to mark (a moon's world drawn in its parent's system); ``moon_worlds``
    maps moon names to built worlds or states, for the ocean and air colours.
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Annulus, Circle

    if view not in ("top", "edge"):
        raise ValueError("view must be 'top' or 'edge'")
    state = world.state if hasattr(world, "state") else world
    if state.system is None:
        raise ValueError(f"{state.name} has no system summary (a moon's world: draw its parent's system "
                         "with highlight=<moon name>)")
    moon_worlds = moon_worlds or {}
    radius = state.bulk.radius_m
    obliquity = state.orbit.obliquity_rad
    sy = state.system
    orbits = moon_orbits(state)
    fig = plt.figure(figsize=(13, 8) if table and sy.moons else (8, 8))
    ax = fig.add_axes([0.02, 0.04, 0.6 if table and sy.moons else 0.96, 0.92])
    ax.set_facecolor("white")
    colour = state.appearance.colour if state.appearance is not None else ROCKY_PLANET_COLOUR
    if view == "top":
        ax.add_patch(Circle((0, 0), 1.0, color=colour, zorder=3))
        for ring in sy.rings:
            r_in, r_out = radial_map([ring.inner_radius_m / radius, ring.outer_radius_m / radius], radial)
            alpha = 1.0 - math.exp(-ring.optical_depth)
            ax.add_patch(Annulus((0, 0), r_out, max(r_out - r_in, 1e-3), color=ring.colour,
                                 alpha=max(alpha, h.SYSTEM_RING_MIN_ALPHA), lw=0, zorder=2))
    else:
        ax.add_patch(Circle((0, 0), 1.0, color=colour, zorder=3))
        axis = planet_to_scene(obliquity) @ np.array([0.0, 0.0, 1.0])
        ax.plot([-1.5 * axis[0], 1.5 * axis[0]], [-1.5 * axis[2], 1.5 * axis[2]], color="0.3", lw=0.8, zorder=4)
        circle = np.linspace(0.0, 2.0 * math.pi, 721)
        for ring in sy.rings:
            alpha = max(1.0 - math.exp(-ring.optical_depth), h.SYSTEM_RING_MIN_ALPHA)
            for r in np.linspace(ring.inner_radius_m, ring.outer_radius_m, 5):
                pts = np.column_stack([r * np.cos(circle), r * np.sin(circle), np.zeros_like(circle)])
                xy = _project(pts, radius, view, obliquity, radial)
                near = (pts @ planet_to_scene(obliquity).T)[:, 0] > 0       # the viewer is on +x
                for front in (False, True):
                    seg = np.where((near == front)[:, None], xy, np.nan)
                    ax.plot(seg[:, 0], seg[:, 1], color=ring.colour, lw=1.2, alpha=alpha, zorder=4 if front else 2)
    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    extent = 1.2
    shades = ["#9e9e9e", "#7f7f7f", "#5b8db8", "#6a9a5b", "#b07a3c", "#555555"]
    for k, (name, r, style) in enumerate(_limits(state)):
        rr = float(radial_map(r / radius, radial))      # limits are spheres: circles in either view
        ax.plot(rr * np.cos(theta), rr * np.sin(theta), style, color=shades[k % len(shades)], lw=0.8, zorder=1,
                label=f"{name} ({r / radius:,.1f} R_p)")
        extent = max(extent, rr)
    positions = {}
    labelled: list[np.ndarray] = []
    spacing = h.SYSTEM_LABEL_SPACING * extent_hint(state, radial)
    for o, moon in zip(orbits, sy.moons):
        xy = _project(o.path(), radius, view, obliquity, radial)
        ax.plot(xy[:, 0], xy[:, 1], "--" if moon.retrograde else "-", color="0.8", lw=0.6, zorder=1)
        p = _project(o.position(time_s)[None, :], radius, view, obliquity, radial)[0]
        positions[moon.name] = p
        kind = moon_kind(moon, moon_worlds.get(moon.name))
        face = h.SYSTEM_MOON_COLOURS["highlight" if moon.name == highlight else kind]
        ax.plot(*p, "o", ms=_marker_size(moon.radius_m), color=face, mec="0.2", mew=0.5, zorder=6)
        extent = max(extent, float(np.abs(xy).max()))
    for moon in sorted(sy.moons, key=lambda m: (m.name != highlight, -m.radius_m)):   # big moons labelled first
        p = positions[moon.name]
        if any(np.hypot(*(p - q)) < spacing for q in labelled):
            continue                          # the table lists every moon; crowded labels are dropped
        labelled.append(p)
        ax.annotate(moon.name + (" (retrograde)" if moon.retrograde else ""), p, xytext=(5, 4),
                    textcoords="offset points", fontsize=7, zorder=7,
                    fontweight="bold" if moon.name == highlight else "normal")
    for moon in sy.moons:
        if moon.resonance and " with " in moon.resonance:
            ratio, other = moon.resonance.split(" with ", 1)
            if other in positions:
                a, b = positions[moon.name], positions[other]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#7a5ea8", lw=0.8, alpha=0.7, zorder=5)
                mid = (a + b) / 2
                ax.text(mid[0], mid[1], ratio, fontsize=7, color="#7a5ea8", ha="center", va="center", zorder=7)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_xlim(-1.05 * extent, 1.05 * extent)
    ax.set_ylim(-1.05 * extent, 1.05 * extent)
    ax.set_aspect("equal")
    ax.axis("off")
    note = ("radius: 1 + log₁₀(distance / planet radius)" if radial == "log" else "distance in planet radii") + \
        (" · view down the spin axis" if view == "top" else " · edge-on, spin axis tilted by the obliquity")
    ax.text(0.0, -0.02, note, transform=ax.transAxes, fontsize=7, color="0.4", va="top")
    if table and sy.moons:
        _moon_table(fig, sy.moons, radius)
    if path is not None:
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    return fig


def _moon_table(fig, moons, planet_radius_m: float) -> None:
    """Add a table of the moons' mass, radius, orbit, period and origin beside the diagram."""
    ax = fig.add_axes([0.64, 0.04, 0.35, 0.92])
    ax.axis("off")
    rows = [[m.name, f"{m.mass_kg / c.M_EARTH:.3g}", f"{m.radius_m / 1e3:,.0f}",
             f"{m.semi_major_axis_m / planet_radius_m:.1f}", f"{m.orbital_period_s / c.SECONDS_PER_DAY:.3g}",
             m.origin + (" (r)" if m.retrograde else "")] for m in moons]
    table = ax.table(cellText=rows, colLabels=["moon", "Earth masses", "km", "R_p", "days", "origin"], loc="upper left",
                     cellLoc="right", colLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.1)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.2)
