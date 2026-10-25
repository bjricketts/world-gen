"""A planet's system in 3D (task 11.9): the planet, its rings and moons lit by the star, with shadows.

The scene frame has z normal to the planet's orbit; the planet's spin axis is
tilted from it by the obliquity, and the star lies in the orbit plane at an
angle set by ``season`` (radians; 0 puts the star on +x, where neither pole
leans towards it). Distances are compressed by default (``Compression``):
true out to just past the rings, logarithmic beyond, with moons inflated to
stay visible; ``scale='true'`` keeps true proportions.

Light is computed per vertex and baked into the colours, so windows, images,
HTML pages and movies all show the same shading: Lambert on the planet and
moons, the rings' shadow on the planet (transmission exp(−τ/μ) along the ray
to the star), the planet's shadow on the rings and moons, and moons' shadows
on the planet and rings, all from true positions and sizes (transit shadows
are therefore where the true moon would cast them). An ambient term keeps the
night side faintly visible. Rings are as opaque as 1 − exp(−τ/μ) for the
camera's view angle μ at the start.

Moons are drawn with their own surfaces when their worlds are given
(``moon_worlds``) or kept in the planet's saved folder (``folder``); others
are spheres coloured by type. Moons are never built here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .. import heuristics as h
from .rings import _rgb, ring_mesh_arrays
from .system import moon_kind
from .system_geometry import Compression, moon_orbits, planet_to_scene, rotation_z


def lambert(normals: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Return cos(incidence) clipped at zero for unit normals and a unit light direction."""
    return np.clip(normals @ light, 0.0, 1.0)


def sphere_lit(points: np.ndarray, light: np.ndarray, centre, radius: float) -> np.ndarray:
    """Return True where the ray from a point towards the light misses a sphere (the point is not in its shadow)."""
    rel = points - np.asarray(centre, dtype=float)
    b = rel @ light
    c = (rel * rel).sum(axis=1) - radius * radius
    disc = b * b - c
    hit = (disc > 0) & (-b + np.sqrt(np.maximum(disc, 0.0)) > 0) & (c > 0)
    return ~hit


def ring_transmission(points: np.ndarray, light: np.ndarray, rings) -> np.ndarray:
    """Return the fraction of starlight passing the rings to points (planet frame): exp(−τ/μ) per ring crossed."""
    out = np.ones(len(points))
    mu = abs(light[2])
    if mu < 1e-6 or not rings:
        return out
    s = -points[:, 2] / light[2]
    hit = points + s[:, None] * light
    r = np.hypot(hit[:, 0], hit[:, 1])
    for ring in rings:
        crossed = (s > 0) & (r >= ring.inner_radius_m) & (r <= ring.outer_radius_m)
        out[crossed] *= math.exp(-ring.optical_depth / mu)
    return out


def star_direction(season: float, obliquity: float) -> np.ndarray:
    """Return the unit direction to the star in the planet frame for an orbital angle ``season`` (rad)."""
    scene = np.array([math.cos(season), math.sin(season), 0.0])
    return planet_to_scene(obliquity).T @ scene


def _shade(rgb: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Return colours dimmed to the ambient level where unlit."""
    return np.clip(rgb * (h.SCENE_AMBIENT + (1.0 - h.SCENE_AMBIENT) * light[:, None]), 0.0, 1.0)


class SystemScene:
    """The planet, rings and moons of a world as PyVista meshes, shaded for a time and season."""

    def __init__(self, world, scale: str = "compressed", season: float = math.radians(60.0), level: int = 1,
                 moon_worlds: Optional[dict] = None, folder: str | Path | None = None):
        from .globe import GlobeViewer, _pyvista

        pv = _pyvista()
        state = world.state
        if state.system is None:
            raise ValueError(f"{state.name} has no system summary (a moon's world: show its parent's system)")
        self.world, self.state = world, state
        self.season = season
        self.obliquity = state.orbit.obliquity_rad
        self.to_scene = planet_to_scene(self.obliquity)
        self.light = star_direction(season, self.obliquity)
        sy = state.system
        self.rings = list(sy.rings)
        outer = max([r.outer_radius_m for r in self.rings], default=0.0)
        self.compression = Compression(state.bulk.radius_m, outer, scale)
        self.orbits = moon_orbits(state)
        self.time_s = 0.0
        self.spin_period_s = state.orbit.rotation_period_s

        globe = GlobeViewer(world, level=level)
        self.radius_m = globe.radius_m
        self._planet_points = np.asarray(globe.terrain.points).copy()
        self._planet_rgb = globe.colours("terrain" if "terrain" in globe.fields else globe.field)[:, :3] / 255.0
        self.planet = globe.terrain.copy()
        self.sea = globe.sea.copy() if globe.sea is not None else None
        self._sea_points = np.asarray(globe.sea.points).copy() if globe.sea is not None else None
        self._sea_rgb = np.asarray(globe.sea["rgba"])[:, :3] / 255.0 if globe.sea is not None else None

        self.ring_mesh = None
        if self.rings:
            v, f, rgba = ring_mesh_arrays(self.rings, h.SCENE_RING_SEGMENTS, h.SCENE_RING_RADIAL_STEPS)
            self._ring_points = v
            self._ring_rgb = rgba[:, :3] / 255.0
            self._ring_tau = -np.log(np.clip(1.0 - rgba[:, 3] / 255.0, 1e-6, 1.0))
            self.ring_mesh = pv.PolyData(v @ self.to_scene.T, np.column_stack([np.full(len(f), 3), f]).ravel())

        moon_worlds = dict(moon_worlds or {})
        if folder is not None:
            moon_worlds.update(_kept_moons(world, folder, exclude=moon_worlds))
        self.moons = []
        for o, moon in zip(self.orbits, sy.moons):
            mw = moon_worlds.get(moon.name)
            drawn = self.compression.moon_radius(moon.radius_m)
            if mw is not None and getattr(mw, "surface", None) is not None:
                g = GlobeViewer(mw, level=0, sea=False, exaggeration=h.SCENE_MOON_EXAGGERATION)
                unit = np.asarray(g.terrain.points) / g.radius_m
                mesh = g.terrain.copy()
                rgb = g.colours("terrain" if "terrain" in g.fields else g.field)[:, :3] / 255.0
            else:
                sphere = pv.Sphere(radius=1.0, theta_resolution=h.SCENE_MOON_SPHERE_RESOLUTION,
                                   phi_resolution=h.SCENE_MOON_SPHERE_RESOLUTION)
                unit = np.asarray(sphere.points)
                mesh = sphere
                rgb = np.tile(_rgb(h.SYSTEM_MOON_COLOURS[moon_kind(moon, mw)]), (len(unit), 1))
            self.moons.append({"name": moon.name, "orbit": o, "unit": unit, "rgb": rgb, "mesh": mesh,
                               "drawn_radius": drawn, "true_radius": moon.radius_m})
        self._plotter = None
        self.update(0.0)

    def _moon_true_positions(self, t_s: float) -> list[np.ndarray]:
        return [m["orbit"].position(t_s) for m in self.moons]

    def update(self, t_s: float) -> None:
        """Move the moons and the planet's spin to a time (s) and recompute the light on every mesh."""
        self.time_s = t_s
        light = self.light
        centres = self._moon_true_positions(t_s)
        spin = rotation_z(2.0 * math.pi * t_s / self.spin_period_s) if self.spin_period_s else np.eye(3)

        def lit_planet_surface(points):
            normals = points / np.linalg.norm(points, axis=1)[:, None]
            shade = lambert(normals, light) * ring_transmission(points, light, self.rings)
            for c, m in zip(centres, self.moons):
                shade = shade * sphere_lit(points, light, c, m["true_radius"])
            return shade

        pts = self._planet_points @ spin.T
        self.planet.points = pts @ self.to_scene.T
        self.planet["shaded"] = _shade(self._planet_rgb, lit_planet_surface(pts))
        if self.sea is not None:
            sp = self._sea_points @ spin.T
            self.sea.points = sp @ self.to_scene.T
            self.sea["shaded"] = _shade(self._sea_rgb, lit_planet_surface(sp))
        if self.ring_mesh is not None:
            lit = sphere_lit(self._ring_points, light, (0.0, 0.0, 0.0), self.radius_m).astype(float)
            for c, m in zip(centres, self.moons):
                lit *= sphere_lit(self._ring_points, light, c, m["true_radius"])
            self.ring_mesh["shaded"] = _shade(self._ring_rgb, lit)
        for c, m in zip(centres, self.moons):
            drawn_centre = self.compression.point(c)[0]
            facing = rotation_z(math.atan2(c[1], c[0]))          # the same face towards the planet
            unit = m["unit"] @ facing.T
            true_pts = c + unit * m["true_radius"]
            shade = lambert(unit, light) * sphere_lit(true_pts, light, (0.0, 0.0, 0.0), self.radius_m)
            shade = shade * ring_transmission(true_pts, light, self.rings)
            m["mesh"].points = (drawn_centre + unit * m["drawn_radius"]) @ self.to_scene.T
            m["mesh"]["shaded"] = _shade(m["rgb"], shade)

    def extent(self) -> float:
        """Return the drawn distance (m) of the farthest moon or ring from the planet's centre."""
        far = [self.radius_m] + [r.outer_radius_m for r in self.rings]
        far += [float(self.compression.distance(m["orbit"].a_m * (1 + m["orbit"].e))) for m in self.moons]
        return max(far)

    def camera(self, distance: Optional[float] = None) -> list:
        """Return a camera on the day side, above the orbit plane, framing the whole system."""
        star = self.to_scene @ self.light
        side = np.cross([0.0, 0.0, 1.0], star)
        eye = star + 0.6 * side + np.array([0.0, 0.0, 0.45])
        eye /= np.linalg.norm(eye)
        distance = distance or h.SCENE_CAMERA_DISTANCE * self.extent()
        return [tuple(eye * distance), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def _ring_opacity(self, eye: np.ndarray) -> np.ndarray:
        """Return the rings' vertex opacity 1 − exp(−τ/μ) for a viewing direction."""
        normal = self.to_scene @ np.array([0.0, 0.0, 1.0])
        mu = max(abs(float(normal @ (eye / np.linalg.norm(eye)))), 0.05)
        return 1.0 - np.exp(-self._ring_tau / mu)

    def _label(self) -> None:
        days = self.time_s / 86400.0
        scale = "compressed scale" if self.compression.mode == "compressed" else "true scale"
        self._plotter.add_text(f"{self.state.name} · {scale} · t = {days:,.2f} days", position="lower_left",
                               font_size=10, name="label")

    def plotter(self, off_screen: bool = False, window_size=(1200, 900), **kwargs):
        """Return a PyVista plotter holding the scene, its camera on the day side."""
        from .globe import _pyvista

        pv = _pyvista()
        pl = pv.Plotter(off_screen=off_screen, window_size=window_size, **kwargs)
        pl.add_mesh(self.planet, scalars="shaded", rgb=True, lighting=False)
        if self.sea is not None:
            pl.add_mesh(self.sea, scalars="shaded", rgb=True, lighting=False, opacity=0.85)
        cam = self.camera()
        if self.ring_mesh is not None:
            self.ring_mesh["opacity"] = self._ring_opacity(np.array(cam[0]))
            rgba = np.column_stack([self.ring_mesh["shaded"], self.ring_mesh["opacity"]])
            self.ring_mesh["rgba"] = np.round(rgba * 255).astype(np.uint8)
            pl.add_mesh(self.ring_mesh, scalars="rgba", rgba=True, lighting=False)
        for m in self.moons:
            pl.add_mesh(m["mesh"], scalars="shaded", rgb=True, lighting=False)
        pl.set_background("black")
        pl.camera_position = cam
        self._plotter = pl
        self._label()
        return pl

    def _refresh_rings(self) -> None:
        """Copy the rings' shading into their RGBA array (opacity kept)."""
        if self.ring_mesh is not None and "opacity" in self.ring_mesh.point_data:
            rgba = np.column_stack([self.ring_mesh["shaded"], self.ring_mesh["opacity"]])
            self.ring_mesh["rgba"] = np.round(rgba * 255).astype(np.uint8)

    def set_time(self, t_s: float) -> None:
        """Move the scene to a time (s), redrawing any open plotter."""
        self.update(t_s)
        self._refresh_rings()
        if self._plotter is not None:
            self._label()
            self._plotter.render()

    def _close(self) -> None:
        self._plotter.close()
        self._plotter = None

    def screenshot(self, path: str | Path, t_s: float = 0.0, window_size=(1200, 900)) -> Path:
        """Write an image of the scene at a time."""
        self.plotter(off_screen=True, window_size=window_size)
        self.set_time(t_s)
        self._plotter.screenshot(str(path))
        self._close()
        return Path(path)

    def save_html(self, path: str | Path, t_s: float = 0.0) -> Path:
        """Write the scene at a time as a standalone interactive HTML page (vtk.js)."""
        pl = self.plotter(off_screen=True)
        self.set_time(t_s)
        exporter = getattr(pl, "trame", None)
        (exporter if exporter is not None and hasattr(exporter, "export_html") else pl).export_html(str(path))
        self._close()
        return Path(path)

    def default_duration_s(self) -> float:
        """Return a movie's default span: one orbit of the outermost moon, or one spin without moons."""
        periods = [m["orbit"].period_s for m in self.moons]
        return max(periods) if periods else self.spin_period_s

    def save_movie(self, path: str | Path, duration_s: Optional[float] = None, frames: int = 120,
                   fps: int = 24, window_size=(960, 720)) -> Path:
        """Write a movie (MP4, or GIF by the suffix) of the system over a span of time (s)."""
        duration_s = duration_s or self.default_duration_s()
        pl = self.plotter(off_screen=True, window_size=window_size)
        path = Path(path)
        if path.suffix.lower() == ".gif":
            pl.open_gif(str(path), fps=fps)
        else:
            pl.open_movie(str(path), framerate=fps)
        for t in np.linspace(0.0, duration_s, frames, endpoint=False):
            self.set_time(float(t))
            pl.write_frame()
        self._close()
        return path

    def show(self, t_s: float = 0.0) -> None:
        """Open a window with a slider for time, over the default movie span."""
        pl = self.plotter()
        self.set_time(t_s)
        span = self.default_duration_s() / 86400.0
        pl.add_slider_widget(lambda days: self.set_time(days * 86400.0), [0.0, span], value=t_s / 86400.0,
                             title="days", pointa=(0.6, 0.08), pointb=(0.95, 0.08))
        pl.show()
        self._plotter = None


def _kept_moons(world, folder: str | Path, exclude: dict) -> dict:
    """Return the moons' worlds kept in a planet's saved folder (those not already given)."""
    from ..giants.system import kept_moons
    from ..world import load_world

    return {name: load_world(path) for name, path in kept_moons(folder, world).items() if name not in exclude}


def system_scene(world, scale: str = "compressed", season: float = math.radians(60.0), level: int = 1,
                 moon_worlds: Optional[dict] = None, folder: str | Path | None = None) -> SystemScene:
    """Return the 3D scene of a world's system (see ``SystemScene``)."""
    return SystemScene(world, scale, season, level, moon_worlds, folder)
