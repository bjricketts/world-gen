"""A 3D globe of a world: any map field as vertex colours, the kept epochs of an organic history, and zoomed regions
draped over it, in a PyVista window, a Jupyter widget, screenshots or a standalone HTML page.

The globe is built once from tile meshes (``export.globe_meshes``: the global
grid resampled onto the cube-sphere, or a pyramid level); switching field or
epoch only recolours its vertices from the map colours (0.01–0.1 s). Regions
are drawn as overlays raised slightly above the globe, and a planet's rings as
annuli in its equatorial plane (``render.rings``). PyVista is needed (the
``globe`` extra); the Jupyter widget also needs ipywidgets and a PyVista
Jupyter backend (trame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .. import heuristics as h
from .explore import available_fields


def _pyvista():
    """Return the pyvista module, or raise with the extra to install."""
    try:
        import pyvista
    except ImportError as err:      # pragma: no cover - depends on the installation
        raise ImportError("the 3D globe needs PyVista: pip install 'worldgen[globe]'") from err
    try:                            # keep VTK warnings (no X display for off-screen images) off the terminal
        from vtkmodules.vtkCommonCore import vtkLogger

        vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)
    except ImportError:             # pragma: no cover
        pass
    return pyvista


def _polydata(meshes):
    """Return one PolyData (planet-centred metres) of several tile meshes, and their stacked colours."""
    pv = _pyvista()
    vertices = np.vstack([m.planet_vertices() for m in meshes])
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes])[:-1]
    faces = np.vstack([m.faces + o for m, o in zip(meshes, offsets)])
    poly = pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    colours = np.vstack([m.colours for m in meshes])
    poly["rgba"] = colours
    return poly, colours


class GlobeViewer:
    """A world's globe with selectable map field, kept epoch, sea and draped regions."""

    def __init__(self, world, level: int = 1, pyramid=None, exaggeration: Optional[float] = None,
                 max_error_m: Optional[float] = None, sea: bool = True):
        """Build the globe's meshes at a quadtree level (from a pyramid if given) with exaggerated heights.

        ``exaggeration`` defaults to GLOBE_EXAGGERATION for solid planets and
        GLOBE_CLOUD_EXAGGERATION for a giant's cloud tops.
        """
        from ..export import globe_meshes

        if world.surface is None:
            raise ValueError(f"{world.state.name} has no solid surface")
        self.world = world
        if exaggeration is None:
            exaggeration = h.GLOBE_CLOUD_EXAGGERATION if world.cloud_tops else h.GLOBE_EXAGGERATION
        self.exaggeration = exaggeration
        self.radius_m = float(world.surface.attrs["radius_m"])
        meshes = globe_meshes(world, level, "elevation", max_error_m, exaggeration, sea, pyramid)
        self.terrain, base = _polydata([m for m in meshes if m.layer == "terrain"])
        sea_meshes = [m for m in meshes if m.layer == "sea"]
        self.sea = _polydata(sea_meshes)[0] if sea_meshes else None
        points = np.asarray(self.terrain.points)
        r = np.linalg.norm(points, axis=1)
        self.directions = points / r[:, None]
        self.heights = (r - self.radius_m) / exaggeration
        self._elevation = base                    # the meshes come coloured by elevation
        system = world.state.system
        from .rings import ring_polydata

        self.rings = ring_polydata(system.rings) if system is not None and system.rings else None
        self.fields = available_fields(world)
        from .epochs import epoch_dates

        self.dates = epoch_dates(world) if world.culture is not None and world.culture.epochs else [0.0]
        self.regions: list = []
        self._sampler = None
        self.field, self.epoch = ("clouds" if world.cloud_tops else "elevation"), 0.0
        self._plotter = None

    def sampler(self):
        """Return the lookup from the globe's vertices to grid cells (built once)."""
        if self._sampler is None:
            from ..export.sampling import world_sampler

            self._sampler = world_sampler(self.world, self.directions, (1, len(self.directions)))
        return self._sampler

    def at(self, years_ago: float):
        """Return the world at a kept epoch (the world itself without an organic history)."""
        if self.world.culture is None or not self.world.culture.epochs or years_ago <= 0:
            return self.world
        from ..culture import world_at_epoch

        return world_at_epoch(self.world, years_ago)

    def colours(self, field: str = "elevation", years_ago: float = 0.0) -> np.ndarray:
        """Return RGBA (uint8) colours of the globe's vertices for a map field at a kept epoch."""
        from ..export.meshes import _rgba
        from .maps import field_colours

        if field == "elevation" or (self.world.cloud_tops and field == "clouds"):
            return self._elevation
        return _rgba(field_colours(self.at(years_ago), field, self.sampler())[..., :3].reshape(-1, 3))

    def show_field(self, field: str = "elevation", years_ago: Optional[float] = None) -> None:
        """Recolour the globe for a field and epoch (the nearest kept epoch at or after ``years_ago``)."""
        years = self.epoch if years_ago is None else min(self.dates, key=lambda d: abs(d - years_ago))
        if self.world.cloud_tops and field == "elevation":
            field = "clouds"            # a giant's 'elevation' view is its cloud tops
        if field not in self.fields:
            raise ValueError(f"'{field}' is not available; choose from {', '.join(self.fields)}")
        self.terrain["rgba"] = self.colours(field, years)
        self.field, self.epoch = field, years
        if self._plotter is not None:
            self._label()
            self._plotter.render()

    def add_region(self, region, mode: str = "relief", lift_m: float = h.GLOBE_REGION_LIFT_M):
        """Drape a zoomed region over the globe, raised ``lift_m`` (before exaggeration) above it; return its mesh."""
        from ..export.meshes import region_meshes

        (mesh, *_) = region_meshes(self.world, region, mode, exaggeration=self.exaggeration, sea=False)
        poly, _ = _polydata([mesh])
        points = np.asarray(poly.points)
        r = np.linalg.norm(points, axis=1)
        poly.points = points * ((r + lift_m * self.exaggeration) / r)[:, None]
        self.regions.append(poly)
        if self._plotter is not None:
            self._plotter.add_mesh(poly, scalars="rgba", rgba=True, smooth_shading=True)
        return poly

    def _label(self) -> None:
        """Write the field and epoch in the window's corner."""
        when = "present" if self.epoch <= 0 else f"{self.epoch:,.0f} years ago"
        self._plotter.add_text(f"{self.field} · {when}", position="lower_left", font_size=10, name="label")

    def plotter(self, off_screen: bool = False, window_size=(1000, 1000), **kwargs):
        """Return a PyVista plotter holding the globe, its sea and draped regions."""
        pv = _pyvista()
        pl = pv.Plotter(off_screen=off_screen, window_size=window_size, **kwargs)
        pl.add_mesh(self.terrain, scalars="rgba", rgba=True, smooth_shading=True)
        if self.sea is not None:
            pl.add_mesh(self.sea, scalars="rgba", rgba=True, opacity=0.85, smooth_shading=True)
        for poly in self.regions:
            pl.add_mesh(poly, scalars="rgba", rgba=True, smooth_shading=True)
        if self.rings is not None:
            pl.add_mesh(self.rings, scalars="rgba", rgba=True, lighting=False)
        pl.set_background("black")
        self._plotter = pl
        self._label()
        return pl

    def look_at(self, lat_deg: float, lon_deg: float, distance: float = 3.0) -> None:
        """Turn the camera to look down on a place from ``distance`` planet radii."""
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        eye = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]) * distance * self.radius_m
        up = (0.0, 0.0, 1.0) if abs(lat_deg) < 80 else (np.cos(lon), np.sin(lon), 0.0)
        self._plotter.camera_position = [tuple(eye), (0.0, 0.0, 0.0), up]

    def screenshot(self, path: str | Path, field: str = "elevation", years_ago: Optional[float] = None,
                   lat_deg: float = 20.0, lon_deg: float = 0.0, distance: float = 3.0, window_size=(1000, 1000)) -> Path:
        """Write an image of the globe showing a field, seen from above a place."""
        self.plotter(off_screen=True, window_size=window_size)
        self.show_field(field, years_ago)
        self.look_at(lat_deg, lon_deg, distance)
        self._plotter.screenshot(str(path))
        self._plotter.close()
        self._plotter = None
        return Path(path)

    def save_html(self, path: str | Path, field: str = "elevation", years_ago: Optional[float] = None,
                  lat_deg: float = 20.0, lon_deg: float = 0.0) -> Path:
        """Write the globe showing a field as a standalone interactive HTML page (vtk.js)."""
        pl = self.plotter(off_screen=True)
        self.show_field(field, years_ago)
        self.look_at(lat_deg, lon_deg)
        exporter = getattr(pl, "trame", None)
        (exporter if exporter is not None and hasattr(exporter, "export_html") else pl).export_html(str(path))
        pl.close()
        self._plotter = None
        return Path(path)

    def show(self, field: str = "elevation", years_ago: Optional[float] = None, lat_deg: float = 20.0,
             lon_deg: float = 0.0) -> None:
        """Open a window: the globe with keys n / p for the next and previous field and an epoch slider."""
        pl = self.plotter()
        self.show_field(field, years_ago)
        self.look_at(lat_deg, lon_deg)

        def step(k):
            i = self.fields.index(self.field) if self.field in self.fields else 0
            self.show_field(self.fields[(i + k) % len(self.fields)])

        pl.add_key_event("n", lambda: step(1))
        pl.add_key_event("p", lambda: step(-1))
        if len(self.dates) > 1:
            pl.add_slider_widget(lambda v: self.show_field(self.field, v), [min(self.dates), max(self.dates)],
                                 value=self.epoch, title="years ago", pointa=(0.6, 0.08), pointb=(0.95, 0.08))
        pl.show()
        self._plotter = None

    def widget(self, backend: str = "trame"):
        """Return a Jupyter widget: field and epoch menus over the interactive globe (``backend`` as PyVista's)."""
        import ipywidgets as widgets

        pl = self.plotter(notebook=True)
        self.look_at(20.0, 0.0)
        field = widgets.Dropdown(options=self.fields, value=self.field, description="field")
        when = widgets.SelectionSlider(options=[(("present" if d <= 0 else f"{d:,.0f} ya"), d) for d in self.dates],
                                       value=self.epoch, description="epoch", continuous_update=False)
        view = pl.show(jupyter_backend=backend, return_viewer=True, auto_close=False)
        redraw = None
        if not isinstance(view, widgets.Widget):
            # a static backend gives an image: show it in an output area and redraw it after each change
            from IPython.display import display

            image = widgets.Output()

            def redraw():
                image.clear_output(wait=True)
                with image:
                    display(pl.show(jupyter_backend=backend, return_viewer=True, auto_close=False))

            redraw()
            view = image

        def update(name, value):
            self.show_field(value if name == "field" else field.value, value if name == "epoch" else when.value)
            if redraw is not None:
                redraw()

        field.observe(lambda change: update("field", change["new"]), names="value")
        when.observe(lambda change: update("epoch", change["new"]), names="value")
        return widgets.VBox([widgets.HBox([field, when]), view])

    def export_views(self, folder: str | Path, fields=("elevation",), epochs=None, fmt: str = "html",
                     lat_deg: float = 20.0, lon_deg: float = 0.0) -> list[Path]:
        """Write the globe for each field and kept epoch as a standalone HTML page (``fmt='html'``, vtk.js) or an
        image (``'png'``); return the files."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for years in (self.dates if epochs is None else epochs):
            for f in fields:
                name = f"{f.replace(':', '_')}_{int(years)}ya.{fmt}"
                if fmt == "png":
                    out.append(self.screenshot(folder / name, f, years, lat_deg, lon_deg))
                    continue
                out.append(self.save_html(folder / name, f, years, lat_deg, lon_deg))
        return out
