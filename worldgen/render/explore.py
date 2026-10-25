"""An interactive explorer of an organic history for Jupyter: one map field at a kept epoch with the planet events of
the period, the values of the cell under the cursor, and a timeline of people, energy and events marking the epoch.

``HistoryExplorer(world).widget()`` returns the widget; it needs ``ipywidgets``, and hover needs the ``ipympl``
backend (``%matplotlib widget``). Without ipympl the maps are redrawn as images and hover is off. The drawing,
cell lookup and cell description work without either (``draw``, ``cell_at``, ``describe``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .culture import CULTURE_FIELDS, HISTORY_LAYER_FIELDS
from .epochs import epoch_dates, plot_chronicle
from .maps import plot_map, projection

PHYSICAL_FIELDS = ("elevation", "terrain", "biomes", "koppen", "temperature", "rainfall", "ice")
CULTURE_VIEW_FIELDS = ("political", "population", "cultures", "polities", "settlements", "roads", "suitability",
                       "resources:best")
HISTORY_VIEW_FIELDS = ("energy", "transport", "communication", "bands", "livestock", "events")
VIEW_LAYERS = {"energy": "development_level", "transport": "development_level", "communication": "development_level",
               "bands": "subpolity_count", "livestock": "livestock"}     # field -> the layer it needs
EXPLORER_WIDTH = 900


def available_fields(world) -> list[str]:
    """Return the map fields a world can show: physical ones its surface has, and its culture's and history's."""
    surface = world.surface
    if world.cloud_tops:
        from .clouds import CLOUD_FIELDS

        return list(CLOUD_FIELDS)
    needs = {"ice": "ice_thickness", "biomes": "biome", "koppen": "koppen", "rainfall": "precipitation"}
    out = [f for f in PHYSICAL_FIELDS if needs.get(f, "elevation") in surface]
    civ = world.culture
    if civ is None:
        return out
    out += [f for f in CULTURE_VIEW_FIELDS if f.split(":")[0] in CULTURE_FIELDS]
    out += [f for f in HISTORY_VIEW_FIELDS if f in HISTORY_LAYER_FIELDS + ("events",)
            and VIEW_LAYERS.get(f, "population_density") in civ.layers]
    if "suitability_component" in civ.layers:
        out += [f"land:{c}" for c in civ.layers["suitability_component"]["component"].values]
    return out


class HistoryExplorer:
    """A world's organic history as maps at its kept epochs, with planet events, cell values and a timeline."""

    def __init__(self, world, projection_name: str = "mollweide", central_longitude: float = 0.0,
                 width: int = EXPLORER_WIDTH):
        """Prepare the epochs, the fields on offer and a cell lookup for the cursor."""
        if world.culture is None:
            raise ValueError(f"{world.state.name} has no culture: the explorer needs an organic history")
        self.world = world
        self.dates = epoch_dates(world)
        self.projection_name = projection_name
        self.central_longitude = central_longitude
        self.proj = projection(projection_name, central_longitude)
        self.width = width
        self.tree = cKDTree(world.grid.points)
        self.fields = self._fields()

    def _fields(self) -> list[str]:
        """Return the map fields the world can show at its epochs."""
        return available_fields(self.world)

    def at(self, years_ago: float):
        """Return the world as it stood at a kept epoch."""
        from ..culture import world_at_epoch

        return world_at_epoch(self.world, years_ago)

    def draw(self, fig, field: str, years_ago: float, events: bool = True):
        """Draw one field at a kept epoch on a figure (cleared first), with that period's planet events; return the
        axes."""
        fig.clear()
        ax = fig.add_subplot(1, 1, 1, projection=self.proj)
        plot_map(self.at(years_ago), field, self.projection_name, ax=ax, central_longitude=self.central_longitude,
                 width=self.width, title="", legend="inside", events=True if events and field != "events" else None)
        ax.text(0.01, 0.97, "present" if years_ago <= 0 else f"{years_ago:,.0f} years ago", transform=ax.transAxes,
                fontsize=9, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
        return ax

    def cell_at(self, x: float, y: float) -> Optional[int]:
        """Return the grid cell under projected map coordinates (None off the globe)."""
        import cartopy.crs as ccrs

        lon, lat = ccrs.Geodetic().transform_point(x, y, self.proj)
        if not (np.isfinite(lon) and np.isfinite(lat)):
            return None
        la, lo = np.radians(lat), np.radians(lon)
        _, cell = self.tree.query([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        return int(cell)

    def describe(self, cell: int, years_ago: float) -> str:
        """Return the values of one cell at a kept epoch, one per line."""
        from ..biosphere.biomes import KOPPEN_CODES, Biome
        from ..culture.development import COMMUNICATION_LEVELS, ENERGY_LEVELS, TRANSPORT_LEVELS
        from .culture import _owner, level_codes

        world = self.at(years_ago)
        s, civ = world.surface, world.culture
        lines = [f"{float(world.grid.lat[cell]):.1f}°, {float(world.grid.lon[cell]):.1f}°"]
        if bool(s["ocean"].values[cell]):
            return "\n".join(lines + ["ocean"])
        lines.append(f"elevation {float(s['elevation'].values[cell]):,.0f} m")
        if "biome" in s:
            lines.append(f"{Biome(int(s['biome'].values[cell])).name.lower().replace('_', ' ')}, "
                         f"Köppen {KOPPEN_CODES[int(s['koppen'].values[cell])] or '–'}")
        density = civ.layers["population_density"]
        people = density.values[:, cell]
        if people.sum() > 0:
            top = int(density["culture"].values[int(np.argmax(people))])
            name = next((c.name for c in civ.cultures if c.id == top), None) or f"culture {top}"
            lines.append(f"{people.sum():,.3g} people/km², mostly {name}")
        else:
            lines.append("no people")
        owner = int(_owner(civ)[cell]) if "polity" in civ.layers else -1
        if owner >= 0:
            polity = next((p for p in civ.polities if p.id == owner), None)
            lines.append(f"state: {polity.name if polity is not None and polity.name else f'state {owner}'}")
        if "development_level" in civ.layers:
            names = {"energy": ENERGY_LEVELS, "transport": TRANSPORT_LEVELS, "communication": COMMUNICATION_LEVELS}
            levels = [names[a][k].replace("_", " ") for a in names if (k := int(level_codes(civ, a)[cell])) >= 0]
            if levels:
                lines.append(" / ".join(levels))
        if "livestock" in civ.layers:
            herds = int(civ.layers["livestock"].values[cell])
            lines.append(("no herds", "herds kept", "herds brought by settlers")[herds] if people.sum() > 0 else "")
            lines = [x for x in lines if x]
        if "suitability_component" in civ.layers:
            comp = civ.layers["suitability_component"]
            lines.append(", ".join(f"{c.replace('_', ' ')} {float(v):.2f}"
                                   for c, v in zip(comp["component"].values, comp.values[:, cell])))
        return "\n".join(lines)

    def timeline(self):
        """Return the timeline figure (people, mean energy and major entries) with a marker line per axis."""
        fig = plot_chronicle(self.world)
        self.markers = [ax.axvline(self.dates[-1], color="#c51b7d", lw=1.2) for ax in fig.axes]
        return fig

    def mark(self, years_ago: float) -> None:
        """Move the timeline's marker lines to a date."""
        for line in getattr(self, "markers", []):
            line.set_xdata([years_ago, years_ago])

    def widget(self, field: str = "political", years_ago: Optional[float] = None):
        """Return an ipywidgets box: field menu, epoch slider, events switch, the map, the cell under the cursor and
        the timeline."""
        import ipywidgets as widgets
        import matplotlib.pyplot as plt

        years_ago = self.dates[-1] if years_ago is None else years_ago
        field_box = widgets.Dropdown(options=self.fields, value=field if field in self.fields else self.fields[0],
                                     description="Field")
        labels = [("present" if d <= 0 else f"{d:,.0f} years ago", d) for d in self.dates]
        epoch_box = widgets.SelectionSlider(options=labels, value=years_ago, description="Epoch",
                                            continuous_update=False, layout=widgets.Layout(width="60%"))
        events_box = widgets.Checkbox(value=True, description="planet events since the previous epoch")
        info = widgets.HTML(value="<i>move the cursor over the map</i>")
        with plt.ioff():
            fig = plt.figure(figsize=(10, 5.6))
            line = self.timeline()
        interactive = _is_widget_canvas(fig.canvas)
        map_out = fig.canvas if interactive else widgets.Output()
        line_out = line.canvas if _is_widget_canvas(line.canvas) else widgets.Output()

        def redraw(*_):
            self.draw(fig, field_box.value, epoch_box.value, events_box.value)
            self.mark(epoch_box.value)
            if interactive:
                fig.canvas.draw_idle()
            else:
                map_out.clear_output(wait=True)
                with map_out:
                    from IPython.display import display

                    display(fig)
            if isinstance(line_out, widgets.Output):
                line_out.clear_output(wait=True)
                with line_out:
                    from IPython.display import display

                    display(line)
            else:
                line.canvas.draw_idle()

        def hover(event):
            if event.inaxes is None or event.xdata is None:
                return
            cell = self.cell_at(event.xdata, event.ydata)
            if cell is not None:
                info.value = "<br>".join(self.describe(cell, epoch_box.value).splitlines())

        for box in (field_box, epoch_box, events_box):
            box.observe(redraw, names="value")
        if interactive:
            fig.canvas.mpl_connect("motion_notify_event", hover)
        else:
            info.value = "<i>hover needs the ipympl backend: %matplotlib widget</i>"
        redraw()
        self._widgets = {"field": field_box, "epoch": epoch_box, "events": events_box, "info": info, "figure": fig,
                         "hover": hover, "timeline": line}
        return widgets.VBox([widgets.HBox([field_box, events_box]), epoch_box,
                             widgets.HBox([map_out, info]), line_out])


def _is_widget_canvas(canvas) -> bool:
    """Return whether a figure canvas is an ipympl widget (so it can be shown and take mouse events)."""
    return type(canvas).__module__.startswith("ipympl")
