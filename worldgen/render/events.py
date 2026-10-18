"""Planet events of an organic history on maps: eruptions, droughts, impacts, pandemics, contact epidemics and sea-level
rises between two dates.

Eruptions are triangles sized by their sulfur, impacts stars sized by their
energy, pandemics crosses at their origin sized by their deaths, droughts
circles of their radius, and contact epidemics shade the landmass they struck;
sea-level rises, which have no place, are noted in a corner.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.lines import Line2D

EVENT_KINDS = ("eruption", "drought", "impact", "pandemic", "epidemic", "sea_level")
EVENT_STYLE = {                      # kind -> (marker, colour, label)
    "eruption": ("^", "#b2182b", "eruptions"),
    "drought": ("o", "#e08214", "droughts"),
    "impact": ("*", "#542788", "impacts"),
    "pandemic": ("X", "#1b7837", "pandemics"),
    "epidemic": ("s", "#c51b7d", "contact epidemics"),
    "sea_level": ("v", "#2166ac", "sea-level rises"),
}


def events_between(civ, since: float, until: float) -> list:
    """Return the chronicle's planet events from ``since`` (exclusive) to ``until`` (inclusive) years ago, oldest
    first."""
    return sorted((e for e in civ.chronicle if e.kind in EVENT_KINDS and until <= e.years_ago < since),
                  key=lambda e: -e.years_ago)


def event_window(world) -> tuple[float, float]:
    """Return (since, until) years ago for a world's events: since its previous kept epoch (or the history's start)."""
    if world.culture is None:
        raise ValueError(f"{world.state.name} has no culture")
    return world.culture.event_window()


def _marker_size(e) -> float:
    """Return a marker area (points²) growing with an event's magnitude."""
    r = e.refs or {}
    if e.kind == "eruption":
        return 20.0 + 6.0 * np.sqrt(max(r.get("sulfur_tg", 1.0), 0.0))
    if e.kind == "impact":
        return 30.0 + 25.0 * max(r.get("energy_mt", 1.0), 0.0) ** (1 / 3)
    if e.kind == "pandemic":
        return 15.0 + 12.0 * np.log10(max(r.get("killed", 1.0), 1.0))
    return 30.0


def draw_events(ax, world, since: float, until: float, proj, legend: Optional[str] = "lower right") -> list:
    """Draw the planet events between two dates on a map axis; return the events drawn."""
    import cartopy.crs as ccrs

    from ..culture.history.population import landmasses
    from .raster import projected_sampler

    civ = world.culture
    chosen = events_between(civ, since, until)
    lat, lon = np.asarray(world.grid.lat), np.asarray(world.grid.lon)
    radius_km = float(world.surface.attrs["radius_m"]) / 1e3
    geo = ccrs.PlateCarree()
    handles = []
    for kind in EVENT_KINDS:
        mine = [e for e in chosen if e.kind == kind]
        if not mine:
            continue
        marker, colour, label = EVENT_STYLE[kind]
        handles.append(Line2D([], [], marker=marker, color=colour, linestyle="none", markersize=7,
                              label=f"{label} ({len(mine)})"))
        if kind == "drought":
            for e in mine:
                c = e.refs.get("cell")
                if c is None:
                    continue
                # cartopy draws on a 6371 km sphere: scale so the circle spans the same angle on this planet
                ax.tissot(rad_km=max(float(e.refs.get("radius_km", 300.0)), 50.0) * 6371.0 / radius_km,
                          lons=[lon[c]], lats=[lat[c]], n_samples=60, facecolor=colour, edgecolor=colour,
                          alpha=0.25, zorder=6)
        elif kind == "epidemic":
            mass = landmasses(world)
            struck = np.isin(mass, [e.refs["landmass"] for e in mine if "landmass" in e.refs])
            sampler = projected_sampler(world.grid, proj, 700)
            img = sampler.sample(struck.astype(float), categorical=True).astype(bool)
            rgba = np.zeros(img.shape + (4,))
            rgba[img] = (*_rgb(colour), 0.35)
            ax.imshow(rgba, transform=proj, extent=sampler.extent, origin="upper", interpolation="nearest", zorder=5)
        elif kind == "sea_level":
            rise = sum(float(e.refs.get("rise_m", 0.0)) for e in mine)
            ax.text(0.02, 0.04, f"sea rises {rise:,.0f} m", transform=ax.transAxes, fontsize=7, color=colour,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5}, zorder=8)
        else:
            cells = [e.refs.get("cell") for e in mine]
            keep = [i for i, c in enumerate(cells) if c is not None]
            if not keep:
                continue
            c = np.array([cells[i] for i in keep])
            ax.scatter(lon[c], lat[c], s=[_marker_size(mine[i]) for i in keep], marker=marker, c=colour,
                       edgecolors="white", linewidths=0.6, transform=geo, zorder=7)
    if handles and legend:
        before = ax.get_legend()
        ax.legend(handles=handles, loc=legend, fontsize=7, framealpha=0.85)
        if before is not None:
            ax.add_artist(before)
    return chosen


def _rgb(colour: str) -> tuple:
    from matplotlib.colors import to_rgb

    return to_rgb(colour)
