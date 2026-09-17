"""Map figures of a world's surface using Cartopy projections."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from ..biosphere.biomes import KOPPEN_CODES, Biome
from ..heuristics import MOUNTAIN_OROGENY_MAX_MYR
from ..surface.fields import Boundary, Crust, Terrain
from ..world import World
from .raster import projected_sampler
from .rivers import LAKE_COLOUR, draw_rivers

PROJECTIONS = ("equirectangular", "mollweide", "robinson", "orthographic", "north_polar", "south_polar")
GLOBE_PROJECTIONS = ("orthographic", "north_polar", "south_polar")
FIELDS = ("elevation", "terrain", "plates", "crust_age", "orogeny_age", "temperature", "rainfall", "basins",
          "biomes", "koppen", "ice")
SEASON_NAMES = {1: "month 1 (northern winter)", 7: "month 7 (northern summer)"}
DEFAULT_WIDTH = {"map": 1100, "globe": 700}

_OCEAN = LinearSegmentedColormap.from_list("ocean", ["#0b1d3a", "#16386b", "#2a5d9c", "#4f8cc9", "#9cc7e8"])
_ICE = LinearSegmentedColormap.from_list("sea_ice", ["#7fa3bb", "#bcd6e6", "#eef6fb"])
_TEMPERATE_LAND = LinearSegmentedColormap.from_list(
    "temperate_land", ["#5a8f4e", "#8fb466", "#d7cf8d", "#b48a5a", "#8a6a55", "#ffffff"])
_BARREN_LAND = LinearSegmentedColormap.from_list(
    "barren_land", ["#4a3c34", "#7a6150", "#a88a6d", "#cdb597", "#eee4d4"])
_FROZEN_LAND = LinearSegmentedColormap.from_list(
    "frozen_land", ["#6f6a66", "#948d86", "#b9b4ae", "#dcdad6", "#ffffff"])

TERRAIN_COLOURS = {
    Terrain.PLAIN: "#9cbf73",
    Terrain.HIGHLAND: "#c7a76c",
    Terrain.MOUNTAIN: "#7b5b43",
    Terrain.VOLCANO: "#c2402f",
    Terrain.RIFT: "#e08a2c",
    Terrain.TRENCH: "#16213d",
    Terrain.RIDGE: "#6fa8dc",
    Terrain.CRATER: "#8c8c8c",
    Terrain.VOLCANIC_PLAIN: "#5a4a48",
    Terrain.SHELF: "#9fd0e3",
    Terrain.ABYSSAL_PLAIN: "#2f5a8a",
    Terrain.LAKE: "#3d7fc4",
}
BIOME_COLOURS = {
    Biome.OCEAN: "#dfe8f0",
    Biome.ICE_SHEET: "#f4f8fb",
    Biome.TUNDRA: "#a9b8a0",
    Biome.BOREAL_FOREST: "#2f6b4f",
    Biome.COLD_DESERT: "#c9bf9a",
    Biome.TEMPERATE_GRASSLAND: "#c2cf7a",
    Biome.WOODLAND_SHRUBLAND: "#9fae54",
    Biome.TEMPERATE_FOREST: "#4f8f3a",
    Biome.TEMPERATE_RAINFOREST: "#1f6f4a",
    Biome.HOT_DESERT: "#e6c98a",
    Biome.SAVANNA: "#c8b44a",
    Biome.TROPICAL_SEASONAL_FOREST: "#6a9e2b",
    Biome.TROPICAL_RAINFOREST: "#16582a",
    Biome.BARREN: "#8a7e72",
    Biome.POLAR_DESERT: "#cfd4d6",
}
VEGETATED = {Biome.BOREAL_FOREST, Biome.TEMPERATE_GRASSLAND, Biome.WOODLAND_SHRUBLAND, Biome.TEMPERATE_FOREST,
             Biome.TEMPERATE_RAINFOREST, Biome.SAVANNA, Biome.TROPICAL_SEASONAL_FOREST, Biome.TROPICAL_RAINFOREST}
EARTH_PIGMENT = "#4e6a2f"
# Köppen–Geiger colours as commonly used for the classification (Beck et al. 2018 style).
KOPPEN_COLOURS = {
    "Af": (0, 0, 255), "Am": (0, 120, 255), "Aw": (70, 170, 250), "BWh": (255, 0, 0), "BWk": (255, 150, 150),
    "BSh": (245, 165, 0), "BSk": (255, 220, 100), "Csa": (255, 255, 0), "Csb": (200, 200, 0),
    "Csc": (150, 150, 0), "Cwa": (150, 255, 150), "Cwb": (100, 200, 100), "Cwc": (50, 150, 50),
    "Cfa": (200, 255, 80), "Cfb": (100, 255, 80), "Cfc": (50, 200, 0), "Dsa": (255, 0, 255),
    "Dsb": (200, 0, 200), "Dsc": (150, 50, 150), "Dsd": (150, 100, 150), "Dwa": (170, 175, 255),
    "Dwb": (90, 120, 220), "Dwc": (75, 80, 180), "Dwd": (50, 0, 135), "Dfa": (0, 255, 255),
    "Dfb": (55, 200, 255), "Dfc": (0, 125, 125), "Dfd": (0, 70, 95), "ET": (178, 178, 178), "EF": (102, 102, 102),
}
BOUNDARY_COLOURS = {Boundary.CONVERGENT: "#d62728", Boundary.DIVERGENT: "#2ca02c", Boundary.TRANSFORM: "#ffbf00"}


def projection(name: str, central_longitude: float = 0.0, central_latitude: float = 20.0) -> ccrs.Projection:
    """Return the Cartopy projection for a name in ``PROJECTIONS``."""
    if name == "equirectangular":
        return ccrs.PlateCarree(central_longitude=central_longitude)
    if name == "mollweide":
        return ccrs.Mollweide(central_longitude=central_longitude)
    if name == "robinson":
        return ccrs.Robinson(central_longitude=central_longitude)
    if name == "orthographic":
        return ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
    if name == "north_polar":
        return ccrs.NorthPolarStereo(central_longitude=central_longitude)
    if name == "south_polar":
        return ccrs.SouthPolarStereo(central_longitude=central_longitude)
    raise ValueError(f"unknown projection '{name}'; choose from {', '.join(PROJECTIONS)}")


def vegetation_colour(world: World, base: str) -> tuple[float, float, float]:
    """Return a vegetation colour shifted from Earth's green toward the planet's pigment colour."""
    life = world.state.biosphere
    if life is None or life.pigment_colour == EARTH_PIGMENT:
        return to_rgb(base)
    shift = np.array(to_rgb(life.pigment_colour)) - np.array(to_rgb(EARTH_PIGMENT))
    return tuple(np.clip(np.array(to_rgb(base)) + shift, 0.0, 1.0))


def land_palette(world: World) -> LinearSegmentedColormap:
    """Return the land colour scheme: vegetation colours only for worlds with surface life."""
    atm = world.state.atmosphere
    life = world.state.biosphere
    if life is not None and life.life == "surface" and atm.surface_water == "liquid":
        low = [vegetation_colour(world, c) for c in ("#5a8f4e", "#8fb466")]
        return LinearSegmentedColormap.from_list("living_land", low + ["#d7cf8d", "#b48a5a", "#8a6a55", "#ffffff"])
    if atm.surface_water == "ice" or atm.surface_temperature_k < 250.0:
        return _FROZEN_LAND
    return _BARREN_LAND


def _hillshade(elev: np.ndarray, valid: np.ndarray, pixel_m: float, relief_m: float) -> np.ndarray:
    """Return hillshade intensities (0–1) for an elevation image."""
    filled = np.where(valid, elev, np.nanmean(elev[valid]) if valid.any() else 0.0)
    exaggeration = 20.0 * pixel_m / max(relief_m, 1.0) * 0.05
    return LightSource(azdeg=315, altdeg=40).hillshade(filled, vert_exag=exaggeration, dx=pixel_m, dy=pixel_m)


def elevation_image(world: World, proj: ccrs.Projection, width: int, hillshade: bool = True,
                    limits: Optional[tuple[float, float]] = None):
    """Return an RGBA image of elevation in a projection, plus the colormap and norm for a colour bar.

    ``limits`` fixes the colour scale (lowest, highest elevation); by default
    it spans the world's elevations.
    """
    ds = world.surface
    sampler = projected_sampler(world.grid, proj, width)
    valid = sampler.valid.reshape(sampler.shape)
    elev = sampler.sample(ds["elevation"].values)
    # Interpolated elevation gives smoother coastlines than the per-cell ocean mask; the mask keeps
    # dry land below sea level (closed depressions) from being drawn as sea.
    if ds.attrs["has_ocean"]:
        near_sea = sampler.sample(ds["ocean"].values, categorical=True).astype(bool)
        ocean = (np.nan_to_num(elev, nan=1.0) < 0.0) & valid & near_sea
    else:
        ocean = np.zeros_like(valid)
    lo, hi = limits if limits is not None else (float(ds["elevation"].min()), float(ds["elevation"].max()))
    land = land_palette(world)

    if not ds.attrs["has_ocean"]:
        norm = Normalize(lo, hi)
        cmap = land
        rgb = land(norm(np.nan_to_num(elev)))[..., :3]
    else:
        norm = TwoSlopeNorm(vmin=min(lo, -1.0), vcenter=0.0, vmax=max(hi, 1.0))
        cmap = ListedColormap(np.vstack([_OCEAN(np.linspace(0, 1, 256)), land(np.linspace(0, 1, 256))]),
                              name="world")
        e = np.nan_to_num(elev)
        depth = 1.0 - np.clip(e / norm.vmin, 0, 1)
        water = _OCEAN(depth)[..., :3]
        if "sea_ice" in ds:
            ice = np.clip(np.nan_to_num(sampler.sample(ds["sea_ice"].values)), 0.0, 1.0)[..., None]
            ice = np.clip((ice - 0.3) / 0.4, 0.0, 1.0)
            water = (1.0 - ice) * water + ice * _ICE(depth)[..., :3]
        elif ds.attrs["frozen_ocean"]:
            water = _ICE(depth)[..., :3]
        rgb = np.where(ocean[..., None], water, land(np.clip(e / norm.vmax, 0, 1))[..., :3])
    if hillshade:
        # Pixel size in metres on this planet (projection units are metres on Earth's radius).
        pixel_m = (sampler.extent[1] - sampler.extent[0]) / width * world.state.bulk.radius_m / 6.371e6
        shade = _hillshade(elev, valid, pixel_m, hi - lo)
        strength = np.where(ocean, 0.2, 0.55)[..., None]
        rgb = np.clip(rgb * (1 - strength + strength * 1.6 * shade[..., None]), 0, 1)
    if "ice_thickness" in ds:
        glacier = sampler.sample((ds["ice_thickness"].values > 0).astype(np.float32)) > 0.5
        glacier &= valid & ~ocean
        rgb[glacier] = 0.25 * rgb[glacier] + 0.75 * np.array(to_rgb("#f4f8fb"))
    if "lake" in ds:
        lake = sampler.sample(ds["lake"].values, categorical=True).astype(bool) & valid
        rgb[lake] = to_rgb(LAKE_COLOUR)
    rgba = np.concatenate([rgb, valid[..., None].astype(float)], axis=-1)
    return rgba, cmap, norm, sampler.extent


def field_image(world: World, field: str, proj: ccrs.Projection, width: int, month: Optional[int] = None):
    """Return an RGBA image of a categorical, age or climate field, plus its legend handles or (cmap, norm).

    ``month`` (1–12) selects a monthly temperature or precipitation field instead of the annual mean.
    """
    ds = world.surface
    sampler = projected_sampler(world.grid, proj, width)
    valid = sampler.valid.reshape(sampler.shape)
    if field == "terrain":
        codes = sampler.sample(ds["terrain"].values, categorical=True)
        table = np.array([to_rgb(TERRAIN_COLOURS.get(Terrain(k), "#000000")) for k in range(len(Terrain))])
        rgb = table[codes]
        # Tint submerged land-type terrain so the coastline stays visible.
        submerged = sampler.sample(ds["ocean"].values, categorical=True).astype(bool)
        submerged &= ~np.isin(codes, [Terrain.SHELF, Terrain.ABYSSAL_PLAIN, Terrain.TRENCH, Terrain.RIDGE])
        rgb[submerged] = 0.45 * rgb[submerged] + 0.55 * np.array(to_rgb("#2f5a8a"))
        present = set(np.unique(ds["terrain"].values).tolist())
        extra = [Patch(color=TERRAIN_COLOURS[t], label=t.name.replace("_", " ").lower())
                 for t in Terrain if int(t) in present]
    elif field == "plates":
        plates = sampler.sample(ds["plate"].values, categorical=True)
        bound = sampler.sample(ds["boundary"].values, categorical=True)
        rgb = plt.get_cmap("tab20")((plates % 20) / 19.0)[..., :3] * 0.75 + 0.25
        if (ds["plate"].values < 0).all():
            rgb[:] = 0.85
        for code, colour in BOUNDARY_COLOURS.items():
            rgb[bound == code] = to_rgb(colour)
        extra = [Patch(color=c, label=b.name.lower()) for b, c in BOUNDARY_COLOURS.items()]
    elif field == "crust_age":
        raw = ds["crust_age"].values
        top = float(np.nanmax(raw)) if np.isfinite(raw).any() else 1.0
        age = sampler.sample(np.nan_to_num(raw, nan=0.0))
        continental = sampler.sample(np.isnan(raw), categorical=True)
        cmap = plt.get_cmap("magma_r")
        rgb = cmap(np.clip(np.nan_to_num(age) / max(top, 1.0), 0, 1))[..., :3]
        rgb[continental] = (0.78, 0.78, 0.78)
        extra = (cmap, Normalize(0, max(top, 1.0)))
    elif field == "rainfall" and "precipitation" not in ds:
        raise ValueError("'rainfall' needs surface water and an atmosphere")
    elif field in ("biomes", "koppen", "ice") and "biome" not in ds:
        raise ValueError(f"'{field}' needs a saved world from milestone 5 or later")
    elif field == "basins" and "drainage_basin" not in ds:
        raise ValueError("'basins' needs rivers, which only planets with liquid surface water have")
    elif field == "temperature":
        values = ds["air_temperature"].values if month is None else ds["monthly_temperature"].values[month - 1]
        temp = sampler.sample(values)
        cmap = plt.get_cmap("RdYlBu_r")
        norm = TEMPERATURE_NORM
        rgb = cmap(norm(np.nan_to_num(temp, nan=273.15)))[..., :3]
        rgb = _shade_ocean(ds, sampler, rgb, 0.25)
        extra = (cmap, norm)
    elif field == "rainfall":
        values = ds["precipitation"].values if month is None else ds["monthly_precipitation"].values[month - 1]
        rain = sampler.sample(values)
        cmap = plt.get_cmap("YlGnBu")
        top = 3.0
        rgb = cmap(np.clip(np.nan_to_num(rain) / top, 0, 1))[..., :3]
        rgb = _shade_ocean(ds, sampler, rgb, 0.55)
        extra = (cmap, Normalize(0, top))
    elif field == "basins":
        basin = sampler.sample(ds["drainage_basin"].values, categorical=True)
        palette = plt.get_cmap("tab20")(np.arange(20))[:, :3] * 0.6 + 0.4
        rgb = palette[(basin * 7) % 20]
        rgb[basin == 0] = to_rgb("#dfe8f0")
        endorheic = sampler.sample(ds["endorheic"].values, categorical=True).astype(bool)
        rgb[endorheic] = 0.55 * rgb[endorheic] + 0.45 * np.array(to_rgb("#b08a50"))
        lake = sampler.sample(ds["lake"].values, categorical=True).astype(bool)
        rgb[lake] = to_rgb(LAKE_COLOUR)
        extra = [Patch(color="#c8a97a", label="no outlet to the sea"), Patch(color=LAKE_COLOUR, label="lake")]
    elif field == "biomes":
        codes = sampler.sample(ds["biome"].values, categorical=True)
        palette = {b: (vegetation_colour(world, c) if b in VEGETATED else to_rgb(c)) for b, c in BIOME_COLOURS.items()}
        table = np.array([palette[Biome(k)] for k in range(len(Biome))])
        rgb = table[codes]
        present = set(np.unique(ds["biome"].values).tolist()) - {int(Biome.OCEAN)}
        extra = [Patch(color=palette[b], label=b.name.replace("_", " ").lower()) for b in Biome if int(b) in present]
    elif field == "koppen":
        codes = sampler.sample(ds["koppen"].values, categorical=True)
        table = np.array([[0.875, 0.91, 0.94]] + [np.array(KOPPEN_COLOURS[k]) / 255 for k in KOPPEN_CODES[1:]])
        rgb = table[codes]
        present = set(np.unique(ds["koppen"].values).tolist()) - {0}
        extra = [Patch(color=table[k], label=KOPPEN_CODES[k]) for k in sorted(present)]
    elif field == "ice":
        thick = sampler.sample(ds["ice_thickness"].values)
        cmap = plt.get_cmap("Blues")
        top = max(float(ds["ice_thickness"].max()), 1.0)
        rgb = np.full(thick.shape + (3,), to_rgb("#b9a58c"))
        glacier = np.nan_to_num(thick) > 0
        rgb[glacier] = cmap(0.2 + 0.8 * np.clip(thick[glacier] / top, 0, 1))[..., :3]
        if "sea_ice" in ds:
            sea = np.nan_to_num(sampler.sample(ds["sea_ice"].values))
            ocean = sampler.sample(ds["ocean"].values, categorical=True).astype(bool)
            base = np.array(to_rgb("#2f5a8a"))
            rgb[ocean] = base + (np.array(to_rgb("#e8f1f7")) - base) * np.clip(sea[ocean], 0, 1)[:, None]
        extra = (cmap, Normalize(0, top))
    elif field == "orogeny_age":
        raw = ds["orogeny_age"].values
        top = MOUNTAIN_OROGENY_MAX_MYR
        age = sampler.sample(np.nan_to_num(raw, nan=top))
        oceanic = sampler.sample(ds["crust"].values != Crust.CONTINENTAL, categorical=True).astype(bool)
        cmap = plt.get_cmap("inferno")
        rgb = cmap(np.clip(np.nan_to_num(age, nan=top) / top, 0, 1))[..., :3]
        rgb[oceanic] = 0.35 * rgb[oceanic] + 0.65 * np.array(to_rgb("#2f5a8a"))
        extra = (cmap, Normalize(0, top))
    else:
        raise ValueError(f"unknown field '{field}'; choose from {', '.join(FIELDS)}")
    rgba = np.concatenate([rgb, valid[..., None].astype(float)], axis=-1)
    return rgba, extra, sampler.extent


TEMPERATURE_NORM = TwoSlopeNorm(vmin=213.15, vcenter=273.15, vmax=318.15)
COLORBAR_LABELS = {"crust_age": "oceanic crust age (Myr)", "rainfall": "precipitation (m/yr)",
                   "ice": "ice-sheet thickness (m); sea shaded by sea-ice cover",
                   "temperature": "air temperature (K)", "orogeny_age": "time since mountain building (Myr)"}


def _shade_ocean(ds, sampler, rgb: np.ndarray, strength: float) -> np.ndarray:
    """Return the image with ocean pixels blended toward grey so land stands out."""
    if not ds.attrs["has_ocean"]:
        return rgb
    ocean = sampler.sample(ds["ocean"].values, categorical=True).astype(bool)
    rgb[ocean] = (1.0 - strength) * rgb[ocean] + strength * np.array(to_rgb("#e4e8ec"))
    return rgb


def _elevation_colorbar(ax, cmap, norm) -> None:
    """Add a horizontal elevation colour bar with ticks in km on both sides of sea level."""
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6)
    lo, hi = norm.vmin, norm.vmax
    if isinstance(norm, TwoSlopeNorm):
        ticks = [lo, lo / 2, 0.0, hi / 2, hi]
    else:
        ticks = list(np.linspace(lo, hi, 5))
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t / 1e3:.1f}" for t in ticks])
    cb.set_label("elevation (km)")


def plot_map(world: World, field: str = "elevation", projection_name: str = "mollweide", ax=None,
             central_longitude: float = 0.0, central_latitude: float = 20.0, width: Optional[int] = None,
             hillshade: bool = True, title: Optional[str] = None, colorbar: bool = True,
             legend: str | None = "right", rivers: bool = True, month: Optional[int] = None):
    """Draw one field of the world's surface on a map projection and return the axes.

    ``legend`` places the legend for categorical fields: "right" (outside the
    map), "inside" (lower left), or None. ``rivers`` draws rivers over the
    elevation, terrain, rainfall and basin maps. ``month`` (1–12) shows one
    month of the temperature or rainfall fields.
    """
    if world.surface is None:
        raise ValueError(f"{world.state.name} has no solid surface to map")
    proj = projection(projection_name, central_longitude, central_latitude)
    globe = projection_name in GLOBE_PROJECTIONS
    if width is None:
        width = DEFAULT_WIDTH["globe" if globe else "map"]
    if ax is None:
        fig = plt.figure(figsize=(7, 7) if globe else (11, 6.5))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()

    if field == "elevation":
        rgba, cmap, norm, extent = elevation_image(world, proj, width, hillshade)
        ax.imshow(rgba, transform=proj, extent=extent, origin="upper", interpolation="bilinear")
        if colorbar:
            _elevation_colorbar(ax, cmap, norm)
    else:
        rgba, extra, extent = field_image(world, field, proj, width, month)
        ax.imshow(rgba, transform=proj, extent=extent, origin="upper", interpolation="nearest")
        if isinstance(extra, list) and legend:
            if field == "plates" and (world.surface["plate"].values < 0).all():
                ax.text(0.5, 0.5, f"no plate tectonics\n({world.state.interior.tectonic_regime.replace('_', ' ')})",
                        transform=ax.transAxes, ha="center", va="center", fontsize=11)
            elif legend == "right":
                ax.legend(handles=extra, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
            elif legend == "inside":
                ax.legend(handles=extra, loc="lower left", fontsize=6, framealpha=0.8)
        elif isinstance(extra, tuple) and colorbar:
            sm = plt.cm.ScalarMappable(norm=extra[1], cmap=extra[0])
            cb = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04, shrink=0.6)
            cb.set_label(COLORBAR_LABELS[field])
    if rivers and field in ("elevation", "terrain", "rainfall", "basins"):
        draw_rivers(ax, world, proj, width_scale=1.0 if not globe else 0.9)

    ax.gridlines(color="white", alpha=0.3, linewidth=0.5)
    ax.set_title(title if title is not None else f"{world.state.name}: {field}", fontsize=11)
    return ax


def plot_overview(world: World):
    """Return a figure with elevation, terrain, plates and two globe views."""
    fig = plt.figure(figsize=(15, 10.5))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.0], hspace=0.12, wspace=0.25)
    moll = projection("mollweide")
    plot_map(world, "elevation", "mollweide", ax=fig.add_subplot(gs[0, 0:3], projection=moll), title="Elevation")
    plot_map(world, "terrain", "mollweide", ax=fig.add_subplot(gs[0, 3:6], projection=moll), title="Terrain")
    plot_map(world, "plates", "mollweide", ax=fig.add_subplot(gs[1, 0:2], projection=moll),
             title="Plates and boundaries", width=700, legend="inside")
    for col, lon, lat in ((2, 0.0, 25.0), (4, 180.0, -25.0)):
        ax = fig.add_subplot(gs[1, col:col + 2], projection=projection("orthographic", lon, lat))
        plot_map(world, "elevation", "orthographic", ax=ax, central_longitude=lon, central_latitude=lat,
                 colorbar=False, title=f"Globe at {lon:.0f}°, {lat:+.0f}°")
    s, sf = world.state, world.state.surface
    fig.suptitle(f"{s.name}  |  {s.archetype or 'custom'}  |  {s.bulk.planet_class}, "
                 f"{s.interior.tectonic_regime.replace('_', ' ')}  |  land {sf.land_fraction:.0%}  |  "
                 f"{s.atmosphere.surface_temperature_k:.0f} K, {_water_phrase(s.atmosphere.surface_water)}",
                 fontsize=13, y=0.97)
    return fig


def plot_climate(world: World):
    """Return a figure with annual and seasonal temperature and precipitation.

    Months 1 and 7 begin at the northern winter and summer solstices.
    """
    if world.surface is None:
        raise ValueError(f"{world.state.name} has no solid surface to map")
    wet = "precipitation" in world.surface
    rows = 3 if wet else 1
    fig = plt.figure(figsize=(15, 3.3 * rows + 0.6))
    moll = projection("mollweide")
    gs = GridSpec(rows, 3, figure=fig, hspace=0.3, wspace=0.08, top=0.9 if wet else 0.8, bottom=0.06)
    fields = ["temperature"] + (["rainfall"] if wet else [])
    for r, field in enumerate(fields):
        for col, month in enumerate((None, 1, 7)):
            name = "annual mean" if month is None else SEASON_NAMES[month]
            plot_map(world, field, "mollweide", ax=fig.add_subplot(gs[r, col], projection=moll), month=month,
                     title=f"{'Temperature' if field == 'temperature' else 'Precipitation'}, {name}",
                     width=700, rivers=False)
    if wet:
        plot_map(world, "biomes", "mollweide", ax=fig.add_subplot(gs[2, 0], projection=moll), width=700,
                 rivers=False, legend="inside", title="Biomes")
        plot_map(world, "koppen", "mollweide", ax=fig.add_subplot(gs[2, 1], projection=moll), width=700,
                 rivers=False, legend=None, title="Köppen–Geiger classes")
        _zonal_profiles(fig.add_subplot(gs[2, 2]), world)
    s = world.state
    features = s.surface.features
    mean_k = features.get("climate_mean_temperature_k", s.atmosphere.surface_temperature_k)
    albedo = features.get("climate_planetary_albedo", s.atmosphere.bond_albedo)
    fig.suptitle(f"{s.name}: climate  |  mean {mean_k:.0f} K  |  planetary albedo {albedo:.2f}", fontsize=13)
    return fig


def _zonal_profiles(ax, world: World) -> None:
    """Plot zonal mean precipitation over land and ocean, with annual temperature on a second axis."""
    ds = world.surface
    lat = ds["lat"].values
    bins = np.linspace(-90, 90, 37)
    centre = 0.5 * (bins[1:] + bins[:-1])
    idx = np.clip(np.digitize(lat, bins) - 1, 0, 35)
    ocean = ds["ocean"].values
    rain = ds["precipitation"].values
    temp = ds["air_temperature"].values

    def zonal(values, mask):
        """Return band means of the values where the mask holds."""
        total = np.bincount(idx[mask], weights=values[mask], minlength=36)
        count = np.bincount(idx[mask], minlength=36)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    ax.plot(centre, zonal(rain, ocean), color="#2a5d9c", label="precipitation, ocean")
    ax.plot(centre, zonal(rain, ~ocean), color="#5a8f4e", label="precipitation, land")
    ax.set_xlabel("latitude (°)")
    ax.set_ylabel("precipitation (m/yr)")
    ax.set_xlim(-90, 90)
    twin = ax.twinx()
    twin.plot(centre, zonal(temp, np.ones_like(ocean)), color="#c2402f", label="air temperature")
    twin.set_ylabel("air temperature (K)")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=7, frameon=False)
    ax.set_title("Zonal means")


def _water_phrase(state: str) -> str:
    """Return a short description of the surface water state."""
    return {"liquid": "liquid water", "ice": "frozen water", "vapour": "water vapour", "none": "no surface water"}[state]


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    """Write a figure to an image file and close it."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
