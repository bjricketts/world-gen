"""Culture layers on global maps: population, cultures, polities, settlements, roads, habitats, resources, the
suitability of the land, and an organic history's development levels, bands and chiefdoms, livestock and the
components of the land's suitability.

The political map combines polity colours and borders with roads, sea routes,
settlements and the names of the largest places and polities.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, PowerNorm, to_rgb
from matplotlib.patches import Patch

from ..culture.habitats import Habitat
from ..culture.model import ROAD_KINDS, SETTLEMENT_FORMS
from .raster import projected_sampler
from .rivers import smooth_line

CULTURE_FIELDS = ("population", "cultures", "polities", "settlements", "roads", "political", "habitat", "resources",
                  "suitability")
# layers only an organic history leaves (task 9.16)
HISTORY_LAYER_FIELDS = ("energy", "transport", "communication", "bands", "livestock", "land")
LEVEL_AXES = {"energy": 0, "transport": 1, "communication": 2}
RESOURCE_BEST_MIN = 0.5          # resources:best colours a cell by its richest resource of at least this potential
LIVESTOCK_COLOURS = {"farming without herds": "#b8d39a", "herds kept": "#8c510a", "herds brought by settlers": "#d8b365"}
OCEAN_COLOUR = "#2f5a8a"
EMPTY_LAND = "#d9d2c3"
ROAD_STYLE = {   # kind: (colour, width, zorder)
    "track": ("#9a8a78", 0.2, 3), "road": ("#8a6a4a", 0.35, 3), "trunk": ("#a0523a", 0.6, 4),
    "highway": ("#8e2418", 0.9, 4), "sea_route": ("#9fc3e6", 0.35, 2),
}
POLITICAL_ROADS = ("road", "trunk", "highway", "sea_route")
POLITICAL_SETTLEMENTS = 300
HABITAT_COLOURS = {Habitat.NONE: "#6b4b3e", Habitat.OPEN: "#8fb466", Habitat.DOME: "#e0b050",
                   Habitat.UNDERGROUND: "#8a6db0"}
LABELLED_SETTLEMENTS = 14
LABELLED_POLITIES = 10


def split_field(field: str) -> tuple[str, Optional[str]]:
    """Return a field name and its option, from "resources:copper" style names."""
    name, _, option = field.partition(":")
    return name, option or None


def is_culture_field(field: str) -> bool:
    """Return whether a map field is one of the culture layers (including an organic history's)."""
    return split_field(field)[0] in CULTURE_FIELDS + HISTORY_LAYER_FIELDS


def _culture_id(civ, option: Optional[str]) -> Optional[int]:
    """Return the id of the culture an option names (by name or id), or None for no option."""
    if option is None:
        return None
    for c in civ.cultures:
        if (c.name and c.name.lower() == option.lower()) or str(c.id) == option:
            return c.id
    names = ", ".join(c.name or str(c.id) for c in civ.cultures)
    raise ValueError(f"no culture '{option}'; choose from {names}")


def _history_layer(civ, name: str):
    """Return a layer an organic history leaves, or raise a clear error."""
    if name not in civ.layers:
        raise ValueError(f"this civilisation has no '{name}' layer: the map needs an organic history")
    return civ.layers[name]


def level_codes(civ, axis: str, culture: Optional[int] = None) -> np.ndarray:
    """Return the development level in force per cell on one axis: the highest among the organic cultures living
    there, or one culture's (−1 where no one lives)."""
    from ..culture.history.state import in_force

    dev = _history_layer(civ, "development_level")
    density = civ.layers["population_density"]
    ids = [int(i) for i in dev["organic"].values if int(i) in set(density["culture"].values.tolist())]
    if culture is not None:
        ids = [i for i in ids if i == culture]
    out = np.full(dev.sizes["cell"], -1)
    for i in ids:
        lived = density.sel(culture=i).values > 0
        level = in_force(dev.sel(organic=i).values[LEVEL_AXES[axis]])
        out = np.where(lived, np.maximum(out, level), out)
    return out


def _civ(world):
    """Return the world's civilisation, or raise if it has none."""
    if world.culture is None:
        raise ValueError(f"{world.state.name} has no culture; add one with `worldgen culture` or a culture section")
    return world.culture


def _palette(n: int, seed: int = 7) -> np.ndarray:
    """Return ``n`` muted, distinct colours."""
    base = np.vstack([plt.get_cmap("tab20")(np.arange(20))[:, :3], plt.get_cmap("Set3")(np.arange(12))[:, :3]])
    order = np.random.default_rng(seed).permutation(len(base))
    colours = base[order][np.arange(max(n, 1)) % len(base)]
    return 0.65 * colours + 0.35 * np.array(to_rgb(EMPTY_LAND))


def _id_colours(ids: np.ndarray, seed: int = 7) -> np.ndarray:
    """Return a colour per id, fixed by the id alone so a polity or culture keeps its colour across epochs."""
    ids = np.asarray(ids, dtype=np.int64)
    base = _palette(32, seed)
    return base[(ids * 2654435761 + seed) % 32]


def _owner(civ) -> np.ndarray:
    """Return, per cell, the id of the polity with the most people there (−1 where none)."""
    polity = civ.layers["polity"].values
    density = civ.layers["population_density"].values
    top = np.argmax(np.where(polity >= 0, density, -1.0), axis=0)
    owner = np.take_along_axis(polity, top[None], axis=0)[0]
    return np.where((polity >= 0).any(axis=0), owner, -1)


def _borders(codes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return pixels where the code differs from the pixel to the right or below (both coded)."""
    edge = np.zeros(codes.shape, dtype=bool)
    right = (codes[:, 1:] != codes[:, :-1]) & (codes[:, 1:] >= 0) & (codes[:, :-1] >= 0)
    down = (codes[1:, :] != codes[:-1, :]) & (codes[1:, :] >= 0) & (codes[:-1, :] >= 0)
    edge[:, 1:] |= right
    edge[1:, :] |= down
    return edge & valid


def _leading_culture(civ) -> np.ndarray:
    """Return, per cell, the id of the culture with the most people there (−1 where none)."""
    density = civ.layers["population_density"]
    values = density.values
    ids = density["culture"].values
    return np.where(values.max(axis=0) > 0, ids[np.argmax(values, axis=0)], -1)


def _livestock_code(civ) -> np.ndarray:
    """Return, per cell, 1 where herds are kept, 2 where settlers brought them, 0 for farming without herds, else −1."""
    herds = _history_layer(civ, "livestock").values
    dev = civ.layers["development_level"]
    density = civ.layers["population_density"]
    ids = [int(i) for i in dev["organic"].values if int(i) in set(density["culture"].values.tolist())]
    farmed = np.zeros(herds.size, dtype=bool)
    for i in ids:
        farmed |= (density.sel(culture=i).values > 0) & (dev.sel(organic=i).values[0] >= 0.5)
    return np.where(herds == 1, 1, np.where(herds == 2, 2, np.where(farmed, 0, -1)))


def _suitability_culture(civ, option: Optional[str]) -> int:
    """Return the culture a suitability field shows: the one named, else the most populous."""
    cid = _culture_id(civ, option)
    if cid is None:
        people = civ.layers["population_density"].sum(dim="cell")
        cid = int(people["culture"].values[int(np.argmax(people.values))])
    if cid not in set(civ.layers["suitability"]["culture"].values.tolist()):
        raise ValueError(f"culture {cid} has no suitability layer (it has no people at this epoch)")
    return cid


def culture_values(world, field: str) -> tuple[np.ndarray, bool, str, Optional[dict]]:
    """Return a culture layer per grid cell: values, whether they are category codes, units, and code labels.

    Category codes are ids (cultures, polities) or indices (habitats, levels,
    resources, livestock), with −1 where the layer is empty. Settlements and
    roads are drawn on maps but have no per-cell values.
    """
    name, option = split_field(field)
    civ = _civ(world)
    if name == "population":
        return civ.layers["population_density"].values.sum(axis=0), False, "people/km2", None
    if name == "cultures":
        return _leading_culture(civ), True, "culture id", {c.id: c.name or f"culture {c.id}" for c in civ.cultures}
    if name in ("polities", "political"):
        return _owner(civ), True, "polity id", {p.id: p.name or f"polity {p.id}" for p in civ.polities}
    if name == "habitat":
        if "habitat" not in civ.layers:
            raise ValueError("this civilisation has no habitat layer; rebuild its culture")
        return civ.layers["habitat"].values, True, "habitat", {int(k): k.name.lower() for k in HABITAT_COLOURS}
    if name == "resources":
        pot = civ.layers["resource_potential"]
        names = [str(n) for n in pot["resource"].values]
        if option == "best":
            values = pot.values
            best = np.where(values.max(axis=0) >= RESOURCE_BEST_MIN, np.argmax(values, axis=0), -1)
            return best, True, "resource", {k: n.replace("_", " ") for k, n in enumerate(names)}
        if option is None or option not in names:
            raise ValueError(f"choose a resource as resources:<name> or resources:best, one of {', '.join(names)}")
        return pot.sel(resource=option).values, False, "potential (0-1)", None
    if name == "suitability":
        cid = _suitability_culture(civ, option)
        return civ.layers["suitability"].sel(culture=cid).values, False, "capacity relative to the best cell", None
    if name in LEVEL_AXES:
        from ..culture.development import COMMUNICATION_LEVELS, ENERGY_LEVELS, TRANSPORT_LEVELS

        names = {"energy": ENERGY_LEVELS, "transport": TRANSPORT_LEVELS, "communication": COMMUNICATION_LEVELS}[name]
        labels = {k: n.replace("_", " ") for k, n in enumerate(names)}
        return level_codes(civ, name, _culture_id(civ, option)), True, f"{name} level", labels
    if name == "bands":
        return _history_layer(civ, "subpolity_count").values.sum(axis=0), False, "polities per cell", None
    if name == "livestock":
        labels = dict(zip((0, 1, 2), LIVESTOCK_COLOURS))
        return _livestock_code(civ), True, "livestock", labels
    if name == "land":
        comp = _history_layer(civ, "suitability_component")
        names = [str(c) for c in comp["component"].values]
        if option is None or option not in names:
            raise ValueError(f"choose a component as land:<name>, one of {', '.join(names)}")
        return comp.sel(component=option).values, False, "suitability component (0-1)", None
    if name in ("settlements", "roads"):
        raise ValueError(f"'{name}' are drawn as overlays and have no per-cell values")
    raise ValueError(f"unknown culture field '{field}'; choose from {', '.join(CULTURE_FIELDS + HISTORY_LAYER_FIELDS)}")


def culture_image(world, field: str, proj, width: int, sampler=None):
    """Return an RGBA image of a culture layer, its legend handles or (cmap, norm), and the image extent.

    ``sampler`` replaces the projected image lookup, so the same colours can be
    sampled at any set of points (a raster, mesh vertices).
    """
    name, option = split_field(field)
    civ = _civ(world)
    ds = world.surface
    sampler = sampler if sampler is not None else projected_sampler(world.grid, proj, width)
    valid = sampler.valid.reshape(sampler.shape)
    ocean = sampler.sample(ds["ocean"].values, categorical=True).astype(bool)
    rgb = np.empty(ocean.shape + (3,))
    rgb[:] = to_rgb(EMPTY_LAND)
    extra = None
    if name in ("settlements", "roads"):
        extra = []
    else:
        values, categorical, _, labels = culture_values(world, field)
        img = sampler.sample(values, categorical=categorical)
    if name == "population":
        norm = LogNorm(vmin=0.01, vmax=max(1.0, float(values.max())))
        cmap = plt.get_cmap("magma_r")
        live = np.nan_to_num(img) > 0.01
        rgb[live] = cmap(norm(img[live]))[..., :3]
        extra = (cmap, norm)
    elif name == "cultures":
        shown = np.unique(img[img >= 0])
        rgb[img >= 0] = _id_colours(img[img >= 0], seed=3)
        colours = dict(zip(shown.tolist(), _id_colours(shown, seed=3)))
        extra = [Patch(color=colours[i], label=labels[i]) for i in shown.tolist()]
    elif name in ("polities", "political"):
        rgb[img >= 0] = _id_colours(img[img >= 0])
        rgb[_borders(img, valid) & ~ocean] = (0.2, 0.2, 0.2)
        extra = []
    elif name == "habitat":
        for k, colour in HABITAT_COLOURS.items():
            rgb[img == k] = to_rgb(colour)
        extra = [Patch(color=c, label=k.name.lower()) for k, c in HABITAT_COLOURS.items()]
    elif name == "resources" and option == "best":
        palette = plt.get_cmap("tab20")(np.arange(len(labels)) % 20)[:, :3]
        rgb[img >= 0] = palette[img[img >= 0]]
        rgb[ocean] = to_rgb(OCEAN_COLOUR)
        rgba = np.concatenate([rgb, valid[..., None].astype(float)], axis=-1)
        shown = np.unique(img[img >= 0])
        return rgba, [Patch(color=palette[k], label=labels[k]) for k in shown], sampler.extent
    elif name == "resources":
        cmap = plt.get_cmap("YlOrBr")
        rgb = cmap(np.clip(np.nan_to_num(img), 0, 1))[..., :3]
        extra = (cmap, plt.Normalize(0, 1))
    elif name == "suitability":
        cmap = plt.get_cmap("viridis")
        norm = PowerNorm(0.5, vmin=0.0, vmax=1.0)          # most land holds a small share of the best cell
        rgb = cmap(norm(np.clip(np.nan_to_num(img), 0, 1)))[..., :3]
        extra = (cmap, norm)
    elif name in LEVEL_AXES:
        palette = plt.get_cmap("YlOrRd")(np.linspace(0.15, 0.95, len(labels)))[:, :3]
        lived = img >= 0
        rgb[lived] = palette[np.clip(img[lived], 0, len(labels) - 1)]
        shown = np.unique(img[lived])
        extra = [Patch(color=palette[k], label=labels[k]) for k in shown if 0 <= k < len(labels)]
    elif name == "bands":
        live = np.nan_to_num(img) >= 1
        norm = LogNorm(vmin=1.0, vmax=max(2.0, float(values.max())))
        cmap = plt.get_cmap("YlGn")
        rgb[live] = cmap(norm(img[live]))[..., :3]
        ruled = sampler.sample((_owner(civ) >= 0).astype(float), categorical=True).astype(bool)
        rgb[ruled & ~live] = (0.72, 0.72, 0.72)
        extra = (cmap, norm)
    elif name == "livestock":
        colours = list(LIVESTOCK_COLOURS.values())
        for k in range(3):
            rgb[img == k] = to_rgb(colours[k])
        extra = [Patch(color=c, label=label) for label, c in LIVESTOCK_COLOURS.items()]
    elif name == "land":
        cmap = plt.get_cmap("YlGnBu")
        rgb = cmap(np.clip(np.nan_to_num(img), 0, 1))[..., :3]
        extra = (cmap, plt.Normalize(0, 1))
    rgb[ocean] = 0.5 * rgb[ocean] + 0.5 * np.array(to_rgb(OCEAN_COLOUR)) if name == "resources" else to_rgb(OCEAN_COLOUR)
    rgba = np.concatenate([rgb, valid[..., None].astype(float)], axis=-1)
    return rgba, extra, sampler.extent


def _project(proj, points: np.ndarray):
    """Return projected coordinates of unit vectors, with a flag for points on the visible map."""
    import cartopy.crs as ccrs

    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1, 1)))
    with np.errstate(invalid="ignore"):
        xy = proj.transform_points(ccrs.Geodetic(), lon, lat)[:, :2]
    return xy, np.isfinite(xy).all(axis=1)


def road_collections(world, proj, width_scale: float = 1.0, kinds=tuple(ROAD_STYLE)) -> list[LineCollection]:
    """Return the roads of the given kinds as smoothed line collections, split where they wrap around the map edge."""
    civ = _civ(world)
    roads = civ.roads
    x0, x1 = proj.x_limits
    y0, y1 = proj.y_limits
    span = max(x1 - x0, y1 - y0)
    points = world.grid.points
    out = []
    for kind, (colour, lw, z) in ROAD_STYLE.items():
        if kind not in kinds:
            continue
        segments = []
        for i in np.flatnonzero(roads.kind == ROAD_KINDS.index(kind)):
            path = points[roads.path(i)]
            if len(path) > 2:
                path = smooth_line(path, passes=2)
            xy, ok = _project(proj, path)
            jump = np.r_[False, np.hypot(*np.diff(xy, axis=0).T) > 0.05 * span]
            piece = []
            for k in range(len(xy)):
                if (not ok[k] or jump[k]) and len(piece) > 1:
                    segments.append(np.array(piece))
                if not ok[k] or jump[k]:
                    piece = []
                if ok[k]:
                    piece.append(xy[k])
            if len(piece) > 1:
                segments.append(np.array(piece))
        if segments:
            out.append(LineCollection(segments, colors=colour, linewidths=lw * width_scale, zorder=z,
                                      linestyles="dashed" if kind == "sea_route" else "solid", capstyle="round"))
    return out


def draw_settlements(ax, world, proj, labels: int = LABELLED_SETTLEMENTS, size_scale: float = 1.0,
                     limit: Optional[int] = None) -> None:
    """Mark settlements (the ``limit`` largest, or all) with dots sized by population and label the largest."""
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe

    st = _civ(world).settlements
    if len(st) == 0:
        return
    order = np.argsort(-st.population)[:limit]
    lon, lat = st.lon_deg[order], st.lat_deg[order]
    size = size_scale * np.clip(np.log10(np.maximum(st.population[order], 1.0)) - 2.5, 0.2, 5) ** 2.2
    colour = np.where(np.isin(st.form[order], [SETTLEMENT_FORMS.index(f) for f in ("dome", "underground",
                                                                                    "outpost", "orbital_anchor")]),
                      "#4b2a7a", "#1a1a1a")
    ax.scatter(lon, lat, s=size, c=list(colour), transform=ccrs.PlateCarree(), zorder=5, linewidths=0)
    shown = 0
    for k in range(len(order)):
        if shown >= labels:
            break
        xy, ok = _project(proj, np.array([[np.cos(np.radians(lat[k])) * np.cos(np.radians(lon[k])),
                                           np.cos(np.radians(lat[k])) * np.sin(np.radians(lon[k])),
                                           np.sin(np.radians(lat[k]))]]))
        if not ok[0]:
            continue
        ax.text(xy[0, 0], xy[0, 1], " " + st.name[order[k]], fontsize=6, zorder=6, transform=proj,
                va="center", ha="left", color="#111111", path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        shown += 1


def label_polities(ax, world, proj, count: int = LABELLED_POLITIES) -> None:
    """Write the names of the largest polities at the population centre of their land."""
    import matplotlib.patheffects as pe

    civ = _civ(world)
    owner = _owner(civ)
    people = civ.layers["population_density"].values.sum(axis=0)
    points = world.grid.points
    for p in sorted(civ.polities, key=lambda p: -(p.area_km2 or 0))[:count]:
        cells = np.flatnonzero(owner == p.id)
        if cells.size == 0 or not p.name:
            continue
        centre = (points[cells] * people[cells, None]).sum(axis=0)
        centre = points[cells[np.argmax(points[cells] @ centre)]]
        xy, ok = _project(proj, centre[None])
        if ok[0]:
            ax.text(xy[0, 0], xy[0, 1], p.name.upper(), fontsize=6.5, zorder=6, transform=proj, ha="center",
                    va="center", color="#222222", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white", alpha=0.7)])


def draw_culture_overlays(ax, world, field: str, proj, globe: bool) -> None:
    """Add the lines, markers and labels a culture field shows over its image."""
    name, _ = split_field(field)
    scale = 0.9 if globe else 1.0
    if name in ("roads", "political"):
        kinds = POLITICAL_ROADS if name == "political" else tuple(ROAD_STYLE)
        for lc in road_collections(world, proj, scale, kinds):
            ax.add_collection(lc)
        if name == "roads":
            present = {ROAD_KINDS[k] for k in np.unique(_civ(world).roads.kind)}
            ax.legend(handles=[plt.Line2D([], [], color=c, lw=max(w * 2, 0.8), label=k.replace("_", " "),
                                          linestyle="--" if k == "sea_route" else "-")
                               for k, (c, w, _) in ROAD_STYLE.items() if k in present],
                      loc="lower left", fontsize=6, framealpha=0.8)
    if name in ("settlements", "political"):
        draw_settlements(ax, world, proj, labels=LABELLED_SETTLEMENTS if name == "political" else 0,
                         size_scale=scale, limit=POLITICAL_SETTLEMENTS if name == "political" else None)
    if name == "political":
        label_polities(ax, world, proj)


def plot_culture(world):
    """Return a 2×2 figure of a world's culture: political map, population, roads, and habitats or resources."""
    from .maps import plot_map, projection

    civ = _civ(world)
    colonists = any(c.inhabitants == "colonist" for c in civ.cultures) and "habitat" in civ.layers
    last = "habitat" if colonists else "resources:" + _richest_resource(civ)
    fig = plt.figure(figsize=(16, 9.5))
    for k, field in enumerate(("political", "population", "roads", last)):
        ax = fig.add_subplot(2, 2, k + 1, projection=projection("mollweide"))
        plot_map(world, field, "mollweide", ax=ax, width=900, legend="inside",
                 title=field.replace(":", ": ").replace("_", " "))
    fig.suptitle(f"{world.state.name}: {', '.join(c.name or f'culture {c.id}' for c in civ.cultures)}", fontsize=13)
    fig.tight_layout()
    return fig


def _richest_resource(civ) -> str:
    """Return the resource class whose potential is highest over the land people live on."""
    pot = civ.layers["resource_potential"]
    lived = civ.layers["population_density"].values.sum(axis=0) > 0
    scores = pot.values[:, lived].mean(axis=1) if lived.any() else pot.values.mean(axis=1)
    return str(pot["resource"].values[int(np.argmax(scores))])


LOCAL_ROAD_STYLE = {"track": ("#8a7560", 0.35), "road": ("#6e4e30", 0.7), "trunk": ("#a0523a", 1.3),
                    "highway": ("#8e2418", 1.8)}
FOOTPRINT_COLOUR = "#7b6a5f"
BORDER_COLOUR = "#3a1f4f"
LOCAL_LABELS = 15


def draw_local_culture(ax, region, local, labels: int = LOCAL_LABELS) -> None:
    """Draw a zoomed region's culture on its local map: footprints, borders, roads, settlements and names."""
    import matplotlib.patheffects as pe

    rows, cols = region.grid.shape
    if (local.footprint >= 0).any():
        img = np.zeros((rows, cols, 4))
        img[..., :3] = to_rgb(FOOTPRINT_COLOUR)
        img[..., 3] = 0.85 * (local.footprint.reshape(rows, cols) >= 0)
        ax.imshow(img, origin="lower", interpolation="nearest", zorder=2)
    if local.polity_map is not None:
        lab = local.polity_map.reshape(rows, cols)
        edge = np.zeros((rows, cols), dtype=bool)
        edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] >= 0) & (lab[:, :-1] >= 0)
        edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] >= 0) & (lab[:-1, :] >= 0)
        if edge.any():
            img = np.zeros((rows, cols, 4))
            img[..., :3] = to_rgb(BORDER_COLOUR)
            img[..., 3] = 0.9 * edge
            ax.imshow(img, origin="lower", interpolation="nearest", zorder=4)
    for kind, (colour, lw) in LOCAL_ROAD_STYLE.items():
        code = ROAD_KINDS.index(kind)
        segs = []
        for nodes, k, _ in local.roads:
            if k == code and len(nodes) > 1:
                r, c = np.divmod(nodes, cols)
                segs.append(np.column_stack([c, r]))
        if segs:
            ax.add_collection(LineCollection(segs, colors=colour, linewidths=lw, zorder=3, capstyle="round"))
    if len(local):
        r, c = np.divmod(local.node, cols)
        small = local.radius_km < 0.5 * region.grid.mean_spacing(region.radius_m) / 1e3
        ax.scatter(c[small], r[small], s=np.clip(np.log10(local.population[small]) - 1.0, 0.3, 3) ** 1.6,
                   c="#2a1f1a", linewidths=0, zorder=5)
        for k in np.argsort(-local.population)[:labels]:
            ax.text(c[k], r[k], " " + str(local.name[k]), fontsize=7, zorder=6, va="center", ha="left",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
