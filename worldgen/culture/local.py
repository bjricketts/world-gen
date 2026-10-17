"""Culture on a zoomed region: villages, footprints, local roads and refined borders on the region lattice.

The global civilisation lists settlements down to about one per grid cell and
keeps the rest of each cell's people as dispersed population. A zoomed region
turns that population into villages and hamlets whose sizes continue the
rank-size law, placed where the local ground suits them; gives every
settlement a footprint that grows with its population (Ortman et al. 2014);
routes local roads with the gravity-and-reuse method of the global roads on
the lattice's real slopes; carries the global trunk roads through; and redraws
polity borders by travel time so they settle on ridges and rivers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .. import heuristics as h
from ..util import named_rng
from .habitats import Habitat, classify_habitats
from .model import ROAD_KINDS, Culture
from .names import Namer
from .roads import TravelGraph, _astar, _segments, classify, route_flows, tobler_kmh
from .settlements import settlement_form


@dataclass
class LocalCulture:
    """Culture on a region lattice: settlements, footprints, roads, polities and habitats per node."""

    node: np.ndarray                 # settlement position (lattice node)
    population: np.ndarray
    culture: np.ndarray              # culture id
    polity: np.ndarray               # polity id (−1: none)
    form: np.ndarray                 # form code (model.SETTLEMENT_FORMS)
    name: np.ndarray
    listed: np.ndarray               # id in the global settlement table, −1 for villages
    radius_km: np.ndarray            # footprint radius
    footprint: np.ndarray            # per node: settlement covering it (−1: none; under half a node: none)
    roads: list = field(default_factory=list)    # (nodes, kind code, culture id) per road segment
    polity_map: np.ndarray = None    # per node: polity id (−1: none)
    culture_map: np.ndarray = None   # per node: culture id of most people (−1: none)
    habitat: np.ndarray = None       # per node: habitat class for colonists

    def __len__(self) -> int:
        """Return the number of settlements."""
        return len(self.node)


def _home_cells(world, grid, land: np.ndarray, people: np.ndarray):
    """Return each node's global cell, and the cell whose people it takes: land nodes of an empty cell (the sea on
    the global grid, land on the refined coastline) join the nearest occupied cell within 1.5 grid spacings."""
    tree = world.grid.tree
    _, parent = tree.query(grid.points)
    home = parent.copy()
    stray = np.flatnonzero(land & (people[parent] <= 0))
    if stray.size:
        spacing = world.grid.mean_spacing(1.0)
        dist, near = tree.query(grid.points[stray], k=7)
        ok = (people[near] > 0) & (dist <= 1.5 * spacing)
        first = np.argmax(ok, axis=1)
        found = ok[np.arange(stray.size), first]
        home[stray[found]] = near[np.arange(stray.size), first][found]
    return parent, home


def footprint_km2(population: np.ndarray, culture: Culture) -> np.ndarray:
    """Return the settled area (km²) of settlements of a culture by their population."""
    population = np.asarray(population, dtype=float)
    a = np.where(population < h.FOOTPRINT_SPLIT, h.FOOTPRINT_SMALL[0], h.FOOTPRINT_LARGE[0])
    b = np.where(population < h.FOOTPRINT_SPLIT, h.FOOTPRINT_SMALL[1], h.FOOTPRINT_LARGE[1])
    area = a * population ** b / 100.0
    if culture.development.energy >= 4:
        density = h.URBAN_DENSITY_KM2 * (population / h.URBAN_DENSITY_REF_POP) ** h.URBAN_DENSITY_EXPONENT
        area = np.where(population >= h.FOOTPRINT_SPLIT, population / density, area)
    return area


def truncated_power_sizes(total: float, low: float, high: float, zeta: float, rng) -> np.ndarray:
    """Return settlement sizes from a power law with pdf exponent 1 + ζ on [low, high] adding up to ``total``."""
    if total < low or high <= low:
        return np.array([total]) if total >= 1.0 else np.empty(0)
    mean = _truncated_mean(low, high, zeta)
    out, left = [], total
    while left >= low:
        n = int(1.3 * left / mean) + 2
        u = rng.random(n)
        if abs(zeta - 1.0) < 1e-9:
            s = low / (1 - u * (1 - low / high))
        else:
            s = (low ** -zeta - u * (low ** -zeta - high ** -zeta)) ** (-1 / zeta)
        keep = np.cumsum(s) <= left
        out.append(s[keep])
        left -= s[keep].sum()
        if not keep.any():
            break
    sizes = np.concatenate(out) if out else np.empty(0)
    if left >= low / 2:
        sizes = np.append(sizes, left)
    return sizes


def _truncated_mean(low: float, high: float, zeta: float) -> float:
    """Return the mean of the truncated power law used for village sizes."""
    x = np.geomspace(low, high, 400)
    pdf = x ** (-1 - zeta)
    return float(np.trapezoid(x * pdf, x) / np.trapezoid(pdf, x))


def node_suitability(region, spacing_km: float, radius_km: float) -> np.ndarray:
    """Return how well each lattice node suits a settlement: flat, dry land near water."""
    grid = region.grid
    land = ~region.ocean & ~region.lake
    adj = grid.neighbours.tocoo()
    rise = np.abs(region.elevation[adj.row] - region.elevation[adj.col]) / (adj.data * radius_km * 1e3)
    slope = np.zeros(grid.size)
    np.maximum.at(slope, adj.row, rise)
    water = (region.river_order >= h.ZOOM_RIVER_ORDER) | region.ocean | region.lake
    near = np.zeros(grid.size, dtype=bool)
    if water.any():
        d = dijkstra(grid.neighbours, indices=np.flatnonzero(water), min_only=True, limit=h.ZOOM_WATER_KM / radius_km)
        near = np.isfinite(d)
    return land * np.exp(-slope / h.ZOOM_SLOPE_SCALE) * np.where(near, h.ZOOM_WATER_BONUS, 1.0)


def _place(sizes, cells, weight, taken, shape, spacing_km, area_km2, rng):
    """Place settlements of the given sizes (largest first) at nodes drawn by weight, keeping them apart."""
    placed = np.full(sizes.size, -1, dtype=np.int64)
    rows, cols = shape
    gap = h.SETTLEMENT_EXCLUSION * np.sqrt(area_km2 / max(sizes.size, 1))
    rad = max(0, int(gap / spacing_km))
    w = weight[cells].astype(float)
    priority = np.log(np.maximum(w, 1e-300)) + rng.gumbel(size=cells.size)      # weighted order without replacement
    order = cells[np.argsort(-priority)]
    k = 0
    for i in np.argsort(-sizes, kind="stable"):
        while k < order.size and (taken.flat[order[k]] or weight[order[k]] <= 0):
            k += 1
        if k >= order.size:
            break
        node = order[k]
        placed[i] = node
        r, c = divmod(int(node), cols)
        taken[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1] = True
    return placed


def crosses(shape, row: np.ndarray, col: np.ndarray, river: np.ndarray) -> np.ndarray:
    """Return which lattice edges cross a river: those stepping onto a river node, and diagonal steps between
    two river nodes that touch at the corner."""
    cols = shape[1]
    r0, c0 = np.divmod(row, cols)
    r1, c1 = np.divmod(col, cols)
    diagonal = (r0 != r1) & (c0 != c1)
    corners = river[r0 * cols + c1] & river[r1 * cols + c0]
    return river[col] | (diagonal & corners)


class LatticeGraph(TravelGraph):
    """Travel times on a region lattice: Tobler on the real slopes, rivers to cross, no boats."""

    def __init__(self, region, transport: int, radius_km: float):
        """Build the travel times of every lattice edge for a transport level."""
        adj = region.grid.neighbours.tocoo()
        self.row, self.col = adj.row.astype(np.int64), adj.col.astype(np.int64)
        self.km = adj.data * radius_km
        self.size = region.grid.size
        water = region.ocean | region.lake
        self.water_cell = water
        r, c = self.row, self.col
        rise = (region.elevation[c] - region.elevation[r]) / (self.km * 1e3)
        hours = self.km / (tobler_kmh(rise) * h.LAND_SPEED_FACTOR[transport])
        big = crosses(region.grid.shape, r, c, (region.discharge_m3_s > h.RIVER_CROSSING_M3S) & ~water)
        hours = hours + np.where(big, h.RIVER_CROSSING_HOURS[transport], 0.0)
        land = ~water[r] & ~water[c]
        self._finish(hours, land, ~land, float(tobler_kmh(np.array(0.0))) * h.LAND_SPEED_FACTOR[transport])


def _local_pairs(points: np.ndarray, weight: np.ndarray, radius_km: float, spacing_km: float):
    """Return each settlement's strongest gravity partners among its nearest neighbours, strongest pair first."""
    n = len(points)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    k = min(h.ZOOM_ROAD_NEIGHBOURS + 1, n)
    _, nb = cKDTree(points).query(points, k=k)
    nb = nb[:, 1:]
    i = np.repeat(np.arange(n), nb.shape[1])
    j = nb.ravel()
    d = np.maximum(radius_km * np.arccos(np.clip(np.einsum("ij,ij->i", points[i], points[j]), -1, 1)), spacing_km)
    g = weight[i] * weight[j] / d**2
    rank = np.argsort(np.argsort(-g.reshape(n, -1), axis=1), axis=1).ravel()
    keep = rank < h.ZOOM_ROAD_PARTNERS
    pairs = np.sort(np.stack([i[keep], j[keep]], axis=1), axis=1)
    pairs, first = np.unique(pairs, axis=0, return_index=True)
    g = g[keep][first]
    order = np.argsort(-g, kind="stable")
    return pairs[order], g[order]


def _refine_borders(region, label: np.ndarray, hours: sparse.csr_matrix, spacing_km: float) -> np.ndarray:
    """Return polity labels with the nodes near borders reassigned to the polity whose interior reaches them first."""
    n = label.size
    adj = region.grid.neighbours.tocoo()
    change = (label[adj.row] != label[adj.col]) & (label[adj.row] >= 0) & (label[adj.col] >= 0)
    if not change.any():
        return label
    edge_nodes = np.unique(np.r_[adj.row[change], adj.col[change]])
    to_border = dijkstra(region.grid.neighbours, indices=edge_nodes, min_only=True) * (spacing_km / np.median(adj.data))
    # a polity's interior lies a band's width from its border, or half its greatest depth if it is narrower
    seeds = np.zeros(n, dtype=bool)
    for p in np.unique(label[label >= 0]):
        mine = label == p
        depth = min(h.ZOOM_BORDER_BAND_KM, 0.5 * float(np.max(np.where(np.isfinite(to_border[mine]),
                                                                      to_border[mine], 0.0))))
        seeds |= mine & (to_border >= depth)
    band = (label >= 0) & ~seeds
    _, _, source = dijkstra(hours, indices=np.flatnonzero(seeds), min_only=True, return_predecessors=True)
    out = label.copy()
    moved = band & (label >= 0) & (source >= 0)
    out[moved] = label[source[moved]]
    return out


def zoom_culture(world, region) -> LocalCulture:
    """Return the culture of a zoomed region, built from the world's civilisation."""
    civ = world.culture
    if civ is None or "population_density" not in civ.layers:
        raise ValueError(f"{world.state.name} has no placed culture to zoom into")
    grid = region.grid
    radius_km = region.radius_m / 1e3
    rows, cols = grid.shape
    spacing_km = float(np.median(grid.neighbours.data)) * radius_km
    node_km2 = spacing_km**2
    land = ~region.ocean & ~region.lake
    layers = civ.layers
    parent, home = _home_cells(world, grid, land, layers["population_density"].values.sum(axis=0))
    suit = node_suitability(region, spacing_km, radius_km)
    density = layers["population_density"].values[:, home]
    # each cell's dispersed people in the region (by its share of the region's nodes) spread over its land nodes
    counted = np.bincount(parent, minlength=world.grid.size).astype(float)
    landed = np.bincount(home[land], minlength=world.grid.size).astype(float)
    spread = np.where(land, counted[home] / np.maximum(landed[home], 1.0), 0.0)
    dispersed = layers["dispersed_density"].values[:, home] * node_km2 * spread
    culture_map = np.where(land & (density.max(axis=0) > 0), np.argmax(density, axis=0), -1)
    ids = np.array([c.id for c in civ.cultures])
    habitat = None
    if world.state is not None:
        habitat = classify_habitats(world.state, region.elevation, region.temperature_k, land)
    st = civ.settlements
    taken = np.zeros((rows, cols), dtype=bool)
    cols_out = {k: [] for k in ("node", "population", "culture", "listed")}
    # listed settlements inside the region
    if len(st):
        la, lo = np.radians(st.lat_deg), np.radians(st.lon_deg)
        sp = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        dist, node = grid.tree().query(sp)
        inside = np.flatnonzero(dist * radius_km < spacing_km)
        for k in inside[np.argsort(-st.population[inside])]:
            n0 = node[k]
            if not land[n0]:                                 # a coastal city stands on the nearest land node
                near = grid.tree().query_ball_point(grid.points[n0], 3 * spacing_km / radius_km)
                near = [x for x in near if land[x]]
                if not near:
                    continue
                n0 = near[int(np.argmax(suit[near]))]
            cols_out["node"].append(n0)
            cols_out["population"].append(st.population[k])
            cols_out["culture"].append(st.culture[k])
            cols_out["listed"].append(st.id[k])
            r, c = divmod(int(n0), cols)
            taken[r, c] = True
    # villages from each cell's dispersed people
    for ci, c in enumerate(civ.cultures):
        rng = named_rng(civ.seed, f"zoom.{region.seed}.culture.{ci}")
        zeta = c.zipf_exponent or h.ZIPF_DEFAULT
        mine = st.population[st.culture == c.id] if len(st) else np.empty(0)
        low, high = (h.FORAGER_CAMP if c.development.energy == 0
                     else (h.SETTLEMENT_SMALLEST, float(mine.min()) if mine.size else h.FOOTPRINT_SPLIT))
        weight = suit * (culture_map == ci) if habitat is None or c.inhabitants != "colonist" \
            else suit * (culture_map == ci) * (habitat != Habitat.NONE)
        order = np.argsort(home, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(home[order]) != 0, True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            cells = order[a:b]
            people = dispersed[ci, cells].sum()
            if people < low / 2:
                continue
            sizes = truncated_power_sizes(people, low, high, zeta, rng)
            at = _place(sizes, cells, weight, taken, (rows, cols), spacing_km, cells.size * node_km2, rng)
            ok = at >= 0
            cols_out["node"] += list(at[ok])
            cols_out["population"] += list(sizes[ok])
            cols_out["culture"] += [c.id] * int(ok.sum())
            cols_out["listed"] += [-1] * int(ok.sum())
    node = np.array(cols_out["node"], dtype=np.int64)
    population = np.array(cols_out["population"], dtype=float)
    culture = np.array(cols_out["culture"], dtype=np.int32)
    listed = np.array(cols_out["listed"], dtype=np.int64)
    by_id = {c.id: c for c in civ.cultures}
    form = np.zeros(node.size, dtype=np.int8)
    radius = np.zeros(node.size)
    for cid, c in by_id.items():
        m = culture == cid
        if m.any():
            form[m] = settlement_form(population[m], c, None if habitat is None else habitat[node[m]])
            radius[m] = np.sqrt(footprint_km2(population[m], c) / np.pi)
    footprint = _footprints(grid, node, population, radius, radius_km, node_km2)
    names = _names(world, region, civ, node, population, culture, listed, spacing_km)
    polity_map = _polity_map(region, layers, home, culture_map, ids, spacing_km, radius_km, civ, by_id)
    polity = np.where(node >= 0, polity_map[node], -1)
    roads = _roads(world, region, civ, node, population, culture, spacing_km, radius_km, parent)
    return LocalCulture(node=node, population=population, culture=culture, polity=polity, form=form, name=names,
                        listed=listed, radius_km=radius, footprint=footprint, roads=roads, polity_map=polity_map,
                        culture_map=np.where(culture_map >= 0, ids[np.maximum(culture_map, 0)], -1),
                        habitat=habitat)


def _footprints(grid, node, population, radius, radius_km, node_km2) -> np.ndarray:
    """Return, per lattice node, the settlement whose footprint covers it (larger settlements win); settlements
    smaller than half a node cover none."""
    out = np.full(grid.size, -1, dtype=np.int64)
    tree = grid.tree()
    big = np.flatnonzero(np.pi * radius**2 >= 0.5 * node_km2)
    for i in big[np.argsort(population[big])]:                 # smaller first, so larger overwrite
        cover = tree.query_ball_point(grid.points[node[i]], radius[i] / radius_km)
        out[cover] = i
        out[node[i]] = i
    return out


def _names(world, region, civ, node, population, culture, listed, spacing_km) -> np.ndarray:
    """Return settlement names: listed settlements keep theirs, villages get new ones from their culture."""
    names = np.full(node.size, "", dtype=object)
    st = civ.settlements
    if len(st):
        index = {int(i): k for k, i in enumerate(st.id)}
        for k in np.flatnonzero(listed >= 0):
            names[k] = st.name[index[int(listed[k])]]
    river = region.discharge_m3_s > h.RIVER_CROSSING_M3S / 10
    grid = region.grid
    adj = grid.neighbours.tocsr()
    for ci, c in enumerate(civ.cultures):
        if not c.phonology:
            continue
        namer = Namer(c.phonology, civ.seed, f"zoom.{region.seed}.names.{ci}")
        namer.used |= set(st.name[st.culture == c.id]) if len(st) else set()
        for k in np.flatnonzero((culture == c.id) & (listed < 0)):
            n0 = node[k]
            nb = adj.indices[adj.indptr[n0]:adj.indptr[n0 + 1]]
            feats = []
            if river[n0]:
                feats.append("river")
            if region.ocean[nb].any():
                feats.append("harbour")
            if region.lake[nb].any():
                feats.append("lake")
            feats.append("hill" if region.elevation[n0] > 800 else "field")
            names[k] = namer.place(tuple(feats))
    return names.astype("U40")


def _polity_map(region, layers, parent, culture_map, ids, spacing_km, radius_km, civ, by_id) -> np.ndarray:
    """Return the polity of each node, with borders redrawn by travel time."""
    if "polity" not in layers:
        return np.full(region.grid.size, -1, dtype=np.int64)
    polity = layers["polity"].values
    label = np.where(culture_map >= 0, polity[np.maximum(culture_map, 0), parent], -1).astype(np.int64)
    label[region.ocean | region.lake] = -1
    top = max((c.development.transport for c in civ.cultures), default=0)
    graph = LatticeGraph(region, top, radius_km)
    hours = graph.base_hours.copy()
    crossing = crosses(region.grid.shape, graph.row, graph.col, region.river_order >= h.ZOOM_BORDER_RIVER_ORDER)
    hours = hours + np.where(crossing, h.ZOOM_BORDER_RIVER_HOURS, 0.0)
    return _refine_borders(region, label, graph.matrix(hours), spacing_km)


def _roads(world, region, civ, node, population, culture, spacing_km, radius_km, parent) -> list:
    """Return local road segments per culture: gravity links routed in waves, classed by flow."""
    out = []
    points = region.grid.points
    for c in civ.cultures:
        mine = np.flatnonzero(culture == c.id)
        if mine.size < 2 or c.development.energy == 0:
            continue
        graph = LatticeGraph(region, c.development.transport, radius_km)
        cells, inv = np.unique(node[mine], return_inverse=True)
        weight = np.bincount(inv, weights=population[mine])
        pairs, flows = _local_pairs(points[cells], weight, radius_km, spacing_km)
        if pairs.size == 0:
            continue
        carried = route_flows(graph, cells, pairs, flows, points, radius_km)
        stops = np.zeros(region.grid.size, dtype=bool)
        stops[cells] = True
        segs = _segments(graph, carried, stops)
        if not segs:
            continue
        flow = np.array([carried[ids].max() for _, ids in segs])
        length = np.array([graph.km[ids].sum() for _, ids in segs])
        kinds = classify(flow, length, np.zeros(len(segs), dtype=bool), min(c.development.transport, 1))
        out += [(p, int(k), c.id) for (p, _), k in zip(segs, kinds)]
        out += _through_roads(world, region, civ, c, graph, parent, radius_km, node[mine], population[mine],
                              carried[graph.undirected] > 0)
    return out


def _through_roads(world, region, civ, culture, graph, parent, radius_km, node=None, population=None,
                   local=None) -> list:
    """Return the global land roads of a culture that cross the region, routed on the lattice with their class.

    Each global cell the road passes is anchored at the largest local settlement within half a grid spacing
    of its centre (else the nearest land node), so trunk roads run through towns. Roads are routed from the
    highest class down; ``local`` road edges and those of roads already routed travel at the reuse discount,
    so through roads follow local roads and merge with each other where these serve.
    """
    roads = civ.roads
    if not len(roads):
        return []
    points = region.grid.points
    land = ~region.ocean & ~region.lake
    tree = cKDTree(points[land])
    land_nodes = np.flatnonzero(land)
    in_region = np.zeros(world.grid.size, dtype=bool)
    in_region[np.unique(parent)] = True
    discount = h.ROAD_REUSE_DISCOUNT
    used = np.zeros(graph.row.size, dtype=bool) if local is None else local.copy()
    best = np.zeros(graph.size)
    came = np.zeros(graph.size, dtype=np.int64)
    stamp = np.zeros(graph.size, dtype=np.int64)
    out, mark = [], 0
    sea = ROAD_KINDS.index("sea_route")
    mine = np.flatnonzero((roads.culture == culture.id) & (roads.kind != sea))
    town_tree = cKDTree(points[node]) if node is not None and node.size else None
    reach = 0.5 * world.grid.mean_spacing(1.0)
    for i in mine[np.argsort(-roads.kind[mine], kind="stable")]:
        cells = roads.path(i)
        inside = in_region[cells]
        if inside.sum() < 2:
            continue
        _, anchor = tree.query(world.grid.points[cells[inside]])
        anchor = land_nodes[anchor]
        if town_tree is not None:
            for j, cell in enumerate(cells[inside]):
                near = town_tree.query_ball_point(world.grid.points[cell], reach)
                if near:
                    anchor[j] = node[near[int(np.argmax(population[near]))]]
        m = graph.matrix(graph.base_hours * np.where(used, discount, 1.0))
        m.sort_indices()
        pieces = []
        for a, b in zip(anchor[:-1], anchor[1:]):
            if a == b:
                continue
            mark += 1
            straight = radius_km * np.arccos(np.clip(points[a] @ points[b], -1, 1))
            path = _astar(m.indptr, m.indices, m.data, points, radius_km, graph.top_speed / discount, a, b,
                          h.ROAD_MAX_DETOUR * straight / graph.flat_speed + 1.0, best, came, stamp, mark)
            if path.size > 1:
                pieces.append(path if not pieces else path[1:])
        if pieces:
            path = np.concatenate(pieces)
            e = graph.edges(path[:-1], path[1:])
            used[e] = True
            used[graph.edges(path[1:], path[:-1])] = True
            out.append((path, int(roads.kind[i]), culture.id))
    return out


def border_river_share(region, local: LocalCulture) -> tuple[float, float]:
    """Return the share of polity-border edges and of all land edges that touch a river of the border order."""
    adj = region.grid.neighbours.tocoo()
    lab = local.polity_map
    both = (lab[adj.row] >= 0) & (lab[adj.col] >= 0) & (adj.row < adj.col)
    border = both & (lab[adj.row] != lab[adj.col])
    river = region.river_order >= h.ZOOM_BORDER_RIVER_ORDER
    at = river[adj.row] | river[adj.col]
    return (float(at[border].mean()) if border.any() else float("nan")), float(at[both].mean())
