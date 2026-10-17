"""Roads and sea routes between settlements, grown by gravity flows over a travel-time surface.

Each culture's settlement cells ask for links to their strongest gravity
partners (P_i P_j / d²) among nearby settlements. The links are routed by least
travel time over the grid, strongest first and in waves; roads built in earlier
waves become cheaper, so later routes merge into them and a trunk hierarchy
forms. The network is then cut into segments between settlements and
junctions, and each segment gets a class from the flow it carries.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError

from .. import heuristics as h
from ..surface.fields import Terrain
from .model import ROAD_KINDS, RoadTable, TradeTable


def tobler_kmh(slope: np.ndarray) -> np.ndarray:
    """Return walking speed (km/h) on a signed slope by Tobler's hiking function."""
    return 6.0 * np.exp(-3.5 * np.abs(slope + 0.05))


class TravelGraph:
    """Directed grid edges with travel times for one transport level, and the medium each edge crosses."""

    def __init__(self, world, transport: int):
        """Build the travel times of every grid edge for a transport level."""
        s, grid = world.surface, world.grid
        radius_km = float(s.attrs["radius_m"]) / 1e3
        adj = grid.neighbours.tocoo()
        self.row, self.col = adj.row.astype(np.int64), adj.col.astype(np.int64)
        self.km = adj.data * radius_km
        self.size = grid.size
        water = s["ocean"].values.astype(bool)
        if "lake" in s:
            water |= s["lake"].values.astype(bool)
        self.water_cell = water
        z = s["elevation"].values.astype(float)
        names = {int(t): t.name for t in Terrain}
        sub = np.array([h.SUBGRID_SLOPE.get(names.get(int(t), ""), h.SUBGRID_SLOPE_OTHER)
                        for t in s["terrain"].values])
        r, c = self.row, self.col
        rise = (z[c] - z[r]) / (self.km * 1e3)
        slope = np.sign(rise) * (np.abs(rise) + 0.5 * (sub[r] + sub[c]))
        speed = tobler_kmh(slope) * h.LAND_SPEED_FACTOR[transport]
        if "ice_thickness" in s:
            ice = s["ice_thickness"].values > 0
            speed = np.where(ice[r] | ice[c], speed * h.ICE_SPEED_SHARE, speed)
        hours = self.km / speed
        if "discharge" in s:
            big = s["discharge"].values > h.RIVER_CROSSING_M3S
            hours = hours + np.where(big[c] & ~water[c], h.RIVER_CROSSING_HOURS[transport], 0.0)
        land = ~water[r] & ~water[c]
        boat = h.WATER_SPEED_KMH[transport]
        if boat > 0:
            port = np.where(water[r] != water[c], h.PORT_HOURS[transport] / 2, 0.0)
            hours = np.where(land, hours, self.km / boat + port)
            usable = np.ones(r.size, dtype=bool)
        else:
            usable = land
        flat = min(v for v in (float(tobler_kmh(np.array(0.0))) * h.LAND_SPEED_FACTOR[transport], boat) if v > 0)
        self._finish(hours, usable, ~land, flat)

    def _finish(self, hours: np.ndarray, usable: np.ndarray, water_edge: np.ndarray, flat_speed: float) -> None:
        """Store the edge times and build the lookups shared by every travel graph."""
        r, c = self.row, self.col
        self.base_hours = hours
        self.usable = usable
        self.water_edge = water_edge
        self.flat_speed = flat_speed
        self.top_speed = float(np.max(self.km[usable] / hours[usable])) if usable.any() else 1.0
        # undirected id of each directed edge: the smaller of its index and its reverse's
        index = sparse.csr_matrix((np.arange(r.size) + 1, (r, c)), shape=(self.size, self.size))
        reverse = np.asarray(index[c, r]).ravel() - 1
        self.undirected = np.minimum(np.arange(r.size), reverse)
        self._index = index

    def matrix(self, hours: np.ndarray) -> sparse.csr_matrix:
        """Return the travel-time matrix for the given edge times (unusable edges left out)."""
        u = self.usable
        return sparse.csr_matrix((hours[u], (self.row[u], self.col[u])), shape=(self.size, self.size))

    def edges(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the directed edge indices from cells ``a`` to cells ``b``."""
        return np.asarray(self._index[a, b]).ravel() - 1


def _near_pairs(points: np.ndarray) -> np.ndarray:
    """Return node pairs that are Delaunay neighbours on the sphere or share a Delaunay neighbour."""
    n = points.shape[0]
    if n < 4:
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64).reshape(-1, 2)
    try:
        simplices = ConvexHull(points).simplices
    except QhullError:
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64).reshape(-1, 2)
    e = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    adj = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8)
    two = ((adj @ adj + adj) > 0).tocoo()
    keep = two.row < two.col
    return np.stack([two.row[keep], two.col[keep]], axis=1).astype(np.int64)


def gravity_pairs(points: np.ndarray, weight: np.ndarray, radius_km: float, spacing_km: float,
                  exponent: float = 2.0) -> tuple:
    """Return the node pairs to link (each node's strongest partners nearby, within the link reach) and their
    gravity (P_i P_j)^μ / d^exponent, strongest first."""
    pairs = _near_pairs(points)
    if pairs.size == 0:
        return pairs, np.empty(0)
    d = radius_km * np.arccos(np.clip(np.einsum("ij,ij->i", points[pairs[:, 0]], points[pairs[:, 1]]), -1, 1))
    near = d <= h.ROAD_MAX_LINK_KM
    pairs, d = pairs[near], d[near]
    g = (weight[pairs[:, 0]] * weight[pairs[:, 1]]) ** h.ROAD_GRAVITY_MASS_EXPONENT / np.maximum(d, spacing_km) ** exponent
    chosen = np.zeros(len(pairs), dtype=bool)
    for side in (0, 1):
        order = np.lexsort((-g, pairs[:, side]))
        node = pairs[order, side]
        first = np.r_[0, np.flatnonzero(np.diff(node)) + 1]
        rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
        chosen[order[rank < h.ROAD_PARTNERS]] = True
    pairs, g = pairs[chosen], g[chosen]
    order = np.argsort(-g, kind="stable")
    return pairs[order], g[order]


@njit(cache=True)
def _astar(indptr, indices, hours, xyz, radius_km, top_speed, source, target, limit, best, came, stamp, mark):
    """Return the least-time path from source to target (cells, source first), or an empty array beyond ``limit`` hours."""
    heap = [(0.0, np.int64(source))]
    best[source] = 0.0
    came[source] = -1
    stamp[source] = mark
    t = xyz[target]
    while heap:
        f, u = heapq.heappop(heap)
        if u == target:
            break
        if f > limit:
            return np.empty(0, dtype=np.int64)
        g = best[u]
        for k in range(indptr[u], indptr[u + 1]):
            v = np.int64(indices[k])
            cand = g + hours[k]
            if stamp[v] != mark or cand < best[v]:
                stamp[v] = mark
                best[v] = cand
                came[v] = u
                p = xyz[v]
                dot = min(1.0, max(-1.0, p[0] * t[0] + p[1] * t[1] + p[2] * t[2]))
                heapq.heappush(heap, (cand + radius_km * np.arccos(dot) / top_speed, v))
    if stamp[target] != mark:
        return np.empty(0, dtype=np.int64)
    n = 1
    v = target
    while v != source:
        v = came[v]
        n += 1
    path = np.empty(n, dtype=np.int64)
    v = target
    for i in range(n - 1, -1, -1):
        path[i] = v
        v = came[v]
    return path


def route_flows(graph: TravelGraph, cells: np.ndarray, pairs: np.ndarray, flows: np.ndarray,
                points: np.ndarray, radius_km: float, old: np.ndarray | None = None,
                routes: np.ndarray | None = None) -> np.ndarray:
    """Route node pairs in waves with reuse discounts; return the flow carried by each undirected edge.

    ``old`` lists directed edges of earlier roads, which cost ROAD_OLD_DISCOUNT of their time from the
    start. ``routes`` (pairs × 2), if given, receives each pair's route length (km) and travel time (hours,
    without discounts); unrouted pairs get NaN.
    """
    hours = graph.base_hours.copy()
    if old is not None and old.size:
        hours[old] *= h.ROAD_OLD_DISCOUNT
    if routes is not None:
        routes[:] = np.nan
    carried = np.zeros(graph.row.size)
    best = np.zeros(graph.size)
    came = np.zeros(graph.size, dtype=np.int64)
    stamp = np.zeros(graph.size, dtype=np.int64)
    top_speed = graph.top_speed / min(h.ROAD_REUSE_DISCOUNT, h.ROAD_OLD_DISCOUNT)   # keeps the bound admissible
    mark = 0
    for wave in np.array_split(np.arange(len(pairs)), h.ROAD_WAVES):
        if wave.size == 0:
            continue
        m = graph.matrix(hours)
        m.sort_indices()
        built = []
        for k in wave:
            i, j = pairs[k]
            a, b = cells[i], cells[j]
            straight = radius_km * np.arccos(np.clip(points[a] @ points[b], -1, 1))
            limit = h.ROAD_MAX_DETOUR * straight / graph.flat_speed + 1.0
            mark += 1
            path = _astar(m.indptr, m.indices, m.data, points, radius_km, top_speed, a, b, limit,
                          best, came, stamp, mark)
            if path.size < 2:
                continue
            e = graph.edges(path[:-1], path[1:])
            carried[graph.undirected[e]] += flows[k]
            built.append(e)
            if routes is not None:
                routes[k] = graph.km[e].sum(), graph.base_hours[e].sum()
        if built:
            e = np.concatenate(built)
            both = np.concatenate([e, graph.edges(graph.col[e], graph.row[e])])
            hours[both] = np.minimum(hours[both], graph.base_hours[both] * h.ROAD_REUSE_DISCOUNT)
    return carried


def _segments(graph: TravelGraph, carried: np.ndarray, stops: np.ndarray) -> list:
    """Return the network cut into chains between stop cells and junctions: (cells, edge ids) per segment."""
    used = np.flatnonzero((carried > 0) & (graph.undirected == np.arange(carried.size)))
    if used.size == 0:
        return []
    a, b = graph.row[used], graph.col[used]
    degree = np.bincount(np.r_[a, b], minlength=graph.size)
    medium = graph.water_edge[used]
    wet = np.zeros(graph.size, dtype=bool)
    dry = np.zeros(graph.size, dtype=bool)
    np.logical_or.at(wet, np.r_[a, b], np.r_[medium, medium])
    np.logical_or.at(dry, np.r_[a, b], np.r_[~medium, ~medium])
    stop = stops | (degree != 2) | (wet & dry)
    nbrs: dict[int, list] = {}
    for k, (x, y) in enumerate(zip(a, b)):
        nbrs.setdefault(int(x), []).append((int(y), k))
        nbrs.setdefault(int(y), []).append((int(x), k))
    seen = np.zeros(used.size, dtype=bool)
    out = []
    starts = [c for c in nbrs if stop[c]] + list(nbrs)          # stops first, then any cell (pure loops)
    for c0 in starts:
        for nxt, k in nbrs[c0]:
            if seen[k]:
                continue
            cells, ids, cur = [c0], [], c0
            while True:
                seen[k] = True
                ids.append(used[k])
                cells.append(nxt)
                cur = nxt
                if stop[cur]:
                    break
                step = [(y, kk) for y, kk in nbrs[cur] if not seen[kk]]
                if not step:
                    break
                nxt, k = step[0]
            out.append((np.array(cells), np.array(ids)))
    return out


def classify(flow: np.ndarray, length: np.ndarray, sea: np.ndarray, transport: int) -> np.ndarray:
    """Return the road kind of each segment: sea route, or a land class filled from the busiest down."""
    classes = h.ROAD_CLASSES_BY_TRANSPORT[transport]
    kinds = np.full(flow.size, ROAD_KINDS.index("sea_route"), dtype=np.int8)
    land = np.flatnonzero(~sea)
    if land.size == 0:
        return kinds
    n = len(classes)
    shares = h.ROAD_CLASS_LENGTH_RATIO ** np.arange(n)                # busiest class first
    bounds = np.cumsum(shares) / shares.sum()
    order = land[np.argsort(-flow[land], kind="stable")]
    cum = np.cumsum(length[order]) / length[order].sum()
    tier = np.minimum(np.searchsorted(bounds, cum - 1e-12), n - 1)   # 0 = busiest
    kinds[order] = [ROAD_KINDS.index(classes[n - 1 - t]) for t in tier]
    return kinds


def road_edges(graph: TravelGraph, roads: RoadTable) -> np.ndarray:
    """Return the directed edges of a graph that roads follow (both directions)."""
    if roads is None or len(roads) == 0:
        return np.empty(0, dtype=np.int64)
    parts = []
    for i in range(len(roads)):
        p = roads.path(i)
        if p.size > 1:
            parts.append(np.stack([p[:-1], p[1:]], axis=1))
    if not parts:
        return np.empty(0, dtype=np.int64)
    pairs = np.unique(np.concatenate(parts), axis=0)
    e = np.concatenate([graph.edges(pairs[:, 0], pairs[:, 1]), graph.edges(pairs[:, 1], pairs[:, 0])])
    e = e[e >= 0]
    return e[graph.usable[e]]


def build_roads(world, settlements, cultures, people: np.ndarray | None = None) -> RoadTable:
    """Return the road segments of every culture's settlement network (see ``build_network``)."""
    return build_network(world, settlements, cultures, people)[0]


def build_network(world, settlements, cultures, people: np.ndarray | None = None,
                  old_roads: RoadTable | None = None) -> tuple[RoadTable, TradeTable]:
    """Return the road segments of every culture's settlement network and the trade links routed over it.

    ``people`` (culture × cell, from the population density) adds each cell without a settlement that holds
    at least ROAD_RURAL_MIN_PEOPLE as a node weighted by its people, so roads reach rural land too.
    Gravity falls with distance as d^ROAD_DISTANCE_EXPONENT[transport]. ``old_roads`` (an earlier epoch's)
    are cheaper to follow (ROAD_OLD_DISCOUNT), so networks persist.
    """
    grid = world.grid
    radius_km = float(world.surface.attrs["radius_m"]) / 1e3
    spacing = grid.mean_spacing(radius_km)
    graphs: dict[int, TravelGraph] = {}
    rows = {k: [] for k in ("a", "b", "kind", "length_km", "culture", "flow")}
    trade = {k: [] for k in ("a", "b", "a_cell", "b_cell", "culture", "flow", "km", "hours")}
    offsets, path_cells = [0], []
    for c in cultures:
        mine = np.flatnonzero(settlements.culture == c.id)
        if mine.size < 2:
            continue
        cells, inv = np.unique(settlements.cell[mine], return_inverse=True)
        weight = np.bincount(inv, weights=settlements.population[mine])
        top = np.full(cells.size, -1, dtype=np.int64)
        order = np.argsort(settlements.population[mine])                   # largest written last
        top[inv[order]] = settlements.id[mine][order]
        if people is not None:
            rural = people[cultures.index(c)]
            extra = np.setdiff1d(np.flatnonzero(rural >= h.ROAD_RURAL_MIN_PEOPLE), cells)
            weight = np.r_[weight, rural[extra]]
            top = np.r_[top, np.full(extra.size, -1, dtype=np.int64)]
            cells = np.r_[cells, extra]
            keep = np.argsort(cells)
            cells, weight, top = cells[keep], weight[keep], top[keep]
        if cells.size < 2:
            continue
        transport = c.development.transport
        if transport not in graphs:
            graphs[transport] = TravelGraph(world, transport)
        graph = graphs[transport]
        exponent = h.ROAD_DISTANCE_EXPONENT[min(transport, len(h.ROAD_DISTANCE_EXPONENT) - 1)]
        pairs, flows = gravity_pairs(grid.points[cells], weight, radius_km, spacing, exponent)
        if pairs.size == 0:
            continue
        routes = np.empty((len(pairs), 2))
        carried = route_flows(graph, cells, pairs, flows, grid.points, radius_km,
                              road_edges(graph, old_roads), routes)
        routed = np.isfinite(routes[:, 0])
        trade["a"].append(top[pairs[routed, 0]])
        trade["b"].append(top[pairs[routed, 1]])
        trade["a_cell"].append(cells[pairs[routed, 0]])
        trade["b_cell"].append(cells[pairs[routed, 1]])
        trade["culture"].append(np.full(routed.sum(), c.id))
        trade["flow"].append(flows[routed])
        trade["km"].append(routes[routed, 0])
        trade["hours"].append(routes[routed, 1])
        stops = np.zeros(grid.size, dtype=bool)
        stops[cells] = True
        segs = _segments(graph, carried, stops)
        if not segs:
            continue
        settlement_at = np.full(grid.size, -1, dtype=np.int64)
        settlement_at[cells] = top
        flow = np.array([carried[ids].max() for _, ids in segs])
        length = np.array([graph.km[ids].sum() for _, ids in segs])
        sea = np.array([graph.water_edge[ids].any() for _, ids in segs])
        rows["kind"].append(classify(flow, length, sea, transport))
        rows["a"].append(np.array([settlement_at[p[0]] for p, _ in segs]))
        rows["b"].append(np.array([settlement_at[p[-1]] for p, _ in segs]))
        rows["length_km"].append(length)
        rows["flow"].append(flow)
        rows["culture"].append(np.full(len(segs), c.id))
        for p, _ in segs:
            path_cells.append(p)
            offsets.append(offsets[-1] + p.size)
    links = TradeTable(**{k: np.concatenate(v) for k, v in trade.items()}) if trade["a"] else TradeTable.empty()
    links = TradeTable(id=np.arange(len(links)), **{k: getattr(links, k) for k in trade}) if len(links) else links
    if not path_cells:
        return RoadTable.empty(), links
    data = {k: np.concatenate(v) for k, v in rows.items()}
    return RoadTable(id=np.arange(data["a"].size), path_offsets=np.array(offsets),
                     path_cells=np.concatenate(path_cells), **data), links
