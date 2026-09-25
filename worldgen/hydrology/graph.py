"""Compiled graph kernels for water on the sphere grid: filling, flow directions, ordering and accumulation.

The grid neighbour graph is passed as CSR arrays (``indptr``, ``indices``,
``lengths`` in radians).
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit


@njit(cache=True)
def pass_levels(z, indptr, indices, seed):
    """Return, for each cell, the lowest water level at which it connects to ``seed`` through flooded cells."""
    return pass_levels_from(z, indptr, indices, np.array([seed], dtype=np.int64))


@njit(cache=True)
def pass_levels_from(z, indptr, indices, seeds):
    """Return, for each cell, the lowest water level at which it connects to any of ``seeds``."""
    n = z.size
    level = np.full(n, np.inf)
    done = np.zeros(n, dtype=np.bool_)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for s in seeds:
        level[s] = z[s]
        heapq.heappush(heap, (z[s], np.int64(s)))
    while heap:
        lv, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if not done[j]:
                cand = max(z[j], lv)
                if cand < level[j]:
                    level[j] = cand
                    heapq.heappush(heap, (cand, np.int64(j)))
    return level


@njit(cache=True)
def priority_flood(z, indptr, indices, outlet, epsilon):
    """Return elevations with every closed depression filled to its spill level.

    ``outlet`` marks cells where water leaves the land (the ocean). Filled
    flats rise by ``epsilon`` per cell toward their outlet so that water on
    them keeps moving.
    """
    n = z.size
    filled = z.copy()
    done = np.zeros(n, dtype=np.bool_)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for i in range(n):
        if outlet[i]:
            done[i] = True
            heapq.heappush(heap, (filled[i], np.int64(i)))
    while heap:
        lv, i = heapq.heappop(heap)
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if not done[j]:
                done[j] = True
                if filled[j] <= lv + epsilon:
                    filled[j] = lv + epsilon
                heapq.heappush(heap, (filled[j], np.int64(j)))
    return filled


@njit(cache=True)
def priority_breach(z, indptr, indices, outlet, max_depth, epsilon):
    """Return elevations with shallow depressions drained by cutting through their sills.

    Cells are reached from the outlets in order of elevation, as in
    priority-flood. A cell found below the level it is reached at lies in a
    depression; the path it was reached along is lowered so the depression
    drains, unless some cell on that path would have to drop by more than
    ``max_depth``, in which case the depression is left to fill as a lake
    (hybrid breaching and filling, after Lindsay 2016).
    """
    n = z.size
    out = z.copy()
    level = z.copy()
    parent = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=np.bool_)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for i in range(n):
        if outlet[i]:
            done[i] = True
            heapq.heappush(heap, (z[i], np.int64(i)))
    while heap:
        _, i = heapq.heappop(heap)
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if done[j]:
                continue
            done[j] = True
            parent[j] = i
            if z[j] >= level[i]:
                level[j] = z[j]
                heapq.heappush(heap, (z[j], np.int64(j)))
                continue
            # How deep a cut does draining j need along the path back to the outlet?
            need = 0.0
            target = z[j] - epsilon
            c = i
            while c >= 0 and not outlet[c] and out[c] > target:
                need = max(need, z[c] - target)
                c = parent[c]
                target -= epsilon
            if need <= max_depth:
                target = z[j] - epsilon
                c = i
                while c >= 0 and not outlet[c] and out[c] > target:
                    out[c] = target
                    level[c] = target
                    c = parent[c]
                    target -= epsilon
                level[j] = z[j]
                heapq.heappush(heap, (z[j], np.int64(j)))
            else:
                level[j] = level[i]
                heapq.heappush(heap, (level[i], np.int64(j)))
    return out


@njit(cache=True)
def steepest_receivers(z, indptr, indices, lengths, outlet):
    """Return each cell's downhill neighbour (itself for outlets and pits) and the distance to it (radians)."""
    n = z.size
    receiver = np.arange(n)
    distance = np.ones(n)
    for i in range(n):
        if outlet[i]:
            continue
        best = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            slope = (z[i] - z[j]) / lengths[k]
            if slope > best:
                best = slope
                receiver[i] = j
                distance[i] = lengths[k]
    return receiver, distance


@njit(cache=True)
def upstream_order(receiver):
    """Return the cells ordered so that each comes after its receiver (outlets first)."""
    n = receiver.size
    count = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        if receiver[i] != i:
            count[receiver[i] + 1] += 1
    start = np.cumsum(count)
    donors = np.empty(start[-1], dtype=np.int64)
    fill = start[:-1].copy()
    for i in range(n):
        r = receiver[i]
        if r != i:
            donors[fill[r]] = i
            fill[r] += 1
    order = np.empty(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    m = 0
    for base in range(n):
        if receiver[base] != base:
            continue
        top = 1
        stack[0] = base
        while top > 0:
            top -= 1
            i = stack[top]
            order[m] = i
            m += 1
            for k in range(start[i], start[i + 1]):
                stack[top] = donors[k]
                top += 1
    if m != n:
        raise ValueError("flow directions contain a loop")
    return order


@njit(cache=True)
def accumulate(order, receiver, values):
    """Return each cell's value plus everything upstream of it."""
    total = values.copy()
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        r = receiver[i]
        if r != i:
            total[r] += total[i]
    return total


@njit(cache=True)
def spread_positive(descending, indptr, indices, lengths, z, values, exponent, outlet):
    """Return a flux carried downhill to every lower neighbour, never below zero.

    ``descending`` lists the nodes from highest to lowest ``z``. Each node
    passes its flux to its lower neighbours in proportion to (drop/length)^exponent,
    so flow spreads over broad slopes instead of gathering into single lines
    (multiple flow directions; Quinn et al. 1991). Nodes in ``outlet`` keep
    what reaches them.
    """
    total = values.copy()
    for k in range(descending.size):
        i = descending[k]
        if total[i] < 0.0:
            total[i] = 0.0
        if outlet[i] or total[i] == 0.0:
            continue
        weight = 0.0
        for m in range(indptr[i], indptr[i + 1]):
            drop = z[i] - z[indices[m]]
            if drop > 0.0:
                weight += (drop / lengths[m]) ** exponent
        if weight == 0.0:
            continue
        for m in range(indptr[i], indptr[i + 1]):
            drop = z[i] - z[indices[m]]
            if drop > 0.0:
                total[indices[m]] += total[i] * (drop / lengths[m]) ** exponent / weight
    return total


@njit(cache=True)
def basin_roots(order, receiver):
    """Return, for each cell, the outlet or pit its water ends at."""
    root = np.arange(receiver.size)
    for k in range(order.size):
        i = order[k]
        if receiver[i] != i:
            root[i] = root[receiver[i]]
    return root
