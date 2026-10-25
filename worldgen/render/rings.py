"""Ring meshes for 3D views: an annulus per ring in the equatorial plane, coloured by the ring and as opaque as
its optical depth makes it (1 − exp(−τ) seen face-on), with denser arcs where a ring has them."""

from __future__ import annotations

import numpy as np

from .. import heuristics as h


def _rgb(hex_colour: str) -> np.ndarray:
    """Return sRGB components (0–1) of a hex colour."""
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def ring_mesh_arrays(rings, segments: int = h.RING_MESH_SEGMENTS,
                     radial_steps: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices (m, planet-centred, z along the spin axis), triangle faces and RGBA (uint8) of the rings.

    Each ring is ``radial_steps`` bands of quads across, so per-vertex shading
    (a shadow's edge) can vary across it.
    """
    angles = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    lon = np.degrees(angles)
    vertices, faces, colours = [], [], []
    offset = 0
    k = np.arange(segments)
    nxt = (k + 1) % segments
    for ring in rings:
        alpha = np.full(segments, 1.0 - np.exp(-ring.optical_depth))
        for start, end in ring.arcs:
            inside = ((lon - start) % 360.0) <= ((end - start) % 360.0)
            alpha[inside] = 1.0 - np.exp(-h.RING_ARC_TAU)
        rgb = _rgb(ring.colour)
        for radius in np.linspace(ring.inner_radius_m, ring.outer_radius_m, radial_steps + 1):
            vertices.append(np.column_stack([radius * cos, radius * sin, np.zeros(segments)]))
            colours.append(np.column_stack([np.tile(rgb, (segments, 1)), alpha]))
        for j in range(radial_steps):
            inner, outer = offset + j * segments, offset + (j + 1) * segments
            faces.append(np.column_stack([inner + k, outer + k, outer + nxt]))
            faces.append(np.column_stack([inner + k, outer + nxt, inner + nxt]))
        offset += (radial_steps + 1) * segments
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4), dtype=np.uint8)
    rgba = np.round(np.clip(np.vstack(colours), 0, 1) * 255).astype(np.uint8)
    return np.vstack(vertices), np.vstack(faces).astype(np.int64), rgba


def ring_polydata(rings, segments: int = h.RING_MESH_SEGMENTS):
    """Return the rings as one PyVista PolyData with an 'rgba' array, or None without rings."""
    from .globe import _pyvista

    vertices, faces, rgba = ring_mesh_arrays(rings, segments)
    if len(vertices) == 0:
        return None
    pv = _pyvista()
    poly = pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    poly["rgba"] = rgba
    return poly
