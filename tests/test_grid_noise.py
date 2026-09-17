import numpy as np
import pytest

from worldgen.grid import RESOLUTIONS, build_grid, fibonacci_points, resolve_resolution
from worldgen.noise import fbm, perlin, ridged, warp


def test_fibonacci_points_are_unit_vectors():
    p = fibonacci_points(1000)
    assert p.shape == (1000, 3)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)
    assert abs(p.mean(axis=0)).max() < 0.01


def test_grid_topology():
    g = build_grid(RESOLUTIONS["preview"])
    degree = np.diff(g.neighbours.indptr)
    assert degree.min() >= 4 and degree.max() <= 8
    assert degree.mean() == pytest.approx(6.0, abs=0.01)
    assert (g.neighbours != g.neighbours.T).nnz == 0
    i, j = g.edges()
    assert (i < j).all() and len(i) == g.neighbours.nnz // 2
    # Euler characteristic of a sphere: V - E + F = 2.
    assert g.size - len(i) + len(g.triangles) == 2


def test_grid_spacing_matches_resolution():
    g = build_grid(RESOLUTIONS["preview"])
    expected = np.sqrt(4 * np.pi / g.size)
    assert g.mean_spacing() == pytest.approx(expected, rel=0.25)


def test_neighbour_lookup_and_lat_lon():
    g = build_grid(RESOLUTIONS["preview"])
    assert set(g.neighbour_indices(0)).issubset(range(g.size))
    assert g.lat.min() > -90 and g.lat.max() < 90
    _, idx = g.nearest(np.array([[0.0, 0.0, 1.0]]))
    assert g.lat[idx[0]] > 89


def test_resolution_names():
    assert resolve_resolution("standard") == 40_000
    assert resolve_resolution(1234) == 1234
    with pytest.raises(ValueError):
        resolve_resolution("ultra")


def test_noise_is_deterministic_and_seed_dependent():
    p = fibonacci_points(2000)
    a = fbm(p, 1, "x")
    assert np.array_equal(a, fbm(p, 1, "x"))
    assert not np.allclose(a, fbm(p, 2, "x"))
    assert not np.allclose(a, fbm(p, 1, "y"))
    assert -1.2 < a.min() < 0 < a.max() < 1.2


def test_noise_is_continuous():
    p = fibonacci_points(2000)
    q = p + 1e-4
    assert np.abs(perlin(p * 3, 0) - perlin(q * 3, 0)).max() < 0.01


def test_ridged_and_warp_ranges():
    p = fibonacci_points(2000)
    r = ridged(p, 0, "r")
    assert r.min() >= 0 and r.max() <= 1
    w = warp(p, 0, "w", 0.2)
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0)
    shift = np.arccos(np.clip((w * p).sum(axis=1), -1, 1))
    assert 0.0 < shift.mean() < 0.3
