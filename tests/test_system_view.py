"""The system diagram and the 3D system scene."""

import math

import numpy as np
import pytest

from tests.test_system import jupiter_spec
from worldgen import generate, generate_world
from worldgen import heuristics as h
from worldgen.render.system import _project, plot_system, radial_map
from worldgen.render.system_geometry import Compression, MoonOrbit, eccentric_anomaly, moon_orbits
from worldgen.spec import PlanetSpec

SATURN = {"name": "Saturn", "seed": 1, "star": {"mass_msun": 1.0, "age_gyr": 4.6},
          "orbit": {"semi_major_axis_au": 9.54, "obliquity_deg": 26.7, "rotation_period_h": 10.66},
          "body": {"mass_mearth": 95.2, "radius_rearth": 9.45, "envelope_mass_fraction": 0.85}}


@pytest.fixture(scope="module")
def jupiter_state():
    return generate(jupiter_spec())[0]


@pytest.fixture(scope="module")
def saturn_state():
    state = generate(PlanetSpec.model_validate(SATURN))[0]
    assert state.system.rings
    return state


def test_kepler_positions():
    orbit = MoonOrbit("x", a_m=1e8, e=0.3, inclination_rad=0.4, node_rad=1.0, periapsis_rad=2.0,
                      mean_anomaly_rad=0.0, period_s=1000.0, radius_m=1e5)
    assert np.linalg.norm(orbit.position(0.0)) == pytest.approx(1e8 * 0.7)          # periapsis at M = 0
    assert np.linalg.norm(orbit.position(500.0)) == pytest.approx(1e8 * 1.3)        # apoapsis half a period on
    assert np.allclose(orbit.position(500.0) / 1.3, -orbit.position(0.0) / 0.7)
    assert np.allclose(orbit.position(1000.0), orbit.position(0.0))
    for m in (0.3, 2.0, 5.5):
        for e in (0.0, 0.5, 0.95):
            ecc = eccentric_anomaly(m, e)
            assert ecc - e * math.sin(ecc) == pytest.approx(math.remainder(m, 2 * math.pi), abs=1e-10)


def test_retrograde_moons_run_backwards_and_orbits_are_seeded(jupiter_state):
    a = moon_orbits(jupiter_state)
    assert [o.name for o in a] == ["Io", "Europa", "Ganymede", "Callisto"]
    assert a == moon_orbits(jupiter_state)
    retro = MoonOrbit("r", 1e8, 0.0, math.radians(157), 0.0, 0.0, 0.0, 1000.0, 1e5)
    p0, p1 = retro.position(0.0), retro.position(10.0)
    assert np.cross(p0, p1)[2] < 0


def test_compression_is_continuous_and_monotonic():
    comp = Compression(7e7, outer_ring_m=1.4e8)
    assert comp.r0 == pytest.approx(max(h.SCENE_COMPRESS_FROM_RP * 7e7, 1.05 * 1.4e8))
    r = np.linspace(7e7, 1e10, 2000)
    d = comp.distance(r)
    assert np.all(np.diff(d) > 0)
    assert comp.distance(comp.r0 * (1 - 1e-9)) == pytest.approx(comp.distance(comp.r0 * (1 + 1e-9)))
    assert np.allclose(comp.distance(r[r <= comp.r0]), r[r <= comp.r0])
    assert Compression(7e7, mode="true").distance(5e9) == 5e9
    assert comp.moon_radius(1.8e6) > 1.8e6
    with pytest.raises(ValueError):
        Compression(7e7, mode="log")


def test_diagram_radii():
    assert radial_map(np.array([0.5, 1.0, 10.0, 100.0])).tolist() == pytest.approx([0.5, 1.0, 2.0, 3.0])
    assert radial_map(np.array([10.0]), "linear")[0] == 10.0
    p = _project(np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 100.0]]), 1.0, "top", 0.3, "log")
    assert np.hypot(*p[0]) == pytest.approx(2.0)
    edge = _project(np.array([[0.0, 0.0, 100.0]]), 1.0, "edge", 0.3, "log")[0]
    assert np.hypot(*edge) == pytest.approx(3.0)                  # a 3D distance maps the same in either view
    assert edge[0] == pytest.approx(-3.0 * math.sin(0.3))         # the spin axis leans by the obliquity


def test_diagram_files(jupiter_state, saturn_state, tmp_path):
    for state, view in ((jupiter_state, "top"), (saturn_state, "edge")):
        path = tmp_path / f"{state.name}_{view}.png"
        fig = plot_system(state, view=view, highlight="Io", path=path)
        assert path.stat().st_size > 10_000
        assert not fig.axes[0].get_title()
    plot_system(jupiter_state, radial="linear", table=False, path=tmp_path / "linear.svg")
    assert (tmp_path / "linear.svg").exists()
    with pytest.raises(ValueError):
        plot_system(jupiter_state, view="side")


# --- the 3D scene ---------------------------------------------------------------

def test_planet_shadow_length_on_rings():
    """The planet's shadow on its ring plane reaches R / sin β along the anti-star direction."""
    from worldgen.render.system_scene import sphere_lit

    radius, beta = 6e7, math.radians(20.0)
    light = np.array([math.cos(beta), 0.0, math.sin(beta)])
    d = np.linspace(1.01, 4.0, 400) * radius
    lit = sphere_lit(np.column_stack([-d, np.zeros_like(d), np.zeros_like(d)]), light, (0, 0, 0), radius)
    edge = d[np.argmax(lit)]
    assert edge == pytest.approx(radius / math.sin(beta), rel=0.01)
    assert sphere_lit(np.array([[2 * radius, 0.0, 0.0]]), light, (0, 0, 0), radius)[0]    # day side is lit


def test_ring_transmission():
    from types import SimpleNamespace

    from worldgen.render.system_scene import ring_transmission

    ring = SimpleNamespace(inner_radius_m=1e8, outer_radius_m=1.5e8, optical_depth=0.8)
    light = np.array([0.0, math.cos(1.0), math.sin(1.0)])
    below = np.array([[0.0, 1.2e8 - 1e7 / math.tan(1.0), -1e7], [0.0, 3e8, -1e7]])
    tr = ring_transmission(below, light, [ring])
    assert tr[0] == pytest.approx(math.exp(-0.8 / math.sin(1.0)))
    assert tr[1] == 1.0


@pytest.fixture(scope="module")
def jupiter_scene():
    pytest.importorskip("pyvista")
    from worldgen.render.system_scene import system_scene

    return system_scene(generate_world(jupiter_spec(), "preview"), level=0)


def test_scene_shading(jupiter_scene):
    sc = jupiter_scene
    sc.update(0.0)
    shaded = np.asarray(sc.planet["shaded"])
    assert shaded.min() >= 0 and shaded.max() <= 1
    night = (sc._planet_points / np.linalg.norm(sc._planet_points, axis=1)[:, None]) @ sc.light < -0.1
    assert np.allclose(shaded[night], sc._planet_rgb[night] * h.SCENE_AMBIENT)
    io = sc.moons[0]
    centre = np.asarray(io["mesh"].points).mean(axis=0)
    true = io["orbit"].position(0.0)
    assert np.linalg.norm(centre) == pytest.approx(float(sc.compression.distance(np.linalg.norm(true))), rel=0.02)


def test_scene_files(jupiter_scene, tmp_path):
    jupiter_scene.screenshot(tmp_path / "scene.png", window_size=(400, 300))
    assert (tmp_path / "scene.png").stat().st_size > 2_000
    jupiter_scene.save_movie(tmp_path / "scene.gif", duration_s=86400.0, frames=4, window_size=(240, 180))
    assert (tmp_path / "scene.gif").stat().st_size > 2_000
    pytest.importorskip("trame_vtk")
    try:
        jupiter_scene.save_html(tmp_path / "scene.html")
    except ImportError as err:
        pytest.skip(f"no HTML exporter: {err}")
    assert (tmp_path / "scene.html").stat().st_size > 10_000


def test_saturn_scene_rings_are_shadowed(saturn_state):
    pytest.importorskip("pyvista")
    from worldgen.render.system_scene import system_scene

    world = generate_world(PlanetSpec.model_validate(SATURN), "preview")
    sc = system_scene(world, level=0, season=math.radians(60))
    rings = np.asarray(sc.ring_mesh["shaded"])
    dark = rings.max(axis=1) <= sc._ring_rgb.max(axis=1) * h.SCENE_AMBIENT + 1e-9
    assert 0 < dark.mean() < 0.3                            # the planet's shadow falls across part of the rings
    behind = (sc._ring_points @ sc.light) < 0
    assert dark[~behind].sum() <= dark.sum() * 0.05         # and only on the side away from the star
