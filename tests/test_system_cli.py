"""The system and moon commands and the giant and moon report (task 11.10)."""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from tests.test_system import jupiter_spec  # noqa: E402
from worldgen import generate, generate_world, save_world  # noqa: E402
from worldgen.cli import app  # noqa: E402
from worldgen.giants import kept_moons, moon_index, parent_folder, rebuilt_parent_spec  # noqa: E402
from worldgen.report import format_report  # noqa: E402

runner = CliRunner()


@pytest.fixture(scope="module")
def jupiter_dir(tmp_path_factory):
    return save_world(generate_world(jupiter_spec(), "preview"), tmp_path_factory.mktemp("sys") / "jupiter")


def _ok(args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def test_giant_report_sections():
    state = generate(jupiter_spec())[0]
    text = format_report(state, kept_moons=["Europa"])
    interior = text[text.index("Interior"):text.index("Appearance")]
    assert "internal heat" in interior and "W/m²" in interior and "mW/m²" not in interior
    atmosphere = text[text.index("\nAtmosphere\n"):text.index("Climate")]
    assert "boiling" not in atmosphere and "surface water" not in atmosphere
    assert "Europa" in text and "kept as a world" in text and "1 of 4 kept" in text
    assert "habitable unaided" not in text
    assert "moon worlds" not in format_report(state)            # without a folder nothing is said


def test_moon_report_tidal_response_and_source(jupiter_dir):
    from worldgen import load_world
    from worldgen.giants import moon_world

    io = moon_world(load_world(jupiter_dir), 0, "preview")
    text = format_report(io.state)
    assert "tidal response" in text and "partially molten" in text
    line = next(row for row in text.splitlines() if row.strip().startswith("tidal heating"))
    assert line.endswith("[heuristic]")


def test_moon_lookup(jupiter_dir):
    from worldgen import load_world

    world = load_world(jupiter_dir)
    assert moon_index(world, "europa") == 1 and moon_index(world, 4) == 3 and moon_index(world, "1") == 0
    with pytest.raises(KeyError, match="Io, Europa"):
        moon_index(world, "Titan")
    with pytest.raises(IndexError):
        moon_index(world, 5)


def test_moon_command_keeps_the_moon(jupiter_dir, tmp_path):
    listing = _ok(["moon", jupiter_dir])
    assert "Callisto" in listing and "kept" not in next(r for r in listing.splitlines() if "Europa" in r)
    out = _ok(["moon", jupiter_dir, "europa", "-r", "preview", "-m", tmp_path / "europa.png", "-f", "terrain"])
    assert "=== Europa ===" in out and "subsurface ocean" in out
    assert (tmp_path / "europa.png").exists()
    from worldgen import load_world

    kept = kept_moons(jupiter_dir, load_world(jupiter_dir))
    assert "Europa" in kept and parent_folder(kept["Europa"]) == jupiter_dir.resolve()
    assert next(r for r in _ok(["moon", jupiter_dir]).splitlines() if "Europa" in r).endswith("kept")
    assert "of 4 kept" in _ok(["report", jupiter_dir])
    result = runner.invoke(app, ["moon", str(jupiter_dir), "Titan"])
    assert result.exit_code == 1 and "no moon 'Titan'" in result.output


def test_system_command_files(jupiter_dir, tmp_path):
    pytest.importorskip("pyvista")
    _ok(["system", jupiter_dir, tmp_path / "top.png"])
    _ok(["system", jupiter_dir, tmp_path / "edge.svg", "--view", "edge", "--linear", "--highlight", "Io"])
    _ok(["system", jupiter_dir, tmp_path / "scene.png", "--3d", "--level", "0", "--time", "1.5"])
    _ok(["system", jupiter_dir, tmp_path / "scene.gif", "--3d", "--level", "0", "--frames", "3", "--duration", "1"])
    for name in ("top.png", "edge.svg", "scene.png", "scene.gif"):
        assert (tmp_path / name).stat().st_size > 2_000
    for args in (["x.bmp"], ["x.txt", "--3d"], ["x.png", "--view", "side"]):
        result = runner.invoke(app, ["system", str(jupiter_dir), str(tmp_path / args[0]), *args[1:]])
        assert result.exit_code == 1


def test_system_of_a_kept_moon_draws_its_parent(jupiter_dir, tmp_path):
    from worldgen import load_world
    from worldgen.giants import moon_folder, moon_world

    world = load_world(jupiter_dir)
    moon_world(world, 3, "preview", folder=jupiter_dir)
    out = _ok(["system", moon_folder(jupiter_dir, world, 3), tmp_path / "callisto_system.png"])
    assert "rebuilt" not in out and (tmp_path / "callisto_system.png").exists()


def test_standalone_moon_rebuilds_its_parent(tmp_path):
    from worldgen.giants import moon_spec

    world = generate_world(jupiter_spec(), "preview")
    moon = generate_world(moon_spec(world, 2), "preview")             # Ganymede, saved on its own
    folder = save_world(moon, tmp_path / "ganymede")
    spec = rebuilt_parent_spec(moon.state, moon.spec)
    parent = generate(spec)[0]
    assert [m.name for m in parent.system.moons] == ["Ganymede"] and not parent.system.rings
    assert parent.system.moons[0].semi_major_axis_m == pytest.approx(moon.state.parent.semi_major_axis_m, rel=1e-6)
    assert parent.bulk.mass_kg == pytest.approx(moon.state.parent.mass_kg, rel=1e-6)
    out = _ok(["system", folder, tmp_path / "ganymede_system.png"])
    assert "rebuilt" in out and (tmp_path / "ganymede_system.png").exists()
