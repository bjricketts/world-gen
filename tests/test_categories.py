"""The test categories cover every test and every source file, and the selection follows them."""

import importlib.util
from pathlib import Path

from tests.categories import CATEGORIES, HEURISTICS, SMOKE, categories_of_source, categories_of_test

ROOT = Path(__file__).resolve().parent.parent


def _selector():
    """Return the selection script as a module."""
    spec = importlib.util.spec_from_file_location("select_tests", ROOT / "scripts" / "select_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_test_module_has_a_category():
    """A new test file must be added to a category (or the smoke set)."""
    for path in (ROOT / "tests").glob("test_*.py"):
        assert categories_of_test(path.stem) or path.stem in SMOKE, f"{path.name} has no category"


def test_every_category_test_exists():
    """Category tables name only test files that exist."""
    for name, cat in CATEGORIES.items():
        for module in cat["tests"]:
            assert (ROOT / "tests" / f"{module}.py").exists(), f"{name} lists missing {module}"


def test_every_source_file_has_a_category():
    """Every package module and script maps to a category (heuristics.py is handled by constant)."""
    paths = [p for p in (ROOT / "worldgen").rglob("*.py") if "__pycache__" not in p.parts]
    paths += [p for p in (ROOT / "scripts").glob("*.py") if p.name != "select_tests.py"]
    missing = [p.relative_to(ROOT).as_posix() for p in paths
               if p.relative_to(ROOT).as_posix() != HEURISTICS and not categories_of_source(p.relative_to(ROOT).as_posix())]
    assert not missing, f"no category for {missing}"


def test_selection_follows_specific_folders_and_consumers():
    """culture/history/ selects organic only; surface changes also run hydrology, zoom and export."""
    sel = _selector()
    cats, tests, uncovered = sel.select(["worldgen/culture/history/wars.py"], "HEAD")
    assert cats == {"organic"} and not uncovered
    cats, _, _ = sel.select(["worldgen/surface/plates.py"], "HEAD")
    assert cats == {"surface", "hydrology", "zoom", "export"}
    cats, tests, _ = sel.select(["DESIGN.md", "notebooks/01_quickstart.ipynb", "tests/test_orbit.py"], "HEAD")
    assert cats == set() and tests == {"test_orbit"}
    _, _, uncovered = sel.select(["worldgen/new_module_nobody_knows.py"], "HEAD")
    assert uncovered


def test_heuristics_constants_map_to_their_users():
    """A changed constant selects the categories of the modules that use it."""
    sel = _selector()
    users = sel.users_of({"ROCHE_GAMMA_RUBBLE"})
    assert "worldgen/giants/limits.py" in users
    assert all(categories_of_source(u) for u in users)
