"""Test categories: which tests cover which part of the package, and which parts consume which.

Each category names the source paths it covers (relative to the repository
root; a trailing slash means a folder) and the test modules that exercise
them. A test module may sit in several categories. ``CONSUMERS`` lists the
categories whose tests build directly on another category's output, so a
change there runs them too. ``scripts/select_tests.py`` uses these tables to
run only the tests relevant to a change; ``conftest.py`` marks every test
with its categories, so ``pytest -m zoom`` runs one category.
"""

from __future__ import annotations

CATEGORIES: dict[str, dict[str, tuple[str, ...]]] = {
    "physics": {
        "sources": ("worldgen/star.py", "worldgen/orbit.py", "worldgen/bulk.py", "worldgen/interior.py",
                    "worldgen/atmosphere.py", "worldgen/water.py", "worldgen/priors/", "worldgen/spec.py",
                    "worldgen/spec_base.py", "worldgen/evolve/", "worldgen/state.py", "worldgen/generate.py",
                    "worldgen/report.py", "worldgen/world.py", "worldgen/constants.py", "worldgen/units.py",
                    "worldgen/util.py", "worldgen/giants/", "worldgen/__init__.py", "pyproject.toml",
                    "scripts/archetype_fidelity.py", "scripts/validate_solar_system.py", "worldgen/data/giant_optics.npz",
                    "scripts/build_giant_optics.py"),
        "tests": ("test_star", "test_orbit", "test_bulk", "test_interior", "test_atmosphere", "test_water",
                  "test_archetype_fidelity", "test_pipeline", "test_system", "test_giant_appearance",
                  "test_giant_colours", "test_cloud_tops", "test_rings", "test_moon_systems", "test_moon_calibration", "test_moon_environment",
                  "test_habitable_moon", "test_captured_moons", "test_solar_system", "test_system_cli"),
    },
    "history": {
        "sources": ("worldgen/history/", "scripts/validate_history.py", "scripts/history_sensitivity.py"),
        "tests": ("test_history_climate", "test_history_life", "test_history_volatiles", "test_history_validation",
                  "test_timeline_output", "test_interior"),
    },
    "surface": {
        "sources": ("worldgen/grid/", "worldgen/noise.py", "worldgen/surface/", "worldgen/tectonics/",
                    "scripts/hypsometry_table.py"),
        "tests": ("test_grid_noise", "test_surface", "test_fabric", "test_relicts", "test_tectonics",
                  "test_tectonic_drive", "test_subsidence", "test_biogenic"),
    },
    "climate": {
        "sources": ("worldgen/climate/", "worldgen/biosphere/", "scripts/validate_earth.py"),
        "tests": ("test_climate", "test_biosphere"),
    },
    "hydrology": {
        "sources": ("worldgen/hydrology/",),
        "tests": ("test_hydrology",),
    },
    "zoom": {
        "sources": ("worldgen/zoom/",),
        "tests": ("test_zoom_assemble", "test_zoom_detail", "test_zoom_erode", "test_zoom_glacier",
                  "test_zoom_inherit", "test_zoom_render", "test_zoom_tiles", "test_culture_local"),
    },
    "culture": {
        "sources": ("worldgen/culture/", "scripts/validate_culture.py"),
        "tests": ("test_culture_density", "test_culture_habitats", "test_culture_local", "test_culture_model",
                  "test_culture_names", "test_culture_polities", "test_culture_render", "test_culture_resources",
                  "test_culture_roads", "test_culture_settlements", "test_validate_culture"),
    },
    "organic": {
        "sources": ("worldgen/culture/history/", "scripts/validate_organic.py"),
        "tests": ("test_culture_history", "test_history_colonists", "test_history_continuity",
                  "test_history_development", "test_history_events", "test_history_explorer", "test_history_livestock",
                  "test_history_outputs", "test_history_polities", "test_history_population",
                  "test_history_suitability", "test_history_wars", "test_validate_organic"),
    },
    "render": {
        "sources": ("worldgen/render/", "scripts/render_examples.py", "scripts/planet_gallery.py"),
        "tests": ("test_world_render", "test_zoom_render", "test_culture_render", "test_globe_viewer",
                  "test_history_explorer", "test_history_outputs", "test_timeline_output", "test_cloud_tops", "test_rings", "test_system_view", "test_system_cli"),
    },
    "export": {
        "sources": ("worldgen/export/", "scripts/validate_export.py"),
        "tests": ("test_export", "test_export_meshes", "test_export_pyramid", "test_export_regions",
                  "test_cli_export", "test_globe_viewer", "test_cloud_tops"),
    },
    "cli": {
        "sources": ("worldgen/cli.py",),
        "tests": ("test_pipeline", "test_cli_export", "test_culture_local", "test_culture_render",
                  "test_history_outputs", "test_timeline_output", "test_world_render", "test_zoom_assemble", "test_system_cli"),
    },
}

# Categories whose tests consume another category's output directly.
CONSUMERS: dict[str, tuple[str, ...]] = {
    "physics": ("history",),
    "surface": ("hydrology", "zoom"),
    "climate": ("hydrology",),
    "hydrology": ("zoom",),
    "zoom": ("export",),
    "culture": ("organic",),
    "render": ("export",),
}

# Tests that run with every selection: an end-to-end world through the pipeline, and the check of these tables.
SMOKE = ("test_pipeline", "test_categories")

# Paths that never need tests.
IGNORED_PREFIXES = ("notebooks/", "papers/", "examples/", "kestrel/", "Claude outputs/", ".gitignore", "LICENSE")
IGNORED_SUFFIXES = (".md",)

# Files of the selection itself: a change runs the check of the tables.
SELECTION_FILES = ("tests/categories.py", "scripts/select_tests.py")

# Constants file: a change selects the categories of the modules that use the changed names.
HEURISTICS = "worldgen/heuristics.py"


def categories_of_test(module: str) -> list[str]:
    """Return the categories a test module belongs to."""
    return [name for name, cat in CATEGORIES.items() if module in cat["tests"]]


def categories_of_source(path: str) -> list[str]:
    """Return the categories whose sources include ``path``, the most specific prefix first."""
    matches = []
    for name, cat in CATEGORIES.items():
        for src in cat["sources"]:
            if path == src or (src.endswith("/") and path.startswith(src)):
                matches.append((len(src), name))
    if not matches:
        return []
    longest = max(m[0] for m in matches)
    # A file in culture/history/ belongs to organic only, not to culture as well.
    return [name for length, name in sorted(matches, reverse=True) if length == longest]
