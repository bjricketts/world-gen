"""Shared fixtures: Solar System reference bodies, the Earth spec and a cache of generated worlds.

Tests that generate worlds are marked ``slow`` (directly, or through the
fixtures in ``SLOW_FIXTURES``); ``pytest -m "not slow"`` skips them.
"""

from pathlib import Path

import pytest

from worldgen import bulk, generate_world, star
from worldgen.spec import load_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Fixtures that generate worlds or run the tectonic simulation.
SLOW_FIXTURES = {"world_cache", "living_world", "aquaplanet", "wet_world", "temperate", "regime_worlds",
                 "result", "simulated_world", "small_world"}


def pytest_collection_modifyitems(items):
    """Mark tests that use a slow fixture as slow."""
    for item in items:
        if SLOW_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def world_cache():
    """Return a function that generates a world once per spec and shares it; tests must not modify it."""
    worlds = {}

    def get(spec, resolution="preview", **options):
        key = (spec.model_dump_json(), str(resolution), tuple(sorted(options.items())))
        if key not in worlds:
            worlds[key] = generate_world(spec, resolution, **options)
        return worlds[key]

    return get


@pytest.fixture
def sun():
    """The present-day Sun."""
    state, _ = star.build_star(1.0, 4.57, 0.0)
    return state


def rocky_body(mass_mearth, radius_rearth, cmf, wmf=0.0):
    """Return a rocky body with a fixed mass and radius."""
    state, _, _ = bulk.build_bulk(mass_mearth, radius_rearth, cmf, wmf, 0.0, wmf_is_user=True)
    return state


@pytest.fixture
def earth_body():
    return rocky_body(1.0, 1.0, 0.325, 2.3e-4)


@pytest.fixture
def earth_spec():
    return load_spec(EXAMPLES / "earth.yaml")
