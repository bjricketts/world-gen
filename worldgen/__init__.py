"""worldgen: physically motivated planet generation for world building."""

from .generate import candidates, generate, generate_world
from .report import format_report, format_timeline, save_state
from .spec import PlanetSpec, load_spec, save_spec
from .world import World, load_world, save_world

__version__ = "0.1.0"

__all__ = [
    "PlanetSpec", "load_spec", "save_spec", "generate", "generate_world", "candidates",
    "format_report", "format_timeline", "save_state", "World", "save_world", "load_world",
]
