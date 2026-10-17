"""worldgen: physically motivated planet generation for world building."""

__version__ = "0.1.0"

from .generate import candidates, generate, generate_world  # noqa: E402
from .report import format_report, format_timeline, save_state  # noqa: E402
from .spec import PlanetSpec, load_spec, save_spec  # noqa: E402
from .world import World, load_world, save_world  # noqa: E402

__all__ = [
    "PlanetSpec", "load_spec", "save_spec", "generate", "generate_world", "candidates",
    "format_report", "format_timeline", "save_state", "World", "save_world", "load_world",
]
