"""Spherical grids for global surface fields."""

from .sphere import RESOLUTIONS, SphereGrid, build_grid, fibonacci_points, resolve_resolution

__all__ = ["RESOLUTIONS", "SphereGrid", "build_grid", "fibonacci_points", "resolve_resolution"]
