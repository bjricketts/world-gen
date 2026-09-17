"""Figures and animations of a world's simulated tectonic history."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ..world import World
from .maps import elevation_image, projection


def has_history(world: World) -> bool:
    """Return whether the world carries tectonic snapshots."""
    return world.surface is not None and "snapshot_elevation" in world.surface


def snapshot_times(world: World) -> np.ndarray:
    """Return the snapshot times (Myr relative to the present surface, negative = past)."""
    if not has_history(world):
        raise ValueError(f"{world.state.name} has no tectonic snapshots; generate it with a snapshot interval")
    return world.surface["time"].values


def snapshot_world(world: World, index: int) -> World:
    """Return a copy of the world whose elevation, ocean and plates are those of one snapshot."""
    snapshot_times(world)
    ds = world.surface
    elevation = ds["snapshot_elevation"].values[index]
    surface = ds[["elevation", "ocean", "plate"]].copy()
    surface["elevation"] = ("cell", elevation)
    surface["ocean"] = ("cell", (elevation < 0.0) & bool(ds.attrs["has_ocean"]))
    surface["plate"] = ("cell", ds["snapshot_plate"].values[index])
    surface.attrs = dict(ds.attrs)
    return World(spec=world.spec, state=world.state, surface=surface)


def _limits(world: World) -> tuple[float, float]:
    """Return the elevation range over all snapshots, so frames share one colour scale."""
    values = world.surface["snapshot_elevation"].values
    return float(values.min()), float(values.max())


def _label(time_myr: float) -> str:
    return "present" if abs(time_myr) < 1e-6 else f"{-time_myr:.0f} Myr ago"


def plot_history(world: World, panels: int = 6, projection_name: str = "mollweide",
                 central_longitude: float = 0.0):
    """Return a figure with elevation maps at evenly spaced snapshots, oldest first."""
    times = snapshot_times(world)
    picks = np.unique(np.linspace(0, len(times) - 1, min(panels, len(times))).round().astype(int))
    cols = min(3, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    proj = projection(projection_name, central_longitude)
    fig = plt.figure(figsize=(5.5 * cols, 3.2 * rows))
    limits = _limits(world)
    for n, k in enumerate(picks):
        ax = fig.add_subplot(rows, cols, n + 1, projection=proj)
        ax.set_global()
        rgba, _, _, extent = elevation_image(snapshot_world(world, k), proj, 600, hillshade=False, limits=limits)
        ax.imshow(rgba, transform=proj, extent=extent, origin="upper", interpolation="bilinear")
        ax.set_title(_label(times[k]), fontsize=10)
    fig.suptitle(f"{world.state.name}: tectonic history", fontsize=13)
    fig.tight_layout()
    return fig


def animate_history(world: World, path: str | Path, fps: int = 4, width: int = 800,
                    projection_name: str = "mollweide", central_longitude: float = 0.0,
                    dpi: Optional[int] = 100) -> Path:
    """Write an animated GIF of the elevation through the simulated history and return its path."""
    times = snapshot_times(world)
    proj = projection(projection_name, central_longitude)
    limits = _limits(world)
    frames = [elevation_image(snapshot_world(world, k), proj, width, hillshade=False, limits=limits)
              for k in range(len(times))]

    fig = plt.figure(figsize=(8, 4.6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    rgba, _, _, extent = frames[0]
    image = ax.imshow(rgba, transform=proj, extent=extent, origin="upper", interpolation="bilinear")
    title = ax.set_title("")

    def draw(k: int):
        """Show frame ``k``."""
        image.set_data(frames[k][0])
        title.set_text(f"{world.state.name}: {_label(times[k])}")
        return image, title

    animation = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    path = Path(path)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
