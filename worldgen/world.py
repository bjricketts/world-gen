"""A generated world: its spec, global state and surface, with saving and loading."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import xarray as xr
import yaml

from .grid import SphereGrid, build_grid
from .report import state_to_plain
from .spec import PlanetSpec, load_spec, save_spec
from .state import PlanetState

FORMAT_VERSION = 3


@dataclass
class World:
    """A planet with its global surface fields (``surface`` is None for planets without a surface)."""

    spec: PlanetSpec
    state: PlanetState
    surface: Optional[xr.Dataset]

    @property
    def grid(self) -> Optional[SphereGrid]:
        """Return the spherical grid the surface lives on."""
        if self.surface is None:
            return None
        return build_grid(int(self.surface.attrs["grid_size"]))


def save_world(world: World, path: str | Path) -> Path:
    """Write a world to a folder containing spec.yaml, state.yaml and surface.zarr; returns the folder."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    save_spec(world.spec, folder / "spec.yaml")
    with open(folder / "state.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"format_version": FORMAT_VERSION, **state_to_plain(world.state)}, f,
                       sort_keys=False, allow_unicode=True)
    if world.surface is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            world.surface.to_zarr(folder / "surface.zarr", mode="w", consolidated=False)
    return folder


def load_world(path: str | Path) -> World:
    """Read a world saved with ``save_world``."""
    folder = Path(path)
    spec = load_spec(folder / "spec.yaml")
    with open(folder / "state.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    version = data.pop("format_version", None)
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported world format version {version} (expected {FORMAT_VERSION})")
    state = PlanetState.from_dict(data)
    surface = None
    if (folder / "surface.zarr").exists():
        surface = xr.open_zarr(folder / "surface.zarr", consolidated=False).load()
    return World(spec=spec, state=state, surface=surface)
