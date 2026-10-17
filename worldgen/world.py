"""A generated world: its spec, global state, surface and history, with saving and loading.

A saved world is a folder: ``spec.yaml``, ``state.yaml``, ``surface.zarr``
and, in history mode, ``timeline.yaml`` (events and sampled series) with one
file per epoch under ``epochs/``. Zoomed regions kept for reuse live under
``regions/`` (format 5). A world with inhabitants keeps its civilisation under
``culture/`` (format 6). Format-4 and format-5 worlds still load; format-4
surfaces gain the structural fabric the zoom needs, computed from the stored
fields.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import xarray as xr
import yaml

from . import constants as c
from .culture import Civilisation, load_civilisation, save_civilisation
from .grid import SphereGrid, build_grid
from .report import state_to_plain, to_plain
from .spec import PlanetSpec, load_spec, save_spec
from .state import PlanetState, Timeline, TimelineEvent

FORMAT_VERSION = 6
READABLE_VERSIONS = (4, 5, 6)
REGIONS_FOLDER = "regions"
CULTURE_FOLDER = "culture"


@dataclass
class World:
    """A planet with its global surface fields (``surface`` is None for planets without a surface) and any civilisation."""

    spec: PlanetSpec
    state: PlanetState
    surface: Optional[xr.Dataset]
    timeline: Optional[Timeline] = None
    culture: Optional[Civilisation] = None

    @property
    def grid(self) -> Optional[SphereGrid]:
        """Return the spherical grid the surface lives on."""
        if self.surface is None:
            return None
        return build_grid(int(self.surface.attrs["grid_size"]))


def _epoch_name(index: int, time_s: float) -> str:
    """Return the file name of one epoch state."""
    return f"{index:02d}_{time_s / c.SECONDS_PER_GYR:.2f}gyr.yaml"


def _write_yaml(data: dict, path: Path) -> None:
    """Write plain data to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_world(world: World, path: str | Path) -> Path:
    """Write a world to a folder containing spec.yaml, state.yaml, surface.zarr, any timeline and any civilisation; returns the folder."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    save_spec(world.spec, folder / "spec.yaml")
    _write_yaml({"format_version": FORMAT_VERSION, **state_to_plain(world.state)}, folder / "state.yaml")
    if world.surface is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            world.surface.to_zarr(folder / "surface.zarr", mode="w", consolidated=False)
    timeline = world.timeline
    if timeline is not None and timeline.series:
        names = [_epoch_name(k, t) for k, t in enumerate(timeline.times_s)]
        _write_yaml({"format_version": FORMAT_VERSION, "times_s": to_plain(list(timeline.times_s)),
                     "epochs": names, "events": to_plain([asdict(e) for e in timeline.events]),
                     "series": to_plain(dict(timeline.series))}, folder / "timeline.yaml")
        epochs = folder / "epochs"
        epochs.mkdir(exist_ok=True)
        for name, state in zip(names, timeline.states):
            _write_yaml(state_to_plain(state), epochs / name)
    if world.culture is not None:
        save_civilisation(world.culture, folder / CULTURE_FOLDER)
    elif (folder / CULTURE_FOLDER).exists():
        shutil.rmtree(folder / CULTURE_FOLDER)
    return folder


def load_world(path: str | Path) -> World:
    """Read a world saved with ``save_world``."""
    folder = Path(path)
    spec = load_spec(folder / "spec.yaml")
    with open(folder / "state.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    version = data.pop("format_version", None)
    if version not in READABLE_VERSIONS:
        readable = ", ".join(map(str, READABLE_VERSIONS))
        raise ValueError(f"unsupported world format version {version} (readable: {readable})")
    state = PlanetState.from_dict(data)
    surface = None
    if (folder / "surface.zarr").exists():
        surface = _with_fabric(xr.open_zarr(folder / "surface.zarr", consolidated=False).load())
    culture = load_civilisation(folder / CULTURE_FOLDER) if (folder / CULTURE_FOLDER).exists() else None
    return World(spec=spec, state=state, surface=surface, timeline=_load_timeline(folder), culture=culture)


def _with_fabric(surface: xr.Dataset) -> xr.Dataset:
    """Return the surface with its structural fabric, computing it for surfaces saved before it was stored."""
    if "fabric" in surface or not {"crust", "crust_age", "orogeny_age"} <= set(surface.data_vars):
        return surface
    import numpy as np

    from .surface.fabric import structural_fabric

    grid = build_grid(int(surface.attrs["grid_size"]))
    fabric = structural_fabric(grid, surface["elevation"].values.astype(float), surface["crust"].values,
                               surface["crust_age"].values.astype(float),
                               surface["orogeny_age"].values.astype(float),
                               float(surface.attrs.get("relief_factor", 1.0)))
    out = surface.assign(fabric=(("vec", "cell"), fabric.T.astype(np.float32)))
    return out.assign_coords(vec=("vec", np.arange(3, dtype=np.int8)))


def _load_timeline(folder: Path) -> Optional[Timeline]:
    """Return the saved timeline of a world, or None if it has none."""
    path = folder / "timeline.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    states = []
    for name in data.get("epochs", []):
        with open(folder / "epochs" / name, encoding="utf-8") as f:
            states.append(PlanetState.from_dict(yaml.safe_load(f)))
    return Timeline(times_s=list(data.get("times_s", [])), states=states,
                    events=[TimelineEvent(**e) for e in data.get("events", [])],
                    series={k: list(v) for k, v in data.get("series", {}).items()})
