"""A generated world: its spec, global state, surface and history, with saving and loading.

A saved world is a folder: ``spec.yaml``, ``state.yaml``, ``surface.zarr``
and, in history mode, ``timeline.yaml`` (events and sampled series) with one
file per epoch under ``epochs/``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import xarray as xr
import yaml

from . import constants as c
from .grid import SphereGrid, build_grid
from .report import state_to_plain, to_plain
from .spec import PlanetSpec, load_spec, save_spec
from .state import PlanetState, Timeline, TimelineEvent

FORMAT_VERSION = 4


@dataclass
class World:
    """A planet with its global surface fields (``surface`` is None for planets without a surface)."""

    spec: PlanetSpec
    state: PlanetState
    surface: Optional[xr.Dataset]
    timeline: Optional[Timeline] = None

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
    """Write a world to a folder containing spec.yaml, state.yaml, surface.zarr and any timeline; returns the folder."""
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
    return World(spec=spec, state=state, surface=surface, timeline=_load_timeline(folder))


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
