"""Save and load a civilisation as a folder: ``culture.yaml`` for the small readable parts, ``culture.zarr`` for tables and layers.

``culture.yaml`` holds the spec, the resolved cultures, the polities and the
chronicle. ``culture.zarr`` holds the per-cell layers (dimension ``cell``), the
settlement table (``settlement``) and the road table with its paths (``road``,
``road_offset``, ``road_vertex``). An organic history's earlier epochs are saved
the same way in ``epochs/<years ago>/``. Format 3 adds the history's series table (``series``); formats 1 and 2
still load.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from .model import ChronicleEntry, Civilisation, Culture, Polity, RoadTable, SeriesTable, SettlementTable, TradeTable
from .spec import CultureSpec

CULTURE_FORMAT = 3
READABLE_FORMATS = (1, 2, 3)
EPOCHS_NAME = "epochs"
YAML_NAME = "culture.yaml"
ZARR_NAME = "culture.zarr"


def _plain(obj):
    """Return a copy of nested data using only built-in Python types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def epoch_folder_name(years_ago: float) -> str:
    """Return the folder name of an epoch: its years before the present, zero-padded so folders sort by date."""
    return f"{years_ago:09.0f}" if float(years_ago).is_integer() else f"{years_ago:011.1f}"


def save_civilisation(civ: Civilisation, path: str | Path) -> Path:
    """Write a civilisation to a folder, with any kept epochs under ``epochs/``; returns the folder."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "culture_format": CULTURE_FORMAT,
        "seed": civ.seed,
        "epoch_years_ago": civ.epoch_years_ago,
        "spec": civ.spec.model_dump(mode="json", exclude_none=True),
        "cultures": [c.to_plain() for c in civ.cultures],
        "polities": [p.to_plain() for p in civ.polities],
        "chronicle": [asdict(e) for e in civ.chronicle],
    }
    with open(folder / YAML_NAME, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(data), f, sort_keys=False, allow_unicode=True)
    ds = xr.merge([civ.layers, civ.settlements.to_dataset(), civ.roads.to_dataset(), civ.trade.to_dataset(),
                   civ.series.to_dataset()],
                  compat="override")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.to_zarr(folder / ZARR_NAME, mode="w", consolidated=False)
    shutil.rmtree(folder / EPOCHS_NAME, ignore_errors=True)
    for epoch in civ.epochs:
        save_civilisation(epoch, folder / EPOCHS_NAME / epoch_folder_name(epoch.epoch_years_ago))
    return folder


def load_civilisation(path: str | Path) -> Civilisation:
    """Read a civilisation saved with ``save_civilisation``."""
    folder = Path(path)
    with open(folder / YAML_NAME, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    version = data.get("culture_format")
    if version not in READABLE_FORMATS:
        raise ValueError(f"unsupported culture format {version} (readable: {', '.join(map(str, READABLE_FORMATS))})")
    ds = xr.open_zarr(folder / ZARR_NAME, consolidated=False).load()
    table_dims = {"settlement", "road", "road_offset", "road_vertex", "trade", "series"}
    layers = ds[[v for v in ds.data_vars if not set(ds[v].dims) & table_dims]]
    layers = layers.drop_dims([d for d in table_dims if d in layers.dims])
    return Civilisation(
        spec=CultureSpec.model_validate(data["spec"]),
        seed=int(data["seed"]),
        cultures=[Culture.from_plain(c) for c in data["cultures"]],
        polities=[Polity(**p) for p in data.get("polities", [])],
        settlements=SettlementTable.from_dataset(ds),
        roads=RoadTable.from_dataset(ds),
        trade=TradeTable.from_dataset(ds),
        series=SeriesTable.from_dataset(ds),
        layers=layers,
        chronicle=[ChronicleEntry(**e) for e in data.get("chronicle", [])],
        epoch_years_ago=float(data.get("epoch_years_ago", 0.0)),
        epochs=_load_epochs(folder / EPOCHS_NAME),
    )


def _load_epochs(folder: Path) -> list[Civilisation]:
    """Return the civilisations saved under an ``epochs/`` folder, oldest first."""
    if not folder.is_dir():
        return []
    epochs = [load_civilisation(sub) for sub in folder.iterdir() if (sub / YAML_NAME).exists()]
    return sorted(epochs, key=lambda c: -c.epoch_years_ago)

