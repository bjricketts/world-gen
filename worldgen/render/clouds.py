"""Map colours of a giant's cloud tops: the cloud model evaluated at the pixels or vertices being drawn.

Fields: ``clouds`` (the colour under the star; ``elevation`` asks for the same
on a giant, so globes and meshes show the clouds), ``temperature`` (upper
troposphere), ``winds`` (zonal wind), ``bands`` (zones and belts),
``vortices`` (by class) and ``relief`` (cloud-top height).
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch

from ..giants.clouds import ANTICYCLONIC, cloud_model, evaluate

CLOUD_FIELDS = ("clouds", "temperature", "winds", "bands", "vortices", "relief")
CLOUD_COLORBAR_LABELS = {"temperature": "upper-tropospheric temperature (K)", "winds": "zonal wind (m/s, eastward +)",
                         "bands": "band phase (zone +1, belt −1)", "relief": "cloud-top height (km)"}
VORTEX_LEGEND = {"white": "white oval (anticyclone)", "red": "red oval (anticyclone)",
                 "brown": "brown barge (cyclone)", "dark": "dark spot (cyclone)", "great": "great oval",
                 "polar": "polar cyclone", "bright": "bright cloud"}
_FIELD_INPUT = {"clouds": "colour", "temperature": "temperature", "winds": "zonal_wind", "bands": "band",
                "vortices": "vortex", "relief": "relief"}


def cloud_field(field: str) -> str:
    """Return the cloud field a map field name asks for on a giant ('elevation' shows the clouds)."""
    name = "clouds" if field == "elevation" else field
    if name not in CLOUD_FIELDS:
        raise ValueError(f"'{field}' is not a field of a giant's cloud tops; choose from {', '.join(CLOUD_FIELDS)}")
    return name


def cloud_colours(world, field: str, sampler) -> tuple[np.ndarray, object]:
    """Return RGBA colours (rows, cols, 4) of a cloud field at a sampler's points, and its colour bar or legend."""
    name = cloud_field(field)
    model = cloud_model(world.surface)
    valid = sampler.valid
    vectors = np.asarray(sampler.vectors, dtype=float)[valid]
    values = evaluate(model, vectors, (_FIELD_INPUT[name],))[_FIELD_INPUT[name]]
    rgb = np.zeros((len(valid), 3))
    extra = None
    if name == "clouds":
        rgb[valid] = values
    elif name == "vortices":
        kinds = model.vortices.get("kind", [])
        grey = 0.82
        rgb[:] = grey
        palette = {k: np.asarray(model.vortex_rgb[k]) for k in VORTEX_LEGEND}
        table = np.array([palette[k] for k in kinds]) if kinds else np.zeros((0, 3))
        sub = np.full((len(values), 3), grey)
        inside = values >= 0
        sub[inside] = table[values[inside]]
        rgb[valid] = sub
        present = sorted(set(kinds), key=list(VORTEX_LEGEND).index)
        extra = [Patch(color=palette[k], label=VORTEX_LEGEND[k]) for k in present]
    else:
        if name == "temperature":
            lo, hi = float(np.min(values)), float(np.max(values))
            cmap, norm = plt.get_cmap("RdYlBu_r"), Normalize(lo, max(hi, lo + 1.0))
        elif name == "winds":
            top = max(float(np.max(np.abs(values))), 1.0)
            cmap, norm = plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-top, vcenter=0.0, vmax=top)
        elif name == "bands":
            cmap, norm = plt.get_cmap("BrBG"), Normalize(-1.0, 1.0)
        else:
            values = values / 1e3
            top = max(float(np.max(np.abs(values))), 1e-3)
            cmap, norm = plt.get_cmap("gray"), Normalize(-top, top)
        rgb[valid] = cmap(norm(values))[:, :3]
        extra = (cmap, norm)
    rgba = np.concatenate([rgb, valid[:, None].astype(float)], axis=1)
    return rgba.reshape(*sampler.shape, 4), extra


def vortex_summary(world) -> dict[str, int]:
    """Return the number of vortices of each class, and of anticyclones and cyclones."""
    kinds = cloud_model(world.surface).vortices.get("kind", [])
    out = {k: kinds.count(k) for k in VORTEX_LEGEND if k in kinds}
    out["anticyclones"] = sum(ANTICYCLONIC[k] > 0 for k in kinds)
    out["cyclones"] = sum(ANTICYCLONIC[k] < 0 for k in kinds)
    return out


__all__ = ["CLOUD_FIELDS", "CLOUD_COLORBAR_LABELS", "cloud_colours", "cloud_field", "vortex_summary"]
