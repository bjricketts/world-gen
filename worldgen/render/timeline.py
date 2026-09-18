"""Figures of a planet's integrated history: climate, interior and life against time.

Each figure draws the sampled series of a history-mode run on a shared time
axis, shades the stretches the planet spent frozen, in a runaway or dry, and
marks the events from the timeline.
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .. import constants as c
from ..state import Timeline
from ..world import World

Source = Union[World, Timeline]

CLIMATE_SHADING = {"snowball": ("#cfe3f5", "snowball"), "runaway": ("#f6cfc8", "runaway"),
                   "dry": ("#e8dcc2", "dry")}
LABELLED = ("origin_of_life", "land_colonisation", "oxygenation", "runaway_onset", "runaway_end", "ocean_loss",
            "ocean_return", "limit_cycle", "snowball_onset", "snowball_exit", "dynamo_shutdown", "regime_change",
            "tidal_locking")
LIFE_LEVELS = ("none", "subsurface", "ocean", "surface")


def has_timeline(source: Source) -> bool:
    """Return whether the world or timeline carries an integrated history."""
    timeline = source.timeline if isinstance(source, World) else source
    return timeline is not None and bool(timeline.series)


def _unpack(source: Source) -> tuple[Timeline, str]:
    """Return the timeline and the planet's name, from a world or a timeline."""
    if isinstance(source, World):
        timeline, name = source.timeline, source.state.name
    else:
        timeline, name = source, source.states[-1].name if source.states else ""
    if timeline is None or not timeline.series:
        raise ValueError(f"{name or 'this planet'} has no integrated history; generate it with mode: history")
    return timeline, name


def _phases(times: np.ndarray, modes: list[str]) -> list[tuple[str, float, float]]:
    """Return the (mode, start, end) stretches the planet spent in each climate mode."""
    spans, start = [], 0
    for k in range(1, len(modes) + 1):
        if k == len(modes) or modes[k] != modes[start]:
            spans.append((modes[start], float(times[start]), float(times[min(k, len(times) - 1)])))
            start = k
    return spans


def _decorate(ax, times: np.ndarray, modes: list[str], events: list, label: bool = False) -> None:
    """Shade the climate phases and mark the events on one panel."""
    for mode, start, end in _phases(times, modes):
        if mode in CLIMATE_SHADING and end > start:
            ax.axvspan(start, end, color=CLIMATE_SHADING[mode][0], zorder=0)
    span = max(float(times[-1] - times[0]), 1e-9)
    last_label, level = -1e9, 0
    for e in events:
        t = e.time_s / c.SECONDS_PER_GYR
        ax.axvline(t, color="0.55", linestyle=":", linewidth=0.8, zorder=1)
        if label and e.kind in LABELLED:
            # Labels of events close in time are staggered so they do not sit on top of each other.
            level = (level + 1) % 3 if t - last_label < 0.08 * span else 0
            last_label = t
            ax.annotate(e.kind.replace("_", " "), (t, 1.0 - 0.3 * level), xycoords=("data", "axes fraction"),
                        xytext=(2, -3), textcoords="offset points", rotation=90, va="top", ha="left",
                        fontsize=7, color="0.35")
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.grid(alpha=0.25)


def _log_if_wide(ax, *series, ratio: float = 20.0) -> None:
    """Use a logarithmic y axis only when the values span more than ``ratio``."""
    values = np.concatenate([np.asarray(s, dtype=float) for s in series])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size and values.max() / values.min() > ratio:
        ax.set_yscale("log")


def _panels(name: str, title: str, rows: int = 2, cols: int = 2):
    """Return a figure and its axes for one history topic."""
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 3.3 * rows), sharex=True)
    fig.suptitle(f"{name}: {title}", fontsize=13)
    return fig, axes.ravel()


def _finish(fig, axes, times: np.ndarray) -> None:
    """Label the time axis of the bottom row and tighten the layout."""
    for ax in axes[-2:]:
        ax.set_xlabel("age of the planet (Gyr)")
    fig.tight_layout()


def plot_climate_history(source: Source):
    """Return a figure of temperature, air, instellation and water over the planet's history."""
    timeline, name = _unpack(source)
    s = timeline.series
    t = np.asarray(s["time_gyr"])
    modes, events = list(s["climate"]), timeline.events
    fig, axes = _panels(name, "climate history")

    ax = axes[0]
    ax.plot(t, s["surface_temperature_k"], color="#b2453a")
    ax.axhline(273.15, color="0.5", linewidth=0.8)
    ax.set_ylabel("mean surface temperature (K)")
    _decorate(ax, t, modes, events, label=True)

    ax = axes[1]
    ax.plot(t, s["pressure_bar"], color="#31527a", label="total")
    ax.plot(t, np.maximum(s["co2_bar"], 1e-12), color="#7a8b31", label="CO₂")
    _log_if_wide(ax, s["pressure_bar"], s["co2_bar"])
    ax.set_ylabel("partial pressure (bar)")
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events)

    ax = axes[2]
    ax.plot(t, s["instellation"], color="#c98a1e", label="instellation")
    ax.plot(t, s["runaway_limit"], color="#b2453a", linestyle="--", label="runaway limit")
    ax.set_ylabel("instellation (Earth = 1)")
    ax.legend(fontsize=8, loc="best")
    twin = ax.twinx()
    twin.plot(t, s["albedo"], color="0.45", linewidth=0.9)
    twin.set_ylabel("planetary albedo", color="0.45", fontsize=9)
    _decorate(ax, t, modes, events)

    ax = axes[3]
    ax.plot(t, s["surface_water_oceans"], color="#31527a", label="surface")
    ax.plot(t, s["mantle_water_oceans"], color="#7a5c31", label="mantle")
    ax.plot(t, np.maximum(s["water_escape"], 1e-12), color="#9b59b6", linestyle=":", label="escape (per Gyr)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, max(2.0, 2 * max(s["surface_water_oceans"] + s["mantle_water_oceans"])))
    ax.set_ylabel("water (Earth oceans)")
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events)

    _finish(fig, axes, t)
    return fig


def plot_interior_history(source: Source):
    """Return a figure of mantle and core temperature, heat flow, melting and the carbon cycle."""
    timeline, name = _unpack(source)
    s = timeline.series
    t = np.asarray(s["time_gyr"])
    modes, events = list(s["climate"]), timeline.events
    fig, axes = _panels(name, "interior history")

    ax = axes[0]
    ax.plot(t, s["mantle_temperature_k"], color="#b2453a", label="mantle")
    ax.plot(t, s["core_temperature_k"], color="#31527a", label="core")
    ax.set_ylabel("temperature (K)")
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events, label=True)

    ax = axes[1]
    ax.plot(t, np.asarray(s["heat_flux_w_m2"]) * 1e3, color="#c98a1e")
    ax.axhline(c.EARTH_HEAT_FLUX * 1e3, color="0.5", linewidth=0.8)
    ax.set_ylabel("surface heat flux (mW/m²)")
    _log_if_wide(ax, s["heat_flux_w_m2"])
    for start, end in _runs(t, [bool(d) for d in s["dynamo"]]):
        ax.axvspan(start, end, color="#dfe8d8", zorder=0)
    _decorate(ax, t, modes, events)
    ax.annotate("shaded: core dynamo", (0.02, 0.06), xycoords="axes fraction", fontsize=8, color="0.35")

    ax = axes[2]
    ax.plot(t, s["melt"], color="#a14a2a", label="melt production")
    ax.plot(t, s["land_fraction"], color="#7a8b31", linestyle="--", label="land fraction")
    ax.set_ylabel("relative to present Earth")
    _log_if_wide(ax, s["melt"], s["land_fraction"])
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events)

    ax = axes[3]
    ax.plot(t, s["weathering"], color="#31527a", label="weathering")
    ax.plot(t, s["outgassing"], color="#b2453a", label="outgassing")
    _log_if_wide(ax, s["weathering"], s["outgassing"])
    ax.set_ylabel("carbon flux (×10¹⁸ kg per Gyr)")
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events)

    _finish(fig, axes, t)
    return fig


def plot_life_history(source: Source):
    """Return a figure of the biosphere: productivity, its gases, its habitat and what it did to the air."""
    timeline, name = _unpack(source)
    s = timeline.series
    t = np.asarray(s["time_gyr"])
    modes, events = list(s["climate"]), timeline.events
    fig, axes = _panels(name, "biosphere history")

    ax = axes[0]
    ax.plot(t, s["productivity"], color="#3f7d3f")
    ax.set_ylabel("productivity (Earth = 1)")
    _decorate(ax, t, modes, events, label=True)

    ax = axes[1]
    ax.plot(t, np.maximum(s["o2_fraction"], 1e-12), color="#31527a", label="O₂")
    ax.plot(t, np.maximum(s["ch4_fraction"], 1e-12), color="#a14a2a", label="CH₄")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 2.0)
    ax.set_ylabel("mole fraction of the air")
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events)

    ax = axes[2]
    ax.plot(t, s["land_fraction"], color="#7a5c31", label="land")
    ax.plot(t, s["open_ocean_fraction"], color="#31527a", label="open ocean")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("share of the surface / of the ocean")
    ax.legend(fontsize=8, loc="best")
    _decorate(ax, t, modes, events)

    ax = axes[3]
    levels = [LIFE_LEVELS.index(k) if k in LIFE_LEVELS else 0 for k in s["life"]]
    ax.step(t, levels, where="post", color="#3f7d3f")
    ax.set_yticks(range(len(LIFE_LEVELS)), LIFE_LEVELS)
    ax.set_ylim(-0.3, len(LIFE_LEVELS) - 0.7)
    ax.set_ylabel("where life lives")
    _decorate(ax, t, modes, events)

    _finish(fig, axes, t)
    return fig


def _runs(times: np.ndarray, flags: list[bool]) -> list[tuple[float, float]]:
    """Return the (start, end) stretches over which a flag is true."""
    spans, start = [], None
    for k, flag in enumerate(flags):
        if flag and start is None:
            start = times[k]
        elif not flag and start is not None:
            spans.append((float(start), float(times[k])))
            start = None
    if start is not None:
        spans.append((float(start), float(times[-1])))
    return spans


TIMELINE_FIGURES = {"climate": plot_climate_history, "interior": plot_interior_history, "life": plot_life_history}


def plot_timeline(source: Source, topic: str = "climate"):
    """Return the requested history figure: 'climate', 'interior' or 'life'."""
    if topic not in TIMELINE_FIGURES:
        raise ValueError(f"unknown history figure '{topic}'; choose from {', '.join(TIMELINE_FIGURES)}")
    return TIMELINE_FIGURES[topic](source)
