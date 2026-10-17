"""Figures of an organic history: a culture field at the kept epochs, an animation of them, and a timeline of the
series and the chronicle's major entries.

Every state and culture keeps one colour across epochs (the colour follows its id), so panels and frames can be
compared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ..culture.build import world_at_epoch
from ..culture.chronicle import entries
from ..culture.development import ENERGY_LEVELS
from .culture import culture_image, split_field
from .maps import plot_map, projection

# Chronicle rows on the timeline: (label, kinds)
TIMELINE_ROWS = (
    ("planet", ("eruption", "drought", "impact", "sea_level", "pandemic", "epidemic")),
    ("states", ("first_state", "great_power", "war", "break_up")),
    ("colonies", ("landing", "colony_state", "overseas_colony", "independence", "creole", "supremacy")),
    ("knowledge", ("farming_origin", "first_energy", "first_transport", "first_communication", "knowledge_loss")),
)
TIMELINE_CULTURES = 8          # cultures drawn by name; the rest are summed as "others"
# Line colours of the timeline: a validated eight-hue categorical order (normal-vision and colour-blind separation
# checked on adjacent pairs), given to the cultures in order of id.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
EPOCH_PANELS = 9


def epoch_dates(world) -> list[float]:
    """Return the kept epochs of a world's organic history (years ago, oldest first, the present last)."""
    civ = world.culture
    if civ is None or not civ.epochs:
        raise ValueError(f"{world.state.name} has no kept epochs; give the culture section history.epochs")
    return [c.epoch_years_ago for c in civ.epochs] + [civ.epoch_years_ago]


def _when(years_ago: float) -> str:
    return "present" if years_ago <= 0 else f"{years_ago:,.0f} years ago"


def _picks(dates: Sequence[float], panels: int) -> list[float]:
    """Return up to ``panels`` dates spread over the kept epochs, always with the first and the present."""
    k = np.unique(np.linspace(0, len(dates) - 1, min(panels, len(dates))).round().astype(int))
    return [dates[i] for i in k]


def plot_epochs(world, field: str = "political", epochs: Optional[Sequence[float]] = None,
                panels: int = EPOCH_PANELS, projection_name: str = "mollweide", central_longitude: float = 0.0):
    """Return a figure of one culture field at the kept epochs (or the given ones), oldest first; each panel is
    labelled with its date."""
    dates = list(epochs) if epochs is not None else _picks(epoch_dates(world), panels)
    cols = min(3, len(dates))
    rows = int(np.ceil(len(dates) / cols))
    proj = projection(projection_name, central_longitude)
    fig = plt.figure(figsize=(5.2 * cols, 2.9 * rows))
    for n, years in enumerate(dates):
        ax = fig.add_subplot(rows, cols, n + 1, projection=proj)
        plot_map(world_at_epoch(world, years), field, projection_name, ax=ax, central_longitude=central_longitude,
                 width=700, title="", colorbar=False, legend=None, rivers=False)
        ax.text(0.01, 0.02, _when(years), transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    fig.tight_layout()
    return fig


def animate_epochs(world, path: str | Path, field: str = "political", fps: float = 1.0, width: int = 900,
                   projection_name: str = "mollweide", central_longitude: float = 0.0,
                   dpi: Optional[int] = 100) -> Path:
    """Write an animated GIF of one culture field through the kept epochs and return its path."""
    dates = epoch_dates(world)
    proj = projection(projection_name, central_longitude)
    if split_field(field)[0] in ("settlements", "roads"):
        raise ValueError("animate a coloured field: political, polities, cultures, population or habitat")
    frames = [culture_image(world_at_epoch(world, d), field, proj, width) for d in dates]
    fig = plt.figure(figsize=(8, 4.6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    rgba, _, extent = frames[0]
    image = ax.imshow(rgba, transform=proj, extent=extent, origin="upper", interpolation="nearest")
    label = ax.text(0.01, 0.02, "", transform=ax.transAxes, fontsize=9, ha="left", va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})

    def draw(k: int):
        """Show frame ``k``."""
        image.set_data(frames[k][0])
        label.set_text(_when(dates[k]))
        return image, label

    animation = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    path = Path(path)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def plot_chronicle(world, since: Optional[float] = None, until: float = 0.0, cultures: int = TIMELINE_CULTURES):
    """Return a figure on one time axis (years ago, the present on the right): people per culture, each culture's
    people-weighted mean energy level, and the chronicle's major entries by row."""
    civ = world.culture
    if civ is None or not len(civ.series):
        raise ValueError(f"{world.state.name} has no history series; build its culture with an organic history")
    s = civ.series
    since = float(s.years_ago.max()) if since is None else since
    keep = (s.years_ago <= since) & (s.years_ago >= until)
    names = {c.id: c.name or f"culture {c.id}" for c in civ.cultures}
    for e in civ.epochs:
        for c in e.cultures:
            names.setdefault(c.id, c.name or f"culture {c.id}")
    peak = {cid: s.people[s.culture == cid].max() for cid in np.unique(s.culture)}
    shown = sorted(sorted(peak, key=lambda cid: -peak[cid])[:min(cultures, len(SERIES_COLOURS))])
    colours = dict(zip(shown, SERIES_COLOURS))               # by id over the whole history, not by the window
    shown = [cid for cid in shown if (keep & (s.culture == cid)).any()]
    fig, (top, mid, low) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
    times = np.unique(s.years_ago[keep])[::-1]
    for cid in shown:
        rows = keep & (s.culture == cid)
        order = np.argsort(-s.years_ago[rows])
        t, p = s.years_ago[rows][order], s.people[rows][order]
        top.plot(t, p, color=colours[cid], lw=1.2, label=names.get(cid, str(cid)))
        mid.plot(t, s.energy[rows][order], color=colours[cid], lw=1.2)
    others = keep & ~np.isin(s.culture, shown)
    if others.any():
        total = np.array([s.people[others & (s.years_ago == t)].sum() for t in times])
        top.plot(times, total, color="#8a8a8a", lw=1.0, ls="--", label="others")
    top.set_yscale("log")
    top.set_ylim(bottom=1e3)
    top.set_ylabel("people")
    top.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    mid.set_yticks(range(len(ENERGY_LEVELS)), [n.replace("_", " ") for n in ENERGY_LEVELS], fontsize=7)
    mid.set_ylabel("mean energy level")
    marks = {"war": "|", "break_up": "|"}
    for k, (label, kinds) in enumerate(TIMELINE_ROWS):
        chosen = entries(civ, since, until, kinds, minor=False)
        x = np.array([e.years_ago for e in chosen])
        if x.size:
            for kind in sorted({e.kind for e in chosen}):
                xs = x[[e.kind == kind for e in chosen]]
                low.scatter(xs, np.full(xs.size, k), s=14, marker=marks.get(kind, "o"), color="#3a3a3a",
                            alpha=0.55, linewidths=0.8)
    low.set_yticks(range(len(TIMELINE_ROWS)), [r[0] for r in TIMELINE_ROWS], fontsize=8)
    low.set_ylim(-0.6, len(TIMELINE_ROWS) - 0.4)
    low.set_xlabel("years ago")
    for ax in (top, mid, low):
        ax.grid(color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    low.set_xlim(since, until)
    for d in [c.epoch_years_ago for c in civ.epochs]:
        if until <= d <= since:
            for ax in (top, mid, low):
                ax.axvline(d, color="#bbbbbb", lw=0.6, ls=":")
    fig.tight_layout()
    return fig
