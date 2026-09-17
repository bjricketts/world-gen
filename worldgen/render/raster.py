"""Resampling of grid fields onto regular latitude–longitude rasters and projected images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import SphereGrid


def raster_shape(grid: SphereGrid, width: int | None = None) -> tuple[int, int]:
    """Return (rows, columns) for a raster about twice as fine as the grid, unless a width is given."""
    if width is None:
        cells_around = 2 * np.pi / grid.mean_spacing()
        width = int(np.clip(2 ** np.ceil(np.log2(cells_around * 2)), 512, 2048))
    return width // 2, width


def lonlat_vectors(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel-centre longitudes, latitudes (degrees, north first) and unit vectors."""
    lon = -180.0 + (np.arange(cols) + 0.5) * 360.0 / cols
    lat = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    lo, la = np.meshgrid(np.radians(lon), np.radians(lat))
    vec = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1).reshape(-1, 3)
    return lon, lat, vec


def to_raster(grid: SphereGrid, values: np.ndarray, width: int | None = None,
              categorical: bool = False) -> np.ndarray:
    """Return a (rows, cols) raster of a grid field, north at the top.

    Continuous fields use inverse-distance weighting of the four nearest
    cells; categorical fields take the nearest cell.
    """
    rows, cols = raster_shape(grid, width)
    _, _, vec = lonlat_vectors(rows, cols)
    values = np.asarray(values)
    if categorical:
        _, idx = grid.nearest(vec, k=1)
        return values[idx].reshape(rows, cols)
    dist, idx = grid.nearest(vec, k=4)
    weights = 1.0 / np.maximum(dist, 1e-12) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    return (values[idx] * weights).sum(axis=1).reshape(rows, cols)


@dataclass(frozen=True)
class Sampler:
    """Precomputed lookup from projected image pixels to grid cells."""

    shape: tuple[int, int]            # (rows, cols)
    extent: tuple[float, float, float, float]
    valid: np.ndarray                 # (rows*cols,) pixels inside the projection
    nearest: np.ndarray               # (rows*cols,) nearest cell
    neighbours: np.ndarray            # (rows*cols, 4) nearest cells
    weights: np.ndarray               # (rows*cols, 4) inverse-distance weights

    def sample(self, values: np.ndarray, categorical: bool = False, fill: float = np.nan) -> np.ndarray:
        """Return a (rows, cols) image of a grid field; pixels outside the projection get ``fill``."""
        values = np.asarray(values)
        if categorical:
            return values[self.nearest].reshape(self.shape)
        out = (values[self.neighbours] * self.weights).sum(axis=1)
        out[~self.valid] = fill
        return out.reshape(self.shape)


_SAMPLERS: dict = {}
_MAX_SAMPLERS = 16


def projected_sampler(grid: SphereGrid, proj, width: int) -> Sampler:
    """Return a cached sampler mapping pixels of a projected image to grid cells."""
    import cartopy.crs as ccrs

    x0, x1 = proj.x_limits
    y0, y1 = proj.y_limits
    height = max(int(round(width * (y1 - y0) / (x1 - x0))), 1)
    key = (grid.size, proj.proj4_init, round(x0), round(x1), round(y0), round(y1), width)
    if key in _SAMPLERS:
        return _SAMPLERS[key]

    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
    xx, yy = np.meshgrid(xs, ys)
    with np.errstate(invalid="ignore"):
        lonlat = ccrs.PlateCarree().transform_points(proj, xx.ravel(), yy.ravel())
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90.0)
    # Pixels the projection maps back to a different position lie outside the globe outline.
    with np.errstate(invalid="ignore"):
        back = proj.transform_points(ccrs.PlateCarree(), np.where(valid, lon, 0.0), np.where(valid, lat, 0.0))
    scale = max(x1 - x0, y1 - y0)
    valid &= np.hypot(back[:, 0] - xx.ravel(), back[:, 1] - yy.ravel()) < 1e-6 * scale
    lo, la = np.radians(np.where(valid, lon, 0.0)), np.radians(np.where(valid, lat, 0.0))
    vec = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    dist, idx = grid.nearest(vec, k=4)
    w = 1.0 / np.maximum(dist, 1e-12) ** 2
    w /= w.sum(axis=1, keepdims=True)
    sampler = Sampler(shape=(height, width), extent=(x0, x1, y0, y1), valid=valid,
                      nearest=idx[:, 0], neighbours=idx, weights=w)
    if len(_SAMPLERS) >= _MAX_SAMPLERS:
        _SAMPLERS.pop(next(iter(_SAMPLERS)))
    _SAMPLERS[key] = sampler
    return sampler
