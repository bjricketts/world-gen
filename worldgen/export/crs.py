"""Coordinate reference systems of a planet: a geographic system on its sphere, and raster georeferencing."""

from __future__ import annotations

import numpy as np

DEGREE = "0.0174532925199433"


def planet_crs_wkt(name: str, radius_m: float) -> str:
    """Return the WKT2 definition of a geographic (latitude, longitude) system on a sphere of the planet's radius.

    Longitudes are planetocentric and positive east, as on the world grid.
    """
    angle = f'ANGLEUNIT["degree",{DEGREE}]'
    label = name.replace('"', "'")
    return (f'GEOGCRS["{label} (sphere)",DATUM["{label} sphere",ELLIPSOID["{label} sphere",{radius_m:.3f},0,'
            f'LENGTHUNIT["metre",1]]],PRIMEM["Reference meridian",0,{angle}],CS[ellipsoidal,2],'
            f'AXIS["latitude",north,ORDER[1],{angle}],AXIS["longitude",east,ORDER[2],{angle}]]')


def equirect_geotransform(rows: int, cols: int) -> tuple[float, float, float, float, float, float]:
    """Return the GDAL geotransform of a global equirectangular raster, north row first, from 180° W."""
    return (-180.0, 360.0 / cols, 0.0, 90.0, 0.0, -180.0 / rows)


def laea_crs_wkt(name: str, radius_m: float, lat0_deg: float, lon0_deg: float) -> str:
    """Return the WKT2 definition of a Lambert azimuthal equal-area grid on the planet's sphere, centred on a point."""
    angle = f'ANGLEUNIT["degree",{DEGREE}]'
    metre = 'LENGTHUNIT["metre",1]'
    base = planet_crs_wkt(name, radius_m).replace("GEOGCRS[", "BASEGEOGCRS[", 1)
    base = base[:base.index(",CS[ellipsoidal")] + "]"
    return (f'PROJCRS["{name.replace(chr(34), chr(39))} (sphere) / LAEA {lat0_deg:.4f} {lon0_deg:.4f}",{base},'
            f'CONVERSION["Lambert azimuthal equal area",METHOD["Lambert Azimuthal Equal Area (Spherical)",ID["EPSG",1027]],'
            f'PARAMETER["Latitude of natural origin",{lat0_deg:.8f},{angle}],'
            f'PARAMETER["Longitude of natural origin",{lon0_deg:.8f},{angle}],'
            f'PARAMETER["False easting",0,{metre}],PARAMETER["False northing",0,{metre}]],'
            f'CS[Cartesian,2],AXIS["easting",east,ORDER[1],{metre}],AXIS["northing",north,ORDER[2],{metre}]]')


def laea_inverse(x: np.ndarray, y: np.ndarray, radius_m: float, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude and longitude (radians) of points on a spherical LAEA grid centred on (lat0, lon0) radians."""
    rho = np.hypot(x, y)
    c = 2.0 * np.arcsin(np.clip(rho / (2.0 * radius_m), -1.0, 1.0))
    safe = np.where(rho > 0, rho, 1.0)
    lat = np.arcsin(np.clip(np.cos(c) * np.sin(lat0) + np.where(rho > 0, y * np.sin(c) * np.cos(lat0) / safe, 0.0),
                            -1.0, 1.0))
    lon = lon0 + np.arctan2(x * np.sin(c), rho * np.cos(lat0) * np.cos(c) - y * np.sin(lat0) * np.sin(c))
    return lat, (lon + np.pi) % (2 * np.pi) - np.pi


def laea_forward(lat: np.ndarray, lon: np.ndarray, radius_m: float, lat0: float, lon0: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return the spherical LAEA coordinates (m) of points given in radians."""
    k = np.sqrt(2.0 / (1.0 + np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)))
    x = radius_m * k * np.cos(lat) * np.sin(lon - lon0)
    y = radius_m * k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
    return x, y
