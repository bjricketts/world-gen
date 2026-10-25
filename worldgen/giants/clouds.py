"""Cloud tops of giant planets: a small model of bands, winds and vortices, evaluated at any points (task 11.3).

The model is drawn once from the planet's state and kept with its surface;
maps, rasters, meshes and the globe evaluate it at their own points, so
vortices smaller than a grid cell still show.

- Winds alternate in direction at the jet cores of the appearance (Rhines
  spacing) and peak at the belt–zone boundaries; gas giants with a prograde
  equator have a double-peaked equatorial jet (Fletcher et al. 2019).
- Zones are where the shear is anticyclonic (−∂u/∂y in the north): high,
  cold, cloudy and bright; belts are cyclonic, warm and dark (Fletcher et al.
  2019). Band edges are warped by stretched noise and turbulence grows
  poleward of ~45° (Sankar et al. 2024).
- Vortices follow the JunoCam catalogue of Sankar et al. (2024): sizes
  log-normal about the deformation radius, aspect ratios about 1.5, white,
  red, brown and dark classes with their latitude preferences; anticyclones
  sit in zones and cyclones in belts. One great oval may fill a tropical zone
  (Sánchez-Lavega et al. 2024); ice giants get dark spots and short-lived
  bright clouds (Melin 2026).
- Poles of gas giants carry a polar cyclone, with a ring of circumpolar
  cyclones at ~84° spaced as Gavriel & Kaspi (2021) find, or a hexagon on a
  prograde jet near 78°.
- Tidally locked giants have a day–night temperature pattern with the hot
  spot east of the substellar point (Showman & Polvani 2011), coloured by the
  cloud class at the local temperature, one broad equatorial jet and
  anticyclonic polar vortices (Skinner & Wei 2025).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import cKDTree

from .. import constants as c
from .. import heuristics as h
from ..noise import fbm
from ..state import PlanetState
from ..util import named_rng
from .appearance import class_at_temperature_rgb, is_ice_like, jet_latitudes

if TYPE_CHECKING:
    import xarray as xr

FIELDS = ("colour", "band", "zonal_wind", "relief", "temperature", "vortex")
VORTEX_CLASSES = ("white", "red", "brown", "dark", "great", "polar", "bright")
ANTICYCLONIC = {"white": 1, "red": 1, "great": 1, "brown": -1, "dark": -1, "polar": -1, "bright": 1}
CHUNK = 500_000


@dataclass
class CloudModel:
    """Everything needed to evaluate a giant's cloud tops at any point; plain data so it saves as JSON."""

    seed: int
    radius_m: float
    regime: str                              # "banded" or "locked"
    ice_like: bool
    effective_temperature_k: float
    scale_height_m: float
    deformation_radius_m: float
    wind_lat_deg: list[float] = field(default_factory=list)   # nodes of the zonal wind profile, south to north
    wind_m_s: list[float] = field(default_factory=list)
    band_contrast: list[float] = field(default_factory=list)  # per band between consecutive wind nodes
    zone_rgb: list[float] = field(default_factory=list)
    belt_rgb: list[float] = field(default_factory=list)
    vortex_rgb: dict = field(default_factory=dict)            # class → sRGB
    vortices: dict = field(default_factory=dict)              # lat_deg, lon_deg, a_m, b_m, kind (lists)
    hexagons: list[dict] = field(default_factory=list)        # lat_deg (signed), rotation_deg
    hot_spot_offset_deg: float = 0.0
    day_night_k: float = 0.0
    jet_half_width_deg: float = 0.0
    locked_temperatures_k: list[float] = field(default_factory=list)
    locked_rgb: list[list[float]] = field(default_factory=list)

    def to_json(self) -> str:
        """Return the model as a JSON string (kept in the surface's attributes)."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "CloudModel":
        """Return a model from ``to_json`` output."""
        return cls(**json.loads(text))

    @property
    def vortex_count(self) -> int:
        """Return the number of vortices (great ovals and polar cyclones included)."""
        return len(self.vortices.get("lat_deg", []))


# --- Building the model ------------------------------------------------------

def deformation_radius(state: PlanetState) -> float:
    """Return the deformation radius N H / f at 45° (m), with N H = R_s √(T/c_p), calibrated to Jupiter's 1800 km."""
    ap = state.appearance
    r_s = c.R_GAS / (state.atmosphere.mean_molecular_weight * 1e-3)
    nh = r_s * math.sqrt(ap.effective_temperature_k / (h.GIANT_CP_OVER_R * r_s))
    f = 2 * (2 * math.pi / abs(state.orbit.rotation_period_s)) * math.sin(math.radians(45.0))
    return h.DEFORMATION_CALIBRATION * nh / f


def _scale_height(state: PlanetState) -> float:
    """Return the pressure scale height at the effective temperature (m)."""
    r_s = c.R_GAS / (state.atmosphere.mean_molecular_weight * 1e-3)
    return r_s * state.appearance.effective_temperature_k / state.bulk.surface_gravity_m_s2


def _equatorial_wind(state: PlanetState, ice: bool) -> tuple[float, float]:
    """Return the equatorial and the off-equatorial jet peak speeds (m/s), signed by direction."""
    if ice:
        return h.WIND_EQUATOR_ICE_M_S, h.WIND_HIGH_LATITUDE_ICE_M_S
    m = state.bulk.mass_kg / c.M_EARTH
    x = np.clip((math.log10(m) - math.log10(95.2)) / (math.log10(317.8) - math.log10(95.2)), 0.0, 1.0)
    eq = 10 ** ((1 - x) * math.log10(h.WIND_EQUATOR_SATURN_M_S) + x * math.log10(h.WIND_EQUATOR_JUPITER_M_S))
    if state.appearance.equatorial_jet == "retrograde":
        eq = -eq
    return eq, abs(eq) * h.WIND_JET_FRACTION_GAS


def wind_nodes(cores_rad: np.ndarray, u_eq: float, u_jet: float, double_peak: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the latitudes (degrees, south to north) and speeds of the zonal-wind nodes.

    The jets alternate in direction outward from the equatorial jet; a
    double-peaked equatorial jet has a dip at the equator between two peaks.
    """
    north_lat, north_u = [], []
    sign = np.sign(u_eq) if u_eq != 0 else 1.0
    for k, lat in enumerate(cores_rad[1:], start=1):
        north_lat.append(math.degrees(lat))
        north_u.append(-sign * u_jet if k % 2 else sign * u_jet)
    if double_peak and len(cores_rad) > 1:
        peak = h.WIND_EQUATOR_PEAK_FRACTION * math.degrees(cores_rad[1])
        centre_lat, centre_u = [0.0], [u_eq * (1 - h.WIND_EQUATOR_DIP)]
        north_lat, north_u = [peak] + north_lat, [u_eq] + north_u
    else:
        centre_lat, centre_u = [0.0], [u_eq]
    lat = [-x for x in north_lat[::-1]] + centre_lat + north_lat
    u = north_u[::-1] + centre_u + north_u
    # close the profile at the poles with calm air
    lat = [-90.0] + lat + [90.0]
    u = [0.0] + u + [0.0]
    return np.array(lat), np.array(u)


def _rgb(hex_colour: str) -> np.ndarray:
    """Return sRGB components (0–1) of a hex colour."""
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _vortex_colours(state: PlanetState, ice: bool, zone: np.ndarray, belt: np.ndarray) -> dict:
    """Return the colour of each vortex class."""
    vortex = state.appearance.vortex_colour
    red = _rgb(vortex) if vortex else 0.8 * zone
    return {
        "white": np.clip(zone * 1.12 + 0.04, 0, 1),
        "red": red,
        "great": red,
        "brown": belt * 0.72,
        "dark": belt * 0.55 if not ice else zone * 0.55,
        "polar": belt * 0.8,
        "bright": np.clip(zone * 1.25 + 0.12, 0, 1),
    }


def _band_index(lat_deg: float, nodes: np.ndarray) -> int:
    """Return the index of the band (between consecutive wind nodes) containing a latitude."""
    return int(np.clip(np.searchsorted(nodes, lat_deg) - 1, 0, len(nodes) - 2))


def _place(rng, lat_nodes: np.ndarray, u: np.ndarray, anticyclone: bool, pref) -> Optional[tuple[float, float]]:
    """Return a latitude (degrees) inside a band of the right vorticity and the band width there, or None."""
    for _ in range(20):
        lat = math.degrees(math.asin(rng.uniform(-0.97, 0.97)))
        if rng.uniform() > pref(abs(lat)):
            continue
        k = _band_index(lat, lat_nodes)
        lo, hi = lat_nodes[k], lat_nodes[k + 1]
        # zone (anticyclonic) where u increases poleward: north ∂u/∂φ > 0, south ∂u/∂φ < 0
        du = u[k + 1] - u[k]
        zone = du * (1 if (lo + hi) > 0 else -1) > 0
        if zone == anticyclone:
            centre = 0.5 * (lo + hi) + rng.normal(0, 0.12) * (hi - lo)
            return float(centre), float(hi - lo)
    return None


CLASS_LATITUDE_PREFERENCE = {
    "white": lambda a: 1.0 if a > 40 else 0.25,
    "dark": lambda a: 1.0 if a > 50 else 0.1,
    "brown": lambda a: 1.0 if 30 <= a <= 50 else 0.15,
}


def _vortex_population(state: PlanetState, seed: int, ld: float, lat_nodes, u, ice: bool) -> dict:
    """Return the vortex table: positions, semi-axes (m), classes."""
    rng = named_rng(seed, "clouds.vortices")
    a_p = state.bulk.radius_m
    area = 4 * math.pi * a_p**2
    density = h.VORTEX_DENSITY_PER_LD2 * (h.VORTEX_DENSITY_ICE_FRACTION if ice else 1.0)
    n = int(rng.poisson(density * area / ld**2))
    out = {"lat_deg": [], "lon_deg": [], "a_m": [], "b_m": [], "kind": []}

    def add(lat, lon, a, b, kind):
        """Append one vortex."""
        out["lat_deg"].append(float(lat))
        out["lon_deg"].append(float(lon))
        out["a_m"].append(float(a))
        out["b_m"].append(float(b))
        out["kind"].append(kind)

    classes = list(h.VORTEX_CLASS_SHARES)
    shares = np.array([h.VORTEX_CLASS_SHARES[k] for k in classes])
    red_bands = rng.uniform(15, 60, size=3)          # red vortices gather at a few latitudes (Sankar et al. 2024)
    prefs = dict(CLASS_LATITUDE_PREFERENCE)
    prefs["red"] = lambda a: 1.0 if np.min(np.abs(red_bands - a)) < 4 else 0.1
    for _ in range(n):
        kind = "dark" if ice else str(rng.choice(classes, p=shares / shares.sum()))
        placed = _place(rng, lat_nodes, u, ANTICYCLONIC[kind] > 0, prefs.get(kind, lambda a: 1.0))
        if placed is None:
            continue
        lat, band_deg = placed
        size = ld * h.VORTEX_SIZE_MEDIAN_LD * math.exp(rng.normal(0, h.VORTEX_SIZE_SIGMA))
        size *= 0.5 + 0.5 * math.cos(math.radians(lat))           # smaller toward the poles (Sankar et al. 2024)
        aspect = math.exp(rng.normal(math.log(h.VORTEX_ASPECT_MEDIAN), h.VORTEX_ASPECT_SIGMA))
        b = min(size / aspect, h.VORTEX_MAX_BAND_FRACTION * math.radians(band_deg) * a_p)
        add(lat, rng.uniform(-180, 180), max(b * aspect, b), b, kind)

    if ice:
        for _ in range(h.ICE_BRIGHT_CLOUDS):
            lat = rng.choice([-1, 1]) * rng.uniform(20, 60)
            b = ld * rng.uniform(0.3, 0.8)
            add(lat, rng.uniform(-180, 180), b * rng.uniform(*h.ICE_BRIGHT_ASPECT), b, "bright")
        if rng.uniform() < h.GREAT_DARK_SPOT_PROBABILITY:
            lat = rng.choice([-1, 1]) * rng.uniform(15, 40)
            lon = rng.uniform(-180, 180)
            b = h.GREAT_DARK_SPOT_LD * ld
            add(lat, lon, 2 * b, b, "dark")
            add(lat + np.sign(lat) * math.degrees(1.15 * b / a_p), lon, 1.5 * b, 0.3 * b, "bright")   # companion cloud
    elif rng.uniform() < h.GREAT_OVAL_PROBABILITY:
        for _ in range(30):
            lat = rng.choice([-1, 1]) * rng.uniform(*h.GREAT_OVAL_LATITUDE_DEG)
            k = _band_index(lat, lat_nodes)
            lo, hi = lat_nodes[k], lat_nodes[k + 1]
            if (u[k + 1] - u[k]) * (1 if (lo + hi) > 0 else -1) > 0:          # a zone
                b = 0.5 * h.GREAT_OVAL_WIDTH_FRACTION * math.radians(hi - lo) * a_p
                e = rng.uniform(*h.GREAT_OVAL_ECCENTRICITY)
                add(0.5 * (lo + hi), rng.uniform(-180, 180), b / math.sqrt(1 - e**2), b, "great")
                break
    return out


def _polar_features(state: PlanetState, seed: int, ld: float, lat_nodes, u, vortices: dict) -> list[dict]:
    """Add a polar cyclone to each pole, with a ring of circumpolar cyclones or a hexagon; return the hexagons."""
    rng = named_rng(seed, "clouds.poles")
    a_p = state.bulk.radius_m
    r_max = h.POLAR_CYCLONE_RMAX_LD * ld
    visible = h.POLAR_CYCLONE_VISIBLE * r_max
    hexagons = []
    for sign in (1, -1):
        jets = [abs(lat) for lat, v in zip(lat_nodes, u) if sign * lat > 0 and v > 0
                and h.HEXAGON_JET_LATITUDE_DEG[0] <= abs(lat) <= h.HEXAGON_JET_LATITUDE_DEG[1]]
        if jets and rng.uniform() < h.HEXAGON_PROBABILITY:
            hexagons.append({"lat_deg": float(sign * jets[0]), "rotation_deg": float(rng.uniform(0, 60))})
            # a large polar cyclone and no ring, as on Saturn (Gavriel & Kaspi 2021)
            vortices_add(vortices, sign * 89.9, 0.0, 2 * visible, 2 * visible, "polar")
            continue
        vortices_add(vortices, sign * 89.9, 0.0, visible, visible, "polar")
        if rng.uniform() < h.POLAR_CYCLONE_RING_PROBABILITY:
            colat = math.radians(90.0 - h.POLAR_RING_LATITUDE_DEG)
            n = int(2 * math.pi * a_p * math.sin(colat) / (h.POLAR_RING_SPACING * r_max))
            start = rng.uniform(0, 360)
            for k in range(n):
                vortices_add(vortices, sign * h.POLAR_RING_LATITUDE_DEG, start + 360.0 * k / n, visible, visible,
                             "polar")
    return hexagons


def vortices_add(vortices: dict, lat, lon, a, b, kind) -> None:
    """Append one vortex to a vortex table."""
    for key, value in (("lat_deg", lat), ("lon_deg", (lon + 180) % 360 - 180), ("a_m", a), ("b_m", b),
                       ("kind", kind)):
        vortices.setdefault(key, []).append(value if key == "kind" else float(value))


def build_cloud_model(state: PlanetState) -> CloudModel:
    """Return the cloud-top model of a giant planet from its state (its appearance must be set)."""
    ap = state.appearance
    if ap is None:
        raise ValueError(f"{state.name} is not a giant planet")
    seed = state.draw_seed
    ice = is_ice_like(state.bulk)
    ld = deformation_radius(state)
    zone, belt = _rgb(ap.zone_colour), _rgb(ap.belt_colour)
    model = CloudModel(seed=seed, radius_m=state.bulk.radius_m, regime=ap.regime, ice_like=ice,
                       effective_temperature_k=ap.effective_temperature_k, scale_height_m=_scale_height(state),
                       deformation_radius_m=ld, zone_rgb=zone.tolist(), belt_rgb=belt.tolist(),
                       jet_half_width_deg=ap.jet_half_width_deg)
    model.vortex_rgb = {k: np.asarray(v).tolist() for k, v in _vortex_colours(state, ice, zone, belt).items()}
    if ap.regime == "locked":
        t_eq = state.atmosphere.equilibrium_temperature_k
        lo, hi = h.DAY_NIGHT_T_EQ_K
        x = np.clip((t_eq - lo) / (hi - lo), 0.0, 1.0)
        model.day_night_k = float(h.DAY_NIGHT_WARM_K + x * (h.DAY_NIGHT_HOT_K - h.DAY_NIGHT_WARM_K))
        model.hot_spot_offset_deg = ap.hot_spot_offset_deg
        t = np.linspace(max(ap.effective_temperature_k - model.day_night_k, 30.0),
                        ap.effective_temperature_k + model.day_night_k, 48)
        model.locked_temperatures_k = t.tolist()
        model.locked_rgb = class_at_temperature_rgb(t, state.bulk.surface_gravity_m_s2, ice,
                                                    state.star.effective_temperature_k,
                                                    ap.atmospheric_metallicity,
                                                    zone if ap.cloud_deck else None).tolist()
        model.wind_lat_deg = [-90.0, -ap.jet_half_width_deg, 0.0, ap.jet_half_width_deg, 90.0]
        model.wind_m_s = [0.0, 0.0, h.WIND_LOCKED_M_S, 0.0, 0.0]
        model.band_contrast = [0.0] * 4
        for sign in (1, -1):
            r = math.radians(h.POLAR_VORTEX_LOCKED_DEG) * state.bulk.radius_m
            vortices_add(model.vortices, sign * 89.9, 0.0, r, r, "white")
        return model

    cores = jet_latitudes(state.bulk.radius_m, state.orbit.rotation_period_s, ap.jet_speed_m_s)
    u_eq, u_jet = _equatorial_wind(state, ice)
    lat_nodes, u = wind_nodes(cores, u_eq, u_jet, double_peak=(not ice and u_eq > 0))
    model.wind_lat_deg, model.wind_m_s = lat_nodes.tolist(), u.tolist()
    rng = named_rng(seed, "clouds.bands")
    model.band_contrast = rng.uniform(*h.BAND_CONTRAST_SPREAD, size=len(lat_nodes) - 1).tolist()
    model.vortices = _vortex_population(state, seed, ld, lat_nodes, u, ice)
    if not ice:
        model.hexagons = _polar_features(state, seed, ld, lat_nodes, u, model.vortices)
    return model


# --- Evaluating the model --------------------------------------------------------

def _lat_lon(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude and longitude (radians) of unit vectors."""
    return np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)), np.arctan2(vectors[:, 1], vectors[:, 0])


def _hexagon_latitude(lat: np.ndarray, lon: np.ndarray, hexagons: list[dict]) -> np.ndarray:
    """Return latitudes distorted so that circles of latitude near a hexagon's jet become hexagons."""
    out = lat.copy()
    for hx in hexagons:
        jet = math.radians(abs(hx["lat_deg"]))
        sign = 1.0 if hx["lat_deg"] > 0 else -1.0
        polar = sign * lat > math.radians(60.0)
        if not polar.any():
            continue
        colat = math.pi / 2 - sign * lat[polar]
        theta = (np.degrees(lon[polar]) - hx["rotation_deg"]) % 60.0 - 30.0
        # radius of the hexagon edge relative to its inscribed circle, so the jet's circle becomes a hexagon
        stretch = np.cos(np.radians(theta)) / math.cos(math.radians(30.0))
        weight = np.clip((sign * lat[polar] - math.radians(60.0)) / (jet - math.radians(60.0)), 0.0, 1.0)
        colat = colat * (1 + (stretch - 1) * weight)
        out[polar] = sign * (math.pi / 2 - colat)
    return out


def _polar_banding(model: CloudModel, lat: np.ndarray) -> np.ndarray:
    """Return the weight of banding (1 to 0) fading into the chaotic polar regions, kept up to any hexagon."""
    lo, hi = h.POLAR_CHAOS_DEG
    weight = np.ones_like(lat)
    for sign in (1.0, -1.0):
        edge = [abs(hx["lat_deg"]) for hx in model.hexagons if hx["lat_deg"] * sign > 0]
        start, end = (edge[0] + 2.0, edge[0] + 6.0) if edge else (lo, hi)
        side = sign * lat > 0
        a = np.degrees(np.abs(lat[side]))
        weight[side] = np.clip((end - a) / (end - start), 0.0, 1.0)
    return weight


def _zonal_profile(model: CloudModel, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the zonal wind, the band phase (+1 zone centre, −1 belt centre) and the band index at latitudes."""
    nodes = np.asarray(model.wind_lat_deg)
    speeds = np.asarray(model.wind_m_s)
    k = np.clip(np.searchsorted(nodes, lat_deg) - 1, 0, len(nodes) - 2)
    lo, hi = nodes[k], nodes[k + 1]
    t = np.clip((lat_deg - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)              # winds peak at the nodes and turn smoothly between them
    u = speeds[k] + (speeds[k + 1] - speeds[k]) * ease
    du = speeds[k + 1] - speeds[k]
    sense = np.where(lat_deg >= 0, 1.0, -1.0)
    span = np.max(np.abs(np.diff(speeds))) or 1.0
    phase = np.sign(du) * sense * np.sin(np.pi * t) * np.minimum(np.abs(du) / span * 4, 1.0)
    contrast = np.asarray(model.band_contrast)[k] if model.band_contrast else np.ones_like(t)
    return u, phase * contrast, k


def _vortex_field(model: CloudModel, vectors: np.ndarray, lat: np.ndarray,
                  lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each point's vortex weight (0–1), its rim weight (0–1) and the vortex index (−1 outside any).

    Ovals are laid out in longitude and latitude so elongated clouds follow
    the flow; polar vortices are round in the plane tangent to the pole.
    """
    v = model.vortices
    n = len(v.get("lat_deg", []))
    weight = np.zeros(len(vectors))
    rim = np.zeros(len(vectors))
    index = np.full(len(vectors), -1, dtype=np.int32)
    if n == 0:
        return weight, rim, index
    lat_c, lon_c = np.radians(v["lat_deg"]), np.radians(v["lon_deg"])
    a, b = np.asarray(v["a_m"]) / model.radius_m, np.asarray(v["b_m"]) / model.radius_m
    centres = np.column_stack([np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)])
    tree = cKDTree(vectors)
    for i in range(n):
        idx = tree.query_ball_point(centres[i], min(1.4 * a[i], 1.9))
        if not idx:
            continue
        idx = np.asarray(idx)
        if abs(v["lat_deg"][i]) > 89.0:
            r = np.linalg.norm(vectors[idx] - centres[i], axis=1) / a[i]
        else:
            dlon = (lon[idx] - lon_c[i] + np.pi) % (2 * np.pi) - np.pi
            x = dlon * np.cos(lat_c[i])
            y = lat[idx] - lat_c[i]
            r = np.hypot(x / a[i], y / b[i])
        w = 1.0 / (1.0 + np.exp((r - 1.0) / 0.1))
        ring = np.exp(-((r - 1.0) / 0.12) ** 2)
        stronger = w > weight[idx]
        weight[idx[stronger]] = w[stronger]
        rim[idx] = np.maximum(rim[idx], ring)
        index[idx[stronger & (w > 0.5)]] = i
    return weight, rim, index


def evaluate(model: CloudModel, vectors: np.ndarray, fields=FIELDS) -> dict[str, np.ndarray]:
    """Return the requested fields of the cloud tops at unit vectors.

    ``colour`` is sRGB (n, 3); ``band`` +1 at zone centres and −1 at belt
    centres; ``zonal_wind`` m/s (eastward positive); ``relief`` m above the
    mean cloud deck; ``temperature`` K in the upper troposphere; ``vortex`` the
    index of the vortex at each point (−1 outside any).
    """
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) > CHUNK:
        parts = [evaluate(model, vectors[s:s + CHUNK], fields) for s in range(0, len(vectors), CHUNK)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    lat, lon = _lat_lon(vectors)
    out: dict[str, np.ndarray] = {}
    if model.regime == "locked":
        return _evaluate_locked(model, vectors, lat, lon, fields)

    # band edges warped by noise stretched along longitude, with turbulence toward the poles
    stretch = vectors * np.array([1.0, 1.0, 6.0])
    stretch /= np.linalg.norm(stretch, axis=1, keepdims=True)
    wave = fbm(stretch, model.seed, "clouds.wave", 3.0, h.CLOUD_NOISE_OCTAVES)
    nodes = np.asarray(model.wind_lat_deg)
    spacing = np.interp(np.degrees(np.abs(lat)), np.abs(nodes[nodes >= 0]),
                        np.gradient(np.abs(nodes[nodes >= 0]))) if (nodes >= 0).sum() > 1 else 10.0
    lat_w = lat + np.radians(h.BAND_EDGE_WARP * spacing) * wave
    turb = np.clip((np.degrees(np.abs(lat)) - h.BAND_TURBULENCE_FROM_DEG) / (90 - h.BAND_TURBULENCE_FROM_DEG), 0, 1)
    poleward = turb > 0
    if poleward.any():
        swirl = fbm(vectors[poleward], model.seed, "clouds.turbulence", 8.0, h.CLOUD_NOISE_OCTAVES)
        lat_w[poleward] += h.BAND_TURBULENCE_WARP * turb[poleward] * swirl
    lat_w = _hexagon_latitude(lat_w, lon, model.hexagons)
    u, phase, _ = _zonal_profile(model, np.degrees(lat_w))
    if not model.ice_like:
        phase = phase * _polar_banding(model, lat)
    weight, rim, index = _vortex_field(model, vectors, lat, lon)
    kinds = model.vortices.get("kind", [])
    sense = np.array([ANTICYCLONIC[k] for k in kinds]) if kinds else np.zeros(0)

    if "band" in fields:
        out["band"] = phase
    if "zonal_wind" in fields:
        out["zonal_wind"] = u
    if "temperature" in fields:
        out["temperature"] = model.effective_temperature_k - 0.5 * h.BAND_TEMPERATURE_CONTRAST_K * phase
    if "relief" in fields:
        relief = 0.5 * h.BAND_RELIEF_SCALE_HEIGHTS * model.scale_height_m * phase
        inside = index >= 0
        relief[inside] += h.VORTEX_RELIEF_SCALE_HEIGHTS * model.scale_height_m * sense[index[inside]] * weight[inside]
        out["relief"] = relief
    if "vortex" in fields:
        out["vortex"] = index
    if "colour" in fields:
        zone, belt = np.asarray(model.zone_rgb), np.asarray(model.belt_rgb)
        mix = (0.5 + 0.5 * phase)[:, None]
        rgb = belt + (zone - belt) * mix
        if np.allclose(zone, belt):
            rgb = rgb * (1 + h.BAND_BRIGHTNESS_CONTRAST * phase)[:, None]
        if kinds:
            table = np.array([model.vortex_rgb[k] for k in kinds])
            inside = index >= 0
            w = 0.9 * weight[inside][:, None]
            rgb[inside] = rgb[inside] * (1 - w) + table[index[inside]] * w
            rgb *= (1 - h.VORTEX_RIM_DARKENING * rim)[:, None]
        grain = fbm(vectors, model.seed, "clouds.grain", 24.0, h.CLOUD_NOISE_OCTAVES)
        rgb = rgb * (1 + 0.06 * grain)[:, None]
        out["colour"] = np.clip(rgb, 0.0, 1.0)
    return out


def _evaluate_locked(model: CloudModel, vectors, lat, lon, fields) -> dict[str, np.ndarray]:
    """Return the fields of a tidally locked giant: day–night temperatures, one equatorial jet, polar vortices."""
    out: dict[str, np.ndarray] = {}
    offset = math.radians(model.hot_spot_offset_deg)
    day = np.cos(lat) * np.cos(lon - offset)
    swirl = fbm(vectors * np.array([1.0, 1.0, 3.0]), model.seed, "clouds.locked", 3.0, h.CLOUD_NOISE_OCTAVES)
    temperature = model.effective_temperature_k + 0.5 * model.day_night_k * (day + 0.15 * swirl)
    jet = np.exp(-(np.degrees(lat) / max(model.jet_half_width_deg, 1.0)) ** 2)
    index = _vortex_field(model, vectors, lat, lon)[2] if "vortex" in fields else None
    # anticyclonic polar vortices (Skinner & Wei 2025), soft-edged
    polar = np.exp(-((90.0 - np.degrees(np.abs(lat))) / h.POLAR_VORTEX_LOCKED_DEG) ** 2)
    if "temperature" in fields:
        out["temperature"] = temperature
    if "zonal_wind" in fields:
        out["zonal_wind"] = h.WIND_LOCKED_M_S * jet
    if "band" in fields:
        out["band"] = 2 * jet - 1
    if "relief" in fields:
        out["relief"] = 0.5 * h.BAND_RELIEF_SCALE_HEIGHTS * model.scale_height_m * day
    if "vortex" in fields:
        out["vortex"] = index
    if "colour" in fields:
        table_t = np.asarray(model.locked_temperatures_k)
        table = np.asarray(model.locked_rgb)
        rgb = np.column_stack([np.interp(temperature, table_t, table[:, j]) for j in range(3)])
        rgb *= (1 + 0.06 * swirl)[:, None]
        rgb *= (1 - 0.1 * polar)[:, None]
        out["colour"] = np.clip(rgb, 0.0, 1.0)
    return out


# --- The cloud-top surface on the global grid -------------------------------------

def cloud_surface(state: PlanetState, resolution: int | str = "standard",
                  model: Optional[CloudModel] = None) -> "xr.Dataset":
    """Return a giant's cloud tops on the global grid, with the model kept in the attributes.

    ``elevation`` is the cloud-top relief, so meshes and the globe work as
    for solid planets; there is no ocean. Maps, rasters and meshes evaluate
    the model (``cloud_model(surface)``) at their own points.
    """
    import xarray as xr

    from ..grid import build_grid, resolve_resolution

    model = model if model is not None else build_cloud_model(state)
    grid = build_grid(resolve_resolution(resolution))
    f = evaluate(model, grid.points)
    spacing_km = grid.mean_spacing(state.bulk.radius_m) / 1e3
    attrs = {"grid": "fibonacci", "grid_size": grid.size, "radius_m": state.bulk.radius_m, "kind": "cloud_tops",
             "planet": state.name, "seed": state.seed, "draw_seed": state.draw_seed, "regime": "fluid",
             "relief_factor": 1.0, "has_ocean": 0, "frozen_ocean": 0, "land_fraction": 1.0, "spacing_km": spacing_km,
             "degrees_per_cell": math.degrees(spacing_km / (state.bulk.radius_m / 1e3)), "tectonics": "none",
             "cloud_model": model.to_json()}
    return xr.Dataset(
        data_vars={
            "elevation": ("cell", f["relief"].astype(np.float32),
                          {"units": "m", "long_name": "cloud-top height above the mean cloud deck"}),
            "ocean": ("cell", np.zeros(grid.size, dtype=bool), {"long_name": "no ocean on a giant"}),
            "cloud_colour": (("vec", "cell"), f["colour"].T.astype(np.float32),
                             {"long_name": "cloud-top colour under the star (sRGB, 0–1)"}),
            "band": ("cell", f["band"].astype(np.float32),
                     {"long_name": "band phase: +1 at zone centres (anticyclonic), −1 at belt centres"}),
            "zonal_wind": ("cell", f["zonal_wind"].astype(np.float32),
                           {"units": "m/s", "long_name": "zonal wind at the cloud tops (eastward positive)"}),
            "cloud_temperature": ("cell", f["temperature"].astype(np.float32),
                                  {"units": "K", "long_name": "upper-tropospheric temperature"}),
            "vortex": ("cell", f["vortex"].astype(np.int32), {"long_name": "vortex index (-1: none)"}),
        },
        coords={
            "lat": ("cell", grid.lat.astype(np.float32), {"units": "degrees_north"}),
            "lon": ("cell", grid.lon.astype(np.float32), {"units": "degrees_east"}),
            "vec": ("vec", np.arange(3, dtype=np.int8), {"long_name": "colour component (r, g, b)"}),
        },
        attrs=attrs,
    )


def is_cloud_surface(surface) -> bool:
    """Return whether a surface dataset holds a giant's cloud tops."""
    return surface is not None and surface.attrs.get("kind") == "cloud_tops"


def cloud_model(surface) -> CloudModel:
    """Return the cloud model kept with a cloud-top surface (cached on the dataset)."""
    cached = surface.attrs.get("_model")
    if isinstance(cached, CloudModel):
        return cached
    return CloudModel.from_json(surface.attrs["cloud_model"])
