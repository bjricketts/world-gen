"""Turn a spec into concrete input values, drawing whatever the user left open."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .. import bulk as bulk_model
from .. import orbit as orbit_model
from .. import star as star_model
from ..spec import PlanetSpec
from ..util import named_rng
from .archetypes import ARCHETYPES, BASE_PRIORS, get_archetype
from .distributions import HZFraction, LogUniform, Uniform

# Fields whose [low, high] ranges are drawn log-uniformly.
LOG_SCALE_FIELDS = frozenset({
    "star.mass_msun",
    "orbit.semi_major_axis_au",
    "orbit.instellation_earth",
    "orbit.rotation_period_h",
    "body.mass_mearth",
    "body.radius_rearth",
    "body.water_mass_fraction",
    "body.radiogenic_abundance",
    "body.tidal_heating_w",
    "body.tidal_q",
    "atmosphere.surface_pressure_bar",
    "atmosphere.volatile_richness",
})

# Oldest allowed draw as a fraction of the star's main-sequence lifetime.
MAX_AGE_FRACTION = 0.9


@dataclass
class ResolvedInputs:
    """Concrete inputs for one planet. ``None`` values are left for physics to derive."""

    values: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    archetype: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def get(self, path: str, default: Any = None) -> Any:
        """Return a resolved value, or ``default`` if unset."""
        v = self.values.get(path)
        return default if v is None else v

    def is_user(self, path: str) -> bool:
        """Return whether the user fixed or constrained this value."""
        return self.provenance.get(path) in ("user", "user_range")


def _spec_values(spec: PlanetSpec) -> dict[str, Any]:
    """Return every spec field as a flat dict keyed by 'section.field'."""
    out = {}
    for section in ("star", "orbit", "body", "interior", "atmosphere", "surface", "biosphere"):
        for name, value in getattr(spec, section).model_dump().items():
            out[f"{section}.{name}"] = value
    return out


def compatible_archetypes(user: dict[str, Any]) -> list[str]:
    """Return the archetypes whose priors do not contradict any user-fixed value or range."""
    names = []
    for name, archetype in ARCHETYPES.items():
        ok = True
        for path, dist in archetype.priors.items():
            value = user.get(path)
            if value is None:
                continue
            low, high = value if isinstance(value, tuple) else (value, value)
            if not dist.allows(low, high):
                ok = False
                break
        if ok:
            names.append(name)
    return names


def _pick_archetype(spec: PlanetSpec, user: dict[str, Any]) -> Optional[str]:
    """Return the requested archetype, or draw one compatible with the user's values (``None`` if none is)."""
    if spec.priors.archetype:
        return get_archetype(spec.priors.archetype).name
    names = compatible_archetypes(user)
    if not names:
        return None
    weights = [ARCHETYPES[n].mix_weight for n in names]
    rng = named_rng(spec.seed, "archetype")
    total = sum(weights)
    return names[int(rng.choice(len(names), p=[w / total for w in weights]))]


def _skip_prior(path: str, user: dict[str, Any]) -> bool:
    """Return whether a field must be left for physics to derive given what the user fixed."""
    def has(p: str) -> bool:
        """Return whether the user set this field."""
        return user.get(p) is not None

    if path == "orbit.instellation_earth":
        return has("orbit.semi_major_axis_au")
    if path == "body.mass_mearth":
        return has("body.radius_rearth")
    if path == "body.core_mass_fraction":
        return has("body.mass_mearth") and has("body.radius_rearth")
    if path == "atmosphere.surface_pressure_bar":
        return has("atmosphere.present") and not user["atmosphere.present"]
    if path == "atmosphere.composition":
        return has("atmosphere.present") and not user["atmosphere.present"]
    return False


def resolve(spec: PlanetSpec, seed: Optional[int] = None) -> ResolvedInputs:
    """Return concrete inputs: user values, draws from user ranges, then archetype and base priors.

    ``seed`` overrides the spec's seed for all draws (used when redrawing to
    satisfy archetype constraints).
    """
    if seed is not None and seed != spec.seed:
        spec = spec.model_copy(update={"seed": seed})
    user = _spec_values(spec)
    archetype = _pick_archetype(spec, user)
    archetype_priors = ARCHETYPES[archetype].priors if archetype else {}
    priors = {**BASE_PRIORS, **archetype_priors}
    result = ResolvedInputs(archetype=archetype)
    if archetype is None:
        result.notes.append("no archetype is compatible with the fixed values; unset values use broad base priors")

    for path, value in user.items():
        if isinstance(value, tuple):
            lo, hi = value
            dist = LogUniform(lo, hi) if path in LOG_SCALE_FIELDS and lo > 0 else Uniform(lo, hi)
            result.values[path] = dist.sample(named_rng(spec.seed, path))
            result.provenance[path] = "user_range"
        elif value is not None:
            result.values[path] = value
            result.provenance[path] = "user"
        else:
            result.values[path] = None

    hz_position = None
    for path, dist in priors.items():
        if result.values.get(path) is not None or _skip_prior(path, user):
            continue
        value = dist.sample(named_rng(spec.seed, path))
        if isinstance(dist, HZFraction):
            hz_position = value
            value = None       # converted to instellation below
        result.values[path] = value
        result.provenance[path] = "archetype" if path in archetype_priors else "prior"

    _clip_age(spec, result)
    if hz_position is not None:
        result.values["orbit.instellation_earth"] = _instellation_from_hz(result, hz_position)
    return result


def _estimated_mass_mearth(result: ResolvedInputs) -> float:
    """Return the planet mass to use for habitable-zone limits before the bulk model runs."""
    mass = result.get("body.mass_mearth")
    if mass is not None:
        return mass
    radius = result.get("body.radius_rearth")
    if radius is not None:
        return bulk_model.rocky_mass(radius, result.get("body.core_mass_fraction", bulk_model.default_cmf()))
    return 1.0


def _instellation_from_hz(result: ResolvedInputs, fraction: float) -> float:
    """Return the instellation for a drawn habitable-zone position around the drawn star."""
    star, _ = star_model.build_star(result.get("star.mass_msun"), result.get("star.age_gyr"),
                                    result.get("star.metallicity_feh", 0.0))
    return orbit_model.instellation_at_hz_fraction(
        star.luminosity_w, star.effective_temperature_k, _estimated_mass_mearth(result), fraction)


def _clip_age(spec: PlanetSpec, result: ResolvedInputs) -> None:
    """Keep drawn ages within the star's main-sequence lifetime."""
    if result.is_user("star.age_gyr"):
        return
    lifetime = star_model.main_sequence_lifetime_gyr(result.values["star.mass_msun"])
    limit = MAX_AGE_FRACTION * lifetime
    if result.values["star.age_gyr"] > limit:
        result.values["star.age_gyr"] = named_rng(spec.seed, "star.age_clip").uniform(0.1, 1.0) * limit
