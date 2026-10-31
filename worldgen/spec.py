"""User-facing planet specification: what the user fixes, constrains or leaves open.

Each numeric parameter can be:
  * a number: fixed value,
  * a two-element list ``[low, high]``: drawn from that range,
  * omitted / ``null``: drawn from the archetype prior or derived from physics.

Units in the spec are user-friendly (solar masses, AU, Earth masses, bar,
hours, degrees); they are converted to SI when the planet is built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .culture.spec import CultureSpec

from .spec_base import Param, Range, Section as _Section  # noqa: F401

TectonicRegime = Literal["mobile_lid", "stagnant_lid", "episodic", "heat_pipe", "inactive", "fluid"]
Composition = Literal["n2_o2", "n2_co2", "co2", "n2_ch4", "h2_he", "none"]
SPEC_VERSION = 1                 # version of the spec file format this release reads
Mode = Literal["snapshot", "history"]
TectonicsMode = Literal["simulated", "heuristic"]
TectonicStart = Literal["supercontinent", "cratons"]
LifeMode = Literal["surface", "ocean", "subsurface", "none"]
Biochemistry = Literal["oxygenic", "methanogenic", "alien"]
AlienProduct = Literal["o2", "ch4", "none"]


class StarSpec(_Section):
    """Host star parameters."""

    mass_msun: Param = None
    age_gyr: Param = None
    metallicity_feh: Param = None
    activity_percentile: Param = None    # rotation percentile at birth (0 slow, 1 fast); sets the XUV history


class OrbitSpec(_Section):
    """Orbit and spin. Give either semi-major axis or instellation, not both."""

    semi_major_axis_au: Param = None
    instellation_earth: Param = None
    eccentricity: Param = None
    obliquity_deg: Param = None
    rotation_period_h: Param = None
    periapsis_longitude_deg: Param = None   # season at periapsis, as solar longitude (Earth ~283)

    @model_validator(mode="after")
    def _one_distance(self):
        """Reject specs that give both a semi-major axis and an instellation."""
        if self.semi_major_axis_au is not None and self.instellation_earth is not None:
            raise ValueError("give either semi_major_axis_au or instellation_earth, not both")
        return self


class BodySpec(_Section):
    """Bulk properties of the planet. Any two of mass, radius and composition fix the third."""

    mass_mearth: Param = None
    radius_rearth: Param = None
    core_mass_fraction: Param = None
    water_mass_fraction: Param = None    # all water, including what the mantle holds (Earth ~0.0006)
    envelope_mass_fraction: Param = None
    radiogenic_abundance: Param = None   # heat-producing elements relative to Earth
    tidal_heating_w: Param = None
    tidal_q: Param = None                # tidal dissipation quality factor
    atmospheric_metallicity: Param = None   # giants: C/H of the envelope relative to protosolar


class InteriorSpec(_Section):
    """Overrides for interior outcomes that are otherwise drawn from heuristics."""

    tectonic_regime: Optional[TectonicRegime] = None
    magnetic_field: Optional[bool] = None


class AtmosphereSpec(_Section):
    """Atmosphere parameters and overrides."""

    present: Optional[bool] = None
    surface_pressure_bar: Param = None
    composition: Optional[Composition] = None
    bond_albedo: Param = None
    surface_temperature_k: Param = None
    volatile_richness: Param = None      # volatile inventory relative to Earth
    weathering_target_k: Param = None    # temperature the carbonate-silicate cycle settles at


class SurfaceSpec(_Section):
    """Surface targets used when building maps."""

    land_fraction: Param = None          # target land fraction; sets the surface water when given
    tectonics: Optional[TectonicsMode] = None      # plate simulation, or the fast heuristic layout
    tectonics_start: Optional[TectonicStart] = None  # initial continents of the simulation
    tectonics_duration_myr: Param = None           # simulated time (Myr), limited by the planet's age


class BiosphereSpec(_Section):
    """Life on the planet. Unset values follow rules from the planet's water, temperature and age."""

    life: Optional[LifeMode] = None               # surface (land and sea), ocean only, subsurface, or none
    biochemistry: Optional[Biochemistry] = None   # oxygenic photosynthesis (default), methanogenic, or alien
    alien_product: Optional[AlienProduct] = None  # gas an alien biosphere releases (default none)
    age_gyr: Param = None                         # time since life began (default: planet age − 0.7 Gyr)
    optimum_temperature_k: Param = None           # growth optimum (Earth-like 298 K)
    temperature_tolerance_k: Param = None         # growth stops this far above the optimum (Earth-like 25 K)
    pigment_absorption_nm: Param = None           # peak absorption of the main pigment (default from the star)
    oxygen_fraction: Param = None                 # O2 mole fraction of the air (default from biosphere age)


class HistorySpec(_Section):
    """Initial conditions and poorly constrained rates for history mode. Unset values are drawn."""

    epochs_gyr: Optional[list[float]] = None      # extra epochs at which full states are stored
    initial_water_mass_fraction: Param = None     # all water at formation (default: body.water_mass_fraction)
    carbon_inventory: Param = None                # total carbon relative to Earth's, per unit planet mass
    nitrogen_inventory: Param = None              # atmospheric nitrogen relative to Earth's, per unit planet mass
    initial_mantle_temperature_k: Param = None    # mantle potential temperature at formation
    outgassing_efficiency: Param = None           # volcanic degassing relative to Earth's at the same melt rate
    weathering_efficiency: Param = None           # silicate weathering relative to Earth's at the same climate
    weathering_temperature_scale_k: Param = None  # e-folding temperature of weathering kinetics
    weathering_co2_exponent: Param = None         # weathering ∝ pCO₂^exponent
    biotic_weathering_factor: Param = None        # speed-up of land weathering by a full land biosphere
    escape_efficiency: Param = None               # heating efficiency of XUV-driven escape
    mantle_activation_energy_kj: Param = None     # activation energy of mantle viscosity
    core_adiabatic_heat_flow: Param = None        # heat the core conducts along its adiabat, relative to Earth's
    life_origin_delay_gyr: Param = None           # time with habitable conditions before life appears
    oxygen_burial_efficiency: Param = None        # organic burial (O₂ source) relative to Earth's
    reductant_decay_gyr: Param = None             # e-folding time of the volcanic reductant flux (mantle oxidation)
    land_colonisation_delay_gyr: Param = None     # time from the origin of surface life to life on land

    @field_validator("epochs_gyr")
    @classmethod
    def _positive_epochs(cls, v):
        """Reject negative epochs."""
        if v is not None and any(t < 0 for t in v):
            raise ValueError("epochs must be non-negative")
        return v


class ParentSpec(_Section):
    """The planet a moon orbits, and the moon's orbit about it. Present only in a moon's spec.

    The ``orbit`` section then describes the parent's orbit about the star.
    """

    name: Optional[str] = None
    mass_mearth: Param = None
    radius_rearth: Param = None          # default: Chen & Kipping mean radius for the mass
    rotation_period_h: Param = None
    obliquity_deg: Param = None          # the parent's; the moon orbits in its equatorial plane
    bond_albedo: Param = None
    tidal_q: Param = None
    love_number_k2: Param = None
    semi_major_axis_rp: Param = None     # the moon's orbit, in parent radii
    eccentricity: Param = None
    inclination_deg: Param = None        # to the parent's equator
    effective_temperature_k: Param = None   # the parent's glow; default: its equilibrium temperature
    magnetic_moment_am2: Param = None    # the parent's dipole; default: by mass from the Solar System planets
    moon_origin: Optional[Literal["regular", "captured", "impact"]] = None   # how the moon came to orbit it


class MoonEntry(BaseModel):
    """A moon listed by the user in a planet's system."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    mass_mearth: float
    radius_rearth: Optional[float] = None
    water_mass_fraction: float = 0.0
    semi_major_axis_rp: float
    eccentricity: float = 0.0
    inclination_deg: float = 0.0
    retrograde: bool = False


class RingEntry(BaseModel):
    """A ring listed by the user in a planet's system."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    inner_radius_rp: float
    outer_radius_rp: float
    optical_depth: float = 0.5
    bond_albedo: float = 0.5
    kind: Literal["icy", "polluted", "dark", "dusty"] = "icy"

    @model_validator(mode="after")
    def _ordered(self):
        """Require the inner edge inside the outer edge and outside the planet."""
        if not 1.0 < self.inner_radius_rp < self.outer_radius_rp:
            raise ValueError("a ring needs 1 < inner_radius_rp < outer_radius_rp")
        return self


RING_REGIMES = ("auto", "none", "broad", "narrow", "dusty warm", "dusty cold")


class SystemSpec(_Section):
    """A planet's moons and rings. Moons left unset are drawn (an empty list means none); rings follow the regime or a list."""

    moons: Optional[list[MoonEntry]] = None
    rings: Union[Literal["auto", "none", "broad", "narrow", "dusty warm", "dusty cold"], list[RingEntry]] = "auto"


class PriorSpec(_Section):
    """How unset parameters are drawn."""

    archetype: Optional[str] = None
    occupiability_weights: Optional[dict[str, float]] = None
    enforce_constraints: bool = True     # redraw until the archetype's required outcomes hold
    max_attempts: int = 200

    @field_validator("max_attempts")
    @classmethod
    def _positive(cls, v: int) -> int:
        """Reject a maximum attempt count below one."""
        if v < 1:
            raise ValueError("max_attempts must be at least 1")
        return v


class PlanetSpec(BaseModel):
    """Complete specification of a planet to generate."""

    model_config = ConfigDict(extra="forbid")

    spec_version: int = 1
    name: str = "Unnamed"
    seed: int = 0
    mode: Mode = "snapshot"
    target_epoch_gyr: Optional[float] = None
    star: StarSpec = StarSpec()
    orbit: OrbitSpec = OrbitSpec()
    body: BodySpec = BodySpec()
    interior: InteriorSpec = InteriorSpec()
    atmosphere: AtmosphereSpec = AtmosphereSpec()
    surface: SurfaceSpec = SurfaceSpec()
    biosphere: BiosphereSpec = BiosphereSpec()
    history: HistorySpec = HistorySpec()
    priors: PriorSpec = PriorSpec()
    culture: Optional[CultureSpec] = None   # inhabitants placed after the planet is built (DESIGN 4.8)
    system: SystemSpec = SystemSpec()       # moons and rings (DESIGN 4.12)
    parent: Optional[ParentSpec] = None     # set when the body is a moon (DESIGN 4.12)

    @field_validator("spec_version")
    @classmethod
    def _known_spec_version(cls, v: int) -> int:
        """Reject spec versions this release does not read."""
        if v != SPEC_VERSION:
            raise ValueError(f"spec_version {v} is not supported by this release (it reads version {SPEC_VERSION})")
        return v

    @model_validator(mode="after")
    def _moon_needs_parent_orbit(self):
        """Reject moons listed around a moon (the parent's mass and orbit may come from a moon archetype)."""
        if self.parent is not None:
            if self.system.moons:
                raise ValueError("a moon cannot list moons of its own")
        return self


# Fields describing the planet's present state rather than its initial
# conditions. In history mode, user values for these are targets/overrides
# applied at the end of the evolution; in snapshot mode they are inputs.
STATE_FIELDS = frozenset(
    {
        "orbit.rotation_period_h",
        "body.water_mass_fraction",
        "interior.tectonic_regime",
        "interior.magnetic_field",
        "atmosphere.present",
        "atmosphere.surface_pressure_bar",
        "atmosphere.composition",
        "atmosphere.bond_albedo",
        "atmosphere.surface_temperature_k",
        "atmosphere.weathering_target_k",
        "surface.land_fraction",
        "biosphere.life",
        "biosphere.oxygen_fraction",
    }
)


def field_role(path: str, mode: Mode) -> str:
    """Return whether a spec field acts as an 'input' or a 'target' in the given mode."""
    if mode == "history" and path in STATE_FIELDS:
        return "target"
    return "input"


def load_spec(path: str | Path) -> PlanetSpec:
    """Read a planet specification from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PlanetSpec.model_validate(data)


def save_spec(spec: PlanetSpec, path: str | Path) -> None:
    """Write a planet specification to a YAML file, omitting unset fields."""
    data = spec.model_dump(mode="json", exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


SPEC_TEMPLATE = """\
# World generator planet specification.
# Each numeric value may be a number (fixed), a [low, high] range (drawn),
# or omitted (drawn from the archetype prior or derived from physics).

spec_version: 1
name: New World
seed: 42
mode: snapshot            # snapshot | history (integrated from formation)

priors:
  archetype: temperate    # see `worldgen archetypes`; omit for a weighted mix
  # occupiability_weights: {temperature: 1, gravity: 1, pressure: 1, water: 1, radiation: 1}
  # enforce_constraints: true     # redraw until the archetype's required outcomes hold
  # max_attempts: 200

star:
  mass_msun: 0.9          # solar masses
  # age_gyr: [3, 6]
  # metallicity_feh: 0.0
  # activity_percentile: 0.5      # birth rotation percentile: 0 slow, 1 fast (XUV history)

orbit:
  # semi_major_axis_au: 1.0   # or give instellation_earth instead
  instellation_earth: [0.8, 1.1]
  # eccentricity: 0.02
  # obliquity_deg: 23
  # rotation_period_h: 24
  # periapsis_longitude_deg: 283   # season of closest approach (solar longitude; 270 = northern winter solstice)

body:
  mass_mearth: 1.3
  # radius_rearth: 1.1
  # core_mass_fraction: 0.32
  # water_mass_fraction: 0.0006   # total water; Earth keeps ~1/3 at the surface
  # envelope_mass_fraction: 0.0
  # radiogenic_abundance: 1.0
  # tidal_heating_w: 0
  # tidal_q: 100                  # tidal dissipation; lower means faster tidal locking
  # atmospheric_metallicity: 10   # giants: envelope C/H relative to protosolar (default: mass trend)

interior: {}
  # tectonic_regime: mobile_lid   # mobile_lid | stagnant_lid | episodic | heat_pipe | inactive
  # magnetic_field: true

surface: {}
  # land_fraction: 0.3            # overrides the water: sea level is set to leave this much land
  # tectonics: simulated          # simulated | heuristic (fast, no history)
  # tectonics_start: cratons      # supercontinent | cratons
  # tectonics_duration_myr: 400   # simulated time, limited by the star's age

biosphere: {}
  # life: surface                 # surface | ocean | subsurface | none (default from water, temperature and age)
  # biochemistry: oxygenic        # oxygenic | methanogenic | alien
  # alien_product: none           # o2 | ch4 | none (alien biochemistry)
  # age_gyr: 3.8                  # time since life began
  # optimum_temperature_k: 298
  # temperature_tolerance_k: 25
  # pigment_absorption_nm: 680    # default: where the star's light peaks at the surface
  # oxygen_fraction: 0.21

history: {}                # used in history mode; unset values are drawn
  # epochs_gyr: [0.5, 1, 2, 3]    # states stored at these planet ages
  # initial_water_mass_fraction: 0.0007
  # carbon_inventory: 1.0         # relative to Earth, per unit planet mass
  # nitrogen_inventory: 1.0
  # initial_mantle_temperature_k: 2000
  # outgassing_efficiency: 1.0
  # weathering_efficiency: 1.0
  # weathering_temperature_scale_k: 20
  # weathering_co2_exponent: 0.45
  # biotic_weathering_factor: 4
  # escape_efficiency: 0.15
  # mantle_activation_energy_kj: 300
  # core_adiabatic_heat_flow: 1.0
  # life_origin_delay_gyr: 0.4
  # oxygen_burial_efficiency: 1.0
  # reductant_decay_gyr: 3.0
  # land_colonisation_delay_gyr: 3.4

atmosphere: {}
  # present: true
  # surface_pressure_bar: 1.2
  # composition: n2_o2            # n2_o2 | n2_co2 | co2 | n2_ch4 | h2_he | none
  # bond_albedo: 0.3
  # surface_temperature_k: 290
  # volatile_richness: 1.0
  # weathering_target_k: 288      # temperature CO2 regulation settles at (plate tectonics + water)

# system:                         # moons and rings; moons left unset are drawn (Jovian or Saturnian regular moons,
#                                 # ring-born moons, or an impact moon for rocky planets); moons: [] means none
#   moons:
#     - name: Selene
#       mass_mearth: 0.0123
#       radius_rearth: 0.273      # default: rocky mass–radius relation with the water fraction
#       water_mass_fraction: 0.0
#       semi_major_axis_rp: 60.3  # in planet radii
#       eccentricity: 0.055
#       inclination_deg: 5.1
#       retrograde: false
#   rings: auto                   # auto (from the ring temperature), none, broad, narrow, 'dusty warm',
#                                 # 'dusty cold', or a list:
#   # rings:
#   #   - name: Broad
#   #     inner_radius_rp: 1.3
#   #     outer_radius_rp: 2.2
#   #     optical_depth: 1.0
#   #     bond_albedo: 0.55
#   #     kind: icy                 # icy, polluted, dark or dusty

# parent:                         # makes this body a moon; 'orbit' is then the parent's orbit about the star
#   name: Jove
#   mass_mearth: 317.8
#   radius_rearth: 11.2           # default: Chen & Kipping radius for the mass
#   rotation_period_h: 9.9
#   obliquity_deg: 3.1            # the moon orbits in the parent's equatorial plane
#   semi_major_axis_rp: 9.6       # the moon's orbit, in parent radii
#   eccentricity: 0.009

# culture:                        # inhabitants, placed after the planet is built; can also be its own file
#   # seed: 7                     # redraw the culture alone (default: the planet seed)
#   # history:                    # for organic cultures: span and the epochs kept (years before the present)
#   #   start_years_ago: 12000
#   #   epochs: [5000, 2000, 500]
#   #   pace: 1.0                 # speeds up (>1) or slows (<1) the growth of knowledge
#   cultures:
#     - name: Vessari
#       mode: scoped              # scoped (placed from this entry) | organic (emerges from a simulated history)
#       inhabitants: native       # native | colonist
#       development:              # axes left out take the mean level of those given
#         energy: farming         # foraging | farming | water_wind | fossil | electric | fusion
#         transport: wheel_and_sail  # foot | pack_and_boat | wheel_and_sail | rail_and_steam | motor_and_air | spaceflight
#         communication: writing  # oral | writing | print | telegraph | digital | interplanetary
#       population: [1.0e7, 5.0e7]
#       # homeland: {lat_deg: 20, lon_deg: 40, radius_km: 1500}
#       settlements: {zipf_exponent: 1.0, largest_population: 2.0e5}
#       polities:
#         count: [3, 8]
#         type: kingdom           # band | tribe | chiefdom | city_state | kingdom | empire | federation | republic | corporate | colony
#         # capitals: [{lat_deg: 21, lon_deg: 38, name: Aru}]
#     - inhabitants: colonist
#       development: {energy: fusion, transport: spaceflight}
#       colony: {founded_years_ago: 150, origin: Sol, landing_sites: [{lat_deg: -5, lon_deg: 100}]}
#       # organic colonies: colony: {..., immigrants_per_year: 1000, immigration_years: 100, independence_years: 200,
#       #                            crowd_diseases: true}   # carry herd-animal diseases to natives without livestock
#     # - name: Oru
#     #   mode: organic           # development, population, settlements and polities emerge from the history
#     #   homeland: {lat_deg: -10, lon_deg: 20, radius_km: 3000}   # where its foragers start
"""
