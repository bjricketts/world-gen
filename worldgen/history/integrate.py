"""The history ODE system and its integration from formation to the target epoch.

The continuous state is mantle and core temperature, surface and mantle
water, carbon in air–ocean, crust and mantle, nitrogen and oxygen. Discrete
modes (tectonic regime, climate branch, life, spin) change at events, where
the integration restarts. Tier 0 climate is quasi-static and comes from
``ClimateTable``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .. import constants as c
from .. import heuristics as h
from .. import star as star_model
from ..biosphere.life import methane_optical_depth
from ..orbit import instellation, runaway_greenhouse_limit
from ..water import land_fraction_estimate
from . import life as life_model
from . import thermal
from . import volatiles as vol
from .climate import ClimatePoint, ClimateTable, grey_climate
from .params import HistoryParams

TM, TC, WS, WM, CS, CK, CM, N2, O2 = range(9)
STATE_NAMES = ("mantle_temperature_k", "core_temperature_k", "surface_water_oceans", "mantle_water_oceans",
               "surface_carbon", "crust_carbon", "mantle_carbon", "nitrogen", "oxygen")
ABS_TOL = np.array([0.01, 0.01, 1e-6, 1e-6, 1e-5, 1e-3, 1e-3, 1e-5, 1e-8])
LID_REGIMES = ("stagnant_lid", "episodic", "heat_pipe")


@dataclass
class Modes:
    """Discrete state of the planet between events."""

    regime: str
    climate: str                        # 'warm', 'snowball', 'runaway', 'dry'
    life: str = "none"                  # 'surface', 'ocean', 'subsurface', 'none'
    synchronous: bool = False
    clock_start: Optional[float] = None  # start of the current habitable stretch (origin-of-life clock)
    clock_elapsed: float = 0.0          # habitable time banked before a pause
    life_origin: Optional[float] = None
    land_colonised: Optional[float] = None
    surface_life: bool = False          # life had reached the surface (resumes after a snowball)


@dataclass
class HistoryEvent:
    """A change in the planet's history."""

    time_gyr: float
    kind: str
    detail: str
    flagged: bool = False               # a user-held value was overridden by physics


@dataclass
class Diagnostics:
    """Derived quantities at one instant."""

    time_gyr: float
    luminosity_w: float
    effective_temperature_k: float
    xuv_fraction: float
    instellation: float
    runaway_limit: float
    air: vol.Air
    optical_depth: float
    methane_fraction: float
    climate: ClimatePoint
    land: float
    ocean: bool
    thermal: thermal.ThermalRates
    carbon: vol.CarbonFluxes
    degassing: float
    regassing: float
    water_escape: float
    bulk_escape: float
    productivity: float
    land_productivity: float
    oxygen_rate: float
    modes: Modes

    @property
    def surface_k(self) -> float:
        """Return the global mean surface temperature."""
        return self.climate.mean_k

    @property
    def liquid(self) -> float:
        """Return how much of the surface water is open (0–1) for weathering."""
        if self.modes.climate != "warm":
            return 0.0
        return min(self.climate.open_ocean / h.OPEN_OCEAN_LIQUID, 1.0)


@dataclass
class Segment:
    """One stretch of integration between events."""

    start: float
    end: float
    solution: object
    modes: Modes


@dataclass
class HistoryResult:
    """The integrated history."""

    params: HistoryParams
    segments: list[Segment]
    events: list[HistoryEvent]
    table_solves: int
    rhs_calls: int
    final_modes: Modes
    t_start: float
    t_end: float

    def segment_at(self, t: float) -> Segment:
        """Return the integration segment containing ``t`` (the first or last one outside the range)."""
        for seg in self.segments:
            if seg.start <= t < seg.end:
                return seg
        return self.segments[-1] if t >= self.segments[-1].start else self.segments[0]

    def state_at(self, t: float) -> np.ndarray:
        """Return the continuous state vector at ``t``."""
        seg = self.segment_at(t)
        return np.asarray(seg.solution(min(max(t, seg.start), seg.end)))


class HistoryModel:
    """Right-hand side, diagnostics and events of the history system for one planet."""

    def __init__(self, params: HistoryParams, table: ClimateTable, relief: float, planet_class: str,
                 held_regime: Optional[str] = None, held_life: Optional[str] = None,
                 eccentricity: float = 0.0):
        self.p = params
        self.table = table
        self.relief = relief
        self.planet_class = planet_class
        self.held_regime = held_regime
        self.held_life = held_life
        self.eccentricity = eccentricity
        self.lifetime = star_model.main_sequence_lifetime_gyr(params.star_mass_msun)
        self.base_luminosity = star_model.main_sequence_luminosity(params.star_mass_msun)
        self.base_radius = star_model.main_sequence_radius(params.star_mass_msun)
        self.min_water = h.SURFACE_WATER_MIN_FRACTION * params.mass_kg / c.EARTH_OCEAN_MASS
        self.thin_air = h.THIN_AIR_BAR / params.pressure_bar(1.0)
        self.calls = 0
        self._cache_key = None
        self._cache = None

    # --- star -------------------------------------------------------------------------------
    def star(self, t: float) -> tuple[float, float, float]:
        """Return luminosity (W), effective temperature (K) and XUV fraction at age ``t``."""
        lum, _, teff = star_model.luminosity_track(self.p.star_mass_msun, t, self.lifetime)
        xuv = star_model.xuv_fraction_history(self.p.star_mass_msun, t, self.p.activity_percentile)
        return lum * c.L_SUN, teff, xuv

    # --- diagnostics ------------------------------------------------------------------------
    def diagnostics(self, t: float, y: np.ndarray, modes: Modes) -> Diagnostics:
        """Return every derived quantity for the state ``y`` at ``t`` in ``modes``."""
        key = (t, y.tobytes())
        if key == self._cache_key and modes == self._cache.modes:
            return self._cache
        p = self.p
        lum, teff, xuv = self.star(t)
        s = instellation(lum, p.semi_major_axis_m, self.eccentricity)
        limit = runaway_greenhouse_limit(teff, p.mass_ratio)
        water = max(y[WS], 0.0)
        steam = modes.climate == "runaway"
        ocean = modes.climate in ("warm", "snowball") and water > self.min_water
        liquid_ocean = water if modes.climate in ("warm", "snowball") else 0.0
        air = vol.partition_air(p, y[CS], liquid_ocean, y[N2], y[O2], steam_oceans=water if steam else 0.0)
        methane = life_model.methane_fraction(p, modes.life, air.o2_fraction)
        tau = vol.optical_depth(air, ocean or steam, methane_optical_depth(methane))
        if ocean:
            volume = water * c.EARTH_OCEAN_MASS / h.SEAWATER_DENSITY
            land = land_fraction_estimate(volume, p.radius_m, self.relief, modes.regime)
        else:
            land = 1.0
        if modes.climate in ("warm", "snowball") and ocean:
            branch = "warm" if modes.climate == "warm" else "cold"
            climate = self.table.lookup(s, tau, max(air.pressure_bar + air.steam_bar, 1e-4), land,
                                        modes.synchronous, branch)
        else:
            climate = grey_climate(s, tau, air.pressure_bar + air.steam_bar, air.pressure_bar > 1e-4,
                                   self.planet_class)
        rates = thermal.thermal_rates(p, t, y[TM], y[TC], modes.regime, climate.mean_k)
        heat_ratio = rates.heat_flux_w_m2 / c.EARTH_HEAT_FLUX
        prod, land_prod = life_model.productivity(p, modes.life, self._colonised(t, modes), climate.mean_k,
                                                  land, climate.open_ocean if modes.climate == "warm" else 0.05)
        liquid = min(climate.open_ocean / h.OPEN_OCEAN_LIQUID, 1.0) if modes.climate == "warm" else 0.0
        carbon = vol.carbon_fluxes(p, max(y[CK], 0.0), max(y[CM], 0.0), air.co2_bar, climate.mean_k, land,
                                   ocean, liquid, modes.regime, rates.melt, rates.spreading, heat_ratio, land_prod)
        degas, regas = vol.water_fluxes(p, water, max(y[WM], 0.0), modes.regime, rates.melt, rates.spreading,
                                        ocean)
        flux = vol.xuv_flux(lum, xuv, p)
        escape = vol.water_escape(p, flux, climate.mean_k, steam) * water / (water + self.min_water)
        if not (ocean or steam):
            escape *= 0.0 if climate.mean_k < 273.0 else 1.0
        bulk = vol.bulk_escape(p, flux, s)
        oxygen = life_model.oxygen_rate(p, t, max(y[O2], 0.0), prod, rates.melt, climate.mean_k, land,
                                        heat_ratio)
        d = Diagnostics(time_gyr=t, luminosity_w=lum, effective_temperature_k=teff, xuv_fraction=xuv,
                        instellation=s, runaway_limit=limit, air=air, optical_depth=tau,
                        methane_fraction=methane, climate=climate, land=land, ocean=ocean, thermal=rates,
                        carbon=carbon, degassing=degas, regassing=regas, water_escape=escape, bulk_escape=bulk,
                        productivity=prod, land_productivity=land_prod, oxygen_rate=oxygen, modes=modes)
        self._cache_key, self._cache = key, replace(d, modes=replace(modes))
        return d

    def _colonised(self, t: float, modes: Modes) -> float:
        """Return the land biosphere's spread (0–1) after colonisation."""
        if modes.land_colonised is None or modes.life != "surface":
            return 0.0
        return min(max((t - modes.land_colonised) / 0.2, 0.0), 1.0)

    def warm_climate(self, d: Diagnostics) -> ClimatePoint:
        """Return the warm-branch climate at the conditions of ``d``."""
        return self.table.lookup(d.instellation, d.optical_depth, max(d.air.pressure_bar, 1e-4), d.land,
                                 d.modes.synchronous, "warm")

    # --- right-hand side --------------------------------------------------------------------
    def rhs(self, t: float, y: np.ndarray, modes: Modes) -> np.ndarray:
        """Return dy/dt (per Gyr)."""
        self.calls += 1
        d = self.diagnostics(t, y, modes)
        p = self.p
        dy = np.zeros_like(y)
        dy[TM], dy[TC] = thermal.temperature_rates(p, d.thermal)
        dy[WS] = d.degassing - d.regassing - d.water_escape
        dy[WM] = d.regassing - d.degassing
        cf = d.carbon
        # Each gas escapes in proportion to its share of the air; a thin air loses gas in proportion to its mass.
        air_mass = max(d.air.co2_mass + max(y[N2], 0.0) + max(y[O2], 0.0), self.thin_air)
        share = lambda m: max(m, 0.0) / air_mass
        dy[CS] = (cf.volcanic + cf.arc + cf.recycling + cf.decomposition - cf.weathering
                  - d.bulk_escape * share(d.air.co2_mass))
        dy[CK] = cf.weathering - cf.subduction - cf.recycling - cf.decomposition
        dy[CM] = cf.subduction - cf.arc - cf.volcanic
        dy[N2] = -d.bulk_escape * share(y[N2])
        left_behind = d.water_escape * c.EARTH_OCEAN_MASS / 1e18 * 8.0 / 9.0
        dy[O2] = d.oxygen_rate + left_behind - d.bulk_escape * share(y[O2])
        return dy

    # --- events -----------------------------------------------------------------------------
    def events(self, modes: Modes) -> list[tuple[str, Callable, bool, int]]:
        """Return (name, function, terminal, direction) for the events that can occur in ``modes``."""
        ev = []

        def add(name, fn, terminal=True, direction=0):
            """Register an event function."""
            ev.append((name, lambda t, y, _fn=fn: _fn(self.diagnostics(t, y, modes), t, y), terminal, direction))

        if modes.climate == "warm":
            add("snowball_onset", lambda d, t, y: open_margin(d.climate), direction=-1)
            add("hot", lambda d, t, y: d.surface_k - h.HABITABLE_MAX_K, direction=0)
        if modes.climate == "snowball":
            add("snowball_exit", lambda d, t, y: min(open_margin(d.climate, 1.5),
                                                     open_margin(self.warm_climate(d))), direction=1)
        if modes.climate in ("warm", "snowball"):
            add("runaway_onset", lambda d, t, y: d.instellation - d.runaway_limit, direction=1)
            add("ocean_loss", lambda d, t, y: y[WS] - 2.0 * self.min_water, direction=-1)
        if modes.climate == "runaway":
            add("ocean_loss", lambda d, t, y: y[WS] - 2.0 * self.min_water, direction=-1)
            add("runaway_end", lambda d, t, y: 0.98 * d.runaway_limit - d.instellation, direction=1)
        if modes.climate == "dry":
            add("ocean_return", lambda d, t, y: y[WS] - 20.0 * self.min_water, direction=1)
        if self.held_regime is None:
            if modes.regime == "mobile_lid":
                add("plates_stop", lambda d, t, y: d.thermal.budget_activity
                    - h.REGIME_PLATE_LOSS_FACTOR * h.ACTIVITY_PLATES_ABOVE, direction=-1)
            if modes.regime in LID_REGIMES:
                add("regime_dead", lambda d, t, y: d.thermal.budget_activity - h.ACTIVITY_DEAD_BELOW,
                    direction=-1)
            heat_pipe = (lambda d, t, y: d.thermal.budget_flux_w_m2 / c.EARTH_HEAT_FLUX
                         - h.HEATPIPE_FLUX_RATIO_ABOVE)
            if modes.regime == "heat_pipe":
                add("heat_pipe_end", heat_pipe, direction=-1)
            elif modes.regime != "inactive":
                add("heat_pipe_start", heat_pipe, direction=1)
        if modes.clock_start is not None and modes.life == "none":
            due = modes.clock_start + self.p.life_delay_gyr - modes.clock_elapsed
            add("origin_of_life", lambda d, t, y, _due=due: t - _due, direction=1)
        if modes.life == "surface" and modes.land_colonised is None and modes.life_origin is not None:
            due = modes.life_origin + self.p.land_delay_gyr
            add("land_colonisation", lambda d, t, y, _due=due: t - _due, direction=1)
        if not modes.synchronous and math.isfinite(self.p.tidal_lock_gyr):
            add("tidal_locking", lambda d, t, y: t - self.p.tidal_lock_gyr, direction=1)
        add("dynamo", lambda d, t, y: d.thermal.dynamo_margin, terminal=False)
        for level in h.OXYGENATION_FRACTIONS:
            add(f"oxygen_{level}", lambda d, t, y, _l=level: math.log(max(d.air.o2_fraction, 1e-30) / _l),
                terminal=False, direction=1)
        return ev

    # --- mode changes -----------------------------------------------------------------------
    def habitable(self, d: Diagnostics, modes: Modes) -> bool:
        """Return whether the origin-of-life clock runs."""
        return modes.climate == "warm" and d.ocean and d.surface_k < h.HABITABLE_MAX_K

    def life_kind(self, d: Diagnostics) -> str:
        """Return the life a newly living (or re-emerging) planet has, from the snapshot rules."""
        if self.held_life is not None:
            return self.held_life
        return "surface" if d.land >= h.LIFE_MIN_LAND_FRACTION else "ocean"

    def update_clock(self, t: float, d: Diagnostics, modes: Modes) -> Modes:
        """Start or pause the origin-of-life clock to match the current conditions."""
        if modes.life != "none":
            return modes
        running = modes.clock_start is not None
        if self.habitable(d, modes) and not running:
            return replace(modes, clock_start=t)
        if not self.habitable(d, modes) and running:
            return replace(modes, clock_start=None, clock_elapsed=modes.clock_elapsed + t - modes.clock_start)
        return modes

    def transition(self, name: str, t: float, y: np.ndarray, modes: Modes,
                   log: Callable[[str, str, bool], None]) -> Modes:
        """Return the modes after event ``name`` and log what happened."""
        d = self.diagnostics(t, y, modes)
        m = replace(modes)
        if name == "snowball_onset":
            m.climate = "snowball"
            log("snowball_onset", f"the oceans freeze over at {d.surface_k:.0f} K", False)
            if m.life in ("surface", "ocean"):
                m.surface_life = True
                m.life = "subsurface"
                log("life_retreat", "life retreats beneath the ice", False)
        elif name == "snowball_exit":
            m.climate = "warm"
            log("snowball_exit", f"the ice melts back with {d.air.co2_bar:.2g} bar of CO₂", False)
            if m.life == "subsurface" and m.surface_life:
                m.life = self.life_kind(self.diagnostics(t, y, m))
                log("life_emerges", f"{m.life} life returns to the open water", False)
        elif name == "hot":
            pass
        elif name == "runaway_onset":
            m.climate = "runaway"
            log("runaway_onset", f"instellation {d.instellation:.2f} passes the runaway limit "
                                 f"{d.runaway_limit:.2f}: the oceans boil", False)
            if m.life != "none":
                m.life = "none"
                m.surface_life = False
                log("extinction", "life ends as the oceans boil", False)
        elif name == "runaway_end":
            m.climate = "warm"
            log("runaway_end", "the steam atmosphere condenses into oceans", False)
        elif name == "ocean_loss":
            m.climate = "dry"
            log("ocean_loss", "the surface water is gone", False)
            if m.life != "none":
                m.life = "none"
                log("extinction", "life ends as the planet dries out", False)
            if m.regime == "mobile_lid":
                m.regime = "stagnant_lid"
                held = self.held_regime is not None
                log("regime_change", "plate tectonics stops without water: stagnant lid", held)
        elif name == "ocean_return":
            m.climate = "warm"
            log("ocean_return", "outgassed water collects at the surface", False)
        elif name == "plates_stop":
            m.regime = "stagnant_lid"
            log("regime_change", f"interior activity {d.thermal.budget_activity:.2f} too low for plates: "
                                 "stagnant lid", False)
        elif name == "heat_pipe_start":
            m.regime = "heat_pipe"
            log("regime_change", f"heat flux {d.thermal.budget_flux_w_m2 / c.EARTH_HEAT_FLUX:.0f} × Earth's: "
                                 "volcanism carries the heat (heat pipe)", False)
        elif name == "heat_pipe_end":
            m.regime = "stagnant_lid"
            log("regime_change", f"heat flux {d.thermal.budget_flux_w_m2 / c.EARTH_HEAT_FLUX:.1f} × Earth's is "
                                 "too low for heat-pipe volcanism: stagnant lid", False)
        elif name == "regime_dead":
            m.regime = "inactive"
            log("regime_change", "the interior falls quiet: inactive", False)
        elif name == "origin_of_life":
            m.life = self.life_kind(d)
            m.life_origin = t
            m.surface_life = m.life in ("surface", "ocean")
            log("origin_of_life", f"{m.life} life appears", False)
        elif name == "land_colonisation":
            m.land_colonised = t
            log("land_colonisation", "life spreads onto land", False)
        elif name == "tidal_locking":
            m.synchronous = True
            log("tidal_locking", "the rotation locks to the orbit", False)
        if m.climate == "warm" and name in ("runaway_end", "ocean_return"):
            # A restored climate can itself be frozen: check the warm branch immediately.
            probe = self.diagnostics(t, y, m)
            if open_margin(probe.climate) < 0.0:
                m.climate = "snowball"
        return self.update_clock(t, self.diagnostics(t, y, m), m)


def pending_event(model: "HistoryModel", t: float, y: np.ndarray, modes: Modes, done: set) -> Optional[str]:
    """Return a terminal event whose condition already holds at (t, y), so it would never be crossed."""
    d = model.diagnostics(t, y, modes)
    for name, fn, terminal, direction in model.events(modes):
        if not terminal or name in done or name == "hot":
            continue
        value = fn(t, y)
        if (direction > 0 and value > 0) or (direction < 0 and value < 0):
            if name == "snowball_onset" and "snowball_exit" in done:
                continue
            return name
    return None


def open_margin(climate: ClimatePoint, factor: float = 1.0) -> float:
    """Return how far the climate is from frozen: negative when both open-water tests fail."""
    return max(climate.open_water - factor * h.OPEN_WATER_LIQUID,
               (climate.open_ocean - factor * h.OPEN_OCEAN_LIQUID) * h.OPEN_WATER_LIQUID / h.OPEN_OCEAN_LIQUID)


def initial_state(p: HistoryParams) -> np.ndarray:
    """Return the state vector at formation."""
    y = np.zeros(9)
    y[TM] = p.mantle_temperature_k
    y[TC] = thermal.initial_core_k(p, p.mantle_temperature_k)
    y[WS] = p.water_oceans * (1.0 - h.INITIAL_MANTLE_WATER_SHARE)
    y[WM] = p.water_oceans * h.INITIAL_MANTLE_WATER_SHARE
    y[CS] = p.carbon * h.INITIAL_DEGASSED_CARBON_SHARE
    y[CK] = 0.0
    y[CM] = p.carbon * (1.0 - h.INITIAL_DEGASSED_CARBON_SHARE)
    y[N2] = p.nitrogen
    y[O2] = 0.0
    return y


def initial_modes(model: HistoryModel, t0: float, y0: np.ndarray, regime: str) -> Modes:
    """Return the modes at formation: the climate branch follows the instellation and the water."""
    modes = Modes(regime=regime, climate="warm", synchronous=t0 >= model.p.tidal_lock_gyr)
    d = model.diagnostics(t0, y0, modes)
    if y0[WS] <= 2.0 * model.min_water:
        modes.climate = "dry"
    elif d.instellation > d.runaway_limit:
        modes.climate = "runaway"
    else:
        # Oceans condense: most of the magma ocean's CO₂ is carbonated within a few Myr (not resolved).
        keep = h.INITIAL_AIR_CARBON_SHARE
        y0[CK] += y0[CS] * (1.0 - keep)
        y0[CS] *= keep
        if open_margin(model.diagnostics(t0, y0, modes).climate) < 0.0:
            modes.climate = "snowball"
    return model.update_clock(t0, model.diagnostics(t0, y0, modes), modes)


def integrate(model: HistoryModel, t_end: float, regime: str, t_start: float = h.HISTORY_START_GYR,
              rtol: float = 1e-4) -> HistoryResult:
    """Integrate the history from ``t_start`` to ``t_end`` (Gyr) and return the segments and events."""
    y = initial_state(model.p)
    t = t_start
    modes = initial_modes(model, t, y, regime)
    events: list[HistoryEvent] = []
    segments: list[Segment] = []
    if modes.climate == "runaway":
        events.append(HistoryEvent(t, "runaway_onset", "the planet forms inside the runaway limit: "
                                                         "its water never condenses"))
    elif modes.climate == "snowball":
        events.append(HistoryEvent(t, "snowball_onset", "the planet starts frozen"))
    dynamo = model.diagnostics(t, y, modes).thermal.dynamo
    if dynamo:
        events.append(HistoryEvent(t, "dynamo_onset", "a core dynamo runs from formation"))

    def settle(t, y, modes):
        """Apply events whose conditions already hold before integrating further."""
        done = set()
        for _ in range(8):
            name = pending_event(model, t, y, modes, done)
            if name is None:
                break
            done.add(name)

            def log(kind, detail, flagged, _t=t):
                """Record an event."""
                events.append(HistoryEvent(_t, kind, detail, flagged))

            modes = model.transition(name, t, y, modes, log)
        return modes

    for _ in range(h.HISTORY_MAX_SEGMENTS):
        if t >= t_end:
            break
        modes = settle(t, y, modes)
        spec = model.events(modes)
        fns = []
        for name, fn, terminal, direction in spec:
            fn.terminal = terminal
            fn.direction = direction
            fns.append(fn)
        current = modes
        sol = solve_ivp(lambda tt, yy: model.rhs(tt, yy, current), (t, t_end), y, method="BDF",
                        events=fns, dense_output=True, rtol=rtol, atol=ABS_TOL,
                        max_step=h.HISTORY_MAX_STEP_GYR, first_step=1e-5)
        # Non-terminal events: dynamo and oxygenation
        terminal_hit = None
        for (name, fn, terminal, direction), times, states in zip(spec, sol.t_events, sol.y_events):
            for te, ye in zip(times, states):
                if te <= t + 1e-9:
                    continue
                if terminal:
                    if terminal_hit is None or te < terminal_hit[1]:
                        terminal_hit = (name, te, ye)
                    continue
                if name == "dynamo":
                    after = model.diagnostics(min(te + 1e-4, sol.t[-1]), sol.sol(min(te + 1e-4, sol.t[-1])),
                                              current).thermal.dynamo
                    if after != dynamo:
                        dynamo = after
                        events.append(HistoryEvent(te, "dynamo_onset" if after else "dynamo_shutdown",
                                                   "the core dynamo starts" if after else "the core dynamo stops"))
                elif name.startswith("oxygen_"):
                    level = float(name.split("_", 1)[1])
                    events.append(HistoryEvent(te, "oxygenation",
                                               f"O₂ passes {level:.1%} of the air"))
        end = sol.t[-1]
        segments.append(Segment(t, end, sol.sol, current))
        if sol.status == 1 and terminal_hit is not None:
            name, te, ye = terminal_hit
            segments[-1].end = te
            t, y = te, np.array(ye)

            def log(kind, detail, flagged, _t=te):
                """Record an event."""
                events.append(HistoryEvent(_t, kind, detail, flagged))

            new = model.transition(name, t, y, current, log)
            if new == current and name != "hot":
                # Guard against an event that changes nothing firing again at once.
                t = t + 1e-6
            modes = model.update_clock(t, model.diagnostics(t, y, new), new) if name == "hot" else new
            continue
        if sol.status < 0:
            events.append(HistoryEvent(end, "integration_failed", sol.message, True))
            break
        t, y = end, sol.y[:, -1]
        if sol.status == 1:
            continue      # a terminal event at the segment start was ignored; carry on
    events.sort(key=lambda e: e.time_gyr)
    return HistoryResult(params=model.p, segments=segments, events=events, table_solves=model.table.solves,
                         rhs_calls=model.calls, final_modes=modes, t_start=t_start, t_end=t_end)
