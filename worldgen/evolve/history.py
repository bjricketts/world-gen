"""History mode: integrate the planet's global state from formation to the target epoch."""

from __future__ import annotations

from typing import Optional

from .. import constants as c
from .. import heuristics as h
from .. import interior as interior_model
from ..biosphere.life import product_of
from ..history.build import HistoryContext, build_state, sample_series
from ..history.climate import OrbitClimate, shared_table
from ..history.integrate import HistoryModel, integrate
from ..history.params import build_params
from ..orbit import instellation, runaway_greenhouse_limit
from ..priors import resolve
from ..priors.constraints import attempt_seed, failed_constraints
from ..spec import PlanetSpec
from ..state import Issue, PlanetState, Timeline, TimelineEvent
from ..star import brightening_factor, main_sequence_lifetime_gyr, main_sequence_luminosity
from ..surface import relief_factor
from ..util import named_rng
from .snapshot import SnapshotEvolver

NO_HISTORY_CLASSES = ("sub-Neptune", "ice giant", "gas giant", "brown dwarf")


class HistoryEvolver:
    """Builds a planet by integrating its interior, volatiles, climate and life from formation."""

    def evolve(self, spec: PlanetSpec, t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
        """Return the planet state at the target epoch and its timeline.

        Draws are screened with the snapshot build; history runs on up to
        ``HISTORY_CANDIDATES`` draws that pass, until one also satisfies the
        archetype's required outcomes (the closest is kept otherwise).
        """
        snapshot = SnapshotEvolver()
        attempts = spec.priors.max_attempts if spec.priors.enforce_constraints else 1
        best = None
        runs = tried = 0
        for attempt in range(attempts):
            tried = attempt + 1
            seed = attempt_seed(spec.seed, attempt)
            base = snapshot.build(spec, seed, t_target_gyr)
            if attempts > 1 and failed_constraints(base) and attempt < attempts - 1:
                continue
            state, timeline = self.run(spec, seed, base, t_target_gyr)
            runs += 1
            failures = failed_constraints(state)
            if best is None or len(failures) < len(best[2]):
                best = (state, timeline, failures)
            if not failures or runs >= h.HISTORY_CANDIDATES:
                break
        state, timeline, failures = best
        state.attempts = tried
        state.constraints_met = not failures
        if failures and spec.priors.enforce_constraints:
            missed = "; ".join(f.description for f in failures)
            state.issues.insert(0, Issue("warning", "conflict", "priors",
                                         f"no history in {runs} runs ({tried} draws) met the "
                                         f"'{state.archetype}' archetype; missing: {missed}"))
        elif tried > 1:
            state.issues.insert(0, Issue("info", "note", "priors",
                                         f"{tried} draws screened, {runs} histories run to meet the "
                                         f"'{state.archetype}' archetype"))
        return state, timeline

    def run(self, spec: PlanetSpec, seed: int, base: PlanetState,
            t_target_gyr: Optional[float] = None) -> tuple[PlanetState, Timeline]:
        """Return the history of one draw whose snapshot build is ``base``."""
        age = base.star.age_s / c.SECONDS_PER_GYR
        if base.bulk.planet_class in NO_HISTORY_CLASSES:
            base.issues.insert(0, Issue("info", "note", "history",
                                        "history mode covers planets with a solid surface; snapshot values used"))
            return base, Timeline(times_s=[base.epoch_s], states=[base])
        if age <= h.HISTORY_START_GYR:
            raise ValueError(f"target epoch {age:.3f} Gyr is before the history starts ({h.HISTORY_START_GYR} Gyr)")
        inputs = resolve(spec, seed)
        v = inputs.get
        star_mass = base.star.mass_kg / c.M_SUN
        life = {"biochemistry": v("biosphere.biochemistry") or "oxygenic",
                "optimum_k": v("biosphere.optimum_temperature_k"),
                "tolerance_k": v("biosphere.temperature_tolerance_k")}
        life["product"] = product_of(life["biochemistry"], v("biosphere.alien_product"))
        params = build_params(v, base.bulk, base.orbit, star_mass, v("body.tidal_heating_w", 0.0), life)
        relief = relief_factor(base.bulk.surface_gravity_m_s2)
        o = base.orbit
        rotation_h = v("orbit.rotation_period_h")
        rotation_s = o.rotation_period_s if rotation_h is None else rotation_h * c.SECONDS_PER_HOUR
        table = shared_table(OrbitClimate(o.eccentricity, o.obliquity_rad, o.periapsis_longitude_rad, rotation_s,
                                          o.orbital_period_s, base.bulk.surface_gravity_m_s2,
                                          base.bulk.planet_class))
        held_regime = v("interior.tectonic_regime") if inputs.is_user("interior.tectonic_regime") else None
        held_life = v("biosphere.life") if inputs.is_user("biosphere.life") else None
        model = HistoryModel(params, table, relief, base.bulk.planet_class, held_regime=held_regime,
                             held_life=held_life, eccentricity=o.eccentricity)
        regime, probabilities = self.initial_regime(params, base, inputs, seed, held_regime)
        result = integrate(model, age, regime)

        issues = [i for i in base.issues if i.subsystem in ("priors", "star", "bulk") and i.kind != "heuristic"]
        issues.append(Issue("info", "heuristic", "history",
                            "XUV history from a drawn rotation percentile (Tu et al. 2015)"))
        for e in result.events:
            if e.flagged:
                issues.append(Issue("warning", "conflict", "history",
                                    f"at {e.time_gyr:.2f} Gyr physics overrode a held value: {e.detail}"))
        ctx = HistoryContext(spec=spec, inputs=inputs, base=base, model=model, result=result,
                             regime_probabilities=probabilities, issues=issues)
        final = build_state(ctx, age, apply_overrides=True)
        epochs = sorted({t for t in (spec.history.epochs_gyr or []) if h.HISTORY_START_GYR <= t < age})
        states = [build_state(ctx, t) for t in epochs] + [final]
        timeline = Timeline(times_s=[t * c.SECONDS_PER_GYR for t in epochs] + [final.epoch_s], states=states,
                            events=[TimelineEvent(e.time_gyr * c.SECONDS_PER_GYR, e.kind, e.detail, e.flagged)
                                    for e in result.events],
                            series=sample_series(ctx))
        return final, timeline

    @staticmethod
    def initial_regime(params, base: PlanetState, inputs, seed: int,
                       held: Optional[str]) -> tuple[str, dict[str, float]]:
        """Return the tectonic regime at formation, drawn from the probabilities at 1 Gyr, and those probabilities."""
        star_mass = params.star_mass_msun
        lifetime = main_sequence_lifetime_gyr(star_mass)
        early_lum = main_sequence_luminosity(star_mass) * brightening_factor(0.0, lifetime) * c.L_SUN
        s0 = instellation(early_lum, params.semi_major_axis_m, base.orbit.eccentricity)
        wet = (params.water_oceans * c.EARTH_OCEAN_MASS / params.mass_kg > h.SURFACE_WATER_MIN_FRACTION
               and s0 < runaway_greenhouse_limit(base.star.effective_temperature_k, params.mass_ratio))
        interior, _, _ = interior_model.build_interior(
            bulk=base.bulk, age_gyr=1.0, radiogenic_abundance=params.radiogenic_abundance,
            tidal_power_w=params.tidal_power_w, surface_temperature_k=min(base.atmosphere.surface_temperature_k,
                                                                          400.0),
            surface_water=wet, regime_override=held, dynamo_override=True,
            regime_rng=named_rng(seed, "interior.tectonic_regime"), dynamo_rng=named_rng(seed, "history.dynamo"))
        regime = interior.tectonic_regime
        if regime == "fluid":
            regime = "inactive"
        return regime, interior.regime_probabilities


__all__ = ["HistoryEvolver"]
