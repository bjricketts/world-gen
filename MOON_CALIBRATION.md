# Moon models: calibration assessment

Written 2026-10-25, after milestone 11, at Ben's request: document the moon models' limitations against best practice, quantify the impact of each error, and leave the small ones as open points. Ben then chose to fix items 1–3 (see "Fixes built"); item 4 and the minor items stay open.

## Method

Each limitation is compared with how the literature treats it, then measured with the package. Each test either moves one assumption toward best practice or measures how close an outcome is to a threshold. Two samples are used:

- **Solar System moons.** Io, Europa, Ganymede, Callisto, Enceladus, Titan and Triton, built from their real masses and orbits as in `scripts/validate_solar_system.py`. Some tests repeat this over 40 seeds.
- **Habitable moons.** 150 states from the `habitable_moon` archetype (seeds 1–150). In paired comparisons, seeds whose number of redraws changed are dropped.

The test scripts are in the scratchpad `p13/`:
- `harness.py`: the moon builds.
- `hab.py`: the archetype sample.
- `exp_tides.py` and `exp_tides2.py`: dissipation.
- `exp_ecc.py`: eccentricity damping.
- `exp_shell.py`: ice shells.
- `exp_air.py`: icy air.

**Impact scale:**
- **Major:** the error changes a visible outcome on a material share of worlds, or misses an observed Solar System value by more than a factor of 2 where the validation should catch it. Visible outcomes are an ocean, an atmosphere, a habitability verdict, a report hazard or a surface feature.
- **Minor:** the effect stays below about 1 K and a few per cent of the energy budget, or only touches a small reported number. Minor items go to OPEN_POINTS.

## Summary

| # | Limitation | Best practice | Measured impact | Class |
|---|---|---|---|---|
| 1 | Ice-shell moons get Io's dissipation (k₂/Q = 0.015) through the melt switch (fixed) | Viscoelastic (Andrade) Love numbers with heat generated in the ice shell; Europa's −Im(k₂) is 3 × 10⁻⁴–2.5 × 10⁻³ (Bagheri et al. 2026) | Europa gets 233 mW/m² and a 2.7 km shell. Observed: 15–35 mW/m² and a 19–39 km conductive shell. With k₂/Q = 10⁻³: 22 mW/m², 29 km | Major |
| 2 | Free eccentricities never damp (fixed) | Eccentricity damps in t_damp = [(21/2)(M_p/m)(R/a)⁵(k₂/Q) n]⁻¹ unless a resonance forces it (Tokadjian & Piro 2023) | 83% of habitable moons should be circular by their age, including all 40 of the 150 with tides above 0.1 W/m². 9% are reported above Io's 2 W/m² ("global volcanism") | Major |
| 3 | Thick ice shells conduct from the bare surface temperature, with no salts (lid fixed; salts open) | Fractured, porous ice conducts poorly down to ~130–140 K (Shoemaker et al. 1982; Nimmo et al. 2003, via Singer et al. 2023). Hydrospheres are modelled with salts (Vance et al. 2018) | Ganymede misses an ocean by 1% of its heat, Callisto by 15%, Triton by 41%. With a 130–140 K lid, Ganymede has a 108–119 km shell over a 152–180 km ocean (estimates: ~150 km ice, ≳100 km ocean) | Major |
| 4 | Icy-moon air is a coin flip at the cosmic shoreline | Unsettled: why Titan alone holds dense air is an open question (Nixon 2024) | Over 40 seeds, Ganymede has air in 52% of draws (up to 0.31 bar), Callisto 28%, Europa 15%, Io 28%. Titan ranges over 0.5–5 bar (10–90%) and 73–120 K | Major (no settled fix) |
| 5 | Triton's N₂ frost sits at the climate model's 41 K | The N₂ ice energy balance εσT⁴ = (1 − A)F/4 with A ≈ 0.7–0.8 and ε = 0.8 (Bertrand et al. 2021) | 104 µbar against the observed 14.5. The frost balance gives 37.9 K and 19 µbar. Near the triple point the error is 2× | Minor |
| 6 | Liquid N₂ is not capped | A liquid-N₂ vapour curve between 63 and 126 K | 0 of 40 Titan draws are cold enough with thick air (the greenhouse keeps them ≥ 87 K) | Minor |
| 7 | Long moon days have no day–night cycle | Diurnal climate | Rotation of 2.5–25 days (10–90%), up to 70. The climate is zonal-mean for every world, so means are unaffected. Day–night swings grow with the day length (not modelled) | Minor |
| 8 | The parent's glow is spread over the whole moon | Sub-parent hemisphere | The glow is ≤ 1.1% of absorbed sunlight (median 0.1%). Concentrating it moves temperatures by ≤ 0.4 K | Minor |
| 9 | Stellar-wind pressure is the present Sun's | Wind scaled by stellar type and age (Grießmeier et al. 2007, unread) | 10× the wind moves the magnetic-shelter fraction by up to 0.69, but no occupiability factor reads it | Minor |
| 10 | Capture heating is not applied | Heating the moon while its orbit circularises | It lasts 10⁴–10⁷ yr against Gyr ages | Minor |
| 11 | Parent albedo by class; Uranus's heat drawn | Measured per planet | The light a parent reflects onto its moon is ≤ 1.4% of direct sunlight and its glow ≤ 1.1%. These errors move them by less | Minor for moons |

## Fixes built (2026-10-25)

Ben chose items 1–3. The changes are in `giants/moon_energy.py`, `giants/parent.py`, `giants/moons.py`, `water.py` and the heuristics block "Moon calibration fixes".

1. **Ice-shell dissipation.**
   - Which moons: those with at least 5% water whose equilibrium temperature at the thin-air albedo is below 180 K.
   - Their k₂/Q is interpolated log-log in x = μ_ice/(ρ_ice g R), with μ_ice = 3 GPa. The interpolation runs between Europa (x 1.59, 10⁻³; Bagheri et al. 2026) and Enceladus (x 115, 0.015; Bagheri et al. 2025: |k₂| 0.027 at a 34° lag).
   - The law is clipped to 3 × 10⁻⁴–0.015.
   - Rocky moons keep the task-11.8 melt switch.
   - Values: Ganymede 6.8 × 10⁻⁴, Callisto 7.9 × 10⁻⁴, Titan 7.2 × 10⁻⁴, Triton 1.5 × 10⁻³.
2. **Free-eccentricity damping.**
   - The rule: e = e₀ exp(−age/t_damp) (Tokadjian & Piro 2023, eq. 20), with the moon's dissipation in use.
   - Drawn systems: moons damp unless they belong to a 2:1 pair, whose members keep their forced e.
   - Moon worlds: they damp unless the spec gives e. Listed and kept moons keep theirs.
   - The report gains an "eccentricity damping" row.
3. **Porous ice lid.** Conduction through solid ice starts at 135 K (the middle of the 130–140 K that Shoemaker et al. 1982 and Nimmo et al. 2003 give, via Singer et al. 2023) when the surface is colder.

**Results.**

| Moon | Before | After | Observed or inferred |
|---|---|---|---|
| Europa heat | 233 mW/m² | 21.7 mW/m² | 15–35 |
| Europa shell / ocean | 2.7 / 126 km | 20 / 116 km | 19–39 km conductive |
| Ganymede | 159 km, no ocean | 120 km over 149 km | ~150 km ice, ≳100 km ocean |
| Callisto | 184 km, no ocean | 149 km over 149 km | ocean (induction) |
| Titan | 109 over 201 km | 94 over 239 km | ≥ 50 under ≥ 50 km |
| Enceladus | 26 km shell, 22.5 GW | 14.7 km shell, 22.4 GW | > 10 km, 15–40 km; 18–28 GW |
| Triton | 290 km, no ocean | 164 km over 209 km | an ocean is possible, not established |
| Io | 9.4 × 10¹³ W, molten | unchanged | 2–4 W/m² |

Over 150 habitable moons:
- No draw keeps tides above 0.1 W/m² (before: 27%).
- 79% of eccentricities fall below a tenth of their drawn value.
- Habitability verdicts are unchanged (25% habitable unaided). Redraws average 2.60 per moon, and 26% of the moons are regular.

`validate_solar_system.py` gains five checks: Europa's heat flux and shell, Ganymede's and Callisto's oceans, and Enceladus's shell above 10 km. All 72 checks pass. Tests: `tests/test_moon_calibration.py` (7).

**Left open by the fixes:**
- Enceladus's shell (14.7 km) sits just below the 15–40 km range: the lid thins it.
- Triton now keeps an ocean.
- Salts and adiabatic oceans are still not modelled.
- Archetype moons, alone in their systems, no longer have tidal heat. A resonant companion would be needed to give them some.

## Major items

### 1. Dissipation in ice shells

- **Model.** A moon whose tides at Io's k₂/Q (0.015) would exceed its radiogenic heat is taken to be partially molten, or to have an ice shell decoupled by an ocean, and gets that k₂/Q. This was task 11.8's melt feedback.
- **Io.** Correct: the measured k₂/Q is 0.0109 ± 0.0054 (Park et al. 2025, via Bagheri et al. 2026), and its tides come out at 9.4 × 10¹³ W.
- **Enceladus.** Its 22 GW sits within the 2× check against 15.8 GW.
- **Europa.** The same switch gives Europa 6.9 × 10¹² W, 226 mW/m², and a 2.7 km shell over a 126 km ocean. The constraints:
  - Juno's microwave radiometer gives a 19–39 km conductive shell (Levin et al. 2026, via Bagheri et al. 2026).
  - The surface heat flux is 15–35 mW/m² (Bagheri et al. 2026, Extended Data Table 1).
  - Ocean-floor heat fluxes in the literature are 6–46 mW/m² (Tobie et al. 2003; Howell 2021, via Daniel et al. 2025).
  - Shell estimates run 15–50 km (Soderlund et al. 2020, via Daniel et al. 2025).
  - Crater ring-graben give 2.5–8 km for the conductive lid at the time of impact (Singer et al. 2023).
- **Best practice.** Bagheri et al. (2026) couple the Laplace resonance to layered thermal models. Their Love numbers use Andrade rheology, and Europa's shell is split into a conductive lid and a dissipative basal layer.
  - Their best models put Europa's −Im(k₂) at 3 × 10⁻⁴–2.5 × 10⁻³, with ≥ 85% of the dissipation in the shell.
  - The shell is 22–33 km in total, 19–23 km of it conductive.
- **Measured.** Europa's shell against its dissipation:

  | k₂/Q | Heat (mW/m²) | Shell (km) | Ocean (km) |
  |---|---|---|---|
  | 3 × 10⁻⁴ | 11 | 54 | 74 |
  | 1 × 10⁻³ | 22 | 29 | 100 |
  | 2.5 × 10⁻³ | 44 | 14 | 114 |
  | 0.015 (now) | 233 | 2.7 | 126 |

  - Turning the switch off gives the drawn rocky Q: 60 mW/m² and 9.5 km.
  - The same change takes Enceladus to 2 mW/m² and freezes its ocean. One dissipation value cannot serve both moons; Enceladus dissipates more for its size.
- **Habitable (rocky) moons.**
  - 23% are switched to molten.
  - Surface temperatures move by ≤ 0.7 K, because the carbonate–silicate thermostat absorbs the extra heat.
  - The tectonic regime and pressure do not change.
  - The activity index rises from 0.9 to 6.7 (median of the switched moons). It shortens relict-feature memory and drives volcanic events in culture histories.
- **Impact.** The validation lets Europa's shell pass: only "ocean present" and "hydrosphere ≤ 200 km" are checked. It misses its shell by a factor of 10.
- **Fix options:**
  - Give ocean-decoupled ice shells a shell dissipation of ~10⁻³ (Bagheri's range), with Enceladus-sized moons kept on the switch.
  - Or solve the shell for a thermal–orbital balance: the tidal heat made in a dissipative basal layer equals what the lid conducts.
  - Add a shell-thickness check to the validation.

### 2. Free eccentricities never damp

- **Model.** The archetype draws e = 10⁻⁴–0.02 and holds it for the moon's whole age. Tidal heat is computed from that e.
- **Best practice.** A lone moon's eccentricity damps in t_damp = [(21/2)(M_p/m)(R/a)⁵(k₂/Q) n]⁻¹ (Tokadjian & Piro 2023, eq. 20). A resonance with another moon can force it, giving an equilibrium e (their eq. 22). Bagheri et al. (2026) likewise treat the Galilean eccentricities as forced.
- **Measured** with the moons' own k₂/Q:
  - t_damp is 0.083 Gyr in the median (10–90%: 3 × 10⁻⁴–12 Gyr).
  - 83% of the habitable moons would be circular by their age, 63% within a tenth of it. That is 100% of regular moons and 77% of captured ones.
  - All 40 moons with tides above 0.1 W/m² damp within their age.
  - 27% of draws report tides above 0.1 W/m² and 9% above 2 W/m², which the report flags as global volcanism.
- **Impact.** Removing these tides leaves the habitability verdicts and the climate unchanged: habitable unaided stays at 25%, and redraws go from 2.72 to 2.60. It lowers the median activity index from 0.63 to 0.37 and removes the volcanism flags. Drawn moon systems that are labelled resonant (Jovian chains) are physically allowed their e. Archetype moons, alone in their system, are not.
- **Fix options:**
  - Damp free eccentricity when t_damp < age.
  - Give resonance-chain members a forced e, and the archetype a companion when it wants tidal heat.

### 3. Thick ice shells sit on a threshold

- **Model.** The shell conducts from the surface temperature to the melting point at its base, with k(T) for solid ice. Freezing through is declared when the melting depth passes the ice Ih limit (208.6 MPa).
- **Observations.**
  - Ganymede has an ocean (induction and auroral evidence; Saur et al. 2015, via Cabanes et al. 2026).
  - Thermodynamic models with 10 wt% MgSO₄ put 26–95 km of ice over a 287–493 km ocean (Vance et al. 2018, via Cabanes et al. 2026).
  - Juno-era fits suggest ~150 km of ice over an ocean ≳ 100 km (Jia et al. 2025, via Cabanes et al. 2026).
  - Callisto's induced field is evidence for an ocean (Khurana et al. 1998, via Daniel et al. 2025).
- **Measured** (scaling of internal heat needed for an ocean):
  - Ganymede needs 1.01× its heat, Callisto 1.15× and Triton 1.41×. So Ganymede's "no ocean" rests on a 1% margin.
  - With 1.25× heat: Ganymede 133 km shell over a 110 km ocean, Callisto 40 km of ocean.
  - A melting point lowered by salts or ammonia (5–20 K) gives Ganymede only 1–9 km of ocean. The ocean floor is set by the high-pressure ice onset.
- **Best practice for the lid.** Fractured, porous ice near the surface conducts poorly. Planetary models treat the lid as starting at an effective 130–140 K, where pores close (Shoemaker et al. 1982; Nimmo et al. 2003; both via Singer et al. 2023). The shell and ocean with that lid:

  | Moon | Now | Lid at 130 K | Lid at 140 K |
  |---|---|---|---|
  | Ganymede | 159 km, no ocean | 119 km over 152 km | 108 km over 180 km |
  | Callisto | 184 km, no ocean | 155 km over 117 km | 140 km over 169 km |
  | Triton | 290 km, no ocean | 172 km over 201 km | 157 km over 217 km |
  | Titan | 109 km over 201 km | 79 km over 274 km | 71 km over 293 km |
  | Enceladus | 26 km over 47 km | 15 km over 58 km | 14 km over 59 km |
  | Europa | ~2–3 km (tide-dominated) | 2 km | 2 km |

- **Convection.** The flagged Rayleigh numbers (10⁸–10⁹) would make shells convect. In the parameterisation Bagheri et al. use (Nu = max[1, (Ra/Ra_c)^0.3], Ra_c = 10³), a convecting pure-water shell loses heat almost independently of its thickness (Nu ∝ Ra^0.3 ∝ d^0.9, so F ∝ d^−0.1; our derivation). It freezes faster unless the lid or the composition holds the heat in, so convection alone would not save these oceans.
- **Impact.** Whether a large icy moon has an ocean is a visible outcome: the report, the water section, the habitability of the subsurface. It flips with changes well inside the uncertainty of the inputs. The model's answer for Ganymede contradicts the observations.
- **Fix options:**
  - A porous conductive lid (effective surface temperature ~130–140 K).
  - Optionally, salts lowering the melting curve.
  - A stagnant-lid convection rule for thick shells.
  - Check Ganymede's ocean in the validation.

### 4. Air on icy moons is a coin flip

- **Model.** Retention is drawn from the cosmic-shoreline logistic (Zahnle & Catling 2017). The inventory is scaled by 1 + 255 × water fraction for cold water-rich worlds, fitted to Titan alone. A pressure scatter applies to every world.
- **Measured over 40 seeds** (seed 1, used by the validation, gives the observed answers):

  | Moon | Has air | Median pressure | Range |
  |---|---|---|---|
  | Ganymede | 52% | 0.047 bar | up to 0.31 bar |
  | Callisto | 28% | — | up to 0.085 bar |
  | Europa | 15% | — | up to 0.016 bar |
  | Io | 28% | — | ≤ 1 mbar |
  | Titan | 98% | 1.8 bar | 0.52–5.0 bar (10–90%), up to 11.8; 73–120 K |
  | Triton | — | 102–106 µbar | (item 5) |

  - Observed: no substantial air on Ganymede, Callisto, Europa or Io. Titan has 1.467 bar.
- **Best practice.** None settled. Why Titan is the only moon with a dense atmosphere is listed as an open question (Nixon 2024). Two origins of its N₂ are debated: enclathrated nebular N₂, or accreted NH₃ ice.
- **Impact.** For a spec that names a Solar System moon, the answer is wrong about half the time for Ganymede. For generated worlds the probabilistic treatment is defensible.
- **Fix options:**
  - A narrower shoreline width for icy moons.
  - Tie the icy inventory to formation temperature (a colder circumplanetary disc keeps NH₃), which needs a source.
  - Run the validation's air checks over seeds.

## Minor items (moved to OPEN_POINTS)

5. **Triton's frost temperature.**
   - The N₂ cap uses the climate model's 41.1 K, giving 104 µbar. The observed value is 14.5 µbar (Sicardy et al. 2024).
   - Bertrand et al. (2021) match Triton with N₂ ice of Bond albedo 0.7–0.8 and emissivity 0.8. Its energy balance gives 39.7, 37.9 and 35.9 K for A = 0.70, 0.75 and 0.80, and 51, 19 and 5 µbar.
   - The fix is a one-line frost balance.
   - Near the triple point the same 3 K error changes the cap by 2× (0.064 against 0.031 bar at 60 K). There it can cross the 0.1 bar haze threshold.
6. **Liquid N₂ is not capped.** Thick N₂ air warms its world: none of 40 Titan draws is below 87 K with more than 1 bar. Worlds in the liquid range with thick air are rare; a liquid vapour curve still needs a source.
7. **Long moon days.**
   - Habitable moons rotate in 2.5–25 days (10–90%), up to 70.
   - The climate tiers already use the rotation period for the Hadley cells and heat transport.
   - What is missing is a day–night cycle. No world in the generator has one, so moons are treated like every other planet.
8. **Parent glow and reflected light.** They are ≤ 1.1% and ≤ 1.4% of the direct sunlight. Eclipses remove ≤ 5%. Concentrating the glow on the sub-parent hemisphere would move temperatures by ≤ 0.4 K.
9. **Magnetosphere numbers.** The shelter fraction is reported but not used by any occupiability factor. 10× or 0.1× the stellar wind changes it by up to 0.69 on some moons and changes nothing else.
10. **Capture heating.** 10⁴–10⁷ yr of heating against Gyr ages (task 11.7b's numbers).
11. **Parent albedo and Uranus's heat.** Through reflected light and glow they shift a moon's energy budget by well under 1%. Their effect on the giants themselves stays in OPEN_POINTS.

## Sources

Read for this assessment:
- Bagheri, A. et al. 2026 — Joint dynamical-geophysical evidence for a limit cycle in the Galilean moons. https://arxiv.org/abs/2608.04226
- Shibley, N. & Goodman, J. 2023 — Europa's coupled ice-ocean system: temporal evolution of a pure ice shell. https://arxiv.org/abs/2309.16821
- Singer, K., McKinnon, W. & Schenk, P. 2023, JGR Planets, 128, e2023JE007928 — Thin ice lithospheres and high heat flows on Europa from large impact structure ring-graben. doi:10.1029/2023JE007928, https://arxiv.org/abs/2401.10380
- Daniel, F., Petitdemange, L. & Gissinger, C. 2025 — New constraint on Europa's ice shell: magnetic signature from the ocean. https://arxiv.org/abs/2509.10903
- Cabanes, S., Gastine, T. & Fournier, A. 2026, GRL (accepted) — Motional induction in Ganymede's ocean. https://arxiv.org/abs/2603.06305
- Tokadjian, A. & Piro, A. 2023 — Tidal heating of exomoons in resonance and implications for detection. https://arxiv.org/abs/2206.11368
- Bertrand, T. et al. 2021, Icarus (accepted) — Volatile transport modeling on Triton with new observational constraints. https://arxiv.org/abs/2110.11992
- Nixon, C. 2024, ACS Earth Space Chem. — The composition and chemistry of Titan's atmosphere. https://arxiv.org/abs/2402.17116

Also read for the fixes:
- Bagheri, A. et al. 2025, PSJ — Exploring the interior structure and mode of tidal heating in Enceladus. https://arxiv.org/abs/2505.14743

Cited through these papers (not read directly): Levin et al. 2026; Park et al. 2025; Tobie et al. 2003; Howell 2021; Soderlund et al. 2020; Shoemaker et al. 1982; Nimmo et al. 2003; Vance et al. 2018; Jia et al. 2025; Saur et al. 2015; Khurana et al. 1998.
