# Open points

Items that remain open at v1.0.0, by subsystem. Impact is the measured effect against the Earth value or other source, where one exists. Sources are in [references.md](references.md); the model is described in [design.md](design.md). *Parked* items were left for after the release.

## Climate and ice

Numbers are the Earth example at standard resolution over six simulated seeds (design.md 2.7).

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Polar temperature | 257–260 K poleward of 70° against 246 K. Earth's value comes from the Antarctic plateau; the seeds have 3–9% of land poleward of 70° and no such plateau | A geography with a polar continent; no model change | Open |
| Land under ice sheets | 2.4–9.4% (mean 5.5%) against ~10%, for the same reason | The same; also ice flow instead of a plastic profile, ice shelves, and ice volume taken from the ocean over time instead of in one step | Open |
| Sea level from large ice sheets | Large sheets on cold worlds lower sea level by hundreds of metres; the cap of half the surface water applies on some cold or dry worlds | Ice-flow model with realistic thickness | Open |
| Snowline erosion | Does not deposit the material it removes | A deposition rule | Open |
| Snowline | 2.3 K too cold against Ohmura et al. (1992) (median over 70 glaciers; 1.3 K at 40–60°, 3.3 K in the tropics); ~350 m too high | Meltwater refreezing (Pfeffer et al. 1991) and tropical sublimation and year-round melt. A 3.5 K daily spread would remove 0.7 K but contradicts the 4.4 K measured at glacier stations (Wake & Marshall 2015) | Open |
| Köppen classes | C 17–28% of land against 13%; D 12–30% against 25%; Dw 0–16% depending on continent layout | Not diagnosed beyond the dependence on layout | Open |
| Ocean heat transport | Fixed zonal shape: no western boundary currents, basin geometry or lobster-shaped open ocean on locked planets. A larger ocean share (0.5) matched the ocean-dominated latitude better but cooled the tropics and warmed the poles by 3–4 K | Basin-resolving transport | Open |
| Sea ice | Keyed to a −7.5 °C air temperature; the ice edge follows the temperature profile, so colder poles and less sea ice cannot both be reached | Not closed by ocean transport | Open |
| Water phase near the tipping point | Where a partly frozen and a snowball state are both stable, the warm one is used; Tier 1 can settle on a different state from Tier 0 (the report warns above 10 K difference) | Decide the state in history mode | Open |
| Slow and synchronous rotators | Circulation not checked against published GCM results; the transport factors are calibrations (day–night contrast ~64 K at 1 bar) | Comparison with GCMs; a Tier 3 backend (ExoPlaSim) if too weak | Open |
| Winds, clouds, currents | Not explicit in the energy-balance climate | Out of scope for v1 | Open |

## History mode

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Carbon cycle C1: crust–mantle partition | Crust ends at 317 against `CARBON_CRUST_EARTH` = 260 (+22%), carbon still moving to the mantle at 38 ×10¹⁸ kg CO₂/Gyr (~30 Gyr relaxation); arc degassing 186 against 153. Pinning the crust at 260 gives 288 ppm | Set the initial split so the crust arrives near 260, or shorten `CARBONATE_TURNOVER_GYR`. At steady state weathering equals twice the volcanic flux whatever the turnover time | Parked |
| Carbon cycle C2: light air | Air ends at 0.965 bar instead of 1.013; τ = 0.827 against 0.850, which costs ~1 K and is paid for in CO₂. Scaling N₂ to 1.013 bar gives 303 ppm and 288.05 K | Find where the N₂ budget loses 48 mbar (initial inventory, escape share, thin-air floor in `rhs`) | Parked |
| Carbon cycle C3: organic carbon | Burial and oxidation are in the O₂ budget (`OXYGEN_BURIAL_EARTH` = 320) but not the carbon budget; the unaccounted net is 24% of the silicate sink today and 28% at 1 Gyr. Sign at present day uncertain | A kerogen reservoir and an oxidised/reduced split of volcanic carbon (Berner 2003); do after C1 and C2 | Parked |
| C1 and C2 together | 254 ppm at 287.73 K, an undershoot; the source needs a small upward adjustment | One task, one recalibration | Parked |
| Earth end state | 343 ppm and 287.31 K against 280 ppm and 288.0 K; the sink is calibrated (305 against `WEATHERING_EARTH` = 310 at 280 ppm); `validate_history.py` still passes | C1 + C2 + C3 | Parked |
| Early Mars | Frozen from formation at 0.31 instellation, so no valley-network rivers (ice relicts instead); 42 mbar of CO₂ against 6.4 | Warming early Mars is unsolved in the literature | Open |
| Snowball limit cycles | Runs of 20–40 s against a 3–5 s median on planets near the tipping point | Fewer integrator restarts | Open |
| Relicts | Marks survive ~100 Myr of erosion on Earth; the climate series is sampled every ~20 Myr, so only snowballs register as past cold epochs | Finer climate sampling | Open |
| Present-state overrides | History is not solved to end at them: 3 bar requested ends at 0.95 bar and 287 K | A two-target solver (pressure via initial volatiles, ocean volume via initial water) and per-subsystem modes; milestone 13, dropped | Dropped |
| Moons | History mode ignores the extra flux from tides and a parent's glow | Pass the flux into the integration | Open |
| Rocky mass–radius below 1 M⊕ | Zeng relation extrapolated; Mars's core fraction 0.46 against ~0.25 | A relation fitted below 1 M⊕ | Open |

## Tectonics and surface

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Sea-floor age | Median 77–118 Myr against ~60 Myr. Old basins enclosed by continents on one plate are never consumed (oldest crust ~550 Myr) | Easier subduction initiation at passive margins, calibrated by sweeps | Open |
| Plate count | 17–27 for the Earth spec against ~15 | More plate fusion, calibrated by sweeps | Open |
| Seed-to-seed spread | Land fraction 23–27% with the same water; highest point 6.4–8.2 km | Inherent to a chaotic system; calibration uses ranges over seeds | Open |
| Continental thickness | No crust thickness, so collisions do not thicken or shorten continents (continental area is held constant); no terrane suturing; hotspots fixed; detail noise fixed in place, so snapshots omit it | A crust-thickness field in the plate simulation | Open |
| Resolution dependence | Plate outlines and belts are coarser at preview. Earth example land fraction 0.15–0.19 at preview against 0.24–0.27 at standard (seed 1: the preview ocean floor averages 3.2 km against 3.4 km and its land 550 m against 725 m, so 22% of continental crust floods against 17%) | A resolution-aware hypsometry, or use standard for any quantitative work | Open |
| Land-fraction estimate | The estimate made before the surface exists differs from the result by up to 0.3 on planets far from Earth-like. A requested land fraction is met within ~0.01; remaining misses come from flat sediment-filled plains that flood at once | More reference seeds | Open |
| Lakes and inland drainage | Lakes cover 0.5–1.7% of land against 2–4%; land draining inland 1–12% against ~18% | Sills and sill erosion for large closed basins | Open |
| Rivers at grid scale | Only rivers draining at least 10⁵ km² appear; courses follow cell centres. Single-cell hollows in mountains can hold small lakes | Handled by zoom | Open |
| Sediment volume | Continental crust holds 0.78–1.05 km on average against 1.46 km (CRUST1.0): interiors lack sag basins (no interior stretching events) and the 400 Myr window is shorter than Earth's accumulation | Interior stretching events in the simulation | Open |
| Ocean sediment pattern | Pelagic rain has no productivity field, so GlobSed's thicker equatorial and high-latitude sediment is absent; the ratio to the GlobSed law varies 0.9–1.8 between seeds. River sediment that finds no room spreads over the whole sea floor instead of staying near the margins | A productivity field; a margin-trapping rule for river sediment | Open |
| Heuristic and non-plate surfaces | No subsidence or biogenic sediment: 1% of continental crust above 1 km sediment on the heuristic layout, so sediment-bound resources are weak | Use simulated tectonics where resources matter | Open |
| Resource calibration | Hydrocarbon factors fit Mann et al. (2003) and give rifts a high factor because rifts are rare on the grid (0–3% of continental crust); strike-slip basins (6% of giant fields) are absent because transforms rarely cross continents. Other classes have no quantitative Earth target; district size and placer transport are heuristic | Quantitative Earth targets for the other resource classes | Open |
| Erosion without wind | Dry plate-tectonic planets keep the uniform decay, frozen ones keep it during the simulation, and dry non-plate planets are not eroded | An aeolian erosion term | Open |
| Minimum mapped sizes | Craters below ~3 grid spacings and volcanoes below ~1.2 spacings are not mapped; heat-pipe volcanoes are small at global scale | Handled by zoom | Open |
| Biosphere rules | Life presence follows fixed rules and is decided before life's effects on the atmosphere, so ~1% of ice-preset draws keep surface life on a frozen surface. O₂ follows an Earth-like history in snapshot mode; methane and alien products use simple scalings. Vegetation responds to annual climate only. The Tier 0 productivity estimate uses a uniform land rain rate | Feedback between presence and atmosphere; seasonal vegetation | Open |

## Zoom

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| One cube face per region | Regions across a cube edge are not supported. Separate regions on two faces disagree where they meet: up to 6 m at level 7, 22 m at level 4, 53 m at level 3 | Cross-face stitching for single regions | Open |
| Relief below a few km | Terrain ruggedness of synthesised relief at 0.9 km is 0.19 + 1.0 × ruggedness, at most ~1.1 in mountains, against 5.0–6.2 for Nepal and Lesotho (Nunn & Puga 2012) (Hurst 0.5) | Exemplar patches or a higher Hurst exponent; defensibility uses sub-grid slopes meanwhile | Open |
| Lakes | 0.5–0.9% of mountain land and almost none on plains without a remembered glacial epoch, against 2–4% on Earth. Temperate history worlds rarely remember one, because the climate series is too coarse for glacial cycles | Finer climate sampling in history mode | Open |
| Flow on a lattice | Steepest of eight neighbours, so breach cuts and flats show short straight stretches, more at fine spacing | Multiple-direction routing | Open |
| Biomes on local maps | Categorical fields take the nearest global cell, so boundaries are straight-edged cells; not reclassified from the downscaled climate | Reclassify from the downscaled climate | Open |
| Pinned edges | Edge bands carry less incision (erosion fades in over ~4.5 km) and a smoother band with a slope kink (median step 28 m per node at the seam against 16 inside, 0.62 km lattice); valleys crossing an edge continue only where global rivers enter | A wider overlap | Open |
| Level consistency | The same ground at levels L and L + 1 differs by 33 m rms at shared nodes and 25 m rms (max 93 m) on tile edges. The blend band makes exported seams exact against the parent, but a finer tile is not a refinement of its parent's heights inside the band | Not planned | Open |
| Erosion at coarse spacing | Scale-dependent: mean depth 84 m at ~9 km, 24 m at ~2.4 km (15 Myr, K = 0.08), and at ~9 km the channel-head rule makes 79% of land nodes channels. Validated at 0.09–1.7 km only; the pyramid re-levels each level onto its parent instead of recalibrating K, so channel fields of coarse levels mean little | Recalibrate K, creep limit and edge fade per spacing | Open |
| Glaciers | Ice flux is balance-routed, not solved from ice flow; outlet glaciers from large caps can run 100 km or more; no ice shelves or moraines; glacial sediment leaves the region | Ice-flow solver | Open |
| Detail | Procedural only; non-plate regimes have no structural grain; channel heads use the lattice slope, so coarser lattices start fewer channels | Exemplar patches | Open |
| Cache validity | The cache is keyed to the request, the world's surface and the package version, not to the heuristics | Key the cache on a heuristics hash; until then delete `regions/` after changing them | Open |

## Culture: user-scoped

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Largest polity | 0.41 of Earth's at the writing level (0.32–0.46), 0.90 print, 0.67 telegraph, 0.76 digital; the count solver fixes the effective number and the influence weight moves the largest polity only weakly. Effective numbers for writing (300) and print (40) are interpolated between Taagepera's dates | A source for writing-level state sizes | Open |
| Hot worlds | Where settlements cluster (hot worlds with people near the poles) the effective number can fall short of its target (Kestrel: 49 against 398 at writing, 20 against 32 at digital) | A polity count search that works with clustered settlements | Open |
| Main-road density elasticity | 1.02 against 0.80 (Meijer et al. 2018); area elasticity 1.06 against 1.01. Noisy at 22 regions per world | More regions per world | Open |
| Roads | Grid road density sits near the one-road-per-edge cap (~0.026 km/km²); degree at grid nodes (4.6) is not comparable with street networks (2.5) because grid nodes are 124 km cells. Cultures route separately, so there is no trade between cultures and roads of neighbours may run in parallel. The route factor (1.13) is a lower bound for real roads | Denser networks belong to zoomed maps; shared routing between cultures | Open |
| Population concentration | Coast shares vary 0.24–0.34 within 100 km and 0.68–0.84 within 400 km between seeds; the person-75th density quantile is 10.8 against 15.0 times the mean, since the grid cannot hold city density | Not possible at 124 km cells | Open |
| Intermediate development levels | Only foragers and the modern set are anchored on data; farming to fossil interpolate and fusion reuses the modern set. Forager density at the 99th percentile is ln 3.1 per 100 km² against ~ln 5 for the densest ethnographic groups, because the HYDE total fixes the mean | Density data for the intermediate levels | Open |
| Uncalibrated population terms | Colonist terms, colony growth (10⁴ founders, 2% a year, 200 years by default) and reach, the fusion population and the homeland edge. Cultures in overlapping ranges share cells instead of competing; resources and defensibility do not enter native capacity | Colonisation data to fit them | Open |
| Settlements | Region area per transport level beyond level 2 is a heuristic ×3 step; the largest modern settlement (50–96 M) has no target; the largest water_wind settlement varies 0.72–1.15 M between seeds. Settlements take people from the nearest cells, so a large city can empty its surroundings; Zipf holds from 50 people up, which is approximate below a few thousand | A modern largest-city target | Open |
| Borders | The 124 km grid cannot show whether borders follow rivers or crests: 18% of border edges touch a large river against 16% of all edges. No dependencies or vassals; cultures do not share polities | Finer grid or zoom-level borders | Open |
| Names | Segments beyond the core are drawn independently, so implicational patterns (voiced stops only with voiceless ones) are not enforced. Inventories stop at 42 of 43 consonants, so very large ones (up to 122) are missing and the mean is 21.8 against 22.7. Complex-syllable cultures read harshly in Latin letters. No tone, stress, morphology beyond place-name elements, or borrowing | Implicational constraints from PHOIBLE | Open |
| Habitats | Capacity shares of sealed habitats (0.3, 0.2), the lava-tube bonus, landing weights, the elevator's maximum taper and latitude are not calibrated. Radiation shielding counts only the air column; a magnetic field does not lower it. Judged on annual means. No orbital or floating habitats, so Venus cannot host colonists | Calibration data; orbital habitats | Open |
| Zoomed culture | Village suitability, the border band and river barrier are heuristic; the barrier order was tuned to Popelka & Smith's 23% and gives 0.32 at present (their river set differs from the lattice's). Villages follow rank-size down to 50 people. Footprints are discs, not street plans. Local roads do not know the global flows. Mean junction degree 3.6 has no like-for-like Earth value | A like-for-like river set and junction statistics | Open |
| Culture maps | Roads follow 124 km cells, so dense regions draw as a mesh and the political map shows only main roads and the 300 largest settlements. Polity colours are not chosen so neighbours differ; labels can overlap | Graph colouring; label layout solver | Open |

## Culture: organic history

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Spread of development timing between worlds | Takes thousands of years at one rate. First fossil 25–525 years ago in the 3 of 6 worlds that industrialise; half the worlds never do (the densest 150 km neighbourhood and fuel near it decide). Population ÷ HYDE falls to 0.40, 0.28 and 0.10 at 1900, 1950 and 2015 | Investigate what sets the spread, then refit | Parked |
| `KNOWLEDGE_RATE` | 0.026 is provisional: 0.022 was fitted over 18 histories (six worlds, seeds 0–2), then raised to keep the timing after early farming's capacity was lowered and knowledge was tied to the people in the sphere (two worlds × two seeds). Timing moves 1000–2000 years for a 15% change | Multi-seed refit | Parked |
| Natives kept after contact | Not measured at present: no natives within 1000 km of the landing site 500 years ago on either colony world, although 3.4 million live elsewhere on that landmass (world 0). Earth: 0.10 (Koch et al. 2019). The cause (regression or landing choice) is not established | Compare with a run before the latest changes; choose the landing among cells with natives | Open |
| Early states | Largest state ÷ land at 500 BCE and 1 CE is 0.090 and 0.097 against 0.041 and 0.047; more than half the worlds have no state at 2800 BCE (median 0). Capitals run ahead of Earth's development (communication 1.1–1.95 at 500 BCE) | Calibrate state formation against early-state data | Parked |
| Largest state in 1900 | 0.142 against 0.226 (Taagepera); half the worlds never industrialise and no tied colonies survive to 1900, while the British Empire of 1900 was mostly colonies | Follows from the timing spread | Parked |
| Arable share | 0.70 (0.62–0.77) against 0.61 (Zabel et al. 2014); 0.63 on the standard Earth example, 0.79 on the preview heuristic world | Recalibrate the arable fuzzy minimum | Parked |
| Steppe belts | 5724 km on average (3730–8480); the largest steppe empire after the first water_wind is met only on worlds with a belt of 6000 km or more | Not planned | Open |
| Polities and wars | States that reached 10⁶ km² last 274 years (Arbesman 220); the conflict rate counts all conflicts where COW counts wars of 1000+ battle deaths; severity is a stand-in whose exponent matches Clauset's but whose scale is arbitrary. Attack and expansion chances are per border edge, so the rate per km of border depends on the grid. Cederman's periods, the 2.5-year rounds, province size, primogeniture adoption and alliances are heuristic. Secessions repeat: the same regions break away from the same state step after step (349 secessions in 520 years around one colony run) | Per-length chances; a memory of recent secession | Parked |
| Settlements through time | Ids and dates belong to grid cells, so the largest settlements date from their cell's first farming (6000–9000 years before each epoch), older than the dates at which cities reached 5000 people (Bosker & Buringh 2017). A settlement changing cell or in-cell order gets a new id. Roads have dates but no stable ids. Trade does not feed back into settlement sizes | Cell-independent settlement identity | Parked |
| Farming front | 1.10 km/yr (1.02–1.17); does not rise with transport level; on the preview grid pack_and_boat crosses no strait (a full ~250 km cell of sea); moves into full cells overshoot capacity by up to ~5% for a step | A mobility per level; strait width below cell size | Parked |
| Development axes | Transport and communication are tied to energy for natives (no literate farming society without wheels); knowledge never declines except through planet events | Independent axes | Parked |
| Colonists | The epidemic speed (500 km/yr), home power (a 10⁶ km² state), war factor for independence (5), adoption lag before 1820, language status rule, bilingual ease (0.80), creole threshold (10⁴ bilinguals) and scope, supremacy chances, overseas rate (0.005/yr per state), reach and levels, and secession share (2/3) and minimum (3 cells) are heuristic. Comin & Hobijn's lags are national adoption lags applied to peoples; Strang's rate is fitted to Western dependencies and capped at his 487 years. The home power has no economy or fleet. Bilinguals do not grow before a creole forms. Settler mortality uses malaria only. The 250-year no-growth window after an epidemic has no source. Off-planet colonies are not validated | Sources for the heuristic rates | Parked |
| Planet events | Drought rates, sizes and durations, the cooling law, rain variability, the capacity cost of cold, Hsiang's effect applied to break-up, shock migration, pandemic scaling (duration, spread, cap, reach), impact damage and knowledge-loss thresholds are heuristic. Marani's law is fitted to the connected world of 1600–1945 and extrapolated; Spanish-flu-intensity pandemics 0.67 per world against ~1.4. Eruptions draw on the planet's activity; other events do not see the planet. The eruption Pareto law ends at the largest on record, so super-eruptions (M8 every 17 kyr, Rougier et al. 2018) are absent unless scripted. Sea level can only rise, by script. No lasting immunity | Sources for the heuristic terms | Parked |
| Outputs | The chronicle's thresholds (10⁷ km², 100 states, 10⁷ deaths) and periods are heuristic and leave ~500 major entries per Holocene; the text is fixed when the history runs, so states never named at a kept epoch stay numbers. Epoch panels crowd labels. Maps exist only at kept epochs. Livestock is stored but not drawn on maps | Chronicle filtering; livestock map layer | Parked |
| Organic and scoped cultures | A scoped native culture without a homeland takes land that organic foragers also hold, and both live there | Share out land between the two modes | Open |
| Warning in `wars.py` | `RuntimeWarning: divide by zero encountered in power` at line 146 when the strength ratio is 0; the result (probability 0) is correct | Wrap in `np.errstate` | Open |

## Export and 3D globe

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Global rasters | Blend the 40k cells (~124 km), smooth beyond ~2048 px; equirectangular rasters oversample the poles; the sampler takes 21 s to build at 4096 × 2048 | Finer global relief comes from the pyramid | Parked |
| 16-bit codes | Heights are quantised to 1/65535 of the world's height scale (0.31 m on the Earth example) | None | Open |
| Closed basins in the pyramid | Breaching leaves depressions deeper than the 300 m limit and basins under ice: up to 0.045% of ice-free land on a level-3 face, 0.023% on chain levels. The pyramid does not mark them as lakes | Mark them as lakes | Parked |
| Pyramid from the command line | `worldgen export pyramid` rebuilds on every run (6.5 min for levels 0–3 on Earth); `build_pyramid` returns it in Python | A saved pyramid | Parked |
| Pyramid chains | Heights differ from a standalone zoom of the same block by 34–57 m rms (relief std 27–198 m); the blend band at a chain's edge is smoother; rivers, lakes and ice are those of the zoom build, not recomputed after re-levelling and breaching; chains need neighbouring tiles at most one level apart | Recompute rivers after re-levelling | Parked |
| Region vectors | A 150 km region around a 2.4-million town holds 54 000 settlements and 103 000 road segments (17.7 and 22.5 MB of GeoJSON); roads are written per segment. Coordinates do not handle a region crossing the antimeridian | Merge segments; antimeridian handling | Parked |
| Engine lattices | A 150 m request on the 127-node lattice gives a 70 m step, below the validated 0.09–1.7 km; Unreal's 8129 and Unity sizes above 1025 exceed the zoom's node limit | Raise the node limit | Parked |
| Meshes | Tiles use their own vertex normals, so lighting can step at tile borders; float32 local vertices limit a level-0 tile to ~0.25 m on Earth; regions off 2ⁿ + 1 lattices are not simplified; the sea layer is opaque; the exact RTIN error costs ~3× Martini's and keeps up to a third more triangles at mid tolerances | Shared border normals | Parked |
| Globe viewer | The trame (Jupyter) path is not covered by tests (no Jupyter server in the cloud workspace); menus and slider unchecked. Draped regions keep their local colours when the field changes and the globe shows through at their edges. HTML pages hold one field and epoch (~20 MB at level 2) | A test with a Jupyter server | Parked |
| Seams across cube edges | See Zoom, one cube face per region | | Parked |
| Native files | GIS tools that warp GCP files with a polynomial misplace large blocks by up to 17 lattice steps; the sidecar names the thin-plate method and the LAEA option avoids it | None | Open |
| Third-largest city on Earth example | Zooming around the 9.8-million coastal city gives no settlements, villages or roads: its node is sea on the refined coastline with no land within three lattice steps, so it is dropped | Snap coastal settlements to land | Open |
| DOI of Losasso & Hoppe (2004) | Not confirmed; `references.md` cites the author's page | A DOI | Open |

## Giants and moons

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Default envelope by mass | `bulk.classify` calls planets of 10–132 M⊕ "ice giants" (Chen & Kipping), so a Saturn-mass planet without an envelope fraction gets 0.15 and behaves as an ice giant | Envelope fraction rising with mass between the classes | Parked |
| Ice-giant internal heat | Drawn within ±0.5 dex of the geometric mean; T_eff 48.8–65.0 K over 20 seeds against Uranus 59.1 K and Neptune 59.3 K; a Uranus spec cannot ask for its faint interior | A spec field for the internal luminosity | Parked |
| Gas-giant albedo by class | Jupiter and Saturn share 0.46 (0.04 from each measured value); the Bond albedo does not use the layered colour model, and Sudarsky's albedos are used for G2V only. Under 1% of a moon's energy budget | Feed the layered albedo back; stellar-type dependence | Parked |
| Classes IV and V | No model spectra in Cahoy et al. (2010): they borrow the shapes of classes III and II; thermal glow of hot giants is not drawn, so most hot Jupiters are uniformly class V grey | Model spectra | Parked |
| Hot-spot offset | Fixed 30° (models span 10–60°; weakly forced planets show none) | A dependence on radiative and advective timescales | Parked |
| Sub-Neptunes | Fixed albedo 0.34 and no appearance | A sub-Neptune appearance model | Parked |
| Giant report | Climate and Biosphere print one line each; the atmosphere section gives only the equilibrium temperature, while the colour model computes a 1-bar temperature | Report the 1-bar temperature | Parked |
| Tuned cloud-top numbers | Vortex density (0.008 per L_d²), great-oval, ring and hexagon probabilities, off-equatorial jet speeds (0.6 of the equatorial peak), band temperature contrast (3 K) and relief set by eye; Saturn's mid-latitude jets are faster than observed | Counts from Voyager and Cassini catalogues | Parked |
| Time dependence | Cloud tops are a single state: no drifting vortices, seasonal storms or transient Neptune clouds | A time-dependent cloud model | Parked |
| Rendering cost | A 2048 × 1024 cloud raster takes ~11 s and a level-2 globe ~7 s; noise dominates | A cached texture | Parked |
| Haze laws | Two-point fits (tropospheric haze through Jupiter and Saturn; methane-level haze through Uranus and Neptune; chromophore amount). Production against UV flux, gravity and temperature is not modelled, so giants unlike these four extrapolate | Haze microphysics or more calibration planets | Parked |
| Adiabat factor | α lowered from 0.85 to 0.6 for metal-rich envelopes lets methane condense on ice giants; ice giants come out ~5 K warm at 1 bar (Neptune 77 against 72 K) | A heavy-envelope lapse rate | Parked |
| Fixed colour-model pieces | Ice giants' deep cloud is a fixed 6 bar floor (no H₂S or NH₄SH chemistry), helium Rayleigh scattering ignored, CH₄ absorption has no temperature dependence, methane above its triple point does not condense; Neptune is a little paler than observed (sRGB distance 10) | Chemistry and temperature-dependent absorption | Parked |
| Tuned ring numbers | Dense and dusty ring probabilities (0.75, 0.9), the 60–80 K window, the 0.3 albedo split, the 10 Myr PR survival threshold and ring spectra come from four giants and Chariklo | More ring systems | Parked |
| Ring mass | Scales with the planet; Ohtsuki (2025) has massive rings converging to ~10¹⁹ kg whatever their start | An absolute ring mass | Parked |
| Rings and moons | Only the Cassini Division responds to moons; Uranus's shepherds, Jupiter's ring-feeding moons and Neptune's arcs are not tied to moons; rings do not respond to ring-born moons; rings lie in the equatorial plane and are unlit in the globe viewer | Moon–ring coupling | Parked |
| Tidal-survival limit on rocky planets | Barnes & O'Brien's limit assumes the star's torque dominates; for Earth–Moon mass ratios it fails, yet drawn moons heavier than the limit are dropped on all planets | Apply it only where its assumption holds | Parked |
| Moon systems as outcomes | Regular moons of gas giants follow Sasaki et al.'s published outcomes scaled by the planet; planets far from Jupiter and Saturn get the same shapes. The synthesis prototype gave too many Jovian moons and too few single Saturnian ones | Formation model; Canup & Ward (2006) in full | Parked |
| Tuned moon numbers | Split of outcomes Sasaki et al. do not give (20% Jovian, 33% Saturnian), template scatter, 5th and 6th moon spacing, nebula aspect ratio and α, water fractions and densities, ring-born disc masses and γ = 0.01–0.5 | Source data | Parked |
| Ring-born moons close in | With giant Q = 10⁵ the outermost reaches 5 R_S at Saturn (Rhea 8.7) and 7.8 R_U at Uranus (Titania 17); Saturn's Q is nearer 2500 (Lainey et al. 2012) | A Q per planet | Parked |
| Impact moons and spin | Recede from 1.3 Roche radii by the planet's Q; formation inside the synchronous orbit, angular-momentum budget, dual lock, and spin change of the planet are not modelled | An angular-momentum budget | Parked |
| Resonance chains | Placed at exact 2:1 period ratios; stability and ring-born crossings not checked | A stability check | Parked |
| Captured moons | Each giant meets exactly one deep binary at 2–10% of orbital speed; member ranges (10⁻⁴–10⁻² M⊕ beyond the ice line, 10⁻³–1 M⊕ inside) are judgement. Rufu & Canup's survival curve is extrapolated beyond three system masses, which alone makes Jupiter and Saturn keep no captured moon. Capture heating (~10 MJ/kg) is reported but does not reach the interior. Irregular moons (Phoebe, Nereid) are not drawn | A binary population model | Parked |
| Uranian systems | Template of Uranus's moons at 50% of ice giants; the impact does not tilt the planet and the dependence on impactor mass is not modelled | More impact simulations | Parked |
| Habitable-moon archetype | Mass splits (0.4 regular, 0.6 captured) are judgement; regular habitable moons need 3–13 M_J parents; those below 0.25 M⊕ still get plate tectonics; about 3% of specs that fix nothing about the body come out as moons; alone in their systems, archetype moons' free eccentricities damp, so none has tides | A resonant companion for a tidally heated habitable moon | Parked |
| Parent glow by mass | A parent below 50 M⊕ is taken as an ice giant for its internal heat | An envelope fraction in the moon spec | Parked |
| Magnetosphere | Dipole interpolated from four planets; solar wind fixed at 2 nPa at 1 AU with no stellar type or age (Grießmeier et al. 2007 unread); shape factor 1.2 from one statement; radiation doses not modelled | Sources for stellar wind and belts | Parked |
| Moon days and parent glow | Moon days of several Earth days (Titan 16) are treated as fast rotation by the climate tiers (rotation 2.5–25 days for 10–90% of moons); the parent's glow (≤ 1.1% of direct sunlight) and reflected light (≤ 1.4%) are spread evenly; concentrating the glow moves temperatures by ≤ 0.4 K | Day–night cycle in the climate tiers | Parked |
| Parent tides | Static bulge and eccentricity tide on the moon not reported (no source read) | A source | Parked |
| Small-body densities | Uncompressed densities (7900, 3400, 1300 kg/m³) and the 0.1–1 M⊕ blend are heuristic; worlds between 0.1 and 1 M⊕ changed radius and hence history | Mass–radius data | Parked |
| Icy-moon air | A coin flip: Ganymede has air in 52% of 40 seeds, Callisto 28%, Europa 15%; Titan ranges over 0.5–5 bar. The icy inventory factor (255 per unit water fraction) is fitted to Titan alone | A narrower shoreline for icy moons, or a formation-temperature inventory (needs a source) | Parked |
| Triton's N₂ frost | The cap uses the climate model's 41.1 K (103 µbar against 14.5 observed); N₂ ice with Bond albedo 0.75 and emissivity 0.8 balances at 37.9 K and 19 µbar (Bertrand et al. 2021). Near the triple point the error is 2×, enough to cross the 0.1 bar haze threshold | A frost albedo for N₂ ice | Parked |
| Liquid N₂ | Not capped between the triple point (63 K) and the critical point (126 K); no Titan draw of 40 is both below 87 K and above 1 bar | A liquid vapour-pressure curve (needs a source) | Parked |
| Enceladus and Triton shells | Enceladus's shell is 15 km, at the edge of the 15–40 km Bagheri et al. (2025) infer; Triton keeps an ocean (164 km of ice over 209 km), possible but not established | Observational constraints | Parked |
| Ice-shell model | No salts, ammonia, convection or adiabatic oceans; the dissipation law runs through two points (Europa and Enceladus) and is heuristic in between and beyond | More anchors | Parked |
| Stellar wind | The shelter fraction responds to the wind (up to 0.69 for 10× the wind), but nothing else reads it | Couple to atmospheric loss | Parked |
| System views | Moon shadows use true positions, so a transit shadow does not line up with the drawn moon in the compressed scale; ring opacity uses the starting camera angle and is not brighter when lit from behind; HTML pages show one time; moons turn about the planet's spin axis and ignore their inclination; crowded diagram labels are dropped; rebuilt parents of standalone moons are stand-ins (default envelope, no rings or other moons, origin "user"); HTML export offline prints a trame traceback but writes the page | Layout solver; forward scattering | Parked |

## Reproducibility

| Item | Measured impact | Information or change required | Status |
|---|---|---|---|
| Worlds differ between library versions | The same spec and seed give a different Earth preview under Python 3.13 with current libraries and under Python 3.10 with the oldest declared libraries (118 against 201 candidate border towns in the culture test world). Cause not identified. Tests that need a particular world select it by property instead of by position | Find the call whose result depends on the version (random streams, tie-breaking in sorts or searches, solver tolerances); until then, state the library versions with any published world | Open |

## Dropped and planned

| Item | Status | Reason |
|---|---|---|
| Tier 2 climate (SWAMPE adapter) | Dropped | It would change winds but not the outputs users see (rainfall, biomes, habitability stay with Tier 1's moisture), at minutes per world, outside history mode, with a second coupling to converge |
| History-mode target solver and per-subsystem mode selection | Dropped | See History mode, present-state overrides |
| Tier 3 climate (ExoPlaSim) | Not planned | Only if Tier 1's circulation proves too weak for locked or slow rotators |
| Ocean circulation and ocean life | Later | Basin-resolving transport would give boundary currents and locked-planet open-ocean shapes; ocean life would give plankton colour |
| Run time | Open | Coupled climate adds 3–7 s per standard world; a zoom region takes 7–15 s |
