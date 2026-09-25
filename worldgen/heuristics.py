"""Tunable constants for relations that have no standard published formula.

Every value here is a heuristic calibrated against Solar System bodies
(Earth, Venus, Mars, Mercury, the Moon, Titan, Io). Modules that use them
flag their outputs as heuristic in the planet report. Change values here to
retune the generator; nothing else in the code hard-codes these numbers.
"""

# --- Star -----------------------------------------------------------------

# Luminosity brightening over the main sequence, generalised from the solar
# relation L(t) = L_now / (1 + 0.4 (1 - t / t_sun)) (Gough 1981).
BRIGHTENING_AMPLITUDE = 0.4
SUN_MS_FRACTION = 0.457          # Sun's age as a fraction of its MS lifetime

# Radius grows with luminosity during the main sequence as R ∝ L^exponent.
RADIUS_LUMINOSITY_EXPONENT = 0.4

# Before the main sequence a contracting star is brighter: L = L_MS (t_ZAMS / t)^exponent,
# capped near the birth line, at the same effective temperature. Fitted to the tracks of
# Baraffe et al. (2015): the main sequence is reached at ~30 Myr for the Sun and ~0.8 Gyr
# at 0.1 M☉, where the star is then ~3 times brighter at 30 Myr.
ZAMS_TIME_SUN_GYR = 0.03
ZAMS_MASS_EXPONENT = -1.4
PRE_MAIN_SEQUENCE_EXPONENT = 0.4
PRE_MAIN_SEQUENCE_MAX = 30.0

# XUV activity: saturated at L_XUV/L_bol = XUV_SATURATED until t_sat, then
# decays as (t / t_sat)^XUV_DECAY_EXPONENT (Ribas et al. 2005 slope).
# t_sat = XUV_SAT_TIME_SUN_GYR * (M / M_sun)^XUV_SAT_MASS_EXPONENT, so
# low-mass stars stay active much longer.
XUV_SATURATED = 10**-3.13
XUV_DECAY_EXPONENT = -1.23
XUV_SAT_TIME_SUN_GYR = 0.1
XUV_SAT_MASS_EXPONENT = -2.0

# --- Orbit and rotation ---------------------------------------------------

# Tidal locking (Gladman et al. 1996) needs an initial spin and tidal
# parameters, which are poorly known for exoplanets.
# Q for rocky planets is drawn per planet from TIDAL_Q_ROCKY_RANGE (log-uniform)
# unless set in the spec; Earth today is ~10 (ocean dissipation), solid bodies ~100.
INITIAL_SPIN_PERIOD_H = 12.0
TIDAL_Q_ROCKY_RANGE = (10.0, 500.0)
TIDAL_Q_ROCKY = 100.0            # used when no value is supplied
TIDAL_K2_ROCKY = 0.3
TIDAL_Q_GIANT = 1e5
TIDAL_K2_GIANT = 0.5

# Locked planets with eccentricity above this settle in a 3:2 spin-orbit
# resonance (as Mercury) instead of synchronous rotation.
RESONANCE_ECCENTRICITY = 0.1

# --- Bulk -----------------------------------------------------------------

# Radius inflation of a rocky planet by a water layer:
# R = R_rock * (1 + WATER_RADIUS_INFLATION * water_mass_fraction).
# Chosen so a 50% water world is ~27% larger than a dry one.
WATER_RADIUS_INFLATION = 0.55

# Moment of inertia factors I / (M R^2).
MOI_FACTOR_ROCKY = 0.33
MOI_FACTOR_GIANT = 0.25

# Planets above this mass without a specified envelope are assumed to have
# accreted a volatile envelope.
ENVELOPE_ASSUMED_ABOVE_MEARTH = 10.0

# --- Interior -------------------------------------------------------------

# Present-day surface heat flow divided by radiogenic heat production
# (inverse Urey ratio). Earth: ~46 TW total vs ~20 TW radiogenic.
INVERSE_UREY_RATIO = 1.0 / 0.43

# Interior activity index a = (heat flux / Earth's) * (M / M_earth)^0.5.
# The mass factor stands in for convective vigour, so small bodies go
# quiescent (Moon, Mercury) even though their heat flux is similar.
ACTIVITY_MASS_EXPONENT = 0.5
ACTIVITY_DEAD_BELOW = 0.05       # Moon, Mercury
ACTIVITY_PLATES_ABOVE = 0.3      # Mars falls below, Venus and Earth above
ACTIVITY_LOG_WIDTH = 0.4         # softness of the dead and heat-pipe thresholds
# Heat-pipe volcanism depends on heat flux alone: Io has ~30x Earth's.
HEATPIPE_FLUX_RATIO_ABOVE = 10.0
ACTIVITY_PLATES_LOG_WIDTH = 0.5  # softness of the plate threshold

# Relative weights for the tectonic regime of an active planet.
PLATES_DRY_FACTOR = 0.15         # multiplier on plates without surface water
PLATES_HOT_FACTOR = 0.2          # multiplier on plates for hot surfaces
HOT_SURFACE_K = 500.0
EPISODIC_BASE = 0.3
EPISODIC_COOL_FACTOR = 0.3       # multiplier on episodic for cool surfaces
STAGNANT_BASE = 0.3

# Mass dependence of plate tectonics is contested in the literature.
# Weight on plates ∝ (M / M_earth)^exponent; 0 is neutral, >0 follows
# Valencia et al. (2007), <0 follows O'Neill & Lenardic (2007).
PLATES_MASS_EXPONENT = 0.0

# Probability of an active core dynamo by tectonic regime.
DYNAMO_P_PLATES = 0.9
DYNAMO_P_OTHER_ACTIVE = 0.3
DYNAMO_P_INACTIVE = 0.05
DYNAMO_MIN_ACTIVITY = 0.3
DYNAMO_MIN_CMF = 0.05

# --- Atmosphere -----------------------------------------------------------

# Cosmic shoreline (Zahnle & Catling 2017): I / I_earth = (v_esc / V0)^4.
# V0 places Mars on the line, which also separates Titan (atmosphere) from
# Callisto and Ganymede (none).
SHORELINE_V0_KMS = 6.2
# Effective insolation is boosted by XUV activity relative to the Sun:
# I_eff = S * (xuv_fraction / xuv_fraction_sun)^SHORELINE_XUV_EXPONENT.
SHORELINE_XUV_EXPONENT = 0.5
# Softness (in dex) of the retention probability around the shoreline.
SHORELINE_LOG_WIDTH = 0.2

# Default surface pressure for a retained atmosphere:
# P = 1 bar * volatile_richness * (M / M_E)^2 / (R / R_E)^4 * (1 - shoreline ratio),
# i.e. column mass ∝ volatile inventory ∝ M, reduced near the shoreline.
PRESSURE_SCATTER_DEX = 0.3
PRESSURE_MIN_SHORELINE_FACTOR = 0.01

# Grey greenhouse: T_s = T_eq (1 + 0.75 tau)^(1/4), tau = TAU0 * (P / 1 bar)^n.
# n from the Venus/Mars pair; TAU0 per composition from Earth, Venus/Mars, Titan.
GREENHOUSE_PRESSURE_EXPONENT = 0.72
TAU0 = {
    "n2_o2": 0.84,
    "n2_co2": 0.84,
    "co2": 5.2,
    "n2_ch4": 0.73,
    "h2_he": 0.0,
    "none": 0.0,
}
MEAN_MOLECULAR_WEIGHT = {
    "n2_o2": 28.97,
    "n2_co2": 29.0,
    "co2": 44.0,
    "n2_ch4": 28.6,
    "h2_he": 2.3,
    "none": 0.0,
}

# Carbonate-silicate cycle: on plate-tectonic planets with liquid water,
# CO2 adjusts toward a target temperature, within these optical-depth limits
# (and never beyond an all-CO2 atmosphere at the surface pressure).
# The target is drawn per planet from this range (uniform) unless set in the
# spec; the real balance point depends on outgassing, land fraction and
# weathering efficiency.
WEATHERING_TARGET_RANGE_K = (260.0, 320.0)
WEATHERING_TARGET_K = 288.0      # used when no value is supplied
WEATHERING_TAU_MIN = 0.2
WEATHERING_TAU_MAX = 20.0

# Default Bond albedos.
ALBEDO_AIRLESS = 0.1
ALBEDO_THIN = 0.25               # below THIN_ATMOSPHERE_BAR (dusty, Mars-like)
ALBEDO_TEMPERATE = 0.27            # ice-free; ice raises Earth-like planets to ~0.3
ALBEDO_THICK = 0.7               # above THICK_ATMOSPHERE_BAR (cloudy, Venus-like)
ALBEDO_ICE = 0.5
ALBEDO_GIANT = 0.34
THIN_ATMOSPHERE_BAR = 0.05
THICK_ATMOSPHERE_BAR = 10.0
ICE_COVER_BELOW_K = 240.0        # global mean surface temperature for ice cover

# Composition defaults.
COLD_NITROGEN_METHANE_BELOW_K = 120.0

# Water present at the surface if the surface water mass fraction exceeds this.
SURFACE_WATER_MIN_FRACTION = 1e-6

# --- Water inventory (Cowan & Abbot 2014) -------------------------------------
# On plate-tectonic planets, mantle and surface exchange water until degassing
# balances regassing: x / x_earth = (P / P_earth)^phi, with seafloor pressure
# P / P_earth = (g / g_earth)^2 * surface water fraction / Earth's.
MANTLE_WATER_EARTH = 5.8e-4          # water mass fraction of Earth's mantle (~1 ocean)
MANTLE_WATER_MAX = 7e-3              # mantle water capacity (~12 oceans)
MANTLE_MASS_FRACTION = 0.68
SURFACE_WATER_EARTH = 2.3e-4         # Earth's ocean mass / Earth's mass
WATER_PRESSURE_EXPONENT = 2.0        # phi: combined pressure dependence of degassing and hydration
# Without subduction, water is not returned to the mantle: the mantle keeps this
# share of the total water, up to Earth's mantle content.
NONPLATE_MANTLE_SHARE = 0.3
SEAWATER_DENSITY = 1025.0
# Closed depressions at least this share of the surface hold enclosed seas at the ocean's level
# (0.5% is 2.5 million km² on Earth); smaller ones follow the lake model.
ENCLOSED_SEA_MIN_AREA = 0.005
WATER_FRACTION_PLAUSIBLE_MAX = 2e-3  # above this, plate-tectonic planets lose their exposed continents
# Below this land fraction, weathering happens only on the sea floor, which is
# assumed not to respond to temperature (Abbot et al. 2012): no CO2 regulation.
WATERWORLD_LAND_FRACTION = 0.01

# --- Surface: relief --------------------------------------------------------

# Mountain height scales with (g_earth / g)^RELIEF_GRAVITY_EXPONENT (crustal
# strength limit). Earth's highest peak is ~8.8 km, Mars's ~22 km.
RELIEF_GRAVITY_EXPONENT = 1.0
MAX_ELEVATION_EARTH_M = 9000.0
RELIEF_FACTOR_RANGE = (0.3, 4.0)     # clamp for very high or very low gravity

# Noise amplitudes below are peak values; fractal noise has a standard
# deviation of about 0.22 times its amplitude.

# --- Surface: plate terrain (heuristic plates, milestone 2) ----------------

PLATES_EARTH = 12                    # typical plate count for Earth
PLATES_RANGE = (4, 40)
PLATE_WARP_RAD = 0.18                # irregularity of plate boundaries
PLATE_SPEED_EARTH_CM_YR = 5.0
CONTINENTAL_SHELF_EXTRA = 0.12       # continental crust area minus land area (Earth: 41% vs 29%)
CONTINENTAL_MAX_FRACTION = 0.75      # plate-tectonic planets keep ocean basins even with little water
CONTINENT_RIDGE_AVOIDANCE = 0.6      # weight pushing continents away from spreading ridges
CONTINENT_RIDGE_SCALE_KM = 1500.0
BOUNDARY_NORMAL_RATIO = 0.35         # closing speed / total speed separating convergent, divergent, transform

CONTINENT_BASE_M = 500.0
CONTINENT_NOISE_M = 2500.0           # standard deviation ~550 m (Earth's continental hypsometry)
# Sea-floor depth with age, GDH1 plate-cooling model (Stein & Stein 1992):
# half-space subsidence up to OCEAN_PLATE_TRANSITION_MYR, then exponential
# approach to OCEAN_MAX_DEPTH_M.
OCEAN_RIDGE_DEPTH_M = -2600.0
OCEAN_SUBSIDENCE_M_PER_SQRT_MYR = -365.0
OCEAN_PLATE_TRANSITION_MYR = 20.0
OCEAN_MAX_DEPTH_M = -5651.0
OCEAN_PLATE_DECAY_M = 2473.0
OCEAN_PLATE_TIMESCALE_MYR = 36.0
OCEAN_NOISE_M = 1000.0               # standard deviation ~220 m (abyssal hills and seamount chains)
MARGIN_SMOOTHING_STEPS = 3           # neighbour-averaging passes across coastlines

COLLISION_HEIGHT_M = 6500.0          # continent-continent (Himalaya-type)
COLLISION_WIDTH_KM = 450.0
ANDEAN_HEIGHT_M = 5000.0             # ocean-continent, overriding side
ANDEAN_WIDTH_KM = 250.0
ANDEAN_OFFSET_KM = 150.0             # peak distance inland from the trench
ARC_HEIGHT_M = 3500.0                # ocean-ocean island arc
ARC_WIDTH_KM = 120.0
TRENCH_DEPTH_M = 4500.0              # extra depth below the surrounding sea floor
TRENCH_WIDTH_KM = 80.0
RIFT_DEPTH_M = 1500.0                # continental rift valley
RIFT_WIDTH_KM = 90.0
HOTSPOTS_EARTH = 8
HOTSPOTS_MAX = 30

# Relict features (milestone 6). A mark on the surface survives RELICT_MEMORY_BASE_MYR divided by the
# erosion the planet applies to it: rain and plate tectonics rework Earth's surface within ~100 Myr,
# while a dry stagnant lid keeps its record for billions of years (Mars's valley networks).
RELICT_MEMORY_BASE_MYR = 6000.0
RELICT_RAIN_EROSION = 50.0           # weight of liquid surface water in that wear
RELICT_TECTONIC_EROSION = 8.0        # weight of the interior activity index
RELICT_SHORELINE_MIN_RISE = 0.02     # the past ocean must exceed the present one by this share
RELICT_SHORELINE_BAND_M = 120.0      # height of the terrace the old coastline leaves (Earth gravity)
RELICT_SHORELINE_MIN_COVER = 0.02    # the old sea must have covered this share of the planet
RELICT_MIN_AGE_MYR = 10.0            # a relict of something still happening is not a relict
RELICT_GLACIAL_COOLING_K = 5.0       # a past epoch this much colder leaves scoured ground
RELICT_PALAEO_RAIN_M = 0.5           # rainfall assumed for the valley networks of a lost wet climate
RELICT_RIVER_MIN_ORDER = 2           # Strahler order a dry valley must reach to be drawn
RELICT_RESURFACING_RATIO = 3.0       # past melting this many times the present resurfaces the plains
RESURFACING_EARTH_MYR = 100.0        # time Earth's melt production takes to bury its whole surface
RELICT_RESURFACED_MAX = 0.9          # cap on the share it can cover

# --- Surface: tectonic simulation (milestone 3) ------------------------------

TECTONIC_DURATION_MYR = 400.0        # default simulated time
# History mode drives the simulation from the integrated interior (milestone 6). Plate speed follows
# the plate creation rate, bounded so a very hot or very cold planet stays in a sensible range; the
# simulated span is scaled so the plates travel about as far as Earth's do in TECTONIC_DURATION_MYR,
# and Earth replaces its sea floor (3 km²/yr over a 3.1 × 10⁸ km² ocean) in about a hundred Myr.
PLATE_SPEED_SPREADING_RANGE = (0.1, 4.0)
TECTONIC_DURATION_RANGE_MYR = (200.0, 800.0)
SEAFLOOR_TURNOVER_EARTH_MYR = 100.0
HOTSPOT_MELT_EXPONENT = 0.5          # hotspot count ∝ melt production^exponent
TECTONIC_STEP_MYR = 2.0
TECTONIC_REMAP_STEPS = 10            # steps between re-mapping crust onto the grid
GAP_FACTOR = 0.75                    # grid cells farther than this × edge length from any crust point are gaps
CONTACT_FACTOR = 0.5                 # crust points of two plates closer than this × edge length interact
MIN_PLATE_FRACTION = 0.006           # plates smaller than this fraction of the surface merge into a neighbour
PLATE_SPEED_MAX_FACTOR = 2.0         # plate speeds are capped at this × the typical speed

# Uplift rates (m/Myr) at typical convergence speed, before the relief cap.
SUBDUCTION_UPLIFT_M_MYR = 100.0
COLLISION_UPLIFT_M_MYR = 150.0
ARC_UPLIFT_M_MYR = 70.0
UPLIFT_SPEED_CAP = 2.0               # maximum speed factor
OROGENY_MIN_RATE_M_MYR = 20.0        # uplift above this resets a point's orogeny age

CONTINENT_EROSION_TIMESCALE_MYR = 120.0   # e-folding time of relief above the continental base
OCEAN_RELIEF_TIMESCALE_MYR = 60.0         # decay of arcs, trenches and seamounts on oceanic crust
CONTINENT_INITIAL_RELIEF_M = 1000.0       # peak amplitude of initial continental variation

HOTSPOT_UPLIFT_M_MYR = 400.0
HOTSPOT_RADIUS_KM = 150.0

SLAB_PULL_RATE = 0.02                # fractional change of plate rotation per Myr toward its subduction zones
COLLISION_COUPLING_PER_MYR = 0.15    # rate at which colliding plates' motions converge to a shared rotation
# Plates fuse after sustained collision: contacts are accumulated with a memory
# of COLLISION_MEMORY_MYR and compared with the smaller plate's size.
MERGE_CONTACT_FRACTION = 0.5
COLLISION_MEMORY_MYR = 30.0
RIFT_TIMESCALE_MYR = 300.0           # mean time between rifting of a fully continental, average-size plate
OCEANIC_RIFT_FACTOR = 0.3            # relative rifting rate of purely oceanic plates
# Old oceanic lithosphere is dense: an ocean basin whose crust is mostly older
# than OLD_OCEAN_MYR detaches from its continent and starts sinking beneath it.
OLD_OCEAN_MYR = 100.0
SUBDUCTION_INITIATION_TIMESCALE_MYR = 25.0
# Plate-count feedback: rifting speeds up and merging slows down when there
# are fewer plates than expected for the planet, and the reverse.
PLATE_COUNT_FEEDBACK = (0.5, 4.0)
RIFT_SPEED_FACTOR = 0.6              # speed of rift opening relative to the typical plate speed
CRATON_COUNT = (4, 8)                # number of continents in the scattered-cratons start
YOUNG_RIFT_MYR = 10.0                # continental cells beside oceanic crust younger than this are rifts
MOUNTAIN_OROGENY_MAX_MYR = 150.0     # older orogens are shown as highlands
# Detail added to the simulated elevation on the grid.
SIM_CONTINENT_NOISE_M = 1500.0
SIM_MARGIN_DROP_M = 1000.0           # continental crust thins toward the ocean, forming shelves
SIM_MARGIN_WIDTH_KM = 250.0
SIM_MOUNTAIN_SHARPNESS = 1.5         # higher values concentrate mountain uplift into narrower, higher ranges

# --- Surface: structural fabric (milestone 7) --------------------------------
# The dominant grain of the surface, stored per cell so zoom can orient detail:
# the belt axis along continental elevation contours, the ridge axis along
# oceanic crust-age contours. Strength runs 0 to 1.
FABRIC_SMOOTH_STEPS = 2               # smoothing of the field before its gradient is taken
FABRIC_OCEAN_AGE_FADE_MYR = 120.0    # abyssal-hill grain fades as sediment buries it (buried by ~120 Myr)
FABRIC_MOUNTAIN_REF_M = 1500.0       # continental relief (above base, at Earth gravity) for full grain strength
FABRIC_OROGENY_FADE_MYR = 300.0      # the fold grain of an orogen subsides as it is worn down
FABRIC_CONFIDENCE_FRACTION = 0.3     # slope, relative to the typical belt slope, needed for a defined grain
FABRIC_MIN_STRENGTH = 0.05           # grains weaker than this are dropped, keeping the field sparse

# --- Zoom: local downscaling (milestone 7) -----------------------------------
# The sub-grid relief added when a region is zoomed carries its own local rain
# shadow, applied on top of the global climate the tile inherits.
ZOOM_OROGRAPHIC_SLOPE = 0.02         # along-wind grade at which the rain-shadow effect is strong
ZOOM_OROGRAPHIC_CAP = 3.0            # most a cell's rain is raised or cut by the local slope
# Sub-grid relief synthesised at zoom, below the global grid scale.
# The detail continues the global surface's relief spectrum below the grid spacing: an octave of
# wavelength λ has standard deviation σ_ref (λ / λ_ref)^H, with σ_ref set by how rugged the ground is.
# H = 0.5 is a k⁻² topographic spectrum; the mountain value gives ~500 m of relief between the grid
# scale and a few km, as in the Alps, and the plain value ~25 m, as in lowland river country.
ZOOM_RELIEF_REF_KM = 10.0            # reference wavelength λ_ref
ZOOM_HURST = 0.5                     # growth of octave amplitude with wavelength
ZOOM_RELIEF_PLAIN_M = 6.0            # σ_ref on flat plains (Earth gravity)
ZOOM_RELIEF_MOUNTAIN_M = 120.0       # σ_ref in mountains
ZOOM_RELIEF_ABYSSAL_M = 10.0         # σ_ref on old, sediment-covered sea floor
ZOOM_RELIEF_SEAFLOOR_M = 40.0        # σ_ref on young or steep sea floor (abyssal hills, slopes)
ZOOM_RUGGED_SLOPE = 0.015            # coarse grid slope counted as fully rugged
ZOOM_RUGGED_HEIGHT_M = 2500.0        # height above the continental base counted as fully rugged
ZOOM_DETAIL_MIN_OCTAVES = 2
ZOOM_DETAIL_MAX_OCTAVES = 10
ZOOM_DETAIL_SMOOTH_STEPS = 3         # passes that elongate ridges along the structural grain
ZOOM_WARP_FRACTION = 0.25            # domain warp, as a fraction of the grid spacing, to break grid alignment
ZOOM_HILLSHADE_EXAGGERATION = 3.0    # vertical exaggeration of the local relief shading
ZOOM_EROSION_MYR = 15.0              # erosion applied to a zoomed region, carving its sub-grid valleys
ZOOM_EROSION_STEP_MYR = 5.0          # erosion step for a zoomed region
# Soil creep at its physical rate (~0.01 m²/yr on soil-mantled hillslopes). The global grid's
# CREEP_M2_PER_MYR is an effective value for 100 km cells and would diffuse away all zoom detail.
ZOOM_CREEP_M2_PER_MYR = 1.0e4
ZOOM_EDGE_BLEND_CELLS = 6            # erosion fades to zero over this many cells at the pinned region edge
# Long fluvial erosion cuts outlets through most small sills; zoom does this directly by breaching
# depressions whose drainage needs a cut no deeper than this, and leaves deeper basins as lakes.
ZOOM_BREACH_MAX_M = 300.0
# A channel forms where enough ground drains to a point, and the steeper the ground the less it needs:
# channel heads follow A·S² ≈ constant (Montgomery & Dietrich 1988, 1992). The constant is set so that
# moderate hill country (ground slope ~0.012 at the lattice scale) maps channels from 5 km², as mapped
# streams do; flat plains then need tens of km² and steep mountains a few lattice cells. A channel, once
# formed, continues downstream. It carries water all year only where the mean flow is large enough;
# channels with less are dry washes, so deserts show their channel pattern without looking wet.
ZOOM_RIVER_MIN_AREA_KM2 = 5.0        # channel-head area at the reference slope
ZOOM_CHANNEL_HEAD_REF_SLOPE = 0.012
ZOOM_CHANNEL_HEAD_MIN_CELLS = 2.0    # smallest channel head, in lattice cells
ZOOM_CHANNEL_HEAD_MAX_KM2 = 1000.0   # largest, on the flattest plains
ZOOM_STREAM_MIN_DISCHARGE_M3_S = 0.1 # mean flow at which a channel is a perennial stream
# Glaciers at zoom follow the glacial stream-power law E = K_g Q_iᵐ S (Hergarten 2021; Liebl et al. 2023):
# Q_i is the ice flux from the mass balance, b = P·min((z − ELA)/ΔH, 1), gained above the snowline and
# lost below it; S is the slope of the ice surface. K_g gives ~1 mm/yr under an Alpine valley glacier (80 km² fed at
# ~1 m/yr, surface slope 0.1), about ten times the fluvial rate, within the range Hallet et al. (1996)
# report for temperate glaciers. The ice surface is the perfectly plastic profile with ICE_YIELD_STRESS_PA
# (Nye 1952; Benn & Hulton 2010) rather than Liebl et al.'s constant thickness-to-width ratio, which puts
# cliffs in the ice surface where glaciers of different size meet.
ZOOM_GLACIER_FULL_ACCUMULATION_M = 500.0   # ΔH: height above the snowline where all precipitation stays as ice
# Below the snowline ice melts by degree days, not in proportion to the snowfall: each metre lower adds
# DEGREE_DAY_FACTOR_M × LAPSE_RATE × (melt-season days) of melt, ~6 mm/yr per metre as on Earth's glaciers.
ZOOM_MELT_SEASON_DAYS = 120.0
# Ice is routed to every lower neighbour in proportion to slope^p (Quinn et al. 1991), as in ice-sheet
# balance-flux calculations (Le Brocq et al. 2006).
ZOOM_ICE_SPREAD_EXPONENT = 1.0
ZOOM_ICE_FILL_PASSES = 4                   # lattice steps ice spreads sideways to fill a valley to its surface
ZOOM_ICE_SHEET_SMOOTH_PASSES = 20          # smoothing of the inherited ice-sheet surface's triangle creases
ZOOM_GLACIER_MIN_THICKNESS_M = 20.0        # thinnest ice where a glacier crosses a bed steeper than its surface
ZOOM_GLACIAL_K = 1.0                       # K_g, m/Myr per (m³/yr)^0.5 of ice flux
ZOOM_GLACIAL_WATER_SHARE = 0.25            # ε: effective flux Q^ε Q_i^(1−ε) keeps the valley network dendritic
# Meltwater climbing an adverse bed slope steeper than ~1.2–1.7 times the ice-surface slope freezes on,
# which halts the deepening of an overdeepening (Alley et al. 2003).
ZOOM_ADVERSE_SLOPE_RATIO = 1.5
ZOOM_GLACIAL_MYR = 1.0                     # time spent under the coldest glaciation (Quaternary-like)
ZOOM_GLACIAL_STEP_MYR = 0.2
ZOOM_ICE_ITERATIONS = 4                    # flux ↔ ice thickness passes when the ice cover is built
ZOOM_NO_SNOWLINE_M = 1.0e5                 # stands for "no snowline" where the global grid has none

# --- Climate: energy balance (Tier 0 and Tier 1) ------------------------------
# Outgoing radiation slope: the grey value (3.3 W m⁻² K⁻¹ for Earth) is lowered by
# water vapour to Earth's all-sky ~2.1 (North et al. 1981); the reduction grows
# with saturation humidity relative to Earth's, up to twice Earth's value.
WATER_VAPOUR_FEEDBACK = 0.19
WATER_VAPOUR_FACTOR_MIN = 0.45
CLIMATE_RELATIVE_HUMIDITY = 0.8
# Moist-static-energy diffusivity for Earth (W m⁻² K⁻¹ on the unit sphere), scaled
# as p · c_p · μ⁻² · Ω^-exponent and capped at a multiple of the radiation slope
# (a nearly isothermal planet). Williams & Kasting (1997) use Ω⁻²; GCMs show a
# weaker rotation dependence (Kaspi & Showman 2015), so the exponent is 1.
# The value is set with the ocean heat transport below (0.27 without it).
# Synchronous planets use a fixed factor instead of the rotation term, set so a
# 1 bar Earth-like planet has a day–night contrast of ~60 K at S = 0.9 (Wordsworth 2015; Yang et al. 2013).
MOIST_DIFFUSIVITY_EARTH = 0.22
TRANSPORT_ROTATION_EXPONENT = 1.0
TRANSPORT_MAX_RATIO = 10.0
SYNCHRONOUS_TRANSPORT_FACTOR = 3.0
# Ocean heat transport: a wind-driven-like poleward flux whose peak is this share of the peak
# atmospheric transport (scaled by ocean width relative to Earth's), reaching this multiple of the
# sea-ice edge angle. Calibrated to Earth's partition (Trenberth & Caron 2001) and sea-ice cover.
OCEAN_TRANSPORT_RATIO = 0.35
OCEAN_ICE_REACH = 1.3
OCEAN_FRACTION_EARTH = 0.71
# Tier 1 applies this factor to transport of the seasonal harmonics, which
# gives continental interiors at 50–70° N a ~30 K annual range (Earth: 30–60 K).
SEASONAL_TRANSPORT_FACTOR = 0.15
LAND_OCEAN_EXCHANGE = 6.0             # W m⁻² K⁻¹ between land and ocean columns (Tier 0), for Earth's transport
OCEAN_MIXED_LAYER_M = 50.0
LAND_HEAT_CAPACITY = 2.0e6            # J m⁻² K⁻¹ of the soil layer that follows the seasons
# Ice-free planetary albedo pattern: + P2(sin φ) × 0.078 (North et al. 1981);
# synchronous planets get bright clouds over the substellar region (Yang et al. 2013).
ALBEDO_ZONAL_P2 = 0.078
SUBSTELLAR_CLOUD_ALBEDO = 0.15
# Surface albedos, seen through the clouds with weight (1 − planetary albedo)².
SURFACE_ALBEDO_OCEAN = 0.07
SURFACE_ALBEDO_LAND = 0.2
SURFACE_ALBEDO_ICE = 0.65
SNOW_COVER_MAX = 0.7                  # snow never hides all land (forests, slopes)
ICE_TRANSITION_K = 1.5                # width of the smooth freezing step
# Sea ice forms below this air temperature, not at the seawater freezing point
# (−1.8 °C): the column temperature is that of the air, and the ocean below
# stays warmer (North et al. 1981 use −10 °C). Calibrated, with ocean heat
# transport, to Earth's ~7% sea-ice cover.
SEA_ICE_AIR_THRESHOLD_K = 273.15 - 7.5
ZONAL_BANDS = 36
CLIMATE_GRID_SIZE = 10_000            # Tier 1 grid (~220 km on Earth), independent of the surface grid
CLIMATE_TIME_SAMPLES = 24
CLIMATE_HARMONICS = 3                 # annual mean and two harmonics
CLIMATE_MAX_ITERATIONS = 120
CLIMATE_RELAXATION = 0.5
CLIMATE_GAIN_LIMIT = 0.8             # ice-albedo gain used in the Tier 0 solve, relative to the radiation slope
CLIMATE_TOLERANCE_K = 0.05
SURFACE_CLIMATE_TOLERANCE_K = 0.2    # Tier 1 stops when no cell changes by more than this
# Moisture (after Siler et al. 2018): Hadley share of the energy flux
# exp(−((x − x_ITCZ)/width)²) with x the sine of latitude, and an upper branch
# carrying a multiple of the largest surface moist static energy. Siler et al.
# use a width of 0.3 and a factor of 1.06 for zonal means; 0.4 and 1.15, with E − P
# spread over ~8 cells, match Earth's zonal ocean rain best on this grid
# (rms error 0.36 m/yr over 10° bands).
HADLEY_WEIGHT_WIDTH = 0.4
HADLEY_EDGE_EARTH_DEG = 30.0
HADLEY_ROTATION_EXPONENT = 0.33       # slower rotation widens the Hadley cells (Kaspi & Showman 2015)
MOISTURE_SPREAD_STEPS = 8            # neighbour-averaging passes on E − P (convective spreading, ~220 km each)
SUBSTELLAR_CELL_WIDTH = 0.6           # synchronous planets: width in (1 − cos substellar angle)
GROSS_MOIST_STABILITY_FACTOR = 1.15
GROSS_MOIST_STABILITY_MIN = 0.02      # relative to the upper-branch value
# Global mean ocean evaporation at 288 K (m/yr; Earth ~1.2), rising 2.5%/K
# (Held & Soden 2006), at most this share of absorbed sunlight as latent heat.
EVAPORATION_EARTH_M = 1.2
HYDROLOGY_SENSITIVITY = 0.025
LATENT_SHARE_MAX = 0.6
SEA_ICE_EVAPORATION_CUT = 0.9
# Water phase: open water on at least this share of the surface, or on at least
# OPEN_OCEAN_LIQUID of the ocean, makes the surface water 'liquid'.
OPEN_WATER_LIQUID = 0.02
OPEN_OCEAN_LIQUID = 0.5
CLIMATE_TARGET_MISS_K = 3.0           # report when a target mean temperature cannot be held
# Archetypes with Earth-like oceans need at least this share of the ocean open.
OPEN_OCEAN_TEMPERATE = 0.5

# --- Climate: precipitation on the surface grid -------------------------------
RAIN_MINIMUM_M = 0.03
MOISTURE_DECAY_KM = 8000.0            # air loses moisture over this distance inland (with recycling by plants and soil)
MOISTURE_MIXING = 0.15                # share of moisture spread by eddies rather than the mean wind
RAIN_SHADOW_HEIGHT_M = 2000.0         # rise over which air loses most of its moisture
RAIN_SHADOW_BASE_M = 1200.0           # terrain below this height does not dry the air
RAIN_TERRAIN_SMOOTHING = 6            # smoothing passes applied to terrain before moisture transport
OROGRAPHIC_BOOST = 1.0                # extra rain on windward slopes, relative
# Surface wind toward regions warmer than their latitude (monsoons), per K/rad of
# temperature gradient relative to the prevailing wind's unit strength. Calibrated
# with the vegetated moisture decay to Earth's land rain and Köppen B and Dw shares.
MONSOON_WIND_PER_K_RAD = 1.0 / 80.0
LAPSE_RATE_K_PER_M = 0.0065
EVAPORATION_M = 1.4                   # potential evaporation at 303 K (m/yr)
EVAPORATION_ZERO_K = 263.0
BUDYKO_W = 2.6                        # Fu–Budyko shape parameter (Zhang et al. 2004)

# --- Ice sheets and glacial erosion -------------------------------------------
# Melt 8 mm water per positive degree-day of ice (Braithwaite 1995: ~8 for ice) and 4.1 mm of snow, the
# value that fits the precipitation–temperature climate at 66 glaciers' equilibrium lines (Braithwaite
# 2008: 4.1 ± 1.5; snow melts at less than half the ice rate, Braithwaite 1995). The year's snow melts
# first. Monthly temperatures are spread by a normal distribution, 4.5 K as measured on glaciers near
# 0 °C (Wake & Marshall 2015); snow below +1 °C.
DEGREE_DAY_FACTOR_M = 0.008
SNOW_DEGREE_DAY_FACTOR_M = 0.0041
PDD_TEMPERATURE_SPREAD_K = 4.5
SNOW_RAIN_THRESHOLD_C = 1.0
# Plastic ice profile h = √(2 τ₀ L / (ρ g)) (Nye 1952); τ₀ = 50 kPa gives ~4.8 km
# 2000 km from the margin. The bedrock sinks under ice by ρ_ice / ρ_mantle.
ICE_YIELD_STRESS_PA = 5.0e4
ICE_ISOSTATIC_SINKING = 0.28
ICE_HEIGHT_ITERATIONS = 3             # mass balance ↔ ice height passes per climate solve
# Glacial buzzsaw (Egholm et al. 2009: peaks cluster within ~1.5 km of the ELA).
# Land above ELA + offset erodes at dx/dt = −x²/(scale·timescale), x its excess height.
GLACIAL_PEAK_OFFSET_M = 1500.0
GLACIAL_EROSION_SCALE_M = 5000.0
GLACIAL_EROSION_TIMESCALE_MYR = 5.0

# --- Hydrology: erosion ------------------------------------------------------
# Stream power E = K Q^m S (Q in m³/yr, E in m/Myr), solved implicitly (Braun & Willett 2013).
STREAM_POWER_K = 0.08
STREAM_POWER_M = 0.5
CREEP_M2_PER_MYR = 5e7               # hillslope diffusivity at grid scale
EROSION_STEP_MYR = 2.0
SIM_EROSION_STEP_MYR = 5.0           # erosion step inside the tectonic simulation
# Net surface change per unit eroded or deposited in the simulation, after isostatic adjustment.
EROSION_ISOSTATIC_FACTOR = 0.35
SEDIMENT_ISOSTATIC_FACTOR = 0.5
FINAL_EROSION_MYR = 10.0             # erosion of the finished surface with its detail
SHELF_DEPTH_M = 60.0                 # sediment builds shelves up to this depth below sea level

# Rivers shown on the global grid.
RIVER_MIN_DISCHARGE_M3_S = 100.0
RIVER_MIN_AREA_KM2 = 1e5              # upstream area
RIVER_MIN_CELLS = 3

# --- Surface: landforms ------------------------------------------------------

VOLCANO_MAX_HEIGHT_EARTH_M = 9000.0  # scaled by the relief factor (Mars: Olympus Mons ~22 km)
VOLCANO_WIDTH_PER_HEIGHT = 27.0      # basal diameter / height (Olympus Mons ~600 km / 22 km)
CALDERA_FRACTION = 0.12              # caldera radius / volcano radius
# Crater size-frequency: N(>D) per km² = CRATER_DENSITY * (D / km)^-2 for a 4.5 Gyr
# surface, scaled linearly with surface age (lunar highlands: ~5000 craters > 20 km).
CRATER_DENSITY = 0.055
CRATER_MAX_DIAMETER_FRACTION = 0.4   # largest basin diameter / planet radius
CRATER_MIN_CELLS = 3.0               # smallest mapped crater diameter in grid spacings
CRATER_MAX_COUNT = 4000
CRATER_SIMPLE_DEPTH_RATIO = 0.2
CRATER_TRANSITION_EARTH_KM = 4.0     # simple-to-complex transition, scaled by g_earth / g
CRATER_WATER_EROSION = 0.2           # fraction of craters surviving on worlds with liquid water

# --- Surface: non-plate regimes ---------------------------------------------

LID_BASE_RELIEF_M = 4000.0           # standard deviation ~900 m before other landforms
DICHOTOMY_HEIGHT_M = 2500.0          # hemispheric highland/lowland step (Mars ~5 km)
RISE_HEIGHT_M = 3000.0               # broad volcanic rise (Tharsis-type)
RISE_RADIUS_RAD = 0.5
STAGNANT_VOLCANOES = 6               # at activity index 1
STAGNANT_RIFT_DEPTH_M = 3500.0
STAGNANT_RIFT_WIDTH_KM = 150.0
STAGNANT_SURFACE_AGE_FRACTION = 0.7
EPISODIC_PLAINS_FRACTION = (0.6, 0.85)   # Venus: ~80% volcanic plains
EPISODIC_SURFACE_AGE_GYR = 0.5
EPISODIC_VOLCANOES = 12
HEATPIPE_VOLCANOES = 80
HEATPIPE_VOLCANO_HEIGHT_M = 2500.0
INACTIVE_MARIA_FRACTION = 0.15       # low, smooth flood-basalt plains

# --- Biosphere ------------------------------------------------------------------
# Life begins this long after the planet forms; surface life needs a planet at least
# LIFE_MIN_PLANET_AGE_GYR old with liquid water (ocean-only if almost no land);
# frozen water worlds get subsurface life.
LIFE_ORIGIN_DELAY_GYR = 0.7
LIFE_MIN_PLANET_AGE_GYR = 1.0
LIFE_MIN_LAND_FRACTION = 0.01
# Oxygen: a first rise (to ~2%) and a second rise to Earth's 21%, at these biosphere
# ages (Earth: ~1.4 and ~3.2 Gyr after life began); ocean-only biospheres make half.
OXYGEN_EARTH = 0.21
OXYGEN_FIRST_RISE = (1.4, 0.1, 0.1)       # (age Gyr, width Gyr, share of the final level)
OXYGEN_SECOND_RISE = (3.2, 0.15)
OCEAN_ONLY_OXYGEN_SHARE = 0.5
OXYGEN_COMPOSITION_ABOVE = 0.01           # n2_co2 air becomes n2_o2 above this O2 fraction
METHANE_BIOTIC = 1e-3                     # CH4 fraction from a mature methanogenic biosphere
METHANE_OPTICAL_DEPTH = 0.15              # extra greenhouse optical depth at METHANE_BIOTIC, ∝ √fraction
BIOTIC_WEATHERING_COOLING_K = 5.0         # land biota speed weathering (Schwartzman & Volk 1989)
EARTH_OPTIMUM_K = 298.0
EARTH_TOLERANCE_K = 25.0
# Pigments absorb at the surface photon-flux peak (Kiang et al. 2007), here λ = 1.04 × 3.67 mm K / T★,
# which gives Earth's vegetation its green; a second band in the blue; a red edge beyond the peak.
PIGMENT_SURFACE_SHIFT = 1.04
PIGMENT_BAND_WIDTH_NM = 60.0
PIGMENT_ACCESSORY_NM = 445.0
PIGMENT_EDGE_OFFSET_NM = 60.0
PIGMENT_BROAD_FROM_NM = 850.0             # peaks beyond this also absorb the visible range below them
PIGMENT_BROAD_WIDTH_NM = 250.0
LEAF_REFLECTANCE_MIN = 0.03
LEAF_REFLECTANCE_GREEN = 0.2
LEAF_REFLECTANCE_EDGE = 0.5
# Vegetation: Miami-model productivity (Lieth 1973, g dry matter m⁻² yr⁻¹) scaled by light;
# full cover at NPP_FULL_COVER. Canopy albedo is Earth's 0.13 scaled by the pigment's
# visible brightness; bare ground brightens with aridity.
NPP_MAX = 3000.0
NPP_FULL_COVER = 1200.0
LIGHT_EXPONENT = 0.5
LIGHT_FACTOR_RANGE = (0.3, 1.5)
CANOPY_ALBEDO_EARTH = 0.13
SOIL_ALBEDO = 0.17
DESERT_BRIGHTENING = 0.18
# Moisture recycling: air loses moisture over this distance over bare and fully vegetated land.
MOISTURE_DECAY_BARE_KM = 5000.0
MOISTURE_DECAY_VEGETATED_KM = 10000.0
PRODUCTIVE_NPP = 300.0                    # land counts as productive above this NPP
PRODUCTIVE_LAND_EARTH = 0.6               # share of Earth's ice-free land above PRODUCTIVE_NPP
LAND_RAIN_SHARE = 0.6                     # land rain relative to global evaporation, for the global estimate
EARTH_ZONAL_NPP = 985.0                   # global-estimate NPP of the Earth spec (normalises productivity)
TIER0_COVER_PER_PRODUCTIVITY = 0.5        # vegetation cover assumed by Tier 0 per unit relative productivity
CLIMATE_TIER_MISMATCH_K = 10.0            # report when Tier 1 and Tier 0 mean temperatures differ more
# Climate–ice–vegetation coupling stops when this share of land cells changes albedo by less than
# the tolerance and few cells change ice cover. When the change grows between passes, the updates
# are damped (halved, down to the minimum) to settle oscillations.
COUPLING_MAX_PASSES = 10
COUPLING_ALBEDO_QUANTILE = 0.99
COUPLING_ALBEDO_TOLERANCE = 0.005
COUPLING_ICE_TOLERANCE = 0.002            # share of cells changing ice cover
COUPLING_MIN_DAMPING = 0.25
# Ice sheets hold at most this share of the surface water; thicker sheets are thinned.
ICE_MAX_WATER_SHARE = 0.5
SEA_LEVEL_RECOUPLE_M = 1.0                 # re-solve the surface climate when ice lowers sea level more than this
CLIMATE_OPTICAL_DEPTH_NOTE = 0.1           # note when Tier 1 needs a greenhouse this much (relative) off the estimate

# --- Occupiability --------------------------------------------------------

DEFAULT_OCCUPIABILITY_WEIGHTS = {
    "temperature": 1.0,
    "gravity": 1.0,
    "pressure": 1.0,
    "water": 1.0,
    "radiation": 1.0,
    "biosphere": 1.0,
}
# Biosphere factor: √(air × food). O₂ partial pressure (bar) is breathable in the first
# range and tolerable (altitude-like or oxygen-rich) in the second; otherwise
# breathing gear is needed. Food is 1 at half Earth's land productivity or more.
OXYGEN_BREATHABLE_BAR = (0.16, 0.5)
OXYGEN_TOLERABLE_BAR = (0.08, 1.0)
BREATHING_GEAR_SCORE = 0.3
NO_FOOD_SCORE = 0.3
XUV_HARSH_RATIO = 10.0

# --- History mode (milestone 6) ------------------------------------------------
# Integration: planet ages in Gyr from HISTORY_START_GYR (after the magma ocean).
HISTORY_START_GYR = 0.03
HISTORY_OUTPUT_POINTS = 200               # timeline samples for figures
HISTORY_MAX_SEGMENTS = 400                # integration restarts at events
HISTORY_CANDIDATES = 10                   # snapshot-screened draws tried in history mode
CYCLE_COLLAPSE_AFTER = 3                  # freezes in quick succession after which the log reports one cycle
CYCLE_FAST_GYR = 0.3                      # freezes closer together than this belong to the same cycle
CYCLE_RECENT_GYR = 0.5                    # a cycle this recent means the planet is still cycling at the end

# Stellar XUV (Tu et al. 2015, 1 M☉): saturation ends at 5.7, 23 and 226 Myr for the
# 10th, 50th and 90th rotation percentiles; tracks join the snapshot relation by
# XUV_CONVERGENCE_GYR (scaled with the saturation time's mass dependence).
XUV_SAT_PERCENTILES = (0.1, 0.5, 0.9)
XUV_SAT_TIMES_SUN_GYR = (0.0057, 0.023, 0.226)
XUV_CONVERGENCE_GYR = 1.0

# Mantle and core thermal evolution (parameterised convection).
MANTLE_HEAT_CAPACITY = 1250.0             # J kg⁻¹ K⁻¹
CORE_HEAT_CAPACITY = 840.0
MANTLE_TEMPERATURE_EARTH_K = 1620.0       # present potential temperature
CORE_TEMPERATURE_EARTH_K = 4100.0         # present core–mantle boundary temperature
SURFACE_HEAT_FLOW_EARTH_W = 46e12
CMB_HEAT_FLOW_EARTH_W = 10e12
MANTLE_VISCOSITY_EARTH = 1e21             # Pa s at the present potential temperature
LOWER_MANTLE_FACTOR = 1.55                # lower-mantle temperature / potential temperature (adiabat)
CORE_SUPERHEAT_EARTH_K = 2500.0           # initial core excess over the lower mantle (Earth)
CORE_SUPERHEAT_EXPONENT = 0.7             # superheat ∝ (depth scale)^exponent
MAGMA_HEAT_EARTH_W = 4e12                 # heat carried by Earth's present melt production
# Mobile lid: Q ∝ ΔT^(1+β) η^(−β) with weak β (Korenaga 2006); stagnant lid: β = 1/3 with a lid factor.
# Mobile lid: a weak temperature dependence (Korenaga 2006). Pure β = 0 leaves no feedback, so a
# small planet with plates would cool without limit; β = 0.07 with the scale below ends Earth's run
# at 1620 K, 82 mW/m² and its present melt production, and grows an inner core at ~3.7 Gyr.
PLATE_FLUX_BETA = 0.07
PLATE_FLUX_SCALE = 0.9
LID_FLUX_BETA = 1.0 / 3.0
LID_FLUX_FACTOR = 0.2                     # stagnant-lid heat flow relative to plates at Earth's mantle state
EPISODIC_FLUX_FACTOR = 0.5
HEAT_PIPE_FLUX_FACTOR = 1.0
INACTIVE_FLUX_FACTOR = 0.1
CMB_FLUX_BETA = 1.0 / 3.0
CRUST_RADIOGENIC_SHARE = 0.3              # share of heat-producing elements in the crust of plate planets
# Dynamo: runs while the core–mantle heat flow exceeds the adiabatic heat flow, or a
# share of it once an inner core grows (compositional buoyancy).
CORE_ADIABATIC_EARTH_W = 7.5e12
INNER_CORE_DYNAMO_SHARE = 0.3
INNER_CORE_ONSET_EARTH_K = 4320.0         # CMB temperature at which an inner core starts to grow (Earth ~1 Ga)
INNER_CORE_PRESSURE_EXPONENT = 0.5       # onset temperature ∝ (central pressure)^exponent
INNER_CORE_RANGE_K = 1500.0               # cooling below onset to freeze most of the core
# Melting: melt production ∝ spreading × (T_m − solidus); solidus at the surface.
MANTLE_SOLIDUS_K = 1400.0
# Melt production grows with the mantle's excess over the solidus but saturates: a mantle far above
# the solidus is largely molten and loses heat by volcanism instead (normalised to 1 for Earth).
MELT_EXCESS_MAX = 4.0
LID_MELT_FACTOR = 0.01                    # stagnant-lid melt production relative to plates at the same excess
                                          # (Mars ~0.01, Venus ~0.05 of Earth's present crust production)
LID_MELT_DECAY_K = 60.0                   # stagnant lids: melt falls off as the lid thickens with cooling
REGIME_PLATE_LOSS_FACTOR = 0.5            # plates stop below this × ACTIVITY_PLATES_ABOVE

# Volatile reservoirs (masses in 10¹⁸ kg). Earth: CO₂ in air and ocean 0.14,
# crustal carbonate 260, mantle 1000; atmospheric N₂ 3.9.
CARBON_SURFACE_EARTH = 0.14
CARBON_CRUST_EARTH = 260.0
CARBON_MANTLE_EARTH = 1000.0
NITROGEN_EARTH = 3.9
# Carbon released by the magma ocean, scaled by the planet's outgassing efficiency (a reduced
# mantle releases less). Venus keeps it as ~80 bar of CO₂; on planets where oceans condense all
# but INITIAL_AIR_CARBON_SHARE of it is carbonated within a few Myr, which the integration skips.
INITIAL_DEGASSED_CARBON_SHARE = 0.35
INITIAL_AIR_CARBON_SHARE = 0.01           # what stays in the air where oceans condense
INITIAL_MANTLE_WATER_SHARE = 0.3
CO2_EARTH_BAR = 2.8e-4                    # pre-industrial partial pressure
OCEAN_CARBON_EARTH = 0.138                # dissolved inorganic carbon (as CO₂) per ocean at CO2_EARTH_BAR
OCEAN_CARBON_EXPONENT = 0.5               # dissolved carbon ∝ pCO₂^exponent
AIR_PARTITION_STEPS = 30                  # bisection steps for the air–ocean carbon split
# Carbon fluxes at present Earth (10¹⁸ kg CO₂ per Gyr; ~7 Tmol C/yr).
WEATHERING_EARTH = 310.0
# The continental sink is calibrated on the Earth a full history produces, not on the nominal
# present-day point: that planet has 26% land, a land biosphere at 91% of full cover, and a crust
# still draining the carbonate the hot early mantle put there, which together need a sink this much
# stronger to hold pre-industrial CO₂.
CONTINENTAL_WEATHERING_SCALE = 1.15
SEAFLOOR_WEATHERING_SHARE = 0.2           # share of Earth's weathering on the sea floor
SEAFLOOR_CO2_EXPONENT = 0.23
SEAFLOOR_TEMPERATURE_SCALE_K = 40.0
WEATHERING_SUPPLY_RATIO = 3.0             # supply limit / present kinetic weathering (Earth)
SEAFLOOR_CAPACITY_RATIO = 30.0            # seafloor uptake limit relative to its present rate
# Earth's carbon input is ~6 Tmol/yr: ridges and plumes ~1.5, arcs ~3.5, with the rest of the
# subducted carbonate (~0.8 Tmol/yr) carried into the mantle.
# These two set the steady state of the cycle: with a share a returned by arcs and a turnover τ,
# the crust settles at C = W τ with W = volcanic / (1 − a), which reproduces Earth's carbonate
# crust (260 × 10¹⁸ kg CO₂) and its ~7 Tmol/yr weathering flux.
ARC_DEGASSING_SHARE = 0.5                 # subducted carbonate returned by arcs
CARBONATE_TURNOVER_GYR = 0.85             # crustal carbonate subduction timescale at Earth's spreading
CARBONATE_DECOMPOSITION_K = 700.0         # crustal carbonate returns to the air above this surface temperature
CARBONATE_DECOMPOSITION_GYR = 0.01
LID_CARBONATE_RECYCLING_GYR = 3.0         # burial and decarbonation of carbonate on active lids, at Earth-like melt
WATER_DEGASSING_EARTH = 1.0               # oceans per Gyr at Earth's melt rate and mantle water
MANTLE_WATER_EARTH_OCEANS = 1.7
# Greenhouse: τ = τ_bg·P^n (with water) + a·ln(1 + p/p₁)·P^n + τ_CO₂·p^n,
# P total and p CO₂ partial pressure in bar. a from Earth's 2×CO₂ forcing (Δτ ≈ 0.034).
GREENHOUSE_BACKGROUND_WET = 0.618
GREENHOUSE_BACKGROUND_DRY = 0.02
GREENHOUSE_CO2_LOG = 0.0358
GREENHOUSE_CO2_LOG_BAR = 1e-6
GREENHOUSE_STEAM = 5.2                    # τ per bar^n of water vapour in a steam atmosphere
# Escape: energy-limited, ṁ = ε π F_XUV R³ / (G M). Water with a wet stratosphere
# (moist greenhouse above MOIST_GREENHOUSE_K) or a steam atmosphere loses hydrogen;
# the oxygen left behind oxidises the crust. Bulk air escapes with BULK_ESCAPE_SHARE
# of the efficiency, switched on around the cosmic shoreline at the current XUV.
MOIST_GREENHOUSE_K = 340.0
STRATOSPHERE_WATER_COLD = 3e-6
DIFFUSION_LIMIT_PER_MIXING = 2.5e17       # H atoms m⁻² s⁻¹ per unit stratospheric H mixing ratio (Hunten 1973)
# Heavy gases escape far less readily than hydrogen: the efficiency below gives a present-day loss
# of ~3 kg/s at Mars (MAVEN: a few kg/s; Jakosky et al. 2018), ~0.5 kg/s at Venus and almost
# nothing at Earth, with ~40× more while the young star's XUV output was saturated.
BULK_ESCAPE_SHARE = 0.002
BULK_ESCAPE_SHORELINE = 0.5               # bulk escape switches on at this shoreline ratio (present-Sun XUV)
THIN_AIR_BAR = 1e-4                       # below this pressure, escape is limited by the gas left
BULK_ESCAPE_WIDTH = 0.1                   # width of the switch (dex)
OXYGEN_CRUST_SINK_GYR = 0.02              # timescale of abiotic O₂ uptake by the crust at Earth-like volcanism
# Tier 0 table used during the integration.
HISTORY_CLIMATE_BANDS = 18
HISTORY_CLIMATE_TOLERANCE_K = 0.2
# Node spacing; halving every step changed Earth's temperatures by < 0.5 K.
HISTORY_TABLE_LOG_S_STEP = 0.08           # in ln(instellation)
HISTORY_TABLE_TAU_STEP = 0.12             # in ln(1 + τ)
HISTORY_TABLE_LOG_P_STEP = 0.7            # in ln(pressure)
HISTORY_TABLE_LAND_STEP = 0.15
HISTORY_TABLE_CACHE = 16                  # tables kept in memory for repeated runs of the same planet
# Oxygen (redox balance, after Goldblatt et al. 2006). Masses in 10¹⁸ kg, fluxes per Gyr.
# Earth: organic burial ~10 Tmol O₂/yr; volcanic reductants (∝ √melt) about a fifth of it today;
# the rest balances oxidative weathering ∝ √O₂ at 1.2 × 10¹⁸ kg of O₂.
OXYGEN_BURIAL_EARTH = 320.0
REDUCTANT_EARTH_SHARE = 0.18
OCEAN_PRODUCTIVITY_SHARE = 0.55           # ocean share of Earth's organic burial
OXYGEN_MASS_EARTH = 1.2
SEAFLOOR_OXIDATION = 0.1                  # O₂ taken up by fresh sea floor, relative to Earth's land weathering
OXYGEN_SMALL = 1e-6                       # O₂ mass below which sinks switch off smoothly
HOT_OXIDATION_K = 500.0                   # hot surfaces take up O₂ on OXYGEN_CRUST_SINK_GYR
OXYGENATION_FRACTIONS = (1e-3, 0.05)      # O₂ fractions logged as the first and second rise
METHANE_ANOXIC_O2 = 1e-4                  # methanogenic CH₄ falls once O₂ exceeds this fraction
# CH₄ ∝ 1 / (1 + (O₂ / METHANE_ANOXIC_O2)^exponent): 10³ ppm in the anoxic Archean (Pavlov et al. 2000),
# ~10 ppm in the mid-Proterozoic at a per cent of present O₂ (Olson et al. 2016), 0.7 ppm pre-industrial.
METHANE_O2_EXPONENT = 0.95
SUBSURFACE_METHANE_SHARE = 0.1            # CH₄ reaching the air from life sealed under ice
HABITABLE_MAX_K = 340.0                   # the origin-of-life clock runs below this mean temperature
LAND_PRODUCTIVITY_WIDTH_K = 30.0
BIOSPHERE_ESTABLISH_GYR = 0.05            # a new biosphere reaches its full productivity over this time
LAND_SPREAD_GYR = 0.2                     # and spreads over the land over this one
HISTORY_MAX_STEP_GYR = 0.02
