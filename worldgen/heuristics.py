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

# --- Surface: tectonic simulation (milestone 3) ------------------------------

TECTONIC_DURATION_MYR = 400.0        # default simulated time
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
# Melt 8 mm water per positive degree-day (Braithwaite 1995: 7–9 for ice), with
# monthly temperatures spread by a normal distribution; snow below +1 °C.
DEGREE_DAY_FACTOR_M = 0.008
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
