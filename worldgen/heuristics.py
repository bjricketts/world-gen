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

# --- Moons and rings (milestone 11) --------------------------------------

# Outermost stable moon orbits as fractions of the Hill radius: prograde from
# Holman & Wiegert's binary integrations as applied by Barnes & O'Brien
# (2002); retrograde from the Solar System's retrograde moons (their Table 1).
MOON_STABLE_FRACTION_PROGRADE = 0.36
MOON_STABLE_FRACTION_RETROGRADE = 0.50

# Ring–moon boundary a_R = (3M / γρ)^(1/3) for strengthless material filling
# its Roche lobe (γ = 1.6, as used by Hedman 2015 and Sicardy et al. 2024),
# for porous icy material of Saturn's small moons' density and rocky
# material of Amalthea's (Sicardy et al. 2024).
ROCHE_GAMMA_RUBBLE = 1.6
RING_DENSITY_ICY = 400.0
RING_DENSITY_ROCKY = 860.0

# --- Giant appearance (task 11.2) ------------------------------------------

# Intrinsic luminosity at 4.5 Gyr (units of 8.710e-10 L_sun, Linder et al.'s L_J) against
# mass (M_E): Jupiter and Saturn from their Cassini-era intrinsic fluxes, 7.485 W/m² (Li et al.
# 2018) and 2.84 W/m² (Wang et al. 2024, over mean radii of 69,911 and 58,232 km; task 11.8);
# ice giants at the geometric mean of Uranus (≤1.01e-3) and Neptune (9.85e-3)
# at their mean mass, with a seeded scatter spanning the two; 5 and 10 M_J
# from the Burrows et al. (1997) cooling tracks at 4.5 Gyr. Interpolated
# log-log; L ∝ t^-1 with age (Linder et al. 2019).
L_JUPITER_W = 8.710e-10 * 3.828e26
GIANT_LUMINOSITY_TABLE = ((15.8, 3.15e-3), (95.2, 0.363), (317.8, 1.379), (1589.0, 20.4), (3178.0, 76.4))
GIANT_LUMINOSITY_AGE_GYR = 4.5
GIANT_LUMINOSITY_MIN_AGE_GYR = 0.01
ICE_GIANT_LUMINOSITY_SCATTER_DEX = 0.5

# Sudarsky, Burrows & Pinto (2000) class boundaries in T_eff (K); class V also
# above the IV boundary when surface gravity is below 10 m/s².
SUDARSKY_BOUNDARIES_K = (150.0, 350.0, 900.0, 1500.0)
SUDARSKY_LOW_GRAVITY_M_S2 = 10.0
# Bond albedos: class I gas giants the mean of Jupiter's 0.503 (Li et al. 2018) and Saturn's 0.41
# (Wang et al. 2024), the Cassini-era values (task 11.8; Voyager's were 0.34); class I ice giants
# between Neptune's 0.29 (Pearl & Conrath 1991) and Uranus's 0.349 (Irwin et al. 2025); classes II–V
# the Sudarsky G2V fiducial values.
GIANT_ALBEDO = {"I gas": 0.46, "I ice": 0.30, "II": 0.81, "III": 0.12, "IV": 0.03, "V": 0.55}
# Half-width (dex in T_eff) over which the albedo blends across a class boundary.
GIANT_ALBEDO_BLEND_DEX = 0.04

# Geometric albedos in B, V, R, I (445, 551, 658, 806 nm) from Cahoy, Marley &
# Fortney (2010) Table 4: class I gas Jupiter 1× at 5 AU, class I ice Neptune
# 10× at 10 AU, II Jupiter 1× at 2 AU, III Jupiter 1× at 0.8 AU. Classes IV
# and V have no model there: IV takes class III's shape and V class II's,
# scaled to their Bond albedos (heuristic).
GIANT_BANDS_NM = (445.0, 551.0, 658.0, 806.0)
GIANT_GEOMETRIC_ALBEDO = {
    "I gas": (0.609, 0.567, 0.531, 0.453),
    "I ice": (0.388, 0.295, 0.228, 0.147),
    "II": (0.733, 0.742, 0.727, 0.673),
    "III": (0.556, 0.322, 0.156, 0.047),
}
# Belt transmission of the chromophores Cahoy's models lack (class I gas
# giants): belts are darker and redder than zones (tuned constant).
CHROMOPHORE_TRANSMISSION = (0.35, 0.55, 0.78, 0.95)
ZONE_CHROMOPHORE_SHARE = 0.3         # zones carry this power of the belt tint
BELT_SHARE = 0.5
# Vortex colour: the chromophore tint raised to this power (redder than belts).
VORTEX_CHROMOPHORE_POWER = 2.5

# --- giant colours from composition (task 11.11) ---------------------------------
# Heavy-element enrichment: the Solar System C/H fit log ε = 8.84 − 1.13 log10(M/M_J)
# (Lothringer et al. 2025), plus the star's [Fe/H]; the scatter is heuristic.
GIANT_METALLICITY_LOG_EPS_MJ = 8.84
GIANT_METALLICITY_SLOPE = -1.13
GIANT_METALLICITY_SCATTER_DEX = 0.2
# Adiabat factor α of the Robinson & Catling (2012) profile: their Jupiter fit 0.85 up to
# 10× protosolar, falling to 0.6 at 50×, which puts the ice giants' 1-bar temperatures near
# the observed 76 and 72 K and their methane cloud at 1.5–2.5 bar (heuristic).
GIANT_ALPHA_SOLAR = 0.85
GIANT_ALPHA_RICH = 0.6
GIANT_ALPHA_METALLICITY = (10.0, 50.0)
# Aerosols, as optical depths at 800 nm; fitted to the Payne et al. (2025) spectra of the
# four giants with two points per law (heuristic):
GIANT_UPPER_HAZE_TAU = 0.40                 # extended small haze above the deck (all four ~0.3–0.5)
GIANT_HAZE_SCATTER_DEX = 0.2
# tropospheric haze over an ammonia cloud, linear in its base pressure (bar), through Jupiter and
# Saturn as the model builds them (the haze saturates their colour above ~40)
GIANT_TROPOSPHERIC_HAZE = ((0.61, 0.0), (1.98, 40.0))
GIANT_TROPOSPHERIC_HAZE_MAX = 80.0
GIANT_CHROMOPHORE_TAU = 0.042               # Jupiter's chromophore layer over an ammonia cloud
# haze at the methane condensation level, falling to none between Uranus's and Neptune's
# internal heat fluxes (W/m², interpolated in log)
GIANT_METHANE_HAZE_TAU = 1.3
GIANT_METHANE_HAZE_FLUX_W_M2 = (0.04, 0.43)
GIANT_CLOUD_TAU = 10.0                      # the visible cloud deck
GIANT_CLOUD_ALBEDO = 0.995
GIANT_FLOOR_REFLECTANCE = 0.9               # below the model, relative to the cloud albedo
GIANT_ICE_CLOUD_FLOOR_BAR = 6.0             # deep (H2S) cloud of ice giants, fixed
GIANT_CLEAR_DECK_BAR = 30.0                 # model floor when nothing condenses
GIANT_RT_LAYERS = 60
GIANT_GEOMETRIC_FACTOR = 0.75               # geometric albedo over the diffuse reflectance
GIANT_BELT_HAZE_TAU = 10.0                  # haze optical depth that halves the belt tint
GIANT_ZONE_CLOUD_SCALE = 1.5                # ice-giant zones: thicker cloud than the disc mean
# Classes III–V: albedo ∝ Z^−b in B, V, R, I, b from Cahoy et al. (2010) Table 4 (clear 0.8 AU
# models, 1×→3× and 10×→30× averaged); Z clipped to this range.
GIANT_HOT_METALLICITY_EXPONENT = (0.17, 0.29, 0.40, 0.41)
GIANT_HOT_METALLICITY_RANGE = (0.3, 100.0)

# Giants with envelope mass fraction below this look and behave like ice
# giants (Uranus and Neptune ~0.1–0.2; Saturn and Jupiter ≳ 0.8).
GIANT_ICE_ENVELOPE_MAX = 0.5

# Speed scale U (m/s) in the Rhines length π√(2U/β): calibrated so Jupiter and
# Saturn get ~20 jets and Uranus and Neptune 3 (Lian & Showman 2010),
# interpolated log-linearly in envelope fraction, with a seeded scatter.
JET_SPEED_GAS_M_S = 15.0
JET_SPEED_ICE_M_S = 200.0
JET_ENVELOPE_ICE = 0.15
JET_ENVELOPE_GAS = 0.9
JET_SPEED_SCATTER_DEX = 0.1
JET_MAX_LATITUDE_DEG = 85.0
# Gas giants with intrinsic heat flux above this have a prograde equatorial jet
# (Liu & Schneider 2010: between Uranus 0.04 and Neptune 0.43 W/m²). Ice giants
# keep a retrograde jet (water condensation, Lian & Showman 2010).
PROGRADE_JET_MIN_FLUX_W_M2 = 0.13

# Tidally locked giants: one superrotating equatorial jet as wide as the
# equatorial deformation radius (Showman & Polvani 2011), capped at the 60°
# their models reach; hot spot east of the substellar point (10–60° in their
# models; middle value).
LOCKED_JET_MAX_LATITUDE_DEG = 60.0
HOT_SPOT_OFFSET_DEG = 30.0
GIANT_CP_OVER_R = 3.5                # c_p / R for an H2-He atmosphere

# --- Giant cloud tops (task 11.3) ------------------------------------------

# Zonal wind: jets alternate in direction at the jet cores of the appearance
# and peak at the belt–zone boundaries (Melin 2026). Equatorial peaks: Jupiter
# ~100 m/s (Fletcher et al. 2019), Saturn ~400 m/s, Neptune −400 m/s with
# ~250 m/s prograde at high latitude (Melin 2026); gas giants interpolate
# log-linearly in mass between Saturn and Jupiter. Off-equatorial jet peaks
# as a fraction of the equatorial one, and the depth of the equatorial dip
# that makes the gas giants' double-peaked equatorial jet (Fletcher et al.
# 2019), are tuned.
WIND_EQUATOR_JUPITER_M_S = 100.0
WIND_EQUATOR_SATURN_M_S = 400.0
WIND_EQUATOR_ICE_M_S = -400.0
WIND_HIGH_LATITUDE_ICE_M_S = 250.0
WIND_JET_FRACTION_GAS = 0.6
WIND_EQUATOR_DIP = 0.4
WIND_EQUATOR_PEAK_FRACTION = 0.35    # equatorial peaks at this fraction of the first jet spacing
WIND_LOCKED_M_S = 1000.0             # superrotating jet of hot locked giants (Showman & Polvani 2011: ~1 km/s)

# Band structure: zones are anticyclonic shear (−∂u/∂y in the north), cold and
# cloudy; belts cyclonic and warm (Fletcher et al. 2019). Contrasts are tuned.
BAND_TEMPERATURE_CONTRAST_K = 3.0    # zone–belt difference in the upper troposphere
BAND_RELIEF_SCALE_HEIGHTS = 0.5      # zone–belt cloud-top relief in scale heights (heuristic, for shading)
BAND_BRIGHTNESS_CONTRAST = 0.07      # zone–belt brightness where zones and belts share a colour
BAND_CONTRAST_SPREAD = (0.45, 1.0)   # per-band contrast drawn from this range (faded and strong belts)
BAND_EDGE_WARP = 0.18                # band-edge waviness as a fraction of the local band width
BAND_TURBULENCE_FROM_DEG = 45.0      # latitude above which filamentary turbulence grows (Sankar et al. 2024)
BAND_TURBULENCE_WARP = 0.04          # its displacement (radians) at the pole
CLOUD_NOISE_OCTAVES = 3
# Above ~70° the banding gives way to a chaotic polar region (Melin 2026), except up to a hexagon's jet.
POLAR_CHAOS_DEG = (65.0, 75.0)

# Vortices (Sankar et al. 2024, JunoCam catalogue): sizes log-normal about
# the deformation radius, aspect ratios log-normal about 1.5, colour classes
# with their shares and latitude preferences. The deformation radius N H / f
# at 45° with N H = R_s √(T/c_p) gives 1450 km on Jupiter; the calibration
# brings it to the catalogue's 1800 km. The count per deformation area is
# tuned (JunoCam saw only part of the planet).
DEFORMATION_CALIBRATION = 1.24
VORTEX_DENSITY_PER_LD2 = 0.008
VORTEX_DENSITY_ICE_FRACTION = 0.1
VORTEX_SIZE_MEDIAN_LD = 0.75          # semi-major axis median, in deformation radii
VORTEX_SIZE_SIGMA = 0.8               # ln-space spread (most between 300 and 10 000 km on Jupiter)
VORTEX_ASPECT_MEDIAN = 1.5
VORTEX_ASPECT_SIGMA = 0.2
VORTEX_MAX_BAND_FRACTION = 0.45       # semi-minor axis at most this fraction of the band it sits in
VORTEX_CLASS_SHARES = {"white": 0.42, "red": 0.22, "brown": 0.09, "dark": 0.27}
VORTEX_RELIEF_SCALE_HEIGHTS = 0.6     # anticyclones domed, cyclones dished (heuristic, for shading)
VORTEX_RIM_DARKENING = 0.12           # darker ring at a vortex's edge (heuristic)
ICE_BRIGHT_CLOUDS = 15                # short-lived bright clouds on ice giants (Melin 2026)
ICE_BRIGHT_ASPECT = (3.0, 6.0)

# Great ovals: one anticyclone as wide as its zone (the Great Red Spot fills
# the gap between its bounding jets; eccentricity 0.6–0.92, Sánchez-Lavega et
# al. 2024); dark spots on ice giants. Probabilities are tuned.
GREAT_OVAL_PROBABILITY = 0.5
GREAT_OVAL_LATITUDE_DEG = (15.0, 35.0)
GREAT_OVAL_ECCENTRICITY = (0.6, 0.92)
GREAT_OVAL_WIDTH_FRACTION = 0.9
GREAT_DARK_SPOT_PROBABILITY = 0.5
GREAT_DARK_SPOT_LD = 1.6              # semi-minor axis in deformation radii (heuristic)

# Polar features of gas giants: a polar cyclone at each pole; a ring of
# circumpolar cyclones at ~84° as on Jupiter, as many as fit at 5.87 radii of
# maximum wind apart (Gavriel & Kaspi 2021; ~1000 km on Jupiter, visible
# diameters 4000–6000 km); or a hexagon on a prograde jet near 78° as on
# Saturn (Allison et al. 1990, via Melin 2026). Probabilities are tuned.
POLAR_CYCLONE_RING_PROBABILITY = 0.5
POLAR_RING_LATITUDE_DEG = 84.0
POLAR_RING_SPACING = 5.87
POLAR_CYCLONE_RMAX_LD = 0.55
POLAR_CYCLONE_VISIBLE = 2.5           # visible radius in radii of maximum wind
HEXAGON_PROBABILITY = 0.5
HEXAGON_JET_LATITUDE_DEG = (65.0, 82.0)

# Tidally locked giants: day–night temperature difference ~100 K for warm
# giants and 500–1000 K for hot Jupiters with T_eq 1600–2500 K (Skinner & Wei
# 2025); interpolated in T_eq between these.
DAY_NIGHT_WARM_K = 100.0
DAY_NIGHT_HOT_K = 750.0
DAY_NIGHT_T_EQ_K = (1300.0, 1600.0)
POLAR_VORTEX_LOCKED_DEG = 20.0        # radius of the anticyclonic polar vortices (heuristic)

# --- Rings (task 11.4) ------------------------------------------------------

# Dense rings form where weak icy material sits near 70 K (Hedman 2015): ring
# temperature T = 283 K (1 − A)^¼ (L/L☉)^¼ (1 AU/a)^½. A planet is in the
# window when some ring albedo between the dark and bright limits brings the
# ring into the band below; the albedo that gives 70 K picks bright broad
# icy rings (Saturn, A ≈ 0.66 needed) or dark narrow ringlets (Uranus, A ≈ 0.02).
RING_TEMPERATURE_K = 70.0
RING_TEMPERATURE_WINDOW_K = (60.0, 80.0)
RING_ALBEDO_RANGE = (0.02, 0.7)
RING_BROAD_ALBEDO_FROM = 0.3          # required albedo above which the dense rings are broad and bright
DENSE_RING_PROBABILITY = 0.75         # chance of dense rings inside the window (both Solar System giants in it have them)
DUSTY_RING_PROBABILITY = 0.9          # chance of dusty rings elsewhere (all four giants have rings)
# Poynting–Robertson drag: τ = 4.19 Gyr (ρ/3000)(r/1 m)(a/1 AU)² (L☉/L) (Hyodo & Ida 2025). Rings need
# 1 cm rock to last at least this long (dust is replenished, but not where PR drag clears it within a few Myr).
RING_MIN_PR_LIFETIME_GYR = 0.01
PR_DECAY_GYR = 4.19

# Saturn-like broad rings (Miller et al. 2023): region edges as fractions of the icy Roche limit
# (outer A edge 136 770 km of 138 760 km), optical depths and Bond albedos of the C, B, Cassini Division
# and A rings; the B ring is split into its thinner inner part and the opaque core.
BROAD_RING_TEMPLATE = (
    # name, inner fraction, outer fraction, optical depth, Bond albedo, kind
    ("C", 0.537, 0.663, 0.08, 0.18, "polluted"),
    ("B inner", 0.663, 0.70, 1.0, 0.55, "icy"),
    ("B", 0.70, 0.847, 3.0, 0.55, "icy"),
    ("Cassini Division", 0.847, 0.879, 0.1, 0.18, "polluted"),
    ("A", 0.879, 0.986, 0.5, 0.55, "icy"),
)
BROAD_RING_MASS_RATIO = 3e-8          # ring/planet mass (Saturn, Sicardy et al. 2024)
CASSINI_RESONANCE_RANGE = (0.75, 0.95)  # a moon's 2:1 resonance in this span of the Roche limit sets the division

# Uranus-like narrow rings (Sicardy et al. 2024; Ohtsuki 2025; Hedman 2015): 5–12 dark rings between
# 0.81 and 1.0 of the Roche limit for ρ = 1.2 g/cm³ (the ε ring at it), widths of a few km, the
# outermost wider; optical depths 0.3–2.5; albedo a few per cent.
NARROW_RING_DENSITY = 1200.0
NARROW_RING_COUNT = (5, 12)
NARROW_RING_SPAN = (0.81, 0.995)
NARROW_RING_WIDTH_KM = (2.0, 10.0)
NARROW_OUTER_WIDTH_KM = (20.0, 100.0)
NARROW_RING_TAU = (0.3, 2.5)
NARROW_RING_ALBEDO = 0.03
NARROW_RING_MASS_RATIO = 2e-10

# Dusty rings fed by small inner moons, in planet radii. Warm (Jupiter, Sicardy et al. 2024): halo
# 1.29–1.71, main 1.71–1.80 (τ 8e-6), gossamer 1.80–2.55 and 2.55–3.16 (τ 1e-8–1e-7). Cold (Neptune):
# narrow dusty rings at 1.69, 2.15, 2.31 and 2.54 (τ 0.003), the outermost with arcs of τ 0.1 (Adams).
DUSTY_WARM_TEMPLATE = (("halo", 1.29, 1.71, 1e-6), ("main", 1.71, 1.80, 8e-6),
                       ("Amalthea gossamer", 1.80, 2.55, 1e-7), ("Thebe gossamer", 2.55, 3.16, 1e-8))
DUSTY_COLD_TEMPLATE = (("Galle", 1.66, 1.72, 1e-4), ("Le Verrier", 2.144, 2.148, 3e-3),
                       ("Lassell", 2.15, 2.30, 1e-4), ("Adams", 2.535, 2.539, 3e-3))
DUSTY_COLD_FROM_T0_K = 70.0           # dusty rings are Neptune-like below this zero-albedo temperature
RING_ARC_TAU = 0.1
RING_ARC_COUNT = (2, 5)
RING_ARC_LENGTH_DEG = (4.0, 15.0)
DUSTY_RING_MASS_RATIO = 5e-22
DUSTY_RING_ALBEDO = 0.05

# Relative reflectance in B, V, R, I of ring materials (shapes only; the albedo sets the level):
# icy rings are reddish from a UV absorber, polluted regions flatter (Miller et al. 2023); dark rings
# neutral (Ohtsuki 2025); dust reddish (heuristic).
RING_MESH_SEGMENTS = 720               # azimuthal segments of a ring mesh on the globe
RING_SPECTRA = {"icy": (0.62, 0.92, 1.0, 1.02), "polluted": (0.85, 0.95, 1.0, 1.03), "dark": (1.0, 1.0, 1.0, 1.0),
                "dusty": (0.75, 0.9, 1.0, 1.05)}

# --- Moon systems (task 11.5) ------------------------------------------------
# Gas-dominated giants take a Jovian or Saturnian architecture of regular moons after Sasaki, Stewart & Ida
# (2010): a planet that opens a gap in its star's disc keeps 4–5 similar moons, the inner ones in resonance
# (four most likely, 39 of 100 runs; 4 or 5 in 80%); one that does not keeps one dominant icy moon (67%).
# The gap criterion is Crida et al. (2006): 3H/(4R_H) + 50/(qℛ) ≤ 1, ℛ = 1/(α h²), for a nebula of
# aspect ratio h at the planet (0.05 at 5.2 AU as in their runs, ∝ a^(1/4), heuristic) and α (heuristic).
NEBULA_ASPECT_RATIO_5AU = 0.05
NEBULA_ASPECT_SLOPE = 0.25
NEBULA_ALPHA = 1e-3
# Jovian architecture: counts (the split of the remaining 20% is heuristic); masses log-normal about the
# Galilean mean, 4.9e-5 M_p, with the Galilean scatter; the innermost at Io's 5.9 R_p; the inner three in
# 2:1 resonances; the fourth trapped in half of the runs, otherwise at a period ratio about Callisto's;
# later moons untrapped beyond (Sasaki et al. set I'). Two moons are rocky in more than half the runs.
JOVIAN_MOON_COUNTS = {3: 0.10, 4: 0.39, 5: 0.41, 6: 0.10}
JOVIAN_MOON_MASS = 4.9e-5             # median mass, fraction of the planet
JOVIAN_MOON_MASS_DEX = 0.2
JOVIAN_INNER_ORBIT_RP = 5.9
JOVIAN_ORBIT_DEX = 0.05
JOVIAN_OUTER_TRAPPED = 0.5
JOVIAN_FREE_PERIOD_RATIO = (2.1, 2.6)
JOVIAN_EXTRA_SPACING = (1.15, 1.3)    # fifth and sixth moons: orbit ratio near Sasaki's 5-Hill-radius trap spacing
JOVIAN_ROCKY_COUNTS = {1: 0.25, 2: 0.55, 3: 0.20}
# Saturnian architecture: the number of moons above 1e-5 M_p (67% one; the rest split heuristically); the
# dominant moon about Titan (2.4e-4 M_p at 20.3 R_p, within 1σ of the runs; scatter heuristic); extra large
# moons inward of it at Sasaki's 1e-5–1e-4.
SATURNIAN_MOON_COUNTS = {1: 0.67, 2: 0.22, 3: 0.08, 4: 0.03}
SATURNIAN_MOON_MASS = 2.4e-4
SATURNIAN_MOON_ORBIT_RP = 20.3
SATURNIAN_MASS_DEX = 0.2
SATURNIAN_ORBIT_DEX = 0.1
SATURNIAN_EXTRA_MASS = (1e-5, 1e-4)
SATURNIAN_EXTRA_SPACING = (1.3, 2.0)  # orbit ratio between successive extra moons (heuristic)
REGULAR_ROCHE_MARGIN = 3.0           # innermost regular moon at least this many fluid Roche radii out (Io: 3.4)
DISC_OUTER_FRACTION = 0.9             # the dominant Saturnian moon stays inside this fraction of the disc
CIRCUMPLANETARY_DISC_RP = 30.0        # infall radius r_c (Canup & Ward 2002; Sasaki et al. 2010)
# Water fractions: icy regular moons as Ganymede, Callisto and Titan; rocky ones as Io and Europa (heuristic
# reading of their densities). Ice is available only where the nebula is below 160 K.
ICY_MOON_WATER_FRACTION = (0.40, 0.55)
ROCKY_MOON_WATER_FRACTION = (0.0, 0.10)
ICE_CONDENSATION_K = 160.0
# Moons spread from a ring past the Roche limit (Crida & Charnoz 2012): each moon twice the mass of the
# next one in; q ∝ Δ^(9/5) inside the 2:1 point Δ = 0.587 and ∝ (1+Δ)^3.9 beyond; the outermost reached by
# tides over the planet's age, r = (K q t)^(2/13), K = 39 k₂ √(GM) R⁵ / 2Q; the ring disc holds 1.5 times
# the moons' mass. The disc mass ratio follows the ring regime, from the Solar System giants' moons
# (Saturn's inside Titan, Uranus's regular moons, Neptune's inner moons, Jupiter's inner moons).
RING_DISC_MASS = {"broad": 1.3e-5, "narrow": 1.6e-4, "dusty cold": 7e-7, "dusty warm": 2e-9}
RING_DISC_MASS_DEX = 0.3
RING_DISC_MOON_RATIO = 1.5
RING_CHAIN_DELTA_21 = 0.587
RING_CHAIN_MIN_DELTA = 0.05
RING_CHAIN_MASS_DEX = 0.15            # scatter of each chain moon's mass (heuristic)
RING_CHAIN_CLEARANCE = 1.5            # chain moons stay inside the innermost disc moon by this orbit ratio
MOON_MIN_MASS_KG = 1e16               # smaller moons are not listed
# Water fraction of ring-born moons by ring regime (heuristic reading of the densities of Saturn's
# mid-sized moons, Uranus's moons, and Jupiter's rocky inner moons).
RING_MOON_WATER_FRACTION = {"broad": (0.6, 0.9), "narrow": (0.4, 0.6), "dusty cold": (0.4, 0.6),
                            "dusty warm": (0.0, 0.1)}
# Drawn moons' densities mix ice and rock at these densities (heuristic; gives Ganymede, Titan and Io
# within 10%).
MOON_ICE_DENSITY = 1300.0
MOON_ROCK_DENSITY = 3400.0
# Impact moons of rocky planets (Elser et al. 2011): about half of planets have a moon-forming impact
# (88 events among ~180 planets); the disc is m_disk/m_tot = 2.8 (0.1/γ)^1.25 (m_pass/m_tot)² (Canup 2008,
# for 0.4 < b < 0.8), the moon 10–55% of the disc (Kokubo et al. 2000). γ log-uniform over 0.01–0.5 is
# calibrated so ~1 in 10 planets hold a moon above half the Moon's mass (Elser: 1 in 12, 1 in 45 to 1 in 4).
IMPACT_MOON_PROBABILITY = 0.49
IMPACT_GAMMA_RANGE = (0.01, 0.5)
IMPACT_B_RANGE = (0.4, 0.8)
IMPACT_EFFICIENCY = (0.10, 0.55)
IMPACT_CORE_RADIUS = 0.5              # impactor core radius / impactor radius, core twice the mantle density
IMPACT_MOON_DENSITY = 3340.0          # mantle material, as the Moon
IMPACT_MOON_START = 1.3               # moons accrete at 1.3 times the Roche limit (Kokubo et al. 2000)
IMPACT_MOON_ECCENTRICITY = (0.0, 0.06)
IMPACT_MOON_INCLINATION_DEG = (0.0, 6.0)
REGULAR_MOON_ECCENTRICITY = (0.0, 0.01)
REGULAR_MOON_INCLINATION_DEG = (0.0, 0.5)

# --- Captured moons and impact systems of giants (task 11.7b) ------------------
# Impact systems of ice giants (Woo et al. 2021): a 3 M⊕ rocky impactor on proto-Uranus leaves a disc that
# spreads and forms ~1e-4 M_p of moons, the mass rising outward from ~4 to 20 R_p, about half ice
# (6.3e-5 ice, 5.5e-5 rock). The outcome is drawn about Uranus's five large moons (masses from JPL GM,
# orbits from the NSSDC fact sheet, in R_U = 25,559 km), with heuristic scatter. Half of ice giants have
# one (heuristic: one of the Solar System's two). An ice giant with an impact system grows no ring-born
# chain beside it (Woo et al.'s disc already spreads from the Roche limit).
URANIAN_IMPACT_PROBABILITY = 0.5
URANIAN_MOONS = ((5.08, 7.4e-7), (7.47, 1.44e-5), (10.41, 1.47e-5), (17.07, 3.92e-5), (22.83, 3.54e-5))  # (R_p, M_p)
URANIAN_MASS_DEX = 0.2
URANIAN_ORBIT_DEX = 0.05
URANIAN_WATER_FRACTION = (0.45, 0.60)
# Binary-exchange capture (Agnor & Hamilton 2006; Williams 2013; Williams & Zugger 2024). Every giant meets
# one binary deep enough to be split (heuristic). The binary's barycentre passes at q log-uniform over
# 4–32 R_p (Nogueira et al. 2011, fig. 5) at 2–10% of the planet's orbital speed (Nogueira et al.: captures
# fall off above 10%; Williams & Zugger's 1.5–3 km/s at 1 AU; lower bound heuristic). Its members are
# log-uniform over the planetesimals of the planet's region (heuristic): Triton- and Pluto-class bodies
# beyond the ice line, embryos inside it. One member is captured if lighter than Williams & Zugger's eq. 1,
# m₁ < 3 M_p [Gπ m₂ / (2 q v_enc (v_enc − v_peri))]^(3/2) − m₂, on an orbit with apoapsis at 0.5 (prograde)
# or 0.9 (retrograde) of the Hill radius; prograde and retrograde captures are equally likely (Porter &
# Grundy 2011).
CAPTURE_PERIAPSIS_RP = (4.0, 32.0)
CAPTURE_SPEED_FRACTION = (0.02, 0.10)
CAPTURE_BODY_MEARTH_ICY = (1e-4, 1e-2)
CAPTURE_BODY_MEARTH_ROCKY = (1e-3, 1.0)
CAPTURE_APOAPSIS_HILL = {"prograde": 0.5, "retrograde": 0.9}
CAPTURE_RETROGRADE_SHARE = 0.5
# Survival of the loose orbit through Kozai cycles and tidal friction, by star mass (Porter & Grundy 2011,
# table 1, averaged over their moons and planets): the rest escape or hit the planet.
CAPTURE_SURVIVAL_BY_STAR = ((0.47, 0.235), (1.0, 0.45), (1.7, 0.65))   # (M☉, surviving share)
# Survival among existing regular moons (Rufu & Canup 2017): a Triton of 2.09e-4 M_N survived 88%, 40%
# and 12% of runs against systems of 0.35e-4, 1.04e-4 and 3.13e-4 M_N, and was then usually the last large
# moon. Interpolated (and extrapolated, clipped) linearly in the log of captured / regular mass; a
# survivor clears the regular moons (ring-born inner moons stay), a failed capture leaves them.
CAPTURE_AMONG_MOONS = ((2.09 / 3.13, 0.12), (2.09 / 1.04, 0.40), (2.09 / 0.35, 0.88))
# The orbit circularises at constant angular momentum, a = q(1 + e) (Goldreich et al. 1989; Nogueira et
# al. 2011), then migrates by the planet's tides: outward if prograde, inward if retrograde. Final
# eccentricity and tilt from the planet's orbital plane are heuristic (Porter & Grundy: near-coplanar).
CAPTURED_FINAL_ECCENTRICITY = (0.0, 0.01)
CAPTURED_TILT_DEG = (0.0, 30.0)
CAPTURED_ICY_WATER_FRACTION = (0.25, 0.50)   # Triton and Pluto are about 70% rock (heuristic range)

# --- Moons through the rocky pipeline (task 11.6) -----------------------------
# Energy on a moon (Heller & Barnes 2013, eq. 22; Heller 2012): starlight lost in eclipses, starlight
# reflected by the parent (π R_p² α_p / 2a² of the direct flux), the parent's thermal emission
# R_p² σ T_p⁴ / a² absorbed over the moon's sphere ((1 − α_IR)/4, with T_p⁴ = T_eq⁴ + T_int⁴ as in Heller
# & Zuluaga 2013), and tidal heat. The moon absorbs the parent's infrared with albedo 0.05 (Heller & Zuluaga).
MOON_INFRARED_ALBEDO = 0.05
# Tidal heat of a synchronous moon, Ė = (21/2) G^1.5 (k₂/Q) M_p^2.5 R⁵ e² / a^7.5 (Tjoa et al. 2020, eq. 1,
# after Henning et al. 2009); Io's k₂/Q ≈ 0.015, Enceladus 0.003–0.013 (via Tjoa et al.).
# Hazard and limits: Io's 2 W/m² of tidal heat drives global volcanism (Heller & Zuluaga 2013).
TIDAL_HAZARD_FLUX_W_M2 = 2.0
# Parent's magnetosphere (Heller & Zuluaga 2013, eq. 1): R_S = (μ₀ f₀² / 8π²)^(1/6) 𝓜^(1/3) P_sw^(-1/6),
# f₀ = 1.3. The magnetopause is a half sphere of radius R_M = 1.2 R_S whose centre lies R_M − R_S behind the
# planet, continued by a tail cylinder of that radius; 1.2 makes 20% of an orbit at 2 R_S lie inside, as
# they state. The stellar wind's dynamic pressure is a present solar value at 1 AU scaled by distance⁻²
# (heuristic; the star's type and the age dependence of Grießmeier et al. 2007 are not used).
MAGNETOPAUSE_FORM_FACTOR = 1.3
MAGNETOPAUSE_SPHERE_RATIO = 1.2
SOLAR_WIND_PRESSURE_1AU_PA = 2.0e-9
# Dipole moments (A m²) of the Solar System planets, interpolated log-log in mass for parents without one
# (heuristic; Heller & Zuluaga's dynamo scaling reproduces them within a factor of 2–6). Rocky parents
# with a dynamo take Earth's moment scaled by mass; without one, none.
PLANET_DIPOLE_MOMENTS = ((1.0, 8.0e22), (14.5, 3.9e24), (17.1, 2.2e24), (95.2, 4.6e25), (317.8, 1.56e27))
# Ice shells over subsurface oceans (Tjoa et al. 2020, eq. 9, conductive): the depth where conduction from
# the surface temperature reaches the melting point with k(T) = 632/T + 0.38 − 0.00197 T W/m/K (Andersson &
# Inaba 2005, as corrected in Carnahan et al. 2021; Tjoa et al. print 0.0197). A melting depth beyond 30%
# of the radius means frozen through (Tjoa et al., after Hussmann et al. 2006). The shell would convect
# above a basal Rayleigh number of 10⁶ (Carnahan et al. 2021; Shibley & Goodman 2023) with ρ = 917 kg/m³,
# c_p = 2106 J/kg/K, k = 2.25 W/m/K (Carnahan et al., Table 1), α = 1.7e-4 /K and η = 10¹⁴ Pa s (the middle
# of their 10¹³–10¹⁵ range).
ICE_MELTING_K = 273.15
ICE_SHELL_MAX_DEPTH_FRACTION = 0.3
ICE_SHELL_DENSITY = 917.0
ICE_SHELL_CP = 2106.0
ICE_SHELL_CONDUCTIVITY = 2.25
ICE_EXPANSION = 1.7e-4
ICE_BASAL_VISCOSITY = 1e14
ICE_CRITICAL_RAYLEIGH = 1e6
SUBSURFACE_WATER_DENSITY = 1000.0

# --- Habitable moons (task 11.7) ------------------------------------------------
# Archetype priors: stars of 0.5–1.2 M☉ (moons of planets around stars below ~0.5 M☉ are disturbed by the
# star; Heller 2012); orbits of 8–30 parent radii (beyond the habitable edge for small eccentricities,
# Heller & Barnes 2013) and e of 1e-4–0.02. Two origins with heuristic shares: regular moons from the
# giant's disc take 3e-5–1e-4 of the parent's mass (the regular-moon budget of Canup & Ward 2006), so only
# 3–13 M_J parents give them habitable sizes (Dencs et al. 2025 form Mars-to-Earth-mass moons around
# 10 M_J); captured moons take 0.1–1 M⊕ around 0.3–3 M_J parents (within Williams & Zugger's eq. 1 for
# embryo binaries). Captured moons end at a = q(1 + e) ≈ 2q for q of 4–32 R_p (Nogueira et al. 2011), and
# closer, ~7 R_p, in Porter & Grundy's (2011) habitable-zone runs, hence 6–60 R_p; half are retrograde,
# tilted up to 30° from the parent's equator plane (heuristic). Giant impacts on giants make moons of at
# most ~1e-4 of the planet's mass (Woo et al. 2021), too small for this archetype.
HABITABLE_MOON_STAR_MSUN = (0.5, 1.2)
HABITABLE_MOON_ORIGINS = {"regular": 0.4, "captured": 0.6}
HABITABLE_MOON_REGULAR_PARENT_MEARTH = (1000.0, 4000.0)
HABITABLE_MOON_MASS_RATIO = (3e-5, 1e-4)
HABITABLE_MOON_CAPTURED_PARENT_MEARTH = (100.0, 1000.0)
HABITABLE_MOON_CAPTURED_MEARTH = (0.1, 1.0)
HABITABLE_MOON_ORBIT_RP = (8.0, 30.0)
HABITABLE_MOON_CAPTURED_ORBIT_RP = (6.0, 60.0)
HABITABLE_MOON_ECCENTRICITY = (1e-4, 0.02)
# A parent's own heat for its glow: giants below this mass are taken as ice giants (Saturn's mass,
# heuristic, since a moon's spec gives no envelope fraction).
JOVIAN_MIN_MEARTH_FOR_GLOW = 50.0

# --- Solar System validation (task 11.8) ---------------------------------------
# Melt feedback: tides that, at the dissipation of a partially molten interior, would exceed a moon's
# radiogenic heat keep it partially molten (or its shell decoupled over an ocean), with Io's measured
# k₂/Q (Lainey et al. 2009; within 2σ of Juno's Im(k₂), Park et al. 2025, via Paris et al. 2026, whose
# models keep Io's melt below ~20%). The threshold is heuristic. It gives Io ~1e14 W and Enceladus 22 GW
# (observed 15.8 ± 3.1 GW, Howett et al. 2011 via Lainey et al. 2012).
TIDAL_MOLTEN_K2_OVER_Q = 0.015
# Icy-body air. Titan's air holds 78× more volatiles per unit mass than Earth's; its N₂ most likely came
# from accreted ammonia ice (Nixon 2024). Cold worlds (equilibrium temperature below
# COLD_NITROGEN_METHANE_BELOW_K) scale the default inventory by 1 + k × water fraction, with k set so
# Titan (water 0.45) reaches its 1.467 bar (Fulchignoni et al. 2005, via Kimura 2025) without scatter.
ICY_NITROGEN_PER_WATER = 255.0
# Thick N₂–CH₄ air is hazy: its Bond albedo is the haze's, 0.25–0.27 for Titan over the Cassini epoch
# (Creecy et al. 2021); thinner air keeps the surface's albedo (threshold heuristic).
HAZE_ALBEDO = 0.26
HAZE_MIN_BAR = 0.1
# N₂ ice caps the air of colder worlds at its vapour pressure (Triton's 14.5 µbar is in vapour-pressure
# equilibrium, Sicardy et al. 2024): Grundy et al. (2024) Table 2, Lobo & Ferreira form
# ln p[bar] = A − B/T + C ln T + D₂T + D₃T² + D₄T³, α below 35.6 K and β up to the 63.15 K triple point.
N2_VAPOUR_ALPHA = (-3.92, 805.3, 4.54794, -0.08002, -5.347e-5, 0.0)
N2_VAPOUR_BETA = (9.4928, 857.5053, 1.00304, -0.04581, 2.630e-4, -1.210e-6)
N2_ALPHA_BETA_K = 35.6
N2_TRIPLE_K = 63.15

# --- Moon calibration fixes (2026-10-25; MOON_CALIBRATION.md) -------------------
# Moons whose tides work an ice shell over an ocean: water-rich (at least this fraction) and frozen
# (equilibrium temperature at the thin-air albedo below this; heuristic).
ICE_SHELL_WORLD_MIN_WATER = 0.05
ICE_SHELL_WORLD_BELOW_K = 180.0
# Their k₂/Q, log-log in x = μ_ice / (ρ_ice g R) between Europa (x 1.59; −Im k₂ 3e-4–2.5e-3, Bagheri et al.
# 2026, central 1e-3) and Enceladus (x 115; |k₂| 0.027 at a 34° lag, so 0.015, Bagheri et al. 2025);
# μ_ice = 3 GPa (Bagheri et al. 2025: 2–4 GPa). The interpolation is heuristic (two points).
ICE_SHELL_RIGIDITY_PA = 3.0e9
ICE_SHELL_DISSIPATION_ANCHORS = ((1.594, 1.0e-3), (114.9, 0.015))
ICE_SHELL_DISSIPATION_MIN = 3.0e-4
# The ice shell conducts as solid ice only below a porous, fractured lid: heat crosses the lid as if the
# surface were at this temperature (Shoemaker et al. 1982: ~130 K; Nimmo et al. 2003: ~140 K, where pores
# close; both via Singer et al. 2023).
ICE_LID_EFFECTIVE_K = 135.0

# Parent rotation assumed when a moon's spec does not give it.
PARENT_ROTATION_GIANT_H = 10.0
PARENT_ROTATION_ROCKY_H = 24.0

# Locked planets with eccentricity above this settle in a 3:2 spin-orbit
# resonance (as Mercury) instead of synchronous rotation.
RESONANCE_ECCENTRICITY = 0.1

# --- Bulk -----------------------------------------------------------------
# Small rocky bodies (task 11.6): below 0.1 M⊕ radii follow uncompressed densities of iron, rock and ice
# (heuristic; from their radii these give the Moon a core fraction of 0, Io 0.06, Mars 0.24 and Mercury
# 0.66), blended log-log into the Zeng-type relation by 1 M⊕. Implied core fractions within 0.1 of 0–1
# are clipped without a warning.
SMALL_BODY_BLEND_MEARTH = (0.1, 1.0)
SMALL_BODY_IRON_DENSITY = 7900.0
SMALL_BODY_ROCK_DENSITY = 3400.0
SMALL_BODY_ICE_DENSITY = 1300.0
CMF_TOLERANCE = 0.1

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
# Zoom erosion runs without uplift, so creep only wears relief down; in nature the roughness below a
# kilometre is renewed as fast as creep removes it. Creep is therefore limited to spreading material by
# about one lattice step over the run (diffusion length √(κt) ≤ this many steps); at ≥0.4 km spacing the
# physical rate is below that limit, and on finer lattices creep would otherwise erase the sub-km detail.
ZOOM_CREEP_MAX_STEPS = 1.0
# Erosion fades to zero toward the pinned region edge over a fixed ground distance, at least a few cells, so
# the fade is as gentle at 150 m spacing as at 750 m (a fixed cell count left a steep rim at fine spacing).
ZOOM_EDGE_BLEND_KM = 4.5
ZOOM_EDGE_BLEND_MIN_CELLS = 6
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
# A zoom request with no resolution or width given: a ~0.75 km lattice over about 150 km, which builds
# in ~15 s. Larger requests are refused past ZOOM_MAX_NODES (memory and time grow with the node count).
ZOOM_DEFAULT_RESOLUTION_M = 750.0
ZOOM_DEFAULT_HALF_WIDTH_KM = 75.0
ZOOM_MAX_NODES = 1_100_000     # admits blocks of 8 × 8 tiles (1017², or 1025² and 1009² on engine lattices)

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
# 0.10 since task 8.2: subsidence and biogenic sediment keep material on the continents, and 0.08 then left
# land ~100 m too high on the Earth example (0.12 makes it too low).
STREAM_POWER_K = 0.10
ZOOM_STREAM_POWER_K = 0.08           # zoomed regions keep the value their erosion was calibrated with
STREAM_POWER_M = 0.5
# Deposition from the river load, G r̄ Q_s / Q (Davy & Lague 2009; Yuan et al. 2019). G/p̃ is of order 1 or
# greater for natural rivers; the implicit solve iterates (Gauss–Seidel) to this tolerance.
STREAM_DEPOSITION_G = 0.0                # off by default (Ben, task 8.2); 1 is typical of natural rivers
DEPOSITION_MAX_ITERATIONS = 50
DEPOSITION_TOLERANCE_M = 0.01
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

# --- Surface record for resources (milestone 8) -------------------------------

# Heuristic plate layouts label a cell with the belt (Andean, arc, collision) that raised it
# most, where that belt raised it by more than this (scaled by the relief factor).
OROGENY_KIND_MIN_UPLIFT_M = 300.0

# --- Culture (milestone 8) --------------------------------------------------

# Development of a culture whose spec names no axis: natives default to an agrarian
# society with writing (roughly Earth's first millennium), colonists to the level that
# crossed space to reach the planet.
CULTURE_DEFAULT_DEVELOPMENT = {
    "native": {"energy": "farming", "transport": "wheel_and_sail", "communication": "writing"},
    "colonist": {"energy": "fusion", "transport": "spaceflight", "communication": "interplanetary"},
}

# --- Organic culture history (milestone 9) -------------------------------------

# An organic history runs from HISTORY_START_YEARS_AGO (the Holocene span of HYDE 3.2, which starts at
# 10 000 BCE) to the present in steps of HISTORY_STEP_YEARS (one generation), shortened to
# HISTORY_LATE_STEP_YEARS once any organic culture reaches HISTORY_LATE_FROM_ENERGY (fossil), so the
# industrial centuries get finer events (Ben's choice in task 9.1).
HISTORY_START_YEARS_AGO = 12_000.0
HISTORY_STEP_YEARS = 25.0
HISTORY_LATE_STEP_YEARS = 5.0
HISTORY_LATE_FROM_ENERGY = 3
# Organic natives start as foragers on foot with oral culture; everything above that emerges.
HISTORY_START_DEVELOPMENT = {"energy": "foraging", "transport": "foot", "communication": "oral"}

# Suitability components of the organic history (task 9.2), each 0–1 per cell.
# Arable land is the fuzzy minimum of four factors after Zabel et al. (2014): crops grow only in months
# above ARABLE_GROWTH_MIN_C (their 5 °C), need a season of ARABLE_SEASON_MONTHS such months (ramp,
# heuristic) and a ratio of precipitation to potential evaporation of ARABLE_MOISTURE (ramp, heuristic),
# the ground must be below ARABLE_MAX_SLOPE (their 16 % for most crops; the share of a cell below it
# assumes half-normal slopes around the landform's sub-grid slope), and permafrost (annual mean at or
# below ARABLE_PERMAFROST_C) is excluded. The ramps are set so 0.63 of ice-free land on the Earth example
# is arable to some degree, against Zabel's 77.7 of 127.5 million km² (0.61) suitable for rainfed crops.
ARABLE_GROWTH_MIN_C = 5.0
ARABLE_SEASON_MONTHS = (4.0, 6.0)
ARABLE_MOISTURE = (0.35, 0.65)
ARABLE_MAX_SLOPE = 0.16
ARABLE_PERMAFROST_C = 0.0
# Fresh water: rivers (log ramp over FRESH_WATER_DISCHARGE_M3S), a lake in or beside the cell, or a wet
# climate (precipitation over potential evaporation, ramp FRESH_WATER_WETNESS); heuristic.
FRESH_WATER_DISCHARGE_M3S = (10.0, 1000.0)
FRESH_WATER_WETNESS = (0.5, 1.0)
# Access: a coastal cell (one with a sea neighbour), closeness to the coast further in (e-folding
# ACCESS_COAST_KM) or a navigable river (log ramp over
# ACCESS_RIVER_M3S); heuristic.
ACCESS_COAST_KM = 50.0
ACCESS_RIVER_M3S = (100.0, 1000.0)
# Resources: the richest potential among the classes a pre-industrial society mines or trades (metals,
# salt, coal); timber is left to the arable and forest terms, and fuels and ores of later levels to the
# development rules (heuristic).
HISTORY_RESOURCES = ("copper", "tin_tungsten", "iron", "gold", "lead_zinc", "evaporites", "coal", "placer_gold")
# Climate: mean annual temperature people favour, full within CLIMATE_COMFORT_C and falling to zero at the
# cold and hot limits (heuristic around Xu et al. 2020: the population's main mode lies at 11–15 °C and
# mean annual temperatures above 29 °C cover 0.8% of land).
CLIMATE_COMFORT_C = (5.0, 25.0)
CLIMATE_COLD_LIMIT_C = -10.0
CLIMATE_HOT_LIMIT_C = 32.0
# Defensibility: rugged ground on the terrain ruggedness index of Nunn & Puga (2012), √8 · slope ·
# RUGGEDNESS_SPACING_M in hundreds of metres (their 30″ grid; plains 0.26, highlands 1.57, mountains 3.93
# from the sub-grid slopes, against 0.04 for the Netherlands and 5.0–6.2 for Nepal and Lesotho), ramped
# over DEFENCE_RUGGEDNESS; sea on a share of the cell's edges adds up to DEFENCE_SEA and a river above the
# crossing threshold DEFENCE_RIVER (heuristic).
RUGGEDNESS_SPACING_M = 926.0
DEFENCE_RUGGEDNESS = (0.25, 4.0)
DEFENCE_SEA = 0.6
DEFENCE_RIVER = 0.3

# Growth and spread of people in the organic history (task 9.3). People grow logistically at
# SPREAD_GROWTH_RATE. Fronts into empty land advance at the time-delayed wave-of-advance speed
# √(am/T)/(1 + aT/2) (Fort & Méndez 1999) = 0.81 km/yr with the middle of Pinhasi et al.'s (2005) ranges
# (a = 0.029–0.035 /yr, m = 900–2200 km² per generation, T = 29–35 yr; 0.6–1.1 km/yr predicted,
# 0.6–1.3 km/yr observed), over each link's travel time times the flat-ground speed. A front leaves cells
# holding SPREAD_ESTABLISHED_SHARE of their capacity and settles a cell with that share, taken from the
# source cell (so a newly settled cell starts a front of its own without a growth lag). Within settled land
# the diffusion equals m/(4T); moves are weighted by exp(−hours / SPREAD_TRAVEL_TAU_HOURS). From
# pack_and_boat, sea routes join coastal cells across at most SPREAD_SEA_REACH_KM of open water per
# transport level (sea-to-sea edges in full, half of each coast edge; heuristic), at the boat speed plus
# the port time; each coastal cell keeps its SPREAD_SEA_LINKS nearest landfalls. A strait narrower than a grid cell counts as a full cell of sea, so on the preview grid
# (~250 km cells) pack_and_boat crosses no strait. Shares above SPREAD_MAX_SHARE are split into sub-steps; cells with fewer than
# SPREAD_MIN_PEOPLE are emptied; a culture reaching SPREAD_LANDFALL_PEOPLE on a new landmass enters the
# chronicle.
SPREAD_GROWTH_RATE = 0.032
SPREAD_MOBILITY_KM2 = 1500.0
SPREAD_GENERATION_YEARS = 32.0
SPREAD_ESTABLISHED_SHARE = 0.1
SPREAD_TRAVEL_TAU_HOURS = 60.0
SPREAD_SEA_REACH_KM = (0.0, 200.0, 800.0, 3000.0, 5000.0, 5000.0)
SPREAD_SEA_LINKS = 12
SPREAD_VOYAGE_CHUNK = 64                  # coastal cells whose voyages are searched at once (memory)
SPREAD_MAX_SHARE = 0.5
SPREAD_MIN_PEOPLE = 1.0
SPREAD_LANDFALL_PEOPLE = 1000.0

# Farming and development in the organic history (task 9.4). Energy is a real-valued level whose anchor
# populations (POPULATION_ANCHOR, HYDE 3.2 mid-era values) apply at the integers; level k is in force from
# k − 0.5 to k + 0.5, and capacity is interpolated log-linearly between anchors.
# Origins: a forager cell starts farming at FARMING_ORIGIN_RATE × people × arable per year (the surplus
# view of Kavanagh et al. 2018) unless farmers live within its interaction sphere, and starts at early
# farming, FARMING_START_ENERGY. Calibrated so the Earth example gets about Larson et al.'s (2014) "at
# least 11" independent centres (see DESIGN.md 4.8).
FARMING_START_ENERGY = 0.5
# Early farming (FARMING_START_ENERGY) holds this share of the log step from foraging to farming capacity
# (0.5 would be the geometric mean of the two anchors), fitted to HYDE 3.2's 19 million people in 5000 BCE over
# two Earth worlds × two history seeds with the slowed farming front below (task 9.12; 0.5 gave 2.1 × HYDE, 0.37
# gave 1.17). Heuristic.
EARLY_FARMING_LOG_SHARE = 0.35
# The front of a level into forager land runs at its speed × the destination's arable component to this power:
# Silva & Steele (2014) found the European Neolithic front slowed to 0.1–0.2 of its speed in forest and from
# ~1.05 to ~0.53 km/yr northward; arable land stands in for the crop package's fit (heuristic).
FARMING_FRONT_ARABLE_EXPONENT = 1.0
FARMING_ORIGIN_RATE = 1e-9
# Knowledge (Kremer 1993 with Jones's damping): in farming cells energy rises at
# KNOWLEDGE_RATE (S/10⁶)^λ A^(φ−1) / Δln K, S the people within the interaction sphere, A the capacity gained since foraging, Δln K the log step
# between the anchors around the cell's level. φ = 0 keeps the feedback from running away (φ = 1, pure
# Kremer, is hyperbolic). Past INDUSTRIAL_FROM_ENERGY (the end of water_wind) energy rises only with
# coal or hydrocarbons of potential INDUSTRIAL_FUEL_MIN within the sphere, INDUSTRIAL_RATE_FACTOR times
# faster; it stops at HISTORY_MAX_ENERGY (electric in force). KNOWLEDGE_RATE was fitted to a median first
# fossil of ~215 years ago over six Earth worlds × history seeds 0–2 (0.022 after task 9.11). Task 9.12 lowered early
# farming's capacity and knowledge grows with the people in the sphere, so g is raised provisionally to keep the
# timing (two worlds × two seeds); the multi-seed refit follows the investigation of the timing spread.
KNOWLEDGE_RATE = 0.026
# The interaction sphere weighs people by exp(−d / L) within KNOWLEDGE_SPHERE_REACH lengths, L =
# KNOWLEDGE_SPHERE_KM × (speed factor of the cell's transport level)^KNOWLEDGE_SPHERE_SPEED_EXP: faster
# travel puts more people in contact (Kremer's regions without contact innovate apart).
KNOWLEDGE_SPHERE_KM = 150.0
KNOWLEDGE_SPHERE_REACH = 3.0
KNOWLEDGE_SPHERE_SPEED_EXP = 0.0
KNOWLEDGE_POPULATION_EXPONENT = 1.0
KNOWLEDGE_DAMPING_PHI = 0.0
INDUSTRIAL_FROM_ENERGY = 2.5
INDUSTRIAL_RATE_FACTOR = 15.0
INDUSTRIAL_FUELS = ("coal", "hydrocarbons")
INDUSTRIAL_FUEL_MIN = 0.5
HISTORY_MAX_ENERGY = 4.45
# Transport and communication of organic natives follow energy through Earth's pairing (energy, transport,
# communication), interpolated: foragers walk and talk, early farmers have boats, farming at its anchor
# goes with wheel and sail and writing (the milestone 8 default native), and later levels move together.
DEVELOPMENT_PAIRING = ((0.0, 0.0, 0.0), (0.5, 1.0, 0.0), (1.0, 2.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0),
                       (4.0, 4.0, 4.0), (5.0, 5.0, 5.0))
# Polities (task 9.5). Before states, occupied cells hold sub-cell polities of SUBPOLITY_FORAGING_KM2
# (bands and tribes) or, once farming is in force, SUBPOLITY_FARMING_KM2 (chiefdoms). A state forms in a
# cell without one whose communication has reached POLITY_FORM_COMMUNICATION (writing) and whose density
# is POLITY_FORM_DENSITY people/km², at POLITY_FORM_RATE per year. Border cells attack a neighbour without a state at
# POLITY_ATTACK_RATE per year with power 1 + β·c·S (Turchin et al. 2013; β = POLITY_POWER_BETA, S area in
# POLITY_UNIT_KM2, Turchin's 100 km cells, c cohesion), the attacker's weakened by exp(−d / reach), the reach
# interpolated log-linearly in the real-valued communication at its capital between POLITY_REACH_KM at the
# integer levels (task 9.13: Taagepera's largest state grew a hundredfold within the writing era); the
# defender adds POLITY_DEFENCE_GAMMA × defensibility. Cohesion changes per POLITY_PERIOD_YEARS by
# POLITY_COHESION_GAIN × frontier share − POLITY_COHESION_LOSS × ln(1 + S), within [POLITY_COHESION_MIN, 1].
# A polity breaks up (at the successions of its rulers) with a chance per period of POLITY_BREAKUP_BASE +
# POLITY_BREAKUP_SIZE·S·(r_writing/r)^POLITY_BREAKUP_REACH_EXP − POLITY_BREAKUP_COHESION·c, r the reach at its
# capital (faster communication holds larger states together). Values fitted
# to Taagepera's largest polities and effective numbers and Arbesman's lifetimes on Earth worlds (task 9.5);
# task 9.13 raised POLITY_FORM_DENSITY from 2 (first states ~7200 years ago) and lowered the reach of oral
# and writing from 1000 and 2500 km (largest state 32 × Taagepera's at 2800 BCE), fitted on two Earth worlds × two
# history seeds against Taagepera's largest states by date; with travel-time distance (follow-up) the density is 7,
# fitted on the six-world validation (8 left most worlds stateless at 2800 BCE, 6 gave 5 × Taagepera's).
SUBPOLITY_FORAGING_KM2 = 1330.0
SUBPOLITY_FARMING_KM2 = 1e4
POLITY_FORM_COMMUNICATION = 1
POLITY_FORM_DENSITY = 7.0
POLITY_FORM_RATE = 0.005
POLITY_ATTACK_RATE = 0.04
POLITY_UNIT_KM2 = 1e4
POLITY_POWER_BETA = 0.15
POLITY_DEFENCE_GAMMA = 2.0
POLITY_REACH_KM = (200.0, 1500.0, 5000.0, 10000.0, 20000.0, 20000.0)
# The distance d in the attacker's power is the travel time from its capital over the travel graph of its
# capital's transport level, counted in km at that level's flat-ground pace (Tobler on level ground times the
# level's speed factor; task 9.13 follow-up, great-circle distance before). Travel times are searched out to POLITY_TRAVEL_LIMIT_REACHES reaches
# and kept for at most POLITY_TRAVEL_CACHE polity-level pairs before ended polities are dropped.
POLITY_TRAVEL_LIMIT_REACHES = 5.0
POLITY_TRAVEL_CACHE = 4000
# Steppe empires (after the task 9.13 follow-up; Ben chose riding and confederations). Turchin (2009): over 90% of
# the 60-odd empires of at least 10⁶ km² (3000 BCE–1800 CE) arose in or next to the Old World's arid belt, from
# the interaction of steppe pastoralists and farmers; Currie et al. (2020) find distance from the steppe a leading
# predictor of imperial density, with horse-based warfare first developed in the steppe's pastoral societies.
# Steppe is Köppen BSk and BSh. States with their capital on the steppe ride STEPPE_RIDE_FACTOR times faster
# between steppe cells from transport STEPPE_RIDE_TRANSPORT (wheel_and_sail, which brings horses) on. In stateless
# occupied steppe cells next to a non-steppe state cell, a confederation forms at STEPPE_CONFEDERATION_RATE per year
# once the culture there has horses and energy STEPPE_CONFEDERATION_ENERGY, with cohesion
# STEPPE_CONFEDERATION_COHESION. Values heuristic, chosen on two steppe worlds × three seeds for peaks near the
# Mongols' 24 Mm² of 133 (riding ×5 alone gave 1.5 ×); confederations wait for water_wind in force (1.5), since
# from energy 1.25 they doubled the largest state of 500 BCE in the six-world validation.
STEPPE_KOPPEN = ("BSk", "BSh")
STEPPE_RIDE_FACTOR = 3.0
STEPPE_RIDE_TRANSPORT = 2
STEPPE_CONFEDERATION_RATE = 0.005
STEPPE_CONFEDERATION_ENERGY = 1.5
STEPPE_CONFEDERATION_COHESION = 1.0
POLITY_PERIOD_YEARS = 25.0
POLITY_COHESION_START = 0.5
POLITY_COHESION_GAIN = 1.0
POLITY_COHESION_LOSS = 0.05
POLITY_COHESION_MIN = 0.05
POLITY_BREAKUP_BASE = 0.05
POLITY_BREAKUP_SIZE = 0.0005
POLITY_BREAKUP_COHESION = 0.04
POLITY_BREAKUP_REACH_EXP = 1.5
# Rulers and provinces (task 9.6). Mean reigns without and with primogeniture (Kokkonen & Sundell 2012,
# European monarchs 1000–1800); primogeniture cuts the break-up hazard by PRIMOGENITURE_BREAKUP_CUT (their
# 75% lower hazard of deposition) and is adopted at PRIMOGENITURE_ADOPT_RATE per year once the capital has
# writing (heuristic). A state breaking up splits into provinces of PROVINCE_KM2 (heuristic).
RULER_REIGN_YEARS = 11.4
RULER_REIGN_PRIMOGENITURE_YEARS = 20.9
PRIMOGENITURE_BREAKUP_CUT = 0.75
PRIMOGENITURE_ADOPT_RATE = 0.0007
PROVINCE_KM2 = 4e5
# Wars between states (task 9.6, after Cederman 2003): onset rate per year at peace (his pAttack 0.01, fitted
# to Clauset's 2018 onset rate), the
# logistic threshold and slope of the power ratio needed to attack (his supThresh 3 and supSlope 20), the
# rate of leaving alert (his pDeactivate), the rate of ending a war (his pDropCampaign); the battle rate per
# front edge and year and the severity per front person and year are heuristic, as is the round length
# WAR_ROUND_YEARS (Cederman's periods are not dated). Alliances are heuristic.
WAR_ROUND_YEARS = 2.5
WAR_ONSET_RATE = 0.02
WAR_THRESHOLD = 3.0
WAR_THRESHOLD_SLOPE = 20.0
WAR_ALERT_DECAY = 0.1
WAR_END_RATE = 0.2
WAR_BATTLE_RATE = 0.3
WAR_DAMAGE = 1e-3
ALLIANCE_THREAT = 2.0
ALLIANCE_RATE = 0.02
ALLIANCE_BREAK_AT_SUCCESSION = 0.3
# Conflicts of at least WAR_CHRONICLE_STATES states enter the chronicle.
WAR_CHRONICLE_STATES = 12
# Colonists and natives (task 9.8). A colonist culture's front moves at the population front speed times the
# land speed factor of its transport level (LAND_SPEED_FACTOR; the US frontier crossed ~4.6 Mkm² in 1790–1890,
# Bazzi et al. 2017). Natives on a landmass colonists reach suffer contact epidemics: an extra death rate that
# removes EPIDEMIC_DEATH_SHARE of them over EPIDEMIC_YEARS (90% in a century in the Americas after 1492, Koch
# et al. 2019), starting at each cell when the epidemic, spreading from first contact at EPIDEMIC_SPEED_KM_YR
# (heuristic), arrives. Epidemics reaching EPIDEMIC_CHRONICLE_PEOPLE natives enter the chronicle.
# Connected cells of a state where another literate culture holds SECESSION_SHARE of the people, at least
# SECESSION_MIN_CELLS of them, break away as a state of that culture (heuristic; lets colonist regions rule
# themselves).
SECESSION_SHARE = 2.0 / 3.0
SECESSION_MIN_CELLS = 3
EPIDEMIC_DEATH_SHARE = 0.9
# Settler mortality (Acemoglu, Johnson & Robinson 2001, AER 91, 1369: European settlement in 1900 falls by 0.11 per
# unit of log settler mortality; malaria and yellow fever caused 80% of European deaths in the tropics; mortality
# per 1000 per year from 8.55 in Australia and New Zealand and 15 in the US to 2004 in Nigeria, via Albouy 2008,
# NBER w14130; Mali's 2940 reflects one yellow-fever epidemic). A cell's malaria suitability is a thermal response
# with its optimum at 25 °C and a steep fall above 28 °C (Mordecai et al. 2013; the 16 and 34 °C ends are
# heuristic) times a wetness ramp (heuristic); mortality runs log-linearly between SETTLER_MORTALITY_MIN and _MAX.
# Colonists die at the excess over SETTLER_MORTALITY_BASELINE (the US), a death rate taken from troop records.
SETTLER_MALARIA_T_C = (16.0, 25.0, 34.0)
SETTLER_MALARIA_WETNESS = (0.5, 1.0)
SETTLER_MORTALITY_MIN = 8.55
SETTLER_MORTALITY_MAX = 2004.0
SETTLER_MORTALITY_BASELINE = 15.0
# Overseas colonies (heuristic): a state whose capital has transport OVERSEAS_TRANSPORT (wheel_and_sail: ocean-going
# sail) and communication OVERSEAS_COMMUNICATION (print, as in Europe around 1500) founds a colony at OVERSEAS_RATE per year (a coast needed), on another landmass within OVERSEAS_REACH_KM of
# its coast by transport level, choosing by what the land holds for colonists after OVERSEAS_MORTALITY_HORIZON_YEARS
# of settler mortality. For scale, Western overseas dependencies rose from 9 in 1500 to ~90 by the mid-1600s (Strang
# 1991), enclaves and islands included; here only landmasses of OVERSEAS_MIN_LANDMASS_KM2 or more are colonised.
OVERSEAS_TRANSPORT = 2
OVERSEAS_COMMUNICATION = 2
OVERSEAS_RATE = 0.005
OVERSEAS_REACH_KM = (0.0, 0.0, 8000.0, 15000.0, 20000.0, 20000.0)
OVERSEAS_MORTALITY_HORIZON_YEARS = 25.0
OVERSEAS_MIN_LANDMASS_KM2 = 1e6
# Independence (Strang 1991, ISQ 35, 429, Table 1 model A: 142 events 1500–1987, Gompertz rate
# exp(g0 + γt + Σβx)). A tied colony breaks away at exp(DECOLONISATION_LOG_RATE + DECOLONISATION_TIME_SLOPE·t
# + DECOLONISATION_WITHIN_EMPIRE·n_in + DECOLONISATION_OUTSIDE_EMPIRE·n_out) per year: t the years since the first
# colony of the history, at most DECOLONISATION_TIME_MAX (Strang counts from 1500, European colonisation beginning
# in the 1490s, to 1987; the Gompertz term is not extrapolated past his range), n_in the earlier
# independences of colonies with the same line of home cultures, n_out the others, each at most
# DECOLONISATION_EVENTS_MAX (his 142 events). Strang found no dependence on the
# colony's age or size. His hegemony and economic-cycle terms have no counterpart here and are left out. Turmoil at
# home multiplies the rate by TURMOIL_WAR_FACTOR while the home state is at war (heuristic; the French invasion of
# Spain in 1808 cut Spanish revenue to a fifth and set off Spanish American independence, EHES WP 130; financial
# crises and defaults cluster with wars, Reinhart & Rogoff 2008, NBER w13882, so war stands for both); a colony
# whose home state ends is independent at once. A home off the planet is at war in ABSTRACT_HOME_WAR_SHARE of the
# years (heuristic) and lends HOME_POWER (the β·c·S of a state of 10⁶ km² at full cohesion); a home state on the
# planet lends its own power projected to the colony's capital. While tied, a home state sends
# COLONY_EMIGRATION_RATE of its people a year (heuristic).
DECOLONISATION_LOG_RATE = -11.61
DECOLONISATION_TIME_SLOPE = 0.012
DECOLONISATION_TIME_MAX = 487.0
DECOLONISATION_EVENTS_MAX = 142
DECOLONISATION_WITHIN_EMPIRE = 0.030
DECOLONISATION_OUTSIDE_EMPIRE = 0.014
TURMOIL_WAR_FACTOR = 5.0
ABSTRACT_HOME_WAR_SHARE = 0.2
HOME_POWER = 15.0
COLONY_EMIGRATION_RATE = 0.001
# Technology passes between cultures under rule: a culture's cell ruled by a state of another culture with a
# higher level in force there takes that level (at its bottom) at a rate 1 / TECH_ADOPTION_LAG_YEARS[level] per
# year (Ben chose rule only: on any contact the natives caught up and colonies stayed enclaves).
# Comin & Hobijn (2010): mean adoption lag 45 years over 15 technologies and 166 countries, 1820–2003; steam
# ships ~120, passenger railways ~97, telegraph ~46, electricity ~56, cars ~44, personal computers ~14 years.
# Levels 1–2 take the steam-ship lag (heuristic, no data before 1820). Natives take energy; their transport and
# communication follow it; colonists take each axis.
TECH_ADOPTION_LAG_YEARS = (float("inf"), 120.0, 120.0, 100.0, 50.0, 14.0)
# Language shift (Abrams & Strogatz 2003): in a cell where cultures X and Y both live, x = X's share of the two
# changes as dx/dt = c·(y·x^a·s − x·y^a·(1 − s)); a = 1.31 (their fit, ±0.25 across 42 regions); c = 0.086 per
# year, fitted to their Welsh (s = 0.43, 0.5 in 1900 to ~0.18 by 1990) and Quechua (s = 0.26, 0.7 in 1930 to ~0.25
# by 1990) curves read from their Fig. 1 (0.089 and 0.084). The fitted minority statuses were 0.26–0.46 (median
# 0.40); the culture at the higher energy level in the cell, or between equals the one ruling it, has status
# LANGUAGE_STATUS, and otherwise both 0.5 (heuristic).
# Supremacy (heuristic; Ben's choice: drawn per ruling, weighted by the gap): the chance that a culture rules the
# people of another as a people apart is SUPREMACY_BASE + SUPREMACY_PER_LEVEL × its lead in energy level.
# Bilingualism (Mira & Paredes 2005, their extension of Abrams & Strogatz): where a culture rules people of another
# without supremacy, of those who take up the other language a share BILINGUAL_EASE become bilingual and the rest
# switch; bilinguals return to one language at the rate at which the other language's monolinguals switch to it.
# They fitted 0.80 for Galician and Castilian (a = 1.50, c = 0.1, s = 0.26); here it applies to any pair
# (heuristic). The bilinguals of a pair are a new creole culture once they number CREOLE_MIN_PEOPLE (heuristic;
# Tria et al. 2015 find a creole emerges when mixed people are a large enough share of the population), which
# takes the higher level of its parents in each cell. CREOLE_CONTACT says which pairs can have bilinguals: "any",
# "once" (a culture joins at most one creole; creoles form none) or "colonial" (a colonist culture in the pair).
# On a two-native standard Earth (seed 0) these gave 120, 19 and 14 cultures and took 688, 168 and 142 s.
SUPREMACY_BASE = 0.1
SUPREMACY_PER_LEVEL = 0.3
BILINGUAL_EASE = 0.80
CREOLE_MIN_PEOPLE = 1e4
CREOLE_CONTACT = "once"
LANGUAGE_SHIFT_EXPONENT = 1.31
LANGUAGE_SHIFT_RATE = 0.086
LANGUAGE_STATUS = 0.60
EPIDEMIC_YEARS = 100.0
EPIDEMIC_SPEED_KM_YR = 500.0
# Livestock and crowd diseases (task 9.14; Ben chose a terrain-weighted draw). Wolfe, Dunavan & Diamond (2007): 8 of
# the 15 major temperate diseases probably came from domestic animals; crowd diseases need several hundred thousand
# people; 13 of the 14 major livestock species are Old World. A farming origin keeps herd animals with probability
# LIVESTOCK_P_CLOSED + (LIVESTOCK_P_OPEN − LIVESTOCK_P_CLOSED) × the share of LIVESTOCK_OPEN_BIOMES (open land where
# large grazers live) within LIVESTOCK_NEIGHBOURHOOD_KM (heuristic). Only colonists that carry herds' diseases cause
# contact epidemics, and only among people of cells without livestock. People struck do not grow for
# EPIDEMIC_NO_GROWTH_YEARS after the epidemic arrives (Ben chose a slower recovery; heuristic: Jones et al. 2021
# find decline continuing as long as contact was sustained, e.g. California's missions 1770–1834, without a rate).
LIVESTOCK_OPEN_BIOMES = ("TEMPERATE_GRASSLAND", "SAVANNA", "WOODLAND_SHRUBLAND")
LIVESTOCK_NEIGHBOURHOOD_KM = 500.0
LIVESTOCK_P_CLOSED = 0.1
LIVESTOCK_P_OPEN = 0.9
EPIDEMIC_NO_GROWTH_YEARS = 250.0
EPIDEMIC_CHRONICLE_PEOPLE = 1e5
# Planet events (task 9.9). Eruptions: VOLCANIC_EVENTS_PER_YEAR large stratospheric sulfur injections (eVolv2k:
# 106 from 500 BCE to 1900 CE, Toohey & Sigl 2017), times the planet's volcanic activity index. Their sulfur follows
# a Pareto law from VOLCANIC_SULFUR_MIN_TG with slope VOLCANIC_SULFUR_SLOPE, fitted to 10.6 events per century,
# 6 of at least Tambora's 28.1 Tg S and 2 of at least 59 Tg S in 2400 years (Wolff et al. 2023: larger than 1257
# about once in 2500 years); draws beyond VOLCANIC_SULFUR_MAX_TG, the largest on record, are set to it (larger
# eruptions lack cooling data). Summer cooling is VOLCANIC_COOLING_K at Tambora's loading (Sigl et al. 2015: 1.1 ± 0.6 K above
# Tambora, 0.6 ± 0.2 K over the 19 largest CE eruptions) and scales as the square root of the sulfur (heuristic),
# for VOLCANIC_YEARS, or VOLCANIC_LONG_YEARS above Tambora ("up to ten years"). An eruption within
# VOLCANIC_TROPICS_DEG of the equator cools both hemispheres, others their own (heuristic after the ~even
# Pinatubo split). Anomalies are counted in local standard deviations: SUMMER_TEMPERATURE_SD_K (the global mean of
# the local JJA standard deviation, 1951–80, Hansen et al. 2012) and RAIN_CV (heuristic).
VOLCANIC_EVENTS_PER_YEAR = 0.106
VOLCANIC_SULFUR_MIN_TG = 2.2
VOLCANIC_SULFUR_SLOPE = 1.47
VOLCANIC_SULFUR_MAX_TG = 60.0
VOLCANIC_COOLING_K = 1.1
VOLCANIC_COOLING_REF_TG = 28.1
VOLCANIC_YEARS = 5.0
VOLCANIC_LONG_YEARS = 10.0
VOLCANIC_TROPICS_DEG = 20.0
SUMMER_TEMPERATURE_SD_K = 0.5
RAIN_CV = 0.25
# Droughts (heuristic rates; Cook et al. 2022: megadroughts on every inhabited continent in the Common Era; the
# 4.2 ka event cut westerly rain 30–50% for 200–300 years, Weiss 2016): DROUGHT_RATE_PER_KM2_YR per km² of land
# (~15 a millennium on Earth), a disc of radius drawn log-uniformly in DROUGHT_RADIUS_KM, lasting a log-uniform
# draw in DROUGHT_YEARS, losing a uniform draw in DROUGHT_RAIN_LOSS of the rain. Land loses capacity by the rain
# lost times its water limitation, a ramp of precipitation over potential evaporation from 1 at
# DROUGHT_WATER_LIMIT[0] to 0 at [1] (heuristic). Cold costs CAPACITY_LOSS_PER_SIGMA of capacity per standard
# deviation (heuristic; a drought or heat disaster cuts a year's national cereal production ~10%, Lesk et al. 2016,
# taken as a 2σ event).
DROUGHT_RATE_PER_KM2_YR = 1e-10
DROUGHT_RADIUS_KM = (500.0, 2000.0)
DROUGHT_YEARS = (10.0, 300.0)
DROUGHT_RAIN_LOSS = (0.2, 0.5)
DROUGHT_WATER_LIMIT = (0.5, 1.5)
CAPACITY_LOSS_PER_SIGMA = 0.05
# Shocks and society: war onset and the break-up chance of states rise by CONFLICT_PER_SIGMA per standard deviation
# of the anomaly (Hsiang et al. 2013: +14% intergroup conflict per σ, median of 60 studies; applying it to
# break-up is heuristic). SHOCK_MIGRATION_SHARE of the people above a stressed cell's capacity move each step to
# less stressed neighbours (heuristic; habitat tracking to the rivers in the 4.2 ka drought, Weiss 2016).
CONFLICT_PER_SIGMA = 0.14
SHOCK_MIGRATION_SHARE = 0.5
# Pandemics (Marani et al. 2021: 395 epidemics 1600–1945, 38% above μ = 10⁻³ ‰ of the world's people a year;
# intensity above μ generalized Pareto with σ = 0.0113 ‰/yr and ξ = 1.40). Epidemics above μ arise at
# PANDEMIC_RATE per year scaled by the farming people over PANDEMIC_REFERENCE_PEOPLE (heuristic: the rate needs
# crowds and trade); only those between PANDEMIC_MIN_INTENSITY and PANDEMIC_MAX_INTENSITY are simulated (the law
# has ξ > 1, an infinite mean, and the data end at Spanish-flu intensity, 5.7 ‰/yr; the cap is heuristic). An epidemic starts in a farming cell
# drawn by people × density, reaches cells within PANDEMIC_YEARS of travel at EPIDEMIC_SPEED_KM_YR times the
# origin's land speed factor, and kills intensity × PANDEMIC_YEARS × the world's people among the reached, spread
# by a log-normal factor of width PANDEMIC_SPREAD_SD (Izdebski et al. 2022: Black Death mortality varied strongly by
# region) and at most PANDEMIC_MAX_DEATH_SHARE per cell (the highest Black Death estimate, 65%, Benedictow 2021).
# Duration, spread width and cap are heuristic. Pandemics killing PANDEMIC_CHRONICLE_PEOPLE enter the chronicle.
PANDEMIC_RATE = 0.435
PANDEMIC_GPD = (1e-6, 1.13e-5, 1.40)
PANDEMIC_MIN_INTENSITY = 1e-3
PANDEMIC_MAX_INTENSITY = 3e-2
PANDEMIC_REFERENCE_PEOPLE = 1e9
PANDEMIC_YEARS = 3.0
PANDEMIC_SPREAD_SD = 0.7
PANDEMIC_MAX_DEATH_SHARE = 0.65
PANDEMIC_CHRONICLE_PEOPLE = 1e6
# Impacts (Brown et al. 2002: log N(>E) = 0.5677 − 0.90 log E impacts a year, E in kt; Tunguska's 10 Mt every 1000 ± 200
# years): energies from IMPACT_MIN_MT to IMPACT_MAX_MT (the fit's range; larger impacts can be scripted). Land within
# the devastated area, IMPACT_AREA_KM2 at 10 Mt scaled as E^(2/3), loses its people (heuristic).
IMPACT_MIN_MT = 1.0
IMPACT_MAX_MT = 10.0
IMPACT_AREA_KM2 = 2000.0
# Knowledge loss (after Henrich 2004: average skill changes as −α + β(ε + ln N), so a skill decays where the pool of
# learners is below N* = e^(α/β − ε); Tasmania, ~4000 people isolated for ~10 kyr, lost bone tools, nets and
# fishing). A cell above farming loses energy at KNOWLEDGE_LOSS_RATE levels a year per e-fold its interaction
# sphere's people fall below KNOWLEDGE_LOSS_PEOPLE at its level (interpolated in log; heuristic above Tasmania's
# anchor), never below FARMING_START_ENERGY. Colonists lose on every axis; colonies tied to a home power draw on
# its learning and lose nothing (heuristic).
KNOWLEDGE_LOSS_PEOPLE = (4e3, 1e4, 1e5, 1e6, 1e7, 1e8)
KNOWLEDGE_LOSS_RATE = 0.002
EVENT_CHRONICLE_PEOPLE = 1e5
# Reading the chronicle (task 9.10): wars and break-ups of CHRONICLE_MAJOR_KM2 or more, wars of
# CHRONICLE_MAJOR_WAR_STATES states or more, and pandemics and impacts killing CHRONICLE_MAJOR_DEATHS or more are
# written out; smaller ones are counted. Periods are 1000 years long
# before 3000 years ago, 500 years to 1000 years ago and 100 years after (CHRONICLE_PERIOD_YEARS: (older than, length)).
# All heuristic, chosen to keep a Holocene chronicle to a few hundred lines.
CHRONICLE_MAJOR_KM2 = 1e7
CHRONICLE_MAJOR_WAR_STATES = 100
CHRONICLE_MAJOR_DEATHS = 1e7
CHRONICLE_PERIOD_YEARS = ((3000.0, 1000.0), (1000.0, 500.0), (0.0, 100.0))
# Settlement ages (task 9.7): the history records each cell's people every SETTLEMENT_RECORD_YEARS.
SETTLEMENT_RECORD_YEARS = 100.0
# The chronicle records each culture's first state and the rise, conquest and break-up of states of
# POLITY_CHRONICLE_KM2.
POLITY_CHRONICLE_KM2 = 1e6

# --- Tectonic subsidence (task 8.2) -------------------------------------------------

# Planets with rivers let continental crust subside where it was stretched (McKenzie 1978) and
# beside mountain belts (DeCelles & Giles 1996); the rivers fill the space with sediment.
CONTINENTAL_CRUST_M = 35e3            # unstretched crustal thickness for the fault-controlled subsidence
MARGIN_BETA = 3.0                     # stretch at the continent–ocean edge
MARGIN_STRETCH_WIDTH_KM = 150.0       # e-folding width of the stretch inland
MARGIN_STRETCH_REACH_KM = 600.0
MARGIN_MIN_STRETCH = 0.2              # β − 1 below which a point is not counted as stretched
MARGIN_ACTIVE_FRONT_MYR = 10.0        # margins above a subduction front this recent are not stretched
FLEXURAL_PARAMETER_KM = 100.0         # α; DeCelles & Giles give 26–150 km (97.6 km in their example)
FLEXURE_MIN_LOAD_M = 500.0            # topography above the continental base that counts as a load
FLEXURE_LOAD_RATIO = 1.0              # peak foredeep deflection per metre of adjacent load

# --- Biogenic sediment (task 8.2) ----------------------------------------------------

# Long-term accumulation of sediment the sea makes itself, on planets with sea life (strength
# ramps up over BIOGENIC_LIFE_GYR of biosphere age). Tropical platforms fill their space up to
# CARBONATE_PLATFORM_TOP_M below sea level where the annual mean temperature stands above the
# tropical factory's limit (coldest month ~20 °C, Schlager 2003) and the ground was within the lit
# zone before subsidence (production falls to ~5% of its maximum near 50 m for k = 0.1 m⁻¹,
# Bosscher & Schlager 1992). Other rates from Milliman (1993), at 1.2 t m⁻³.
BIOGENIC_LIFE_GYR = (0.2, 1.0)
CARBONATE_TROPICAL_ANNUAL_K = 295.15
CARBONATE_PHOTIC_DEPTH_M = 50.0
CARBONATE_PLATFORM_TOP_M = 5.0
SHELF_EDGE_DEPTH_M = 200.0
SLOPE_BASE_DEPTH_M = 3000.0
SHELF_CARBONATE_M_MYR = 13.0              # shelves: (0.1 + 0.3) Gt yr⁻¹ over 25.3 million km²
SLOPE_CARBONATE_M_MYR = 15.0              # slopes: 0.58 Gt yr⁻¹ over 32 million km²
# Pelagic rain, six times slower below the compensation depth, calibrated so the mean sediment on oceanic
# crust younger than 83 Myr matches the age–latitude law of GlobSed (Straume et al. 2019),
# Z = √τ (52 − 2.46 φ + 0.045 φ²) m, on three Earth seeds (together with the river sediment that reaches
# the deep sea). Milliman's deep-sea rate (3.8 g m⁻² yr⁻¹, ~3 m Myr⁻¹) would double the oceanic sediment.
CARBONATE_COMPENSATION_DEPTH_M = 4500.0
PELAGIC_ABOVE_CCD_M_MYR = 1.1
PELAGIC_BELOW_CCD_M_MYR = 0.18

# --- Culture: tectonic settings and resources (task 8.2) --------------------------

# Settings are read from distances to plate boundaries on the finished surface.
GEOLOGY_DISTANCE_LIMIT_KM = 2500.0
SAG_BASIN_MIN_SEDIMENT_M = 300.0          # continental interior with this much fill is a sag basin, else craton
PASSIVE_MARGIN_WIDTH_KM = 300.0           # coastal plain and shelf counted as passive margin
PASSIVE_MARGIN_MAX_ELEVATION_M = 200.0
STRIKE_SLIP_WIDTH_KM = 300.0
RIFT_WIDTH_KM = 400.0
FORELAND_WIDTH_KM = 800.0                 # foreland basins reach several hundred km from a collision front
SUBDUCTION_MARGIN_WIDTH_KM = 400.0
STABLE_CRUST_MYR = 500.0                  # no mountain building for this long: stable crust
RESOURCE_MIN_ELEVATION_M = -3000.0        # submerged continental crust above this depth can hold deposits

# Deep time: the first ocean is taken to condense this long after formation; with no
# history log, land life appears this long after the origin of life (the history-mode
# default land colonisation delay).
OCEAN_FORMED_GYR = 0.5
LAND_LIFE_DELAY_GYR = 3.4

# Ore belts. A deposit's belt value comes from the nearest belt cell of the right kind,
# preserved as exp(-age / preservation) and fading with distance from the belt.
BELT_WIDTH_KM = 150.0
OLD_BELT_PRESERVATION = 0.3               # belts older than the simulated window
PORPHYRY_PRESERVATION_MYR = 150.0         # shallow porphyry systems erode away quickly
ACTIVE_ARC_WIDTH_KM = 250.0
OROGENIC_GOLD_PRESERVATION_MYR = 1500.0   # mid-crustal: survives in old belts
GOLD_ACCRETIONARY_WEIGHT = 0.6
GOLD_ARC_WEIGHT = 0.3
GREENSTONE_GOLD = 0.4                     # gold on cratons of planets old enough to have greenstone belts
GREENSTONE_PLANET_AGE_GYR = (2.5, 3.5)
GRANITE_PRESERVATION_MYR = 800.0
GRANITE_BELT_WIDTH_KM = 250.0
TIN_ANDEAN_WEIGHT = 0.3
IRON_FORMATION_ANOXIC_GYR = (0.2, 1.0)    # liquid, anoxic ocean this long gives full iron-formation potential
OXIDISED_GYR = 0.5                        # time after oxygenation for oxidised brines and groundwater
LEAD_ZINC_SEDIMENT_M = (500.0, 3000.0)

# Hydrocarbons: burial into the oil window and a setting factor for heat and source rocks,
# calibrated so the potential on three Earth-example seeds shares out like the giant fields
# of Mann et al. (2003): passive margins 35%, rifts and sags 31%, forelands and accreted
# terranes 28%, strike-slip 6%, subduction margins <1%.
HYDROCARBON_BURIAL_M = (1000.0, 4000.0)
HYDROCARBON_LIFE_GYR = (0.2, 1.0)
# Fitted after subsidence and biogenic sediment (task 8.2); strike-slip (6%) cannot be reached because transforms
# rarely cross continents on the grid, so the other targets are renormalised over 95%. Rifts cover only 0–3% of
# continental crust, which is why their factor is high: it is a calibration, not a ranking of settings.
HYDROCARBON_SETTING_FACTOR = {"PASSIVE_MARGIN": 0.21, "SAG_BASIN": 0.5, "RIFT": 1.0, "FORELAND": 0.24,
                              "STRIKE_SLIP": 0.32, "SUBDUCTION_MARGIN": 0.005, "CRATON": 0.2, "OROGEN": 0.03,
                              "VOLCANIC": 0.0, "OCEAN": 0.0}

# Coal: wet subsiding basins after land plants appear.
COAL_SEDIMENT_M = (200.0, 2000.0)
COAL_WET_MM = (400.0, 1200.0)
COAL_WET_FLOOR = 0.3                      # the climate that buried the peat need not be today's
COAL_LAND_LIFE_GYR = (0.05, 0.3)
COAL_SETTING_FACTOR = {"FORELAND": 1.0, "SAG_BASIN": 0.8, "RIFT": 0.6, "PASSIVE_MARGIN": 0.6, "STRIKE_SLIP": 0.6,
                       "SUBDUCTION_MARGIN": 0.3, "CRATON": 0.2, "OROGEN": 0.2, "VOLCANIC": 0.0, "OCEAN": 0.0}

# Evaporites: arid ground (precipitation / potential evaporation below the ratio range).
EVAPORITE_ARIDITY_RATIO = (0.2, 0.6)
EVAPORITE_RIFT = 0.7
EVAPORITE_MARGIN = 0.4

# Laterites: tropical, humid, low and long stable ground.
LATERITE_TEMPERATURE_C = (18.0, 24.0)
LATERITE_PRECIPITATION_MM = (1000.0, 1800.0)
LATERITE_STABLE_MYR = 50.0

URANIUM_SEDIMENT_M = (200.0, 1500.0)
URANIUM_GRANITE = 0.4
OPHIOLITE_WEIGHT = 0.7
OPHIOLITE_PRESERVATION_MYR = 1000.0
OPHIOLITE_WIDTH_KM = 100.0
INTRAPLATE_MIN_ARC_DISTANCE_KM = 800.0
RIFT_ALKALINE_WIDTH_KM = 200.0
ION_CLAY_WEIGHT = 0.5

# Geothermal: heat near volcanoes, rifts and arcs, scaled by the planet's heat flow
# (GEOTHERMAL_BASE + (1 - GEOTHERMAL_BASE) × q/q⊕, capped at 1).
GEOTHERMAL_VOLCANIC_KM = 100.0
GEOTHERMAL_RIFT_KM = 150.0
GEOTHERMAL_ARC_KM = 200.0
GEOTHERMAL_ARC = 0.8
GEOTHERMAL_BASE = 0.3

# Placer gold: gold carried downstream, keeping exp(-distance / PLACER_TRANSPORT_KM).
PLACER_TRANSPORT_KM = 300.0
PLACER_MIN_DISCHARGE_M3_S = 10.0

# Metallogenic districts: metal classes are concentrated where a smooth noise field (fbm with
# features ~DISTRICT_SCALE_KM across, standard deviation ~0.2) rises through the range below.
DISTRICT_SCALE_KM = 400.0
DISTRICT_NOISE_RANGE = (0.05, 0.35)

# --- Culture: suitability and population density (task 8.3) -------------------------

# Carrying capacity per cell: ln K = Σ terms. Two sets are anchored on Earth data and levels in
# between are interpolated on the energy axis (weight = level / 4, capped at 1):
# - foragers: ln density rises 0.003 per g m⁻² yr⁻¹ of climatic NPP up to 1360, then levels off
#   (Tallavaara et al. 2018); the other terms are taken from the modern set;
# - electric (modern Earth): fitted on three Earth-spec worlds to the population shares below 100 m
#   (33.5%, Cohen & Small 1998), within 100/200/400 km of the coast (38/50/67%, Small & Cohen 2004),
#   the mode of the temperature niche (11–15 °C, Xu et al. 2020) and the density quantiles of
#   Cohen & Small (1998) relative to the mean (people: median 6.1×, 75th percentile 15×; land:
#   median 0.10×, 75th percentile 0.57×), on six seeds. Fusion keeps the modern set. A fit on three
#   seeds gave a stronger coast term (0.26 over 159 km) and weaker clustering (0.40) but missed the
#   coast shares on the other three; the terms trade off, so individual values are loosely constrained.
DENSITY_NPP_PLATEAU = 1360.0              # g m⁻² yr⁻¹
DENSITY_DISCHARGE_SCALE_M3S = 100.0
DENSITY_TERMS_FORAGING = {"log_npp": 0.0, "linear_npp": 0.003, "discharge": 0.05, "coast": 0.06,
                          "coast_km": 24.0, "elevation_m": 492.0, "heat": 0.52, "heat_from_c": 24.0,
                          "clustering": 0.0}
DENSITY_TERMS_MODERN = {"log_npp": 1.17, "linear_npp": 0.0, "discharge": 0.05, "coast": 0.06,
                        "coast_km": 24.0, "elevation_m": 492.0, "heat": 0.52, "heat_from_c": 24.0,
                        "clustering": 0.71}
# Colonists live in built habitats: no dependence on productivity or heat; they seek water, low
# flat ground, coasts and resources (mean potential of the metal and energy classes), and can settle
# ice at a cost. Heuristic.
DENSITY_TERMS_COLONIST = {"log_npp": 0.0, "linear_npp": 0.0, "discharge": 0.05, "coast": 0.06,
                          "coast_km": 24.0, "elevation_m": 492.0, "heat": 0.0, "heat_from_c": 24.0,
                          "clustering": 0.71, "resources": 2.0, "ice": -2.0, "habitat": True}
COLONIST_RESOURCES = ("copper", "gold", "iron", "hydrocarbons", "coal", "uranium", "rare_earths", "geothermal")
# People gather where others already live (market access): D = K (A / Ā)^clustering, with
# A = Σ D exp(−d / length) over cells within DENSITY_CLUSTER_REACH lengths.
DENSITY_CLUSTER_LENGTH_KM = 150.0
DENSITY_CLUSTER_REACH = 3.0
DENSITY_CLUSTER_ITERATIONS = 150
DENSITY_CLUSTER_TOLERANCE = 1e-5
# Default population of a native culture, per km² of productive land (Σ area × min(NPP, plateau) /
# plateau): Earth's population at the epoch of each energy level (HYDE 3.2, Klein Goldewijk et al.
# 2017: 10 000 BCE, 1 CE, 1700, 1900, 2015) over the productive land of the Earth example spec
# (7.7e7 km², three seeds). Fusion assumes one more step of the same size as fossil → electric.
EARTH_PRODUCTIVE_LAND_KM2 = 7.7e7
POPULATION_ANCHOR = {"foraging": 4.4e6, "farming": 2.32e8, "water_wind": 5.92e8, "fossil": 1.643e9,
                     "electric": 7.26e9, "fusion": 3.2e10}
# Homelands fade outside their radius over this share of the radius.
HOMELAND_EDGE = 0.25
HOMELAND_TAIL = 0.01                      # the fading edge ends where it falls below this share
# Colonies: founders grow exponentially until the land's capacity at their level (per km² of
# habitable land; Earth's ice-free land is 1.33e8 km², Cohen & Small 1998), within a reach that widens
# with age. Heuristic.
COLONY_FOUNDERS = 1e4
COLONY_GROWTH_PER_YEAR = 0.02
COLONY_DEFAULT_AGE_YEARS = 200.0
COLONY_REACH_KM = 500.0
COLONY_SPREAD_KM_PER_YEAR = 5.0
EARTH_HABITABLE_LAND_KM2 = 1.33e8

# --- Culture: settlements (task 8.4) -----------------------------------------------------

# Each culture's land is split into settlement regions (population-weighted k-means). A region's
# whole population lives in a rank-size (Zipf) hierarchy from SETTLEMENT_SMALLEST people up; its
# largest settlement follows from its population. Zipf holds within national systems on Earth
# (Rozenfeld et al. 2010: ζ = 0.97 ± 0.03 for US city clusters above 12 000, 1.07 ± 0.03 for
# Britain above 5000); Ortman et al. (2014) record settlements down to ~10 people.
ZIPF_DEFAULT = 1.0
SETTLEMENT_SMALLEST = 50.0
# Region area by transport level. Level 2 (wheel_and_sail) is set so the largest settlement on the
# Earth example at water_wind is ~1 million on average over six seeds (Beijing ~1 million in 1800,
# Reba et al. 2016; single seeds 0.6–1.7 million); each level multiplies the area by REGION_AREA_STEP
# (heuristic).
REGION_AREA_KM2_LEVEL2 = 6e5
REGION_AREA_STEP = 3.0
REGION_KMEANS_ITERATIONS = 15
# Central-place spacing: a settlement of rank r keeps SETTLEMENT_EXCLUSION × √(region area / r) from
# larger settlements of its region, where land allows.
SETTLEMENT_EXCLUSION = 0.5
# The global grid lists settlements down to the size at which there are about this many per occupied
# cell (never below SETTLEMENT_SMALLEST); smaller ones stay as each cell's dispersed population.
SETTLEMENTS_PER_CELL = 1.0
# Listed settlements sit within this share of the grid spacing of their cell's centre.
SETTLEMENT_JITTER = 0.3
# Settlement forms by population (heuristic; 5000 is the amorphous/networked split of Ortman et al. 2014).
FORM_LIMITS = {"hamlet": 100.0, "village": 5000.0, "town": 1e5, "city": 1e6}

# --- Culture: roads (task 8.5) -------------------------------------------------------------

# Travel time on the grid. On land, speed follows Tobler's hiking function, v = 6 e^(−3.5 |g + 0.05|)
# km/h (Goodchild 2020), times a factor per transport level; the slope g is the relief between cells
# plus a sub-grid slope by landform (heuristic), since 124 km cells hide the slopes a road climbs.
# Large rivers add a crossing time, ice slows travel. From pack_and_boat on, boats cross water at a
# speed per level, with a port time at each landing. Heuristic speeds.
LAND_SPEED_FACTOR = (1.0, 1.2, 1.6, 5.0, 12.0, 12.0)            # by transport level
WATER_SPEED_KMH = (0.0, 3.0, 5.0, 15.0, 30.0, 30.0)             # 0: no boats
PORT_HOURS = (0.0, 24.0, 12.0, 6.0, 3.0, 3.0)
SUBGRID_SLOPE = {"PLAIN": 0.01, "HIGHLAND": 0.06, "MOUNTAIN": 0.15, "VOLCANO": 0.15, "RIFT": 0.05,
                 "CRATER": 0.05, "VOLCANIC_PLAIN": 0.02}
SUBGRID_SLOPE_OTHER = 0.03
RIVER_CROSSING_M3S = 1000.0
RIVER_CROSSING_HOURS = (6.0, 4.0, 2.0, 0.5, 0.2, 0.2)
ICE_SPEED_SHARE = 0.2
# Links: every settlement cell asks for its ROAD_PARTNERS strongest gravity partners ((P_i P_j)^μ / d²,
# with d at least one grid spacing) among its Delaunay neighbours and theirs. Pairs are routed by
# least time in ROAD_WAVES waves, strongest first; after each wave the roads built so far cost
# ROAD_REUSE_DISCOUNT of their time, so later routes merge into trunks (densification, Strano et al.
# 2012). A route may take at most ROAD_MAX_DETOUR times the straight-line time at flat-ground speed
# (or boat speed, if slower).
ROAD_PARTNERS = 3
ROAD_GRAVITY_MASS_EXPONENT = 1.0          # μ in (P_i P_j)^μ / d²; 0.5 and 0.3 were tried (8.5 report)
ROAD_WAVES = 5
# Cells without a listed settlement join the road network as nodes, weighted by their people, once they
# hold ROAD_RURAL_MIN_PEOPLE (task 8.11), so main roads reach rural land as they do on Earth.
ROAD_RURAL_MIN_PEOPLE = 500.0
ROAD_REUSE_DISCOUNT = 0.6
# Gravity falls with distance as d^ROAD_DISTANCE_EXPONENT[transport level] (task 9.7): 1.9 for overland
# trade in the Bronze Age (Barjamovic et al. 2019) up to wheel_and_sail, 0.91 for modern trade (Disdier &
# Head 2008) from rail_and_steam on. Roads of an earlier kept epoch cost ROAD_OLD_DISCOUNT of their time
# when an organic history routes the next epoch (heuristic, the reuse discount), so networks persist.
ROAD_DISTANCE_EXPONENT = (1.9, 1.9, 1.9, 0.91, 0.91, 0.91)
ROAD_OLD_DISCOUNT = 0.6
ROAD_MAX_DETOUR = 4.0
ROAD_MAX_LINK_KM = 2500.0                  # longer Delaunay links (across empty oceans) are not asked for
# Road classes by carried flow: the classes a transport level has are filled from the busiest road
# down, each class holding about twice the length of the one above (Meijer et al. 2018: road length
# nearly doubles per class from highways to tertiary roads).
ROAD_CLASSES_BY_TRANSPORT = (("track",), ("track", "road"), ("track", "road", "trunk"),
                             ("track", "road", "trunk", "highway"), ("track", "road", "trunk", "highway"),
                             ("track", "road", "trunk", "highway"))
ROAD_CLASS_LENGTH_RATIO = 2.0

# --- Culture: polities and borders (task 8.6) -------------------------------------------------

# Territories follow XTENT (Renfrew & Level 1979), I = C^a − k d, on travel time: each cell goes to the
# capital with the highest influence, POLITY_INFLUENCE_KM × (P / P_largest settlement)^a converted to
# hours at the transport level's flat-ground speed, minus the travel time to the cell. The weight was
# scanned (150–10 000 km) on two Earth seeds against the largest polity of Taagepera (1978) at each
# communication level; 3000 km gives 0.65–0.89 of it (1000 km: 0.37–0.76; 10 000 km: 0.67–1.26). The capital
# size is taken relative to the culture's largest settlement, so the weight does not grow with the count.
POLITY_INFLUENCE_KM = 3000.0
POLITY_INFLUENCE_EXPONENT = 0.5
# Effective number of polities, N_A = A² / Σ A_i², for Earth's 133 Mm² of dry land by communication level
# (Taagepera 1997): pre-state fragmentation ~10⁵ (his lower estimate); ~600 around 500 BCE falling to
# 30 by the Mongol empire (writing: 300, interpolated to 1 CE); print 40 (above the "below 30 since
# 1800" of the telegraph era, heuristic); telegraph 15 (between 9 in 1925 and 30 around 1800); digital 24
# (1995); interplanetary 10 (heuristic). A culture's polity count is solved so the effective number of
# its own polities matches this, scaled by its occupied area over 133 Mm².
EARTH_DRY_LAND_KM2 = 1.33e8
# The count is searched upward by POLITY_COUNT_STEP until the effective number reaches its target; the
# search stops after POLITY_COUNT_STALL_STEPS steps that each raise it by less than POLITY_COUNT_GROWTH_MIN,
# keeping the smallest count within that share of the best (on worlds where settlements cluster, a few
# vast, thinly settled territories cap the effective number).
POLITY_COUNT_STEP = 1.3
POLITY_COUNT_STALL_STEPS = 3
POLITY_COUNT_GROWTH_MIN = 0.05
POLITY_EFFECTIVE_NUMBER = (1e5, 300.0, 40.0, 15.0, 24.0, 10.0)
# Polity types when the spec gives none (heuristic): by communication level, with empires and
# federations for the polities larger than POLITY_LARGE_KM2.
POLITY_LARGE_KM2 = 1e6

# --- Culture: names (task 8.7) -----------------------------------------------------------

# Each culture draws a sound system. Inventory sizes follow WALS (Maddieson 2013): consonants in 563
# languages (6–14: 89, 15–18: 122, 19–25: 201, 26–33: 94, 34+: 57, capped here at the offered set),
# vowel qualities in 564 (2–4: 93, 5–6: 287, 7–14: 184), syllable structure in 486 (simple (C)V: 61,
# moderately complex: 274, complex: 151). Every inventory holds the near-universal core; the rest is
# drawn with weight equal to each segment's share of the 3020 PHOIBLE 2.0 inventories (Moran et al.
# 2019), with spellings in Latin letters (first spelling not taken by a more frequent segment).
CONSONANT_CLASSES = (((6, 14), 89), ((15, 18), 122), ((19, 25), 201), ((26, 33), 94), ((34, 42), 57))
VOWEL_CLASSES = (((2, 4), 93), ((5, 6), 287), ((7, 14), 184))
SYLLABLE_CLASSES = (("simple", 61), ("moderate", 274), ("complex", 151))
CORE_CONSONANTS = ("m", "k", "p", "n", "t")
CORE_VOWELS = ("a", "i", "u")          # a two-vowel inventory keeps a and i
CONSONANTS = {   # segment: (PHOIBLE share, spellings)
    "m": (0.963, ("m",)), "k": (0.901, ("k", "c")), "j": (0.895, ("y", "j")), "p": (0.846, ("p",)),
    "w": (0.815, ("w", "u")), "n": (0.777, ("n",)), "t": (0.683, ("t",)), "l": (0.672, ("l",)),
    "s": (0.665, ("s",)), "ŋ": (0.623, ("ng", "ñ")), "b": (0.623, ("b",)), "ɡ": (0.553, ("g",)),
    "h": (0.549, ("h",)), "d": (0.448, ("d",)), "r": (0.433, ("r", "rr")), "f": (0.417, ("f", "ph")),
    "ɲ": (0.411, ("ny", "gn")), "t̠ʃ": (0.394, ("ch", "tch")), "ʔ": (0.358, ("'",)), "ʃ": (0.352, ("sh", "x")),
    "z": (0.285, ("z",)), "d̠ʒ": (0.265, ("j", "dj")), "v": (0.26, ("v",)), "ɾ": (0.251, ("r", "rr")),
    "ts": (0.219, ("ts", "tz")), "kʰ": (0.199, ("kh", "k'")), "pʰ": (0.193, ("ph", "p'")), "x": (0.182, ("kh", "x")),
    "ʒ": (0.148, ("zh", "j")), "ɣ": (0.141, ("gh",)), "tʰ": (0.133, ("th", "t'")), "c": (0.134, ("ky", "tj")),
    "kʷ": (0.123, ("kw", "qu")), "ɟ": (0.119, ("gy", "dj")), "mb": (0.103, ("mb",)), "β": (0.1, ("bh", "v")),
    "nd": (0.095, ("nd",)), "ŋɡ": (0.094, ("ngg", "ng")), "q": (0.082, ("q",)), "χ": (0.069, ("kh", "qh")),
    "ð": (0.05, ("dh",)), "ɬ": (0.048, ("hl", "lh")), "θ": (0.039, ("th",)),
}
VOWELS = {
    "i": (0.92, ("i",)), "u": (0.875, ("u",)), "a": (0.861, ("a",)), "e": (0.607, ("e",)), "o": (0.601, ("o",)),
    "ɛ": (0.37, ("e", "è")), "ɔ": (0.352, ("o", "ò")), "ə": (0.22, ("e", "ë")), "ɨ": (0.161, ("y", "ï")),
    "ɪ": (0.146, ("i", "ì")), "ʊ": (0.135, ("u", "ù")), "æ": (0.072, ("ae", "ä")), "ɑ": (0.074, ("a", "â")),
    "ɯ": (0.064, ("ü", "uu")), "y": (0.057, ("ü", "yu")), "ø": (0.031, ("ö", "oe")), "ʌ": (0.044, ("u", "ô")),
}
LONG_VOWEL_SHARE = 0.371                  # PHOIBLE inventories with any long vowel
CLUSTER_SECOND = ("l", "r", "ɾ", "w", "j")  # second member of the onset clusters of moderately complex syllables
# Consonants allowed in codas of moderately complex syllables (heuristic: nasals, liquids, s and plain stops);
# complex syllables allow any.
CODA_CONSONANTS = ("m", "n", "ŋ", "l", "r", "ɾ", "s", "p", "t", "k", "j", "w")
NAME_MIN_LETTERS = 3
# Syllable shape probabilities by class (heuristic): onset missing (word-initial, later), a two-consonant
# onset, a three-consonant onset, a coda, a second coda consonant.
SYLLABLE_SHAPE = {"simple": (0.3, 0.1, 0.0, 0.0, 0.0, 0.0), "moderate": (0.25, 0.08, 0.15, 0.0, 0.35, 0.0),
                  "complex": (0.2, 0.08, 0.25, 0.08, 0.5, 0.2)}
SYLLABLES_PER_NAME = {"simple": ((2, 3, 4), (0.5, 0.35, 0.15)), "moderate": ((1, 2, 3), (0.2, 0.55, 0.25)),
                      "complex": ((1, 2, 3), (0.35, 0.5, 0.15))}
# Place-name elements: each culture coins a word for each, attached as a suffix (share of cultures
# PLACE_ELEMENT_SUFFIX_SHARE) or prefix; a settlement whose cell has the feature takes the element with
# probability PLACE_ELEMENT_SHARE. Polities take the capital's name or a new word, with the "land" element
# on POLITY_LAND_ELEMENT_SHARE of them.
PLACE_ELEMENTS = ("river", "hill", "harbour", "lake", "field", "new", "great", "land")
PLACE_ELEMENT_SUFFIX_SHARE = 0.6
PLACE_ELEMENT_SHARE = 0.35
POLITY_NAMED_AFTER_CAPITAL = 0.5
POLITY_LAND_ELEMENT_SHARE = 0.5
NAME_MAX_LETTERS = 24

# --- Culture: habitats of colonists (task 8.8) ------------------------------------------------

# Each land cell gets the habitat colonists need there. Open: breathable air (O₂ partial pressure in
# OXYGEN_BREATHABLE_BAR), total pressure and temperature in the shirt-sleeve bands of the occupiability
# score, and enough air above to shield from cosmic rays. Dome: shielded but some other condition fails.
# Underground: too little air above (below the column of 2 m of rock, 560 g/cm², which Hassler et al.
# 2014 scale to 15 mSv/yr of dose equivalent against 232 mSv/yr on the Martian surface). None: annual
# mean temperature outside the survivable band. Pressure falls with elevation over the scale height.
HABITAT_OPEN_PRESSURE_BAR = (0.5, 3.0)
HABITAT_OPEN_TEMPERATURE_K = (263.0, 310.0)
HABITAT_SURVIVABLE_TEMPERATURE_K = (150.0, 400.0)
RADIATION_SHIELD_G_CM2 = 560.0
# Sealed habitats carry fewer people per unit of capacity (heuristic); lava tubes, 1–3 orders of
# magnitude more voluminous on Mars and the Moon than on Earth (Sauro et al. 2020), make volcanic ground
# at low gravity better for underground habitats.
HABITAT_CAPACITY = {"open": 1.0, "dome": 0.3, "underground": 0.2}
LAVA_TUBE_BONUS = 2.0
LAVA_TUBE_MAX_GRAVITY_EARTH = 0.5
# Colonist settlements below this population are outposts.
OUTPOST_MAX_POPULATION = 1000.0
# Landing sites (Golombek et al. 2012, MSL: latitude within ±30°, low elevation, low relief and slopes):
# away from the band or on rough landforms a site is drawn with these weights.
LANDING_MAX_LATITUDE_DEG = 30.0
LANDING_OFF_LATITUDE_WEIGHT = 0.2
LANDING_ROUGH_WEIGHT = 0.2
# Space elevator (Pearson 1975): feasible when the synchronous orbit lies inside the Hill sphere and the
# taper ratio for a material of characteristic height 2150 km on Earth (graphite whiskers, his design
# point, taper 10 on Earth) stays below ELEVATOR_MAX_TAPER (heuristic); the anchor must be within
# ELEVATOR_MAX_LATITUDE_DEG of the equator (heuristic).
ELEVATOR_CHARACTERISTIC_HEIGHT_KM = 2150.0
ELEVATOR_MAX_TAPER = 100.0
ELEVATOR_MAX_LATITUDE_DEG = 30.0

# --- Culture on zoomed maps (task 8.10) ---------------------------------------------------------

# Villages and hamlets: each global cell's people outside the listed settlements are split into
# settlements whose sizes follow a truncated power law (pdf exponent 1 + ζ, the rank-size law continued)
# from SETTLEMENT_SMALLEST up to the smallest listed settlement, placed at lattice nodes drawn by local
# suitability: e^(−slope / ZOOM_SLOPE_SCALE), times ZOOM_WATER_BONUS within ZOOM_WATER_KM of a river of
# order ZOOM_RIVER_ORDER or more, a lake or the sea (heuristic), and kept SETTLEMENT_EXCLUSION × √(area /
# count) from each other. Forager camps hold FORAGER_CAMP people (heuristic).
ZOOM_SLOPE_SCALE = 0.08
ZOOM_WATER_BONUS = 2.0
ZOOM_WATER_KM = 3.0
ZOOM_RIVER_ORDER = 2
FORAGER_CAMP = (25.0, 150.0)
# Settled area (ha) = a N^b, Ortman et al. (2014) for the Basin of Mexico: 0.265 N^0.652 below 5000
# people, 0.294 N^0.724 above. From electric energy on, towns of FOOTPRINT_SPLIT people or more take the
# density of Angel et al. (2011) for the year 2000: 60 people/ha overall in cities above 100 000 whose
# median population is 209 468 (their Table 3.3, world), rising 16% for each doubling of population
# (their regression on 3646 cities), D = URBAN_DENSITY_KM2 (N / URBAN_DENSITY_REF_POP)^URBAN_DENSITY_EXPONENT.
# That gives ~2700 people/km² at 5000 people and ~4400 at 49 000, against their 3000 for cities of
# 10 000–100 000 (median 48 940).
FOOTPRINT_SMALL = (0.265, 0.652)
FOOTPRINT_LARGE = (0.294, 0.724)
FOOTPRINT_SPLIT = 5000.0
URBAN_DENSITY_KM2 = 6000.0
URBAN_DENSITY_REF_POP = 209_468.0
URBAN_DENSITY_EXPONENT = 0.2141            # log2(1.16)
# Local roads: each settlement asks for ZOOM_ROAD_PARTNERS gravity partners among its ZOOM_ROAD_NEIGHBOURS
# nearest settlements; routing, waves and reuse follow the global roads.
ZOOM_ROAD_PARTNERS = 3
ZOOM_ROAD_NEIGHBOURS = 8
# Borders: nodes within ZOOM_BORDER_BAND_KM of a change of polity are reassigned to the polity whose
# interior reaches them first in travel time, where crossing a river of order ZOOM_BORDER_RIVER_ORDER or
# more costs ZOOM_BORDER_RIVER_HOURS (heuristic), so borders settle on ridges and rivers.
ZOOM_BORDER_BAND_KM = 60.0
ZOOM_BORDER_RIVER_ORDER = 5
ZOOM_BORDER_RIVER_HOURS = 24.0

# --- Export (milestone 10) ---
# 16-bit heightmaps of one world share one height scale: the world's elevation range widened by this share
# of it on each side, so zoomed detail and erosion, which add up to a few hundred metres above the global
# grid's peaks and below its deeps, fit inside it (heuristic; 0.31 m steps on the Earth example).
EXPORT_HEIGHT_MARGIN_FRACTION = 0.15
# Native lattices are georeferenced by ground control points at every tile corner, and at least this many
# along each side; a thin-plate-spline warp through them places a 5900 km continent block within 0.3 lattice
# steps (task 10.2 prototype; GDAL's default polynomial is off by up to 17 steps on such blocks).
EXPORT_GCP_MIN_PER_SIDE = 17
# Tile pyramid (task 10.3). Tiles of 129 nodes (2ⁿ + 1, 128 quads). The global part is built at
# PYRAMID_TOP_LEVEL (~9 km on Earth: one block per cube face) and coarser levels keep every other node.
# Each level is re-levelled onto its parent: the part of (level − parent) smoother than
# PYRAMID_RELEVEL_SIGMA parent spacings (a Gaussian) is removed, since zoom erosion lowers coarse lattices
# more (−84 m at ~9 km, −20 m at ~2.4 km on the Earth example). Where a finer level's coverage ends its
# heights ramp linearly to the parent's over PYRAMID_BLEND_FRACTION of a tile (as geometry clipmaps do).
PYRAMID_NODES_PER_TILE = 129
PYRAMID_TOP_LEVEL = 3
PYRAMID_RELEVEL_SIGMA = 0.5
PYRAMID_BLEND_FRACTION = 0.1
# Meshes (task 10.4): a tile's default vertical tolerance is this share of its node spacing (31 m at ~0.6 km,
# 490 m at ~9.8 km), measured on the exaggerated heights plus the chord's sag below the sphere (heuristic).
MESH_ERROR_FRACTION = 0.05
# Globe viewer (task 10.5): default height exaggeration of the 3D globe (relief is ~0.1% of the radius on Earth),
# and how far a draped region is raised above the globe, in metres before exaggeration, so it draws over the
# globe tiles beneath it (heuristic).
GLOBE_EXAGGERATION = 20.0
GLOBE_CLOUD_EXAGGERATION = 3.0          # cloud-top relief of giants is shading only
GLOBE_REGION_LIFT_M = 5.0

# --- System diagram and 3D system scene (task 11.9) ----------------------------
# Diagram: moon markers scale with log radius; colours by type (heuristic palette).
SYSTEM_MOON_MARKER_PT = (3.0, 14.0)         # marker size (points) for moons of 10 km and 3000 km radius
SYSTEM_MOON_COLOURS = {"rocky": "#a0785a", "icy": "#b9d3e8", "ocean": "#4f8fc9", "air": "#d9a441",
                       "highlight": "#e0402a"}
SYSTEM_ICY_WATER_FRACTION = 0.1             # drawn moons with more water than this are drawn icy
SYSTEM_LABEL_SPACING = 0.07                 # diagram moon labels closer than this fraction of the half-width are dropped
SYSTEM_RING_MIN_ALPHA = 0.06               # faintest drawn ring (dusty rings of τ ~ 1e-6 stay just visible)
# Scene: compressed scale keeps true distances out to past the rings, then logarithmic beyond,
# r' = r₀ (1 + g ln(r/r₀)), and inflates moon radii to at least a share of the planet's, ∝ r^(1/3).
SCENE_COMPRESS_FROM_RP = 2.5                # r₀, or 1.05 × the outermost ring if larger
SCENE_COMPRESS_GAIN = 1.0
SCENE_MOON_RADIUS_RP = 0.15                 # a 1000 km moon drawn this many planet radii across (compressed)
SCENE_RING_SEGMENTS = 720
SCENE_RING_RADIAL_STEPS = 16                # steps across each ring, so the planet's shadow edge is smooth
SCENE_AMBIENT = 0.03                        # light on the night side, so the shape stays visible
SCENE_MOON_SPHERE_RESOLUTION = 48
SCENE_CAMERA_DISTANCE = 3.4                # default camera distance in drawn system extents (30° view, 4:3 window; a moon at
                                            # greatest elongation stays in frame)
SCENE_MOON_EXAGGERATION = 3.0              # relief exaggeration of kept moon surfaces (small globes look spiky at 20×)
