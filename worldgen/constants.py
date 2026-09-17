"""Physical constants and reference values, all in SI units."""

# Fundamental constants
G = 6.674_30e-11            # gravitational constant [m^3 kg^-1 s^-2]
SIGMA_SB = 5.670_374e-8     # Stefan-Boltzmann constant [W m^-2 K^-4]
K_B = 1.380_649e-23         # Boltzmann constant [J K^-1]
M_H = 1.673_5e-27           # hydrogen atom mass [kg]
R_GAS = 8.314_462           # molar gas constant [J mol^-1 K^-1]

# Time
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 3.155_76e7   # Julian year
SECONDS_PER_GYR = SECONDS_PER_YEAR * 1e9

# Sun (IAU 2015 nominal values where available)
M_SUN = 1.988_47e30         # [kg]
R_SUN = 6.957e8             # [m]
L_SUN = 3.828e26            # [W]
T_SUN = 5772.0              # effective temperature [K]
AGE_SUN_GYR = 4.57

# Earth
M_EARTH = 5.972_2e24        # [kg]
R_EARTH = 6.371_0e6         # mean radius [m]
G_EARTH = G * M_EARTH / R_EARTH**2
AU = 1.495_978_707e11       # [m]
S_EARTH = L_SUN / (4 * 3.141_592_653_589_793 * AU**2)   # solar constant [W m^-2]
P_EARTH = 1.013_25e5        # mean surface pressure [Pa]
T_SURFACE_EARTH = 288.0     # global mean surface temperature [K]
EARTH_CMF = 0.325           # core mass fraction
EARTH_OCEAN_MASS = 1.4e21   # [kg]
EARTH_HEAT_FLUX = 0.087     # mean surface heat flux [W m^-2]

# Jupiter
M_JUPITER = 1.898_13e27     # [kg]
R_JUPITER = 6.991_1e7       # volumetric mean radius [m]

# Unit conversions
BAR = 1e5                   # [Pa]
KM = 1e3                    # [m]
