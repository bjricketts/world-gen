"""Build worldgen/data/giant_optics.npz: the optical tables of the giant-colour model.

Reads the methane absorption of Karkoschka (1994) from the data folder (default
``giant_colours/``) and computes the Mie properties of the aerosol types with
miepython, which is needed only here:

  python scripts/build_giant_optics.py [--data giant_colours]

Sources
- CH4 absorption cross-sections, 406–1000 nm: Karkoschka (1994).
- Haze imaginary indices: Irwin et al. (2022, arXiv:2201.04516), Table 3, Uranus "Mean 1" (Aerosol-3, the extended small haze; Aerosol-2, the
  haze at the methane condensation level); real index 1.4.
- Chromophore imaginary index: Braude et al. (2020), Table 3
  (r = 0.05 µm, real index 1.4).
- Cloud particles: real index 1.42, non-absorbing (as used by Braude et al. 2020).
The particle radii and the dilution of the chromophore in Saturn's tropospheric
haze are the calibration (heuristic).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WL_NM = np.arange(380.0, 1001.0, 5.0)
NORM_NM = 800.0

IRWIN_NM = np.array([300, 400, 500, 600, 700, 800, 900, 1000.0])
IRWIN_A3 = np.array([3.07e-3, 1.50e-3, 1.91e-4, 1.70e-4, 1.38e-3, 7.74e-3, 7.87e-3, 1.52e-3])
IRWIN_A2 = np.array([1.10e-2, 1.61e-3, 1.35e-3, 9.65e-4, 1.37e-3, 4.89e-3, 4.24e-3, 1.13e-2])
BRAUDE_NM = np.array([450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950.0])
BRAUDE_K = np.array([0.033, 0.023, 0.016, 0.0107, 0.0077, 0.0081, 0.0078, 0.0065, 0.0091, 0.0083, 0.007])

# name: (median radius µm, real index, imaginary index on WL_NM)
LOGNORMAL_WIDTH = 0.3


def k_on_grid(wl: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Interpolate an imaginary index in log–log, extending the first interval's slope blueward."""
    out = np.exp(np.interp(WL_NM, wl, np.log(k)))
    slope = np.log(k[1] / k[0]) / np.log(wl[1] / wl[0])
    return np.where(WL_NM < wl[0], k[0] * (WL_NM / wl[0]) ** slope, out)


def particles() -> dict[str, tuple[float, float, np.ndarray]]:
    """Return the aerosol types: median radius (µm), real index and imaginary index on the grid."""
    braude = k_on_grid(BRAUDE_NM, BRAUDE_K)
    return {
        "haze_upper": (0.05, 1.4, k_on_grid(IRWIN_NM, IRWIN_A3)),
        "haze_methane": (0.6, 1.4, k_on_grid(IRWIN_NM, IRWIN_A2)),
        "haze_troposphere": (1.0, 1.4, 0.03 * braude),
        "chromophore": (0.05, 1.4, braude),
        "cloud": (3.0, 1.42, np.full(len(WL_NM), 1e-6)),
    }


def mie_average(radius_um: float, n_real: float, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the extinction cross-section (normalised at 800 nm), single-scattering albedo and asymmetry
    of a lognormal size distribution."""
    import miepython

    z = np.linspace(-2, 2, 15)
    weights = np.exp(-0.5 * z**2)
    weights /= weights.sum()
    ext, sca, gsum = (np.zeros(len(WL_NM)) for _ in range(3))
    for r, wt in zip(radius_um * np.exp(z * LOGNORMAL_WIDTH), weights):
        x = 2 * np.pi * r * 1e3 / WL_NM
        qe, qs, _, g = miepython.efficiencies_mx(n_real - 1j * k, x)
        area = np.pi * (r * 1e-6) ** 2
        ext += wt * qe * area
        sca += wt * qs * area
        gsum += wt * qs * area * g
    i = int(np.argmin(abs(WL_NM - NORM_NM)))
    return ext / ext[i], sca / ext, gsum / sca


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("giant_colours"))
    default_out = Path(__file__).parents[1] / "worldgen" / "data" / "giant_optics.npz"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    table = np.loadtxt(args.data / "karkoschka(1994).txt")
    fine = np.arange(WL_NM[0] - 2.0, WL_NM[-1] + 3.0, 1.0)             # 1 nm, averaged over each 5 nm bin
    on_fine = np.interp(fine, table[:, 0], table[:, 1], left=0.0, right=table[-1, 1])
    sig_ch4 = on_fine.reshape(len(WL_NM), 5).mean(axis=1) * 1e-4                       # cm² → m²
    arrays = {"wavelength_nm": WL_NM, "ch4_cross_section_m2": sig_ch4}
    for name, (r, n, k) in particles().items():
        ext, omega, g = mie_average(r, n, k)
        arrays[f"{name}_extinction"], arrays[f"{name}_albedo"], arrays[f"{name}_asymmetry"] = ext, omega, g
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
