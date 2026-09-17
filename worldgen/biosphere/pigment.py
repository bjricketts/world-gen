"""Colour of photosynthetic life under its star's light.

Pigments tend to absorb most strongly near the wavelength where the
surface photon flux peaks (Kiang et al. 2007); on Earth that is ~680 nm
for chlorophyll, with a second band in the blue. Beyond the absorption
band leaves reflect strongly (the red edge). Around stars whose photon
flux peaks in the infrared, pigments are expected to absorb across the
visible range, so vegetation looks dark.

The reflectance is a simple model with those features. Its colour is the
reflected starlight converted to sRGB, with the eye adapted to the
starlight itself (so a white surface looks white). Colour matching uses the
multi-lobe Gaussian fit of the CIE 1931 functions by Wyman, Sloan & Shirley
(2013).
"""

from __future__ import annotations

import numpy as np

from .. import heuristics as h

WAVELENGTHS_NM = np.arange(380.0, 781.0, 5.0)
WIEN_PHOTON_NM_K = 3.670e6          # photon-flux peak: λ T = 3.67 mm K
_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def _lobe(x: np.ndarray, mu: float, left: float, right: float) -> np.ndarray:
    """Return a Gaussian with different widths either side of its peak."""
    sigma = np.where(x < mu, left, right)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def colour_matching(nm: np.ndarray) -> np.ndarray:
    """Return the CIE 1931 colour-matching functions (x̄, ȳ, z̄) at the given wavelengths (Wyman et al. 2013)."""
    x = 1.056 * _lobe(nm, 599.8, 37.9, 31.0) + 0.362 * _lobe(nm, 442.0, 16.0, 26.7) - 0.065 * _lobe(nm, 501.1, 20.4, 26.2)
    y = 0.821 * _lobe(nm, 568.8, 46.9, 40.5) + 0.286 * _lobe(nm, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(nm, 437.0, 11.8, 36.0) + 0.681 * _lobe(nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z])


def blackbody(nm: np.ndarray, temperature_k: float) -> np.ndarray:
    """Return the relative spectral radiance of a black body."""
    lam = nm * 1e-9
    return 1.0 / (lam**5 * (np.exp(1.4388e-2 / (lam * temperature_k)) - 1.0))


def absorption_peak(star_temperature_k: float) -> float:
    """Return the expected pigment absorption peak (nm): the surface photon-flux peak, shifted red by the air."""
    return h.PIGMENT_SURFACE_SHIFT * WIEN_PHOTON_NM_K / star_temperature_k


def reflectance(nm: np.ndarray, peak_nm: float) -> np.ndarray:
    """Return a model leaf reflectance with absorption at ``peak_nm`` and in the blue, and a red edge beyond."""
    main = np.exp(-0.5 * ((nm - peak_nm) / h.PIGMENT_BAND_WIDTH_NM) ** 2)
    blue = np.exp(-0.5 * ((nm - h.PIGMENT_ACCESSORY_NM) / h.PIGMENT_BAND_WIDTH_NM) ** 2)
    absorbed = np.maximum(main, blue)
    # Pigments tuned to infrared light also soak up the visible range below their peak.
    broad = np.clip((peak_nm - h.PIGMENT_BROAD_FROM_NM) / h.PIGMENT_BROAD_WIDTH_NM, 0.0, 1.0)
    absorbed = np.where(nm < peak_nm, np.maximum(absorbed, broad), absorbed)
    reflect = h.LEAF_REFLECTANCE_MIN + (h.LEAF_REFLECTANCE_GREEN - h.LEAF_REFLECTANCE_MIN) * (1.0 - absorbed)
    edge = 1.0 / (1.0 + np.exp(-(nm - peak_nm - h.PIGMENT_EDGE_OFFSET_NM) / 8.0))
    return reflect + (h.LEAF_REFLECTANCE_EDGE - reflect) * edge


def pigment_colour(peak_nm: float, star_temperature_k: float) -> tuple[str, float]:
    """Return the vegetation colour (hex sRGB) under the star's light and its visible albedo."""
    light = blackbody(WAVELENGTHS_NM, star_temperature_k)
    cmf = colour_matching(WAVELENGTHS_NM)
    white = cmf @ light
    leaf = cmf @ (light * reflectance(WAVELENGTHS_NM, peak_nm))
    # Adapt to the starlight: scale each channel so a perfect reflector is white.
    rgb_white = _XYZ_TO_RGB @ (white / white[1])
    rgb = (_XYZ_TO_RGB @ (leaf / white[1])) / rgb_white
    rgb = np.clip(rgb, 0.0, None)
    brightness = float(leaf[1] / white[1])
    encoded = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    encoded = np.clip(encoded, 0.0, 1.0)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in encoded), brightness
