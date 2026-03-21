"""Beam metric utilities: moments, uniformity, encircled energy and BPP."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass


def d4sigma(img: np.ndarray, pixel_size_um: float = 1.0) -> dict[str, float]:
    """Compute D4σ beam diameters from first/second image moments."""
    total = float(np.sum(img))
    if total <= 0:
        raise ValueError("Input image intensity must be positive.")

    cy, cx = center_of_mass(img)
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]

    mu_xx = np.sum((x - cx) ** 2 * img) / total
    mu_yy = np.sum((y - cy) ** 2 * img) / total

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "D_x": float(4 * np.sqrt(max(mu_xx, 0.0)) * pixel_size_um),
        "D_y": float(4 * np.sqrt(max(mu_yy, 0.0)) * pixel_size_um),
        "center_intensity": float(img[int(cy), int(cx)]),
    }


def uniformity(img: np.ndarray, center: tuple[float, float]) -> dict[str, float]:
    """Calculate 4-quadrant RMS uniformity around a beam center."""
    cx, cy = int(center[0]), int(center[1])
    q1 = img[:cy, :cx].sum()
    q2 = img[:cy, cx:].sum()
    q3 = img[cy:, cx:].sum()
    q4 = img[cy:, :cx].sum()
    q_array = np.asarray([q1, q2, q3, q4], dtype=float)
    rms = float(np.mean(np.sqrt((q_array - np.mean(q_array)) ** 2)))
    return {"rms_4_quadrant": rms}


def radius(intensity: np.ndarray, center: tuple[float, float], energy: float = 0.99) -> dict[str, float]:
    """Compute encircled-energy radius for a target energy fraction."""
    if not (0 < energy <= 1):
        raise ValueError("energy must be in (0, 1].")

    y, x = np.indices(intensity.shape)
    x0, y0 = center
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    i_sorted = intensity.ravel()[order]
    cumsum = np.cumsum(i_sorted)
    target = energy * cumsum[-1]
    idx = np.searchsorted(cumsum, target)
    return {"power99_radius": float(r_sorted[min(idx, len(r_sorted) - 1)])}


def calculate_bpp_from_pupil_and_focal(
    pupil_diameter_mm: float,
    focal_diameter_mm: float,
    focal_length_mm: float = 3e3,
) -> dict[str, float]:
    """Calculate beam parameter product (BPP) in mm·mrad."""
    w_pupil = pupil_diameter_mm / 2.0
    w_focal = focal_diameter_mm / 2.0
    theta_mrad = (w_focal / focal_length_mm) * 1000.0
    return {"BPP_mm_mrad": float(w_pupil * theta_mrad)}
