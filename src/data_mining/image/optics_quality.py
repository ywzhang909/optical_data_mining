"""Optical quality computations such as Strehl ratio and image recentering."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.interpolate import RegularGridInterpolator


def shift_to_center_fft(image: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Shift a beam image to the geometric center via Fourier shift."""
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    dx = w // 2 - cx
    dy = h // 2 - cy

    u = np.fft.fftfreq(w).reshape(1, -1)
    v = np.fft.fftfreq(h).reshape(-1, 1)
    phase = np.exp(-2j * np.pi * (u * dx + v * dy))

    shifted = np.real(ifft2(fft2(image) * phase))
    return np.clip(shifted, 0, None)


def calculate_strehl_ratio_with_energy_conservation(
    pupil_img: np.ndarray,
    focus_img: np.ndarray,
    pixel_size_pupil_um: float = 5.5,
    pixel_size_focus_um: float = 5.5,
    n_ratio: float = 1 / 32,
    f_mm: float = 3_000,
    wavelength_um: float = 1.064,
    roi_fraction: float = 0.8,
) -> tuple[float, np.ndarray]:
    """Compute Strehl ratio with ideal-focus energy matching."""
    dx_pupil_phys_um = pixel_size_pupil_um / n_ratio
    h, w = pupil_img.shape
    lx_pupil_um = w * dx_pupil_phys_um
    ly_pupil_um = h * dx_pupil_phys_um

    pupil_field = np.sqrt(np.maximum(pupil_img, 0)) + 0j
    ideal_focus_intensity = np.abs(fftshift(fft2(pupil_field))) ** 2

    f_um = f_mm * 1000.0
    dx_ideal_um = (wavelength_um * f_um) / lx_pupil_um
    dy_ideal_um = (wavelength_um * f_um) / ly_pupil_um

    x_ideal = np.linspace(-w * dx_ideal_um / 2, w * dx_ideal_um / 2 - dx_ideal_um, w)
    y_ideal = np.linspace(-h * dy_ideal_um / 2, h * dy_ideal_um / 2 - dy_ideal_um, h)

    hf, wf = focus_img.shape
    x_actual = (np.arange(wf) - wf // 2) * pixel_size_focus_um
    y_actual = (np.arange(hf) - hf // 2) * pixel_size_focus_um
    xa, ya = np.meshgrid(x_actual, y_actual)

    interp_func = RegularGridInterpolator(
        (y_ideal, x_ideal), ideal_focus_intensity, bounds_error=False, fill_value=0.0
    )
    ideal_on_actual = interp_func(np.stack([ya.ravel(), xa.ravel()], axis=-1)).reshape(hf, wf)

    h_roi = int(hf * roi_fraction)
    w_roi = int(wf * roi_fraction)
    y0 = (hf - h_roi) // 2
    x0 = (wf - w_roi) // 2
    actual_roi = focus_img[y0 : y0 + h_roi, x0 : x0 + w_roi]
    ideal_roi = ideal_on_actual[y0 : y0 + h_roi, x0 : x0 + w_roi]

    total_energy_ideal = np.sum(ideal_roi)
    if total_energy_ideal == 0:
        raise ValueError("Ideal focus energy is zero; check pupil image input.")

    scale = np.sum(actual_roi) / total_energy_ideal
    ideal_energy_matched = ideal_on_actual * scale
    peak_ideal = np.max(ideal_energy_matched[y0 : y0 + h_roi, x0 : x0 + w_roi])
    strehl = float(np.max(actual_roi) / (peak_ideal + 1e-12))
    return strehl, ideal_energy_matched
