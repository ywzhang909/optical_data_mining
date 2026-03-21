import numpy as np

from data_mining.image.beam_metrics import calculate_bpp_from_pupil_and_focal, d4sigma, radius, uniformity
from data_mining.image.optics_quality import shift_to_center_fft


def _gaussian_image(size: int = 64, sigma: float = 8.0, offset=(0.0, 0.0)) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    cx = (size - 1) / 2 + offset[0]
    cy = (size - 1) / 2 + offset[1]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))


def test_d4sigma_returns_positive_diameters():
    img = _gaussian_image()
    result = d4sigma(img)
    assert result["D_x"] > 0
    assert result["D_y"] > 0


def test_shift_to_center_fft_moves_peak_toward_center():
    img = _gaussian_image(offset=(8, -6))
    shifted = shift_to_center_fft(img, cx=40, cy=26)
    peak_y, peak_x = np.unravel_index(np.argmax(shifted), shifted.shape)
    assert abs(peak_x - shifted.shape[1] // 2) <= 1
    assert abs(peak_y - shifted.shape[0] // 2) <= 1


def test_uniformity_outputs_nonnegative_rms():
    img = _gaussian_image()
    out = uniformity(img, center=(32, 32))
    assert out["rms_4_quadrant"] >= 0


def test_radius_and_bpp_return_positive_values():
    img = _gaussian_image()
    radius_out = radius(img, center=(32, 32), energy=0.9)
    bpp_out = calculate_bpp_from_pupil_and_focal(2.0, 0.2)
    assert radius_out["power99_radius"] > 0
    assert bpp_out["BPP_mm_mrad"] > 0
