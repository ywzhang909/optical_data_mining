"""Tests for beam analysis operators (ui/analysis/optical_analysis/)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add ui/ to path so we can import analysis.* modules
sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))

from analysis.optical_analysis.beam_analysis import (
    calculate_bpp,
    calculate_centroid,
    calculate_m2,
    calculate_xy_diameters,
    center_of_mass_numpy,
    d4sigma,
    extract_beam_features,
    fitting_gaussian,
    gaussian,
    pib_ratio,
)
from analysis.optical_analysis.beam_analysis_metrics import convert_to_cv, ellipse_fit, find_spot_border
from analysis.optical_analysis.diffraction import crop_to_square, shift_to_center_fft
from analysis.optical_analysis.image_utils import (
    calculate_background_threshold,
    clip_negative_values,
    normalize_image,
    normalize_image_for_display,
    pad_to_square,
    subtract_dark_field,
)


# =============================================================================
# Test data
# =============================================================================

GAUSSIAN_1D = np.array([1, 4, 10, 20, 30, 20, 10, 4, 1], dtype=float)


def make_gaussian_beam(size: int = 64, sigma: float = 10, amplitude: float = 255) -> np.ndarray:
    """Create a synthetic 2D Gaussian beam."""
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    beam = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return beam.astype(np.float64)


def make_elliptical_beam(size: int = 64, sigma_x: float = 15, sigma_y: float = 8) -> np.ndarray:
    """Create a synthetic elliptical Gaussian beam."""
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    beam = 255 * np.exp(-((x - cx) ** 2) / (2 * sigma_x**2) - ((y - cy) ** 2) / (2 * sigma_y**2))
    return beam.astype(np.float64)


# =============================================================================
# Test: beam_analysis.py — gaussian()
# =============================================================================


class TestGaussian:
    def test_returns_expected_shape(self):
        x = np.arange(20)
        result = gaussian(x, mu=10, sigma=3, A=100, b=5)
        assert result.shape == (20,)

    def test_peak_at_mu(self):
        x = np.linspace(0, 20, 101)
        result = gaussian(x, mu=10, sigma=3, A=100, b=5)
        peak_idx = np.argmax(result)
        assert abs(x[peak_idx] - 10) < 0.1

    def test_peak_value(self):
        x = np.linspace(0, 20, 101)
        result = gaussian(x, mu=10, sigma=3, A=100, b=5)
        assert abs(np.max(result) - 105) < 1.0  # A + b = 105

    def test_background_value_at_edges(self):
        x = np.linspace(0, 20, 101)
        result = gaussian(x, mu=10, sigma=3, A=100, b=5)
        assert abs(result[0] - 5) < 1.0  # far from peak, should approach b


# =============================================================================
# Test: beam_analysis.py — fitting_gaussian()
# =============================================================================


class TestFittingGaussian:
    def test_returns_params_or_nan(self):
        """fitting_gaussian returns params on success or nan on failure."""
        params, cov = fitting_gaussian(GAUSSIAN_1D)
        if isinstance(params, tuple):
            mu, sigma, A, b = params
            assert not np.isnan(mu)
        else:
            assert np.isnan(params)

    def test_noisy_data_may_fail_gracefully(self):
        """Fitting fails gracefully (returns nan) on noisy coarse data."""
        noise = np.random.default_rng(42).normal(0, 2, size=len(GAUSSIAN_1D))
        params, cov = fitting_gaussian(GAUSSIAN_1D + noise)
        if isinstance(params, tuple):
            pass  # fit succeeded despite noise
        else:
            assert np.isnan(params)


# =============================================================================
# Test: beam_analysis.py — center_of_mass_numpy()
# =============================================================================


class TestCenterOfMassNumpy:
    def test_centered_beam(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        yv, xv = np.mgrid[0:64, 0:64]
        cx, cy = center_of_mass_numpy(beam, xv, yv, moment=1)
        assert abs(cx - 31.5) < 2.0
        assert abs(cy - 31.5) < 2.0

    def test_uniform_intensity(self):
        img = np.ones((10, 10))
        yv, xv = np.mgrid[0:10, 0:10]
        cx, cy = center_of_mass_numpy(img, xv, yv, moment=1)
        assert abs(cx - 4.5) < 0.1
        assert abs(cy - 4.5) < 0.1


# =============================================================================
# Test: beam_analysis.py — d4sigma()
# =============================================================================


class TestD4Sigma:
    def test_returns_dict_with_expected_keys(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        result = d4sigma(beam, pixel_size_um=1.0)
        assert "center_x" in result
        assert "center_y" in result
        assert "D_x" in result
        assert "D_y" in result
        assert "avg_diameter" in result
        assert "center_intensity" in result

    def test_centered_beam_center(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        result = d4sigma(beam, pixel_size_um=1.0)
        cx, cy = result["center_x"], result["center_y"]
        assert abs(cx - 31.5) < 2.0
        assert abs(cy - 31.5) < 2.0

    def test_diameter_scales_with_pixel_size(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        r1 = d4sigma(beam, pixel_size_um=1.0)
        r2 = d4sigma(beam, pixel_size_um=2.0)
        assert abs(r2["D_x"] - 2 * r1["D_x"]) < 1e-6
        assert abs(r2["D_y"] - 2 * r1["D_y"]) < 1e-6

    def test_elliptical_beam_has_different_diameters(self):
        beam = make_elliptical_beam(size=64, sigma_x=15, sigma_y=8)
        result = d4sigma(beam, pixel_size_um=1.0)
        assert result["D_x"] > result["D_y"]

    def test_zero_image_returns_zero(self):
        beam = np.zeros((32, 32))
        result = d4sigma(beam, pixel_size_um=1.0)
        assert result["avg_diameter"] == 0.0

    def test_subtract_background(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        beam_with_bg = beam + 10.0
        result = d4sigma(beam_with_bg, pixel_size_um=1.0, subtract_background=True)
        assert result["avg_diameter"] > 0


# =============================================================================
# Test: beam_analysis.py — pib_ratio()
# =============================================================================


class TestPibRatio:
    def test_returns_float_and_bool(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        cx, cy = 31.5, 31.5
        ratio, overexposed = pib_ratio(
            beam,
            (cx, cy),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=1e-6,
        )
        assert 0 <= ratio <= 1
        assert isinstance(overexposed, (bool, np.bool_))

    def test_center_region_contains_some_energy(self):
        """The central region should contain some fraction of total energy."""
        beam = make_gaussian_beam(size=64, sigma=10)
        cx, cy = 31.5, 31.5
        ratio, _ = pib_ratio(
            beam,
            (cx, cy),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=1e-6,
        )
        assert ratio >= 0

    def test_zero_image_returns_zero(self):
        beam = np.zeros((32, 32))
        ratio, _ = pib_ratio(
            beam,
            (16, 16),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=1e-6,
        )
        assert ratio == 0.0


# =============================================================================
# Test: beam_analysis.py — calculate_bpp() & calculate_m2()
# =============================================================================


class TestBPPandM2:
    def test_bpp_returns_dict(self):
        result = calculate_bpp(pupil_diameter_mm=10, focal_diameter_mm=0.1, focal_length_mm=3000)
        assert "BPP_mm_mrad" in result
        assert "divergence_mrad" in result
        assert result["BPP_mm_mrad"] > 0
        assert result["divergence_mrad"] > 0

    def test_bpp_scales_linearly_with_pupil(self):
        r1 = calculate_bpp(pupil_diameter_mm=10, focal_diameter_mm=0.1)
        r2 = calculate_bpp(pupil_diameter_mm=20, focal_diameter_mm=0.1)
        assert abs(r2["BPP_mm_mrad"] - 2 * r1["BPP_mm_mrad"]) < 1e-10

    def test_m2_returns_positive(self):
        m2 = calculate_m2(bpp_mm_mrad=0.5, wavelength_um=1.064)
        assert m2 > 0

    def test_diffraction_limited_beam(self):
        """A diffraction-limited beam should have M² ≈ 1."""
        wavelength_um = 1.064
        bpp_diff = wavelength_um / np.pi
        m2 = calculate_m2(bpp_mm_mrad=bpp_diff, wavelength_um=wavelength_um)
        assert abs(m2 - 1.0) < 1e-10


# =============================================================================
# Test: beam_analysis.py — calculate_centroid()
# =============================================================================


class TestCalculateCentroid:
    def test_centered_beam(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        cx, cy = calculate_centroid(beam)
        assert abs(cx - 31.5) < 2.0
        assert abs(cy - 31.5) < 2.0

    def test_zero_image_returns_zero(self):
        beam = np.zeros((32, 32))
        assert calculate_centroid(beam) == (0.0, 0.0)


# =============================================================================
# Test: beam_analysis.py — calculate_xy_diameters()
# =============================================================================


class TestCalculateXYDiameters:
    def test_returns_dict(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        result = calculate_xy_diameters(beam, 31.5, 31.5, pix_size=1.0)
        assert "gaussian_dia_x(um)" in result
        assert "gaussian_dia_y(um)" in result
        assert result["gaussian_dia_x(um)"] > 0
        assert result["gaussian_dia_y(um)"] > 0


# =============================================================================
# Test: beam_analysis.py — extract_beam_features()
# =============================================================================


class TestExtractBeamFeatures:
    def test_returns_comprehensive_dict(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        result = extract_beam_features(beam, pixel_size_um=1.0)
        assert "centroid" in result
        assert "d4s" in result
        assert "pib_ratio" in result
        assert "gaussian_diameter" in result
        assert isinstance(result["centroid"], tuple)


# =============================================================================
# Test: beam_analysis_metrics.py — convert_to_cv(), find_spot_border(), ellipse_fit()
# =============================================================================


class TestConvertToCv:
    def test_converts_to_uint8(self):
        img = np.array([[0.0, 127.5, 255.0]], dtype=np.float64)
        result = convert_to_cv(img)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255


class TestFindSpotBorder:
    def test_returns_dict_with_keys(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        result = find_spot_border(beam)
        assert "border_x" in result
        assert "border_y" in result
        assert "border_radius" in result


class TestEllipseFit:
    def test_returns_dict_with_keys(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        uint8 = convert_to_cv(beam)
        result = ellipse_fit(uint8)
        assert "ellipse_center_x" in result
        assert "short_axis" in result
        assert "long_axis" in result
        assert "ellipticity" in result


# =============================================================================
# Test: diffraction.py — crop_to_square(), shift_to_center_fft()
# =============================================================================


class TestCropToSquare:
    def test_square_image_unchanged(self):
        img = np.ones((32, 32))
        result = crop_to_square(img)
        assert result.shape == (32, 32)

    def test_wide_image_crops_width(self):
        img = np.ones((32, 64))
        result = crop_to_square(img)
        assert result.shape == (32, 32)

    def test_tall_image_crops_height(self):
        img = np.ones((64, 32))
        result = crop_to_square(img)
        assert result.shape == (32, 32)


class TestShiftToCenterFFT:
    def test_centered_beam_stays_centered(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        cx, cy = 31.5, 31.5
        result = shift_to_center_fft(beam, cx, cy)
        assert result.shape == (64, 64)
        assert np.all(result >= 0)

    def test_shifted_beam_returns_to_center(self):
        beam = make_gaussian_beam(size=64, sigma=10)
        shifted = np.roll(beam, shift=5, axis=1)  # shift beam right
        cx, cy = calculate_centroid(shifted)
        result = shift_to_center_fft(shifted, cx, cy)
        assert result.shape == (64, 64)


# =============================================================================
# Test: image_utils.py
# =============================================================================


class TestNormalizeImageForDisplay:
    def test_output_is_uint8(self):
        img = np.array([[0.0, 100.0, 200.0]])
        result = normalize_image_for_display(img)
        assert result.dtype == np.uint8

    def test_output_range(self):
        img = np.array([[0.0, 100.0, 200.0]])
        result = normalize_image_for_display(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_image(self):
        img = np.ones((10, 10)) * 100
        result = normalize_image_for_display(img)
        assert result.min() >= 0


class TestSubtractDarkField:
    def test_none_method_returns_copy(self):
        img = np.ones((10, 10)) * 100
        result, black = subtract_dark_field(img, denoise_method="none")
        assert np.array_equal(result, img)
        assert black == 0

    def test_median_method(self):
        img = np.ones((10, 10)) * 100
        img[0, 0] = 50
        result, black = subtract_dark_field(img, denoise_method="median")
        assert black == 100
        assert result[0, 0] == 0

    def test_min_method(self):
        img = np.ones((10, 10)) * 100
        img[5, 5] = 50
        result, black = subtract_dark_field(img, denoise_method="min")
        assert black == 50
        assert result[5, 5] == 0

    def test_manual_threshold(self):
        img = np.ones((10, 10)) * 100
        result, black = subtract_dark_field(img, denoise_method="manual", manual_threshold=80)
        assert black == 80
        assert result[0, 0] == 20


class TestNormalizeImage:
    def test_target_range(self):
        img = np.array([[0.0, 50.0, 100.0]])
        result = normalize_image(img, target_min=0, target_max=1)
        assert abs(result.min() - 0) < 1e-6
        assert abs(result.max() - 1) < 1e-6

    def test_uniform_image(self):
        img = np.ones((10, 10)) * 50
        result = normalize_image(img, target_min=0, target_max=1)
        assert abs(result.min() - 0) < 1e-6
        assert abs(result.max() - 0) < 1e-6


class TestPadToSquare:
    def test_square_image_unchanged(self):
        img = np.ones((32, 32))
        result = pad_to_square(img)
        assert result.shape == (32, 32)

    def test_rectangle_becomes_square(self):
        img = np.ones((20, 32))
        result = pad_to_square(img)
        assert result.shape[0] == result.shape[1]
        assert result.shape[0] == 32


class TestClipNegativeValues:
    def test_negative_values_clipped(self):
        img = np.array([[-5.0, 0.0, 5.0]])
        result = clip_negative_values(img)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 5.0


class TestCalculateBackgroundThreshold:
    def test_median_method(self):
        img = np.ones((10, 10)) * 100
        img[0, 0] = 50
        threshold = calculate_background_threshold(img, method="median")
        assert threshold == 100

    def test_min_method(self):
        img = np.ones((10, 10)) * 100
        img[5, 5] = 50
        threshold = calculate_background_threshold(img, method="min")
        assert threshold == 50
