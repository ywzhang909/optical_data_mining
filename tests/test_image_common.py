"""Tests for image common utilities (ui/analysis/image/common.py)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ui"))

from analysis.image.common import (
    cartesian_to_polar,
    convert_to_cv,
    crop_to_square,
    fourier_shift_to_center,
    get_profiles,
    normalize_data,
    polar_to_cartesian,
    process_time_columns,
    read_tiff_to_numpy,
)


# =============================================================================
# Test: read_tiff_to_numpy()
# =============================================================================


class TestReadTiffToNumpy:
    def test_nonexistent_file_returns_none(self):
        result = read_tiff_to_numpy("/nonexistent/file.tiff")
        assert result is None

    def test_reads_png_as_fallback(self):
        """read_tiff_to_numpy uses PIL.Image.open, which reads any image type."""
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
            img.save(f.name)
            result = read_tiff_to_numpy(f.name)
            assert result is not None
            assert result.shape == (10, 10)


# =============================================================================
# Test: process_time_columns()
# =============================================================================


class TestProcessTimeColumns:
    def test_requires_path_column(self):
        """process_time_columns expects a 'path' column with Path objects."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            process_time_columns(df)

    def test_processes_path_column(self):
        """With a 'path' column, produces 'time' and 'info' columns."""
        df = pd.DataFrame({"path": [Path("20260326 17:18:52.73(高斯-全子束-弱光).TIFF")]})
        result = process_time_columns(df)
        assert "time" in result.columns
        assert "info" in result.columns


# =============================================================================
# Test: coordinate conversions
# =============================================================================


class TestCartesianToPolar:
    def test_origin(self):
        r, theta = cartesian_to_polar(0, 0, 0, 0)
        assert r == 0
        assert theta == 0

    def test_positive_x(self):
        r, theta = cartesian_to_polar(1, 0, 0, 0)
        assert abs(r - 1) < 1e-10
        assert abs(theta) < 1e-10  # 0 degrees

    def test_positive_y(self):
        r, theta = cartesian_to_polar(0, 1, 0, 0)
        assert abs(r - 1) < 1e-10
        assert abs(theta - np.pi / 2) < 1e-10

    def test_custom_origin(self):
        r, theta = cartesian_to_polar(5, 5, 0, 0)
        assert abs(r - np.sqrt(50)) < 1e-10


class TestPolarToCartesian:
    def test_origin(self):
        x, y = polar_to_cartesian(0, 0, 0, 0)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10

    def test_roundtrip(self):
        x0, y0 = 3.0, 4.0
        r, theta = cartesian_to_polar(x0, y0, 0, 0)
        x1, y1 = polar_to_cartesian(r, theta, 0, 0)
        assert abs(x1 - x0) < 1e-10
        assert abs(y1 - y0) < 1e-10

    def test_custom_origin(self):
        cx, cy = 10, 20
        x, y = polar_to_cartesian(5, 0, cx, cy)
        assert abs(x - (cx + 5)) < 1e-10
        assert abs(y - cy) < 1e-10


# =============================================================================
# Test: normalize_data()
# =============================================================================


class TestNormalizeData:
    def test_explicit_range(self):
        """normalize_data(data, min_val=0, max_val=100) scales to [0,1]."""
        data = np.array([0.0, 50.0, 100.0])
        result = normalize_data(data, min_val=0.0, max_val=100.0)
        assert abs(result.min() - 0) < 1e-6
        assert abs(result.max() - 1) < 1e-6

    def test_auto_range_with_none(self):
        """With None for both, uses actual data min/max."""
        data = np.array([10.0, 30.0, 50.0])
        result = normalize_data(data, min_val=None, max_val=None)
        assert abs(result.min() - 0) < 1e-6
        assert abs(result.max() - 1) < 1e-6

    def test_uniform_data_returns_original(self):
        data = np.ones(10) * 50
        result = normalize_data(data)
        np.testing.assert_array_equal(result, data)


# =============================================================================
# Test: convert_to_cv()
# =============================================================================


class TestConvertToCv:
    def test_returns_uint8(self):
        img = np.array([[0.0, 127.5, 255.0]], dtype=np.float64)
        result = convert_to_cv(img)
        assert result.dtype == np.uint8

    def test_preserves_shape(self):
        img = np.random.default_rng(42).uniform(0, 255, (20, 30))
        result = convert_to_cv(img)
        assert result.shape == (20, 30)


# =============================================================================
# Test: crop_to_square()
# =============================================================================


class TestCropToSquare:
    def test_square_unchanged(self):
        img = np.ones((32, 32))
        result = crop_to_square(img)
        assert result.shape == (32, 32)

    def test_wide_crops(self):
        img = np.ones((32, 64))
        result = crop_to_square(img)
        assert result.shape == (32, 32)

    def test_tall_crops(self):
        img = np.ones((64, 32))
        result = crop_to_square(img)
        assert result.shape == (32, 32)


# =============================================================================
# Test: get_profiles()
# =============================================================================


class TestGetProfiles:
    def test_returns_dict_with_vertical_horizontal(self):
        img = np.ones((32, 32))
        result = get_profiles(img, center=(16, 16), line_width=3)
        assert "vertical" in result
        assert "horizontal" in result

    def test_profile_lengths(self):
        img = np.ones((32, 32))
        result = get_profiles(img, center=(16, 16), line_width=3)
        assert len(result["vertical"]) == 32
        assert len(result["horizontal"]) == 32


# =============================================================================
# Test: fourier_shift_to_center()
# =============================================================================


class TestFourierShiftToCenter:
    def test_centered_image_unchanged(self):
        img = np.zeros((32, 32))
        img[14:18, 14:18] = 1
        result = fourier_shift_to_center(img, 16, 16)
        assert result.shape == (32, 32)

    def test_shifted_image(self):
        img = np.zeros((32, 32))
        img[14:18, 14:18] = 1
        shifted = np.roll(img, shift=3, axis=1)
        result = fourier_shift_to_center(shifted, 19, 16)
        assert result.shape == (32, 32)
