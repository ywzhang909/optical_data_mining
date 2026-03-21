#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the image processing functions in process.py
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

# Patch the FunctionRegistry decorator to avoid import issues
with patch('data_mining.image.common.FunctionRegistry'):
    try:
        from data_mining.image.process import (
            find_centroid,
            calculate_beam_width,
            fit_hyperbola,
            calculate_m2_factor,
            calculate_strehl_ratio,
            encircled_energy,
            calculate_ellipticity,
            calculate_symmetry,
            calculate_uniformity
        )
    except ImportError:
        # Try alternative import path
        from image.process import (
            find_centroid,
            calculate_beam_width,
            fit_hyperbola,
            calculate_m2_factor,
            calculate_strehl_ratio,
            encircled_energy,
            calculate_ellipticity,
            calculate_symmetry,
            calculate_uniformity
        )

# Import SpotImage normally
try:
    from image.common import SpotImage
except ImportError:
    from data_mining.image.common import SpotImage


@pytest.fixture
def sample_spot_image():
    """Create a sample SpotImage for testing."""
    # Create a simple 2D Gaussian-like image for testing
    size = 50
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    xx, yy = np.meshgrid(x, y)
    
    # Create a Gaussian-like intensity distribution
    image_array = np.exp(-(xx**2 + yy**2) / 2)
    
    # Create SpotImage instance
    spot_image = SpotImage(image_array, "test_image")
    return spot_image


@pytest.fixture
def sample_linear_data():
    """Create sample linear data for hyperbola fitting."""
    z_positions = [0, 10, 20, 30, 40, 50]
    # Create some quadratic data: d^2 = 10 + 0.1*z + 0.01*z^2
    d_sq_values = [10 + 0.1*z + 0.01*z**2 for z in z_positions]
    return z_positions, d_sq_values


def test_find_centroid(sample_spot_image):
    """Test the find_centroid function."""
    image_data, (x_centroid, y_centroid) = find_centroid(sample_spot_image)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # For a symmetric Gaussian-like image, centroid should be near the center
    size = sample_spot_image.image.shape[0]
    expected_center = (size - 1) / 2
    assert abs(x_centroid - expected_center) < 2
    assert abs(y_centroid - expected_center) < 2


def test_calculate_beam_width(sample_spot_image):
    """Test the calculate_beam_width function."""
    image_data, result = calculate_beam_width(sample_spot_image)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # Check that all expected keys are in the result
    expected_keys = [
        'total_power', 'centroid', 'sigma_x_sq', 'sigma_y_sq', 
        'sigma_xy_sq', 'beam_width_x', 'beam_width_y'
    ]
    for key in expected_keys:
        assert key in result
    
    # Check that beam widths are positive
    assert result['beam_width_x'] > 0
    assert result['beam_width_y'] > 0
    
    # Check that total power is positive
    assert result['total_power'] > 0


def test_fit_hyperbola(sample_linear_data):
    """Test the fit_hyperbola function."""
    z_positions, d_sq_values = sample_linear_data
    result = fit_hyperbola(z_positions, d_sq_values)
    
    # Check that all expected keys are in the result
    expected_keys = [
        'fitted_coefficients', 'waist_position', 'waist_diameter', 
        'divergence_angle', 'fit_covariance'
    ]
    for key in expected_keys:
        assert key in result
    
    # Check that fitted coefficients are present
    coeffs = result['fitted_coefficients']
    assert 'A' in coeffs
    assert 'B' in coeffs
    assert 'C' in coeffs
    
    # Check that numerical values are reasonable
    assert isinstance(result['waist_position'], (int, float))
    assert result['waist_diameter'] >= 0
    assert result['divergence_angle'] >= 0


def test_calculate_m2_factor():
    """Test the calculate_m2_factor function."""
    # Test with typical values for a HeNe laser
    wavelength = 632.8e-9  # 632.8 nm
    waist_diameter = 1e-3  # 1 mm
    divergence_angle = 1e-3  # 1 mrad
    
    m2 = calculate_m2_factor(wavelength, waist_diameter, divergence_angle)
    
    # Check that M² is positive
    assert m2 > 0
    
    # For a perfect Gaussian beam, M² should be close to 1
    # Our test values should give M² close to 1
    assert m2 >= 1


def test_calculate_strehl_ratio():
    """Test the calculate_strehl_ratio function."""
    # Test normal case
    sr = calculate_strehl_ratio(0.8, 1.0)
    assert sr == 0.8
    
    # Test perfect case
    sr = calculate_strehl_ratio(1.0, 1.0)
    assert sr == 1.0
    
    # Test zero ideal intensity
    sr = calculate_strehl_ratio(0.5, 0.0)
    assert sr == 0.0
    
    # Test clipping behavior
    sr = calculate_strehl_ratio(1.5, 1.0)  # Should be clipped to 1.0
    assert sr == 1.0


def test_encircled_energy(sample_spot_image):
    """Test the encircled_energy function."""
    image_data, (radii, encircled_energy_values) = encircled_energy(sample_spot_image)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # Check that we get arrays back
    assert isinstance(radii, np.ndarray)
    assert isinstance(encircled_energy_values, np.ndarray)
    
    # Check that arrays have the same length
    assert len(radii) == len(encircled_energy_values)
    
    # Check that encircled energy starts at 0 and approaches 1
    assert encircled_energy_values[0] >= 0
    assert encircled_energy_values[-1] <= 1


def test_calculate_ellipticity(sample_spot_image):
    """Test the calculate_ellipticity function."""
    image_data, result = calculate_ellipticity(sample_spot_image)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # Check that all expected keys are in the result
    expected_keys = [
        'major_diameter', 'minor_diameter', 'ellipticity', 'orientation',
        'sigma_x_sq', 'sigma_y_sq', 'sigma_xy_sq'
    ]
    for key in expected_keys:
        assert key in result
    
    # Check that diameters are positive
    assert result['major_diameter'] > 0
    assert result['minor_diameter'] > 0
    
    # Check that ellipticity is between 0 and 1
    assert 0 <= result['ellipticity'] <= 1


def test_calculate_symmetry(sample_spot_image):
    """Test the calculate_symmetry function."""
    image_data, result = calculate_symmetry(sample_spot_image)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # Check that all expected keys are in the result
    expected_keys = [
        'quadrant_energies', 'symmetry_coefficient_of_variation', 
        'symmetry_max_deviation'
    ]
    for key in expected_keys:
        assert key in result
    
    # Check that we have 4 quadrant energies
    assert len(result['quadrant_energies']) == 4
    
    # Check that symmetry measures are non-negative
    assert result['symmetry_coefficient_of_variation'] >= 0
    assert result['symmetry_max_deviation'] >= 0


def test_calculate_uniformity(sample_spot_image):
    """Test the calculate_uniformity function."""
    image_data, result = calculate_uniformity(sample_spot_image)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # Check that all expected keys are in the result
    expected_keys = [
        'uniformity_cv', 'mean_intensity_in_flat_region', 
        'std_intensity_in_flat_region', 'threshold_value', 'peak_intensity'
    ]
    for key in expected_keys:
        assert key in result
    
    # Check that uniformity coefficient of variation is non-negative
    assert result['uniformity_cv'] >= 0
    
    # Check that peak intensity is the maximum value in the image
    assert result['peak_intensity'] == np.max(sample_spot_image.image)


def test_calculate_uniformity_with_custom_threshold(sample_spot_image):
    """Test the calculate_uniformity function with a custom threshold."""
    image_data, result = calculate_uniformity(sample_spot_image, threshold_percent=90.0)
    
    # Check that the returned image data is the same as the input
    assert np.array_equal(image_data, sample_spot_image.image)
    
    # Check that all expected keys are in the result
    expected_keys = [
        'uniformity_cv', 'mean_intensity_in_flat_region', 
        'std_intensity_in_flat_region', 'threshold_value', 'peak_intensity'
    ]
    for key in expected_keys:
        assert key in result


if __name__ == "__main__":
    pytest.main([__file__])
