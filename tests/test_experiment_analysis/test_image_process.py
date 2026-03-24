import numpy as np
import pytest
from data_mining.image.common import SpotImage
from data_mining.image.process import (
    find_centroid,
    calculate_beam_width,
    encircled_energy,
    calculate_ellipticity,
    calculate_symmetry,
    calculate_uniformity,
    fit_hyperbola,
    calculate_m2_factor,
    calculate_strehl_ratio
)


@pytest.fixture
def sample_spot_image():
    # Simple 2D Gaussian-like image
    size = 21
    x = np.arange(size) - (size // 2)
    y = np.arange(size) - (size // 2)
    xx, yy = np.meshgrid(x, y)
    sigma = 3.0
    img = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return SpotImage(img, "test_spot")


def test_find_centroid(sample_spot_image):
    image_data, (cx, cy) = find_centroid(sample_spot_image)
    assert image_data is sample_spot_image.image
    # Centroid near center (0,0) in our centered coordinate system
    assert abs(cx) <= 1.0 and abs(cy) <= 1.0


def test_beam_width(sample_spot_image):
    image_data, result = calculate_beam_width(sample_spot_image)
    assert image_data is sample_spot_image.image
    for k in ["total_power", "centroid", "sigma_x_sq", "sigma_y_sq", "sigma_xy_sq", "beam_width_x", "beam_width_y"]:
        assert k in result
    assert result['beam_width_x'] > 0 and result['beam_width_y'] > 0


def test_encircled_energy(sample_spot_image):
    image_data, (radii, enc) = encircled_energy(sample_spot_image)
    assert isinstance(radii, np.ndarray)
    assert isinstance(enc, np.ndarray)
    assert len(radii) == len(enc)
    assert enc[0] >= 0 and enc[-1] <= 1


def test_ellipticity(sample_spot_image):
    image_data, result = calculate_ellipticity(sample_spot_image)
    assert 'major_diameter' in result
    assert 'minor_diameter' in result
    assert 'ellipticity' in result
    assert 0 <= result['ellipticity'] <= 1


def test_symmetry(sample_spot_image):
    image_data, result = calculate_symmetry(sample_spot_image)
    assert 'quadrant_energies' in result
    assert len(result['quadrant_energies']) == 4
    assert 'symmetry_coefficient_of_variation' in result
    assert result['symmetry_coefficient_of_variation'] >= 0


def test_uniformity(sample_spot_image):
    image_data, result = calculate_uniformity(sample_spot_image)
    assert 'uniformity_cv' in result
    assert result['peak_intensity'] == np.max(sample_spot_image.image)


def test_hyperbola_fit_and_m2():
    z = [0.0, 1.0, 2.0, 3.0]
    d2 = [1.0, 1.2, 1.5, 1.9]
    res = fit_hyperbola(z, d2)
    assert 'fitted_coefficients' in res
    assert 'A' in res['fitted_coefficients']
    # M2 factor sanity
    m2 = calculate_m2_factor(1e-6, 1e-3, 1e-3)
    assert m2 > 0


def test_strehl_ratio_basic():
    sr = calculate_strehl_ratio(0.8, 1.0)
    assert pytest.approx(sr, rel=1e-6) == 0.8
    assert calculate_strehl_ratio(1.0, 1.0) == 1.0
    assert calculate_strehl_ratio(0.5, 0.0) == 0.0
