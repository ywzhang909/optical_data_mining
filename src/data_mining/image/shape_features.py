"""Shape and profile fitting features for optical beam images."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import curve_fit
from skimage.measure import regionprops


def gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float, baseline: float) -> np.ndarray:
    """Gaussian profile function for beam cross-section fitting."""
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + baseline


def fitting_gaussian(data: np.ndarray) -> tuple[tuple[float, float, float, float], np.ndarray | float]:
    """Fit a Gaussian to a 1D profile and return parameters/covariance."""
    x_data = np.arange(len(data))
    initial_guess = [float(np.argmax(data)), 10.0, float(np.max(data)), 0.0]
    try:
        (mu, sigma, amplitude, baseline), covariance = curve_fit(
            gaussian, x_data, data, p0=initial_guess
        )
    except RuntimeError:
        return (np.nan, np.nan, np.nan, np.nan), np.nan
    return (float(mu), float(sigma), float(amplitude), float(baseline)), covariance


def calculate_xy_diameters(image: np.ndarray, center_x: float, center_y: float) -> dict[str, float]:
    """Fit Gaussian profiles along X/Y and return diameters (2σ)."""
    y_data = image[:, int(center_x)]
    x_data = image[int(center_y), :]

    (_, sigma_x, _, _), _ = fitting_gaussian(x_data)
    (_, sigma_y, _, _), _ = fitting_gaussian(y_data)
    return {"gaussian_dia_x": float(2 * sigma_x), "gaussian_dia_y": float(2 * sigma_y)}


def find_spot_border(image: np.ndarray) -> dict[str, float]:
    """Estimate spot enclosing circle from thresholded largest connected component."""
    blurred = gaussian_filter(image.astype(float), sigma=1.0)
    threshold = np.max(blurred) * (1 / np.e)
    binary = blurred > threshold
    labeled, num_features = label(binary)

    if num_features == 0:
        return {"border_x": np.nan, "border_y": np.nan, "border_radius": np.nan}

    areas = [(labeled == idx).sum() for idx in range(1, num_features + 1)]
    largest_id = int(np.argmax(areas)) + 1
    ys, xs = np.where(labeled == largest_id)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    radius = float(np.sqrt(len(xs) / np.pi))
    return {"border_x": center_x, "border_y": center_y, "border_radius": radius}


def ellipse_fit(image: np.ndarray) -> dict[str, float]:
    """Estimate ellipse and uniformity from thresholded beam mask."""
    threshold = np.max(image) * 0.3
    binary = image > threshold
    labeled, num_features = label(binary)
    if num_features == 0:
        return {
            "ellipse_center_x": np.nan,
            "ellipse_center_y": np.nan,
            "short_axis": np.nan,
            "long_axis": np.nan,
            "ellipticity": np.nan,
            "angle": np.nan,
            "uniformity": np.nan,
        }

    largest_id = 1 + int(np.argmax([(labeled == idx).sum() for idx in range(1, num_features + 1)]))
    props = regionprops((labeled == largest_id).astype(np.uint8), intensity_image=image)
    if not props:
        return {
            "ellipse_center_x": np.nan,
            "ellipse_center_y": np.nan,
            "short_axis": np.nan,
            "long_axis": np.nan,
            "ellipticity": np.nan,
            "angle": np.nan,
            "uniformity": np.nan,
        }

    region = props[0]
    short_axis = float(region.axis_minor_length if hasattr(region, "axis_minor_length") else region.minor_axis_length)
    long_axis = float(region.axis_major_length if hasattr(region, "axis_major_length") else region.major_axis_length)
    image_intensity = region.image_intensity if hasattr(region, "image_intensity") else region.intensity_image
    mean_intensity = float(np.mean(image_intensity[region.image]))
    std_intensity = float(np.std(image_intensity[region.image]))

    return {
        "ellipse_center_x": float(region.centroid[1]),
        "ellipse_center_y": float(region.centroid[0]),
        "short_axis": short_axis,
        "long_axis": long_axis,
        "ellipticity": float(long_axis / short_axis) if short_axis > 0 else np.nan,
        "angle": float(region.orientation),
        "uniformity": float(std_intensity / mean_intensity) if mean_intensity > 0 else np.nan,
    }
