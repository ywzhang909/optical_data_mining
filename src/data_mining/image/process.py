from typing import Any, Callable, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from skimage import filters, measure, morphology

from .common import SpotImage, FunctionRegistry


def _get_grids(spot_image: SpotImage) -> Tuple[np.ndarray, np.ndarray]:
    """Safely obtain coordinate grids (xv, yv) from SpotImage.
    Falls back to creating grids if not present on the image object.
    """
    image_data = spot_image.image
    if hasattr(image_data, 'xv') and hasattr(image_data, 'yv'):
        xv, yv = image_data.xv, image_data.yv
        return xv, yv
    # fallback to spot_image providing xv/yv directly
    if hasattr(spot_image, 'xv') and hasattr(spot_image, 'yv'):
        return spot_image.xv, spot_image.yv
    # final fallback: create grids based on shape
    h, w = image_data.shape[:2]
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    return xv, yv

def find_centroid(spot_image: SpotImage) -> tuple[SpotImage, tuple[float, float]]:
    """Compute centroid of the spot image intensities with robust grid access."""
    image_data = spot_image.image
    xv, yv = _get_grids(spot_image)
    total_intensity = np.sum(image_data)
    if total_intensity == 0:
        return spot_image, (0.0, 0.0)
    x_centroid = np.sum(image_data * xv) / total_intensity
    y_centroid = np.sum(image_data * yv) / total_intensity
    return image_data, (float(x_centroid), float(y_centroid))

def calculate_beam_width(spot_image: SpotImage) -> tuple[SpotImage, dict]:
    image_data = spot_image.image
    xv, yv = _get_grids(spot_image)
    total_power = np.sum(image_data)
    if total_power == 0:
        return image_data, {
            'total_power': 0,
            'centroid': (0.0, 0.0),
            'sigma_x_sq': 0.0,
            'sigma_y_sq': 0.0,
            'sigma_xy_sq': 0.0,
            'beam_width_x': 0.0,
            'beam_width_y': 0.0
        }
    x_centroid = np.sum(image_data * xv) / total_power
    y_centroid = np.sum(image_data * yv) / total_power
    sigma_x_sq = np.sum(image_data * (xv - x_centroid)**2) / total_power
    sigma_y_sq = np.sum(image_data * (yv - y_centroid)**2) / total_power
    sigma_xy_sq = np.sum(image_data * (xv - x_centroid) * (yv - y_centroid)) / total_power
    d_x = 4 * np.sqrt(sigma_x_sq)
    d_y = 4 * np.sqrt(sigma_y_sq)
    result = {
        'total_power': float(total_power),
        'centroid': (float(x_centroid), float(y_centroid)),
        'sigma_x_sq': float(sigma_x_sq),
        'sigma_y_sq': float(sigma_y_sq),
        'sigma_xy_sq': float(sigma_xy_sq),
        'beam_width_x': float(d_x),
        'beam_width_y': float(d_y)
    }
    return image_data, result

def encircled_energy(spot_image: SpotImage, max_radius: Optional[float] = None) -> tuple[SpotImage, tuple[np.ndarray, np.ndarray]]:
    image_data = spot_image.image
    xv, yv = _get_grids(spot_image)
    total_intensity = np.sum(image_data)
    if total_intensity == 0:
        return image_data, (np.array([]), np.array([]))
    cx, cy = ndimage.center_of_mass(image_data)
    radii_map = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
    if max_radius is None:
        max_radius = radii_map.max()
    radius_steps = max(2, int(np.ceil(max_radius)))
    radii = np.linspace(0, max_radius, radius_steps)
    encircled = np.zeros_like(radii)
    for i, r in enumerate(radii):
        mask = radii_map <= r
        encircled[i] = image_data[mask].sum()
    encircled = encircled / total_intensity
    return image_data, (radii, encircled)

def calculate_ellipticity(spot_image: SpotImage) -> tuple[SpotImage, dict]:
    image_data = spot_image.image
    xv, yv = _get_grids(spot_image)
    total_power = np.sum(image_data)
    x_centroid = np.sum(image_data * xv) / total_power if total_power > 0 else 0.0
    y_centroid = np.sum(image_data * yv) / total_power if total_power > 0 else 0.0
    sigma_x_sq = np.sum(image_data * (xv - x_centroid) ** 2) / total_power if total_power > 0 else 0.0
    sigma_y_sq = np.sum(image_data * (yv - y_centroid) ** 2) / total_power if total_power > 0 else 0.0
    sigma_xy_sq = np.sum(image_data * (xv - x_centroid) * (yv - y_centroid)) / total_power if total_power > 0 else 0.0
    cov = np.array([[sigma_x_sq, sigma_xy_sq], [sigma_xy_sq, sigma_y_sq]])
    w, v = np.linalg.eigh(cov)
    w = w[::-1]
    v = v[:, ::-1]
    major = 4 * np.sqrt(w[0])
    minor = 4 * np.sqrt(w[1])
    ellipse_ratio = minor / major if major > 0 else 0
    orientation = np.arctan2(v[1, 0], v[0, 0])
    result = {
        'major_diameter': major,
        'minor_diameter': minor,
        'ellipticity': ellipse_ratio,
        'orientation': orientation,
        'sigma_x_sq': float(sigma_x_sq),
        'sigma_y_sq': float(sigma_y_sq),
        'sigma_xy_sq': float(sigma_xy_sq),
    }
    return image_data, result

def calculate_symmetry(spot_image: SpotImage) -> tuple[SpotImage, dict]:
    image_data = spot_image.image
    xv, yv = _get_grids(spot_image)
    total_power = np.sum(image_data)
    cx = np.sum(image_data * xv) / total_power
    cy = np.sum(image_data * yv) / total_power
    q1 = image_data[(xv > cx) & (yv > cy)].sum()
    q2 = image_data[(xv <= cx) & (yv > cy)].sum()
    q3 = image_data[(xv <= cx) & (yv <= cy)].sum()
    q4 = image_data[(xv > cx) & (yv <= cy)].sum()
    quadrant_energies = np.array([q1, q2, q3, q4])
    mean_energy = quadrant_energies.mean()
    symmetry_cv = (quadrant_energies.std() / mean_energy) if mean_energy > 0 else 0
    symmetry_dev = (np.max(np.abs(quadrant_energies - mean_energy)) / total_power) if total_power > 0 else 0
    result = {
        'quadrant_energies': quadrant_energies.tolist(),
        'symmetry_coefficient_of_variation': float(symmetry_cv),
        'symmetry_max_deviation': float(symmetry_dev),
    }
    return image_data, result

def calculate_uniformity(spot_image: SpotImage, threshold_percent: float = 80.0) -> tuple[SpotImage, dict]:
    image_data = spot_image.image
    peak = float(image_data.max())
    threshold_value = np.percentile(image_data[image_data > 0], threshold_percent) if image_data.size > 0 else 0
    flat_mask = image_data >= threshold_value
    if np.any(flat_mask):
        flat_vals = image_data[flat_mask]
        mean_v = flat_vals.mean()
        std_v = flat_vals.std()
        cv = std_v / mean_v if mean_v > 0 else 0
    else:
        cv = 0
        mean_v = 0
        std_v = 0
    result = {
        'uniformity_cv': float(cv),
        'mean_intensity_in_flat_region': float(mean_v),
        'std_intensity_in_flat_region': float(std_v),
        'threshold_value': float(threshold_value),
        'peak_intensity': peak,
    }
    return image_data, result


def fit_hyperbola(z_positions: List[float], d_sq_values: List[float]) -> dict:
    z_positions = np.asarray(z_positions, dtype=float)
    d_sq_values = np.asarray(d_sq_values, dtype=float)
    z_ref = z_positions[0]
    z_rel = z_positions - z_ref
    def hyperbola(z, A, B, C):
        return A + B * z + C * z**2
    popt, pcov = optimize.curve_fit(hyperbola, z_rel, d_sq_values)
    A, B, C = popt
    z0 = z_ref - B / (2 * C) if C != 0 else z_ref
    d0_sq = A - B**2 / (4 * C) if C != 0 else A
    d0 = float(np.sqrt(max(d0_sq, 0))) if d0_sq is not None else 0.0
    theta = float(np.sqrt(C)) if C >= 0 else 0.0
    return {
        'fitted_coefficients': {'A': A, 'B': B, 'C': C},
        'waist_position': z0,
        'waist_diameter': d0,
        'divergence_angle': theta,
        'fit_covariance': pcov
    }

def calculate_m2_factor(wavelength: float, waist_diameter: float, divergence_angle: float) -> float:
    m2 = (np.pi / wavelength) * (waist_diameter * divergence_angle) / 4.0
    return m2

def calculate_strehl_ratio(actual_peak_intensity: float, ideal_peak_intensity: float) -> float:
    if ideal_peak_intensity <= 0:
        return 0.0
    sr = actual_peak_intensity / ideal_peak_intensity
    return float(np.clip(sr, 0.0, 1.0))


# =============================================================================
# ISO 11146 标准光斑分析方法
# 参考论文:
# - ISO 11146-1:2021 Lasers and laser-related equipment — Test methods for laser beam widths, divergence angles and beam propagation ratios
# - Scott Prahl, laserbeamsize Python库 (https://github.com/scottprahl/laserbeamsize)
# - "M^2 Factor" - RP Photonics Encyclopedia (https://rp-photonics.com/m2_factor.html)
# - "Beam quality M^2(ψ) factor, spot rotation angle, and angular speed in general laser beams" - arXiv:2411.07879
# =============================================================================

def iso11146_beam_diameter(
    spot_image: SpotImage,
    pixel_size: float = 1.0,
    background_threshold: Optional[float] = None
) -> tuple[SpotImage, dict]:
    """
    ISO 11146 D4σ 方差法计算光束直径
    
    基于ISO 11146-1:2021标准，使用二阶矩计算光束直径。
    D4σ (Diameter 4 Sigma) = 4 * sqrt(方差)
    
    参考: ISO 11146-1:2021, Section 5.2.2 - Variance method (D4σ)
    """
    image_data = spot_image.image.astype(np.float64)
    xv, yv = _get_grids(spot_image)
    
    if background_threshold is None:
        background_threshold = np.percentile(image_data, 10)
    
    image_bg_sub = image_data - background_threshold
    image_bg_sub = np.maximum(image_bg_sub, 0)
    
    total_power = np.sum(image_bg_sub)
    if total_power == 0:
        return spot_image, {
            'd4sigma_x': 0.0,
            'd4sigma_y': 0.0,
            'd4sigma_major': 0.0,
            'd4sigma_minor': 0.0,
            'rotation_angle': 0.0,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'total_power': 0.0,
            'pixel_size': pixel_size
        }
    
    cx = np.sum(image_bg_sub * xv) / total_power
    cy = np.sum(image_bg_sub * yv) / total_power
    
    dx = xv - cx
    dy = yv - cy
    
    sigma_x_sq = np.sum(image_bg_sub * dx**2) / total_power
    sigma_y_sq = np.sum(image_bg_sub * dy**2) / total_power
    sigma_xy = np.sum(image_bg_sub * dx * dy) / total_power
    
    d4sigma_x = 4.0 * np.sqrt(sigma_x_sq) * pixel_size
    d4sigma_y = 4.0 * np.sqrt(sigma_y_sq) * pixel_size
    
    cov_matrix = np.array([[sigma_x_sq, sigma_xy], [sigma_xy, sigma_y_sq]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    major_axis = 4.0 * np.sqrt(max(eigenvalues[1], 0)) * pixel_size
    minor_axis = 4.0 * np.sqrt(max(eigenvalues[0], 0)) * pixel_size
    
    rotation_angle = 0.5 * np.arctan2(2 * sigma_xy, sigma_x_sq - sigma_y_sq)
    
    return spot_image, {
        'd4sigma_x': float(d4sigma_x),
        'd4sigma_y': float(d4sigma_y),
        'd4sigma_major': float(major_axis),
        'd4sigma_minor': float(minor_axis),
        'rotation_angle': float(rotation_angle),
        'centroid_x': float(cx * pixel_size),
        'centroid_y': float(cy * pixel_size),
        'total_power': float(total_power),
        'pixel_size': pixel_size,
        'sigma_x_sq': float(sigma_x_sq * pixel_size**2),
        'sigma_y_sq': float(sigma_y_sq * pixel_size**2),
        'sigma_xy': float(sigma_xy * pixel_size**2)
    }


def iso11146_m2_fitting(
    z_positions: np.ndarray,
    diameters: np.ndarray,
    wavelength: float,
    pixel_size: float = 1.0
) -> dict:
    """
    基于ISO 11146标准进行M²因子拟合
    
    使用双曲线拟合 d²(z) = A + B*z + C*z²
    从拟合参数计算束腰位置、束腰直径、发散角和M²因子
    
    参考: ISO 11146-1:2021, Section 8 - Beam propagation ratio M²
    """
    z_positions = np.asarray(z_positions, dtype=float)
    diameters = np.asarray(diameters, dtype=float)
    
    if len(z_positions) < 3:
        return {'error': '需要至少3个测量点'}
    
    d_squared = diameters ** 2
    
    z_mean = np.mean(z_positions)
    d_sq_mean = np.mean(d_squared)
    
    z_centered = z_positions - z_mean
    d_sq_centered = d_squared - d_sq_mean
    
    slope = np.sum(z_centered * d_sq_centered) / np.sum(z_centered ** 2) if np.sum(z_centered ** 2) > 0 else 0
    intercept = d_sq_mean - slope * z_mean
    
    C = slope / 2
    A = d_sq_mean - slope * z_mean
    
    z0 = z_mean - C / (2 * C**2) if abs(C) > 1e-10 else z_mean
    d0_sq = A - (slope**2) / (4 * C) if abs(C) > 1e-10 else A
    d0 = np.sqrt(max(d0_sq, 0))
    
    z_far = z_positions[-1]
    d_far_sq = d_squared[-1]
    theta = np.sqrt(max((d_far_sq - d0_sq) / (z_far - z0)**2, 0)) / 2 if abs(z_far - z0) > 0 else 0
    
    w0 = d0 / 2
    theta_rad = theta
    k = 2 * np.pi / wavelength
    
    if w0 > 0 and theta_rad > 0:
        m2 = (k * w0 * theta_rad) / 2
    else:
        m2 = 1.0
    
    return {
        'waist_position': float(z0),
        'waist_diameter': float(d0),
        'waist_radius': float(w0),
        'divergence_angle': float(theta_rad),
        'm2_factor': float(m2),
        'beam_parameter_product': float(w0 * theta_rad),
        'fit_coefficients': {'A': float(A), 'B': float(slope), 'C': float(C)},
        'wavelength': wavelength,
        'pixel_size': pixel_size
    }


def multi_scale_edge_detection(
    spot_image: SpotImage,
    scales: List[float] = [1.0, 2.0, 3.0]
) -> tuple[SpotImage, dict]:
    """
    多尺度自适应卷积边缘提取与定位
    
    参考: "Optimization of laser spot edge extraction and localization based on 
    multi-scale adaptive convolution" - Frontiers in Physics (2025)
    """
    image_data = spot_image.image.astype(np.float64)
    
    edges = []
    for scale in scales:
        kernel_size = int(2 * scale + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        smoothed = ndimage.gaussian_filter(image_data, sigma=scale)
        
        sobel_x = ndimage.sobel(smoothed, axis=1)
        sobel_y = ndimage.sobel(smoothed, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        edges.append(edge_magnitude)
    
    edge_avg = np.mean(edges, axis=0)
    
    threshold = np.percentile(edge_avg, 95)
    binary_edge = edge_avg > threshold
    
    labeled = measure.label(binary_edge)
    regions = measure.regionprops(labeled)
    
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        boundary = largest_region.perimeter
        area = largest_region.area
        circularity = 4 * np.pi * area / (boundary**2) if boundary > 0 else 0
    else:
        circularity = 0
    
    return spot_image, {
        'edge_magnitude': edge_avg,
        'binary_edge': binary_edge,
        'circularity': float(circularity),
        'scales_used': scales
    }


def calculate_beam_quality_product(
    spot_image: SpotImage,
    wavelength: float,
    pixel_size: float = 1.0
) -> tuple[SpotImage, dict]:
    """
    计算光束参数乘积 (Beam Parameter Product, BPP)
    
    BPP = w0 * θ
    其中 w0 是束腰半径，θ 是远场发散角
    
    参考: "M^2 Factor" - RP Photonics Encyclopedia
    """
    image_data = spot_image.image
    xv, yv = _get_grids(spot_image)
    
    total_power = np.sum(image_data)
    if total_power == 0:
        return spot_image, {'bpp': 0.0, 'm2': 0.0, 'error': 'no signal'}
    
    cx = np.sum(image_data * xv) / total_power
    cy = np.sum(image_data * yv) / total_power
    
    dx = xv - cx
    dy = yv - cy
    
    sigma_x_sq = np.sum(image_data * dx**2) / total_power
    sigma_y_sq = np.sum(image_data * dy**2) / total_power
    
    w0_x = 2 * np.sqrt(sigma_x_sq) * pixel_size
    w0_y = 2 * np.sqrt(sigma_y_sq) * pixel_size
    w0 = (w0_x + w0_y) / 2
    
    h, w = image_data.shape[:2]
    max_radius = np.sqrt((w/2)**2 + (h/2)**2) * pixel_size
    
    radii_map = np.sqrt((xv - cx)**2 + (yv - cy)**2) * pixel_size
    
    outer_radius = max_radius * 0.8
    mask_outer = radii_map > outer_radius
    outer_intensity = image_data[mask_outer].sum()
    
    theta_x = np.sqrt(outer_intensity / (total_power * np.pi * outer_radius**2)) if outer_radius > 0 else 0
    theta_y = theta_x
    
    bpp = w0 * (theta_x + theta_y) / 2
    
    k = 2 * np.pi / wavelength
    m2 = (k * w0 * theta_x) / 2 if w0 > 0 and theta_x > 0 else 1.0
    
    return spot_image, {
        'waist_radius': float(w0),
        'divergence_x': float(theta_x),
        'divergence_y': float(theta_y),
        'beam_parameter_product': float(bpp),
        'm2_factor': float(m2),
        'wavelength': wavelength,
        'pixel_size': pixel_size
    }