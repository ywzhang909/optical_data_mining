"""
光束质量分析模块
=================
提供光斑分析和光束质量评估的核心算法

功能:
- D4σ直径计算（一阶矩和二阶矩）
- PIB占比计算
- 高斯拟合直径
- BPP (Beam Parameter Product) 计算
- 斯特列尔比(Strehl Ratio) 计算辅助
"""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, Any


def gaussian(x: np.ndarray, mu: float, sigma: float, A: float, b: float) -> np.ndarray:
    """
    高斯函数
    
    Args:
        x: 自变量数组
        mu: 中心位置
        sigma: 标准差
        A: 振幅
        b: 基线偏移
    
    Returns:
        高斯函数值数组
    """
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b


def fitting_gaussian(data: np.ndarray) -> Tuple[Tuple[float, float, float, float], Any]:
    """
    拟合高斯函数
    
    Args:
        data: 输入一维数据数组
    
    Returns:
        (mu, sigma, A, b): 拟合参数元组
        covariance: 协方差矩阵
    """
    x_data = np.arange(len(data))
    initial_guess = [np.argmax(data), 10, np.max(data), 0]
    try:
        (mu, sigma, A, b), covariance = curve_fit(gaussian, x_data, data, p0=initial_guess)
        return (mu, sigma, A, b), covariance
    except RuntimeError:
        return (np.nan, np.nan, np.nan, np.nan), np.nan

def center_of_mass_numpy(intensity:np.ndarray, xv:np.ndarray, yv:np.ndarray, moment:int=1) -> tuple[float, float]:
    """
    计算光强的中心位置

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param moment: 中心位置的阶数
    :return center_x, center_y: 光强的中心位置
    """
    _intensity = intensity.copy().astype(np.float32)**moment
    total_intensity = np.sum(_intensity)
    c_x = np.sum(xv * _intensity) / total_intensity
    c_y = np.sum(yv * _intensity) / total_intensity
    return (float(c_x), float(c_y))

def d4sigma(
    img: np.ndarray,
    pixel_size_um: float = 1.0,
    subtract_background: bool = False
) -> Dict[str, float]:
    """
    计算 D4σ 光斑直径（符合 ISO 11146 标准）
    
    Args:
        img: 输入光强图像（2D array, 应为非负）
        pixel_size_um: 像素尺寸（微米）
        subtract_background: 是否自动扣除背景（推荐 True）
    
    Returns:
        dict with keys: center_x, center_y, D_x, D_y, avg_diameter, center_intensity
    """
    if img.ndim != 2:
        raise ValueError("Input image must be 2D")
    
    # 确保非负（数值误差可能导致极小负值）
    _img = np.maximum(img.copy().astype(np.float64), 0.0)
    
    if subtract_background:
        edge_pixels = np.concatenate([
            _img[0, :], _img[-1, :], _img[:, 0], _img[:, -1]
        ])
        background = np.median(edge_pixels)
        _img = np.maximum(_img - background, 0.0)
    
    total = _img.sum()
    if total == 0 or not np.isfinite(total):
        return {
            'center_x': 0.0,
            'center_y': 0.0,
            'D_x': 0.0,
            'D_y': 0.0,
            'avg_diameter': 0.0,
            'center_intensity': 0.0,
        }
    
    h, w = _img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    
    cx, cy = center_of_mass_numpy(_img, x, y, 5)
    
    # 二阶中心矩
    mu_xx = np.sum((x - cx)**2 * _img) / total
    mu_yy = np.sum((y - cy)**2 * _img) / total
    
    # D4σ = 4 * σ
    Dx = 4.0 * np.sqrt(max(mu_xx, 0.0)) * pixel_size_um
    Dy = 4.0 * np.sqrt(max(mu_yy, 0.0)) * pixel_size_um
    avg_diameter = np.sqrt(Dx * Dy)
    
    # 中心强度（双线性插值更准，但这里用最近邻）
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    if 0 <= cy_int < h and 0 <= cx_int < w:
        center_intensity = float(_img[cy_int, cx_int])
    else:
        center_intensity = 0.0
    
    return {
        'center_x': float(cx),
        'center_y': float(cy),
        'D_x': float(Dx),
        'D_y': float(Dy),
        'avg_diameter': float(avg_diameter),
        'center_intensity': center_intensity,
    }


def pib_ratio(
    img: np.ndarray,
    center: Tuple[float, float],
    wavelength_m: float,
    focal_length_m: float,
    aperture_diameter_m: float,
    pixel_size_m: float,
) -> Tuple[float, bool]:
    """
    计算 PIB 占比 (Power In Bucket)
    以质心为圆心，计算半径为衍射极限半径的圆内能量占全部能量的比值
    
    Args:
        img: 输入图像（2D数组）
        center: 中心坐标 (x, y)
        wavelength_m: 波长 (米)
        focal_length_m: 焦距 (米)
        aperture_diameter_m: 入瞳直径 (米)
        pixel_size_m: 像素尺寸 (米)
    
    Returns:
        tuple: (pib_ratio, is_overexposed)
            - pib_ratio: PIB 占比 (0-1之间)
            - is_overexposed: 是否过曝
    """
    cx, cy = center
    h, w = img.shape
    
    # 计算衍射极限半径 (Airy disk radius)
    # r_airy = 1.22 * λ * f / D (物理尺寸)
    r_airy_m = 1.22 * wavelength_m * focal_length_m / aperture_diameter_m
    # 转换为像素
    r_airy_pixels = r_airy_m / pixel_size_m
    
    y, x = np.mgrid[0:h, 0:w]
    mask = (x - cx)**2 + (y - cy)**2 <= r_airy_pixels**2
    pib_intensity = img[mask].sum()
    total_intensity = img.sum()
    
    if total_intensity == 0:
        return 0.0, False
    
    # 检查是否过曝：如果中心区域像素值接近或达到最大值
    max_pixel_value = img.max()
    # 假设16位图像最大值为65535，8位为255
    bit_depth = 16 if max_pixel_value > 255 else 8
    max_possible = 65535 if bit_depth == 16 else 255
    is_overexposed = max_pixel_value >= max_possible * 0.95
    
    return pib_intensity / total_intensity, is_overexposed


def calculate_xy_diameters(
    image: np.ndarray, 
    center_x: float, 
    center_y: float, 
    pix_size: float = 1.0
) -> Dict[str, float]:
    """
    计算x和y方向的高斯直径
    
    Args:
        image: 输入图像（2D数组）
        center_x: 中心X坐标
        center_y: 中心Y坐标
        pix_size: 像素尺寸
    
    Returns:
        dict: 包含x和y方向直径的字典
            - gaussian_dia_x(um): X方向高斯直径 (μm)
            - gaussian_dia_y(um): Y方向高斯直径 (μm)
    """
    # 提取x和y方向的数据
    y_data = image[:, int(center_x)]
    x_data = image[int(center_y), :]
    
    # 计算x方向直径
    (mu, sigma, A, b), conv = fitting_gaussian(x_data)
    x_diameter = 2 * sigma * pix_size if not np.isnan(sigma) else 0
    
    # 计算y方向直径
    (mu, sigma, A, b), conv = fitting_gaussian(y_data)
    y_diameter = 2 * sigma * pix_size if not np.isnan(sigma) else 0
    
    return {'gaussian_dia_x(um)': x_diameter, 'gaussian_dia_y(um)': y_diameter}


def calculate_bpp(
    pupil_diameter_mm: float, 
    focal_diameter_mm: float, 
    focal_length_mm: float = 3000
) -> Dict[str, float]:
    """
    计算BPP (Beam Parameter Product)
    
    Args:
        pupil_diameter_mm: 出瞳直径 (mm)
        focal_diameter_mm: 焦斑直径 (mm)
        focal_length_mm: 透镜焦距 (mm)
    
    Returns:
        dict: BPP结果
            - BPP_mm_mrad: 光束参数乘积
            - divergence_mrad: 发散角 (mrad)
    """
    w_pupil = pupil_diameter_mm / 2.0
    w_focal = focal_diameter_mm / 2.0
    f = focal_length_mm
    
    theta_rad = w_focal / f
    theta_mrad = theta_rad * 1000.0
    bpp_mm_mrad = w_pupil * theta_mrad
    
    return {
        "BPP_mm_mrad": bpp_mm_mrad,
        "divergence_mrad": theta_mrad,
    }


def calculate_m2(bpp_mm_mrad: float, wavelength_um: float) -> float:
    """
    计算M²光束质量因子
    
    Args:
        bpp_mm_mrad: 光束参数乘积
        wavelength_um: 波长 (微米)
    
    Returns:
        float: M²值
    """
    bpp_diffraction = wavelength_um / np.pi
    return bpp_mm_mrad / bpp_diffraction


def calculate_centroid(img: np.ndarray) -> Tuple[float, float]:
    """
    计算图像的质心
    
    Args:
        img: 输入图像（2D数组）
    
    Returns:
        tuple: (center_x, center_y) 质心坐标
    """
    total = img.sum()
    if total == 0:
        return (0.0, 0.0)
    
    cy, cx = center_of_mass(img)
    try:
        cx = float(np.ravel(np.asarray(cx))[0])
        cy = float(np.ravel(np.asarray(cy))[0])
    except (TypeError, IndexError, ValueError):
        cx, cy = float(cx), float(cy)
    
    return (cx, cy)


def extract_beam_features(
    img: np.ndarray,
    pixel_size_um: float = 1.0,
    wavelength_m: float = 1064e-9,
    focal_length_m: float = 3.0,
    aperture_diameter_m: float = 0.1,
) -> Dict[str, Any]:
    """
    提取完整的光束特征
    
    Args:
        img: 输入图像（2D数组）
        pixel_size_um: 像素尺寸（微米）
        wavelength_m: 波长（米）
        focal_length_m: 焦距（米）
        aperture_diameter_m: 入瞳直径（米）
    
    Returns:
        dict: 包含所有光束特征的字典
    """
    # D4σ特征
    d4s_features = d4sigma(img, pixel_size_um)
    
    # 质心
    centroid = (d4s_features['center_x'], d4s_features['center_y'])
    
    # PIB占比（使用衍射极限半径）
    pixel_size_m = pixel_size_um * 1e-6
    pib, is_overexposed = pib_ratio(
        img, centroid,
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        aperture_diameter_m=aperture_diameter_m,
        pixel_size_m=pixel_size_m,
    )
    
    # 高斯拟合直径
    gaussian_dia = calculate_xy_diameters(
        img, 
        d4s_features['center_x'], 
        d4s_features['center_y'],
        pixel_size_um
    )
    
    return {
        'centroid': centroid,
        'd4s': d4s_features,
        'pib_ratio': pib,
        'pib_overexposed': is_overexposed,
        'gaussian_diameter': gaussian_dia,
    }
