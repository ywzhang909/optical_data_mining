from typing import Any, Callable, Optional, Tuple, List
from functools import wraps

import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from skimage import filters, measure, morphology

from .common import SpotImage, FunctionRegistry


def _get_coordinate_grids(spot_image: SpotImage) -> tuple[np.ndarray, np.ndarray]:
    """Return meshgrid coordinates for a :class:`SpotImage`."""
    return spot_image.xx, spot_image.yy


def find_centroid(spot_image: SpotImage) -> tuple[SpotImage, tuple[float, float]]:
    """
    计算光斑图像的质心坐标。
    
    该函数使用强度加权平均法计算光斑图像的质心位置。质心是光强分布的重心，
    对于理想的高斯光束，质心位于光强最大的位置。对于非对称或有缺陷的光束，
    质心可能偏离几何中心，这有助于检测光束的偏移或畸变。

    Args:
        spot_image (SpotImage): 输入的光斑图像对象，包含图像数据和坐标信息。

    Returns:
        tuple[SpotImage, tuple[float, float]]: 包含原始图像数据和质心坐标的元组，
        其中坐标格式为 (x_centroid, y_centroid)，表示光斑质心的x和y坐标。
    """

    # 获取图像数据
    image_data = spot_image.image

    # 计算总光强
    total_intensity = np.sum(image_data)

    # 计算质心的x和y坐标
    xx, yy = _get_coordinate_grids(spot_image)
    x_centroid = np.sum(image_data * xx) / total_intensity
    y_centroid = np.sum(image_data * yy) / total_intensity

    return image_data, (x_centroid, y_centroid)

def calculate_beam_width(spot_image: SpotImage) -> tuple[SpotImage, dict]:
    """
    使用二阶矩法（ISO 11146标准）计算光束宽度。
    
    该方法基于光强分布的统计矩来定义光束宽度，是国际标准ISO 11146中规定的标准方法。
    光束宽度定义为4倍的标准差，这样对于理想高斯光束，光束宽度正好等于光束半径。
    此方法考虑了整个光束截面的能量分布，能够有效处理非高斯型光束的情况。

    Args:
        spot_image (SpotImage): 输入的光斑图像对象，包含图像数据和坐标信息。

    Returns:
        tuple[SpotImage, dict]: 包含原始图像数据和光束宽度参数的字典，字典中包含：
            - total_power: 总光功率（零阶矩）
            - centroid: 质心坐标 (x, y)
            - sigma_x_sq: X方向二阶中心矩
            - sigma_y_sq: Y方向二阶中心矩
            - sigma_xy_sq: XY方向二阶混合矩
            - beam_width_x: X方向光束宽度（直径）
            - beam_width_y: Y方向光束宽度（直径）
    """
    image_data = spot_image.image
    
    # 计算总功率（零阶矩）
    total_power = np.sum(image_data)
    
    # 计算质心（一阶矩）
    xx, yy = _get_coordinate_grids(spot_image)
    x_centroid = np.sum(image_data * xx) / total_power
    y_centroid = np.sum(image_data * yy) / total_power
    
    # 计算二阶中心矩
    sigma_x_sq = np.sum(image_data * (xx - x_centroid)**2) / total_power
    sigma_y_sq = np.sum(image_data * (yy - y_centroid)**2) / total_power
    sigma_xy_sq = np.sum(image_data * (xx - x_centroid) * (yy - y_centroid)) / total_power
    
    # 按照ISO 11146标准定义的光束宽度（直径）
    d_x = 4 * np.sqrt(sigma_x_sq)
    d_y = 4 * np.sqrt(sigma_y_sq)
    
    result = {
        'total_power': total_power,
        'centroid': (x_centroid, y_centroid),
        'sigma_x_sq': sigma_x_sq,
        'sigma_y_sq': sigma_y_sq,
        'sigma_xy_sq': sigma_xy_sq,
        'beam_width_x': d_x,
        'beam_width_y': d_y
    }
    
    return image_data, result

def fit_hyperbola(z_positions: List[float], d_sq_values: List[float]) -> dict:
    """
    将光束宽度平方数据拟合到双曲线方程以提取光束参数。
    
    该函数用于分析光束传播特性，通过对不同位置测量的光束宽度进行双曲线拟合，
    可以得到光束的重要参数，如束腰位置、束腰宽度和远场发散角。这些参数对于
    计算光束质量因子M²至关重要。
    
    拟合方程: d^2(z) = A + B*(z - z_ref) + C*(z - z_ref)^2
    
    其中z_ref是参考位置（通常取第一个测量点），通过拟合可得系数A、B、C，
    进而推导出光束参数。

    Args:
        z_positions: 测量位置列表（沿光束传播方向的z坐标）
        d_sq_values: 对应位置的光束宽度平方值列表
        
    Returns:
        dict: 包含拟合参数和推导光束特性的字典：
            - fitted_coefficients: 拟合得到的系数 {'A': A, 'B': B, 'C': C}
            - waist_position: 束腰位置 z0
            - waist_diameter: 束腰处光束直径 d0
            - divergence_angle: 远场发散角（全角）theta
            - fit_covariance: 拟合协方差矩阵
    """
    
    z_positions = np.array(z_positions)
    d_sq_values = np.array(d_sq_values)
    
    # 使用第一个z位置作为参考
    z_ref = z_positions[0]
    z_rel = z_positions - z_ref
    
    # 定义用于拟合的双曲函数
    def hyperbola(z, A, B, C):
        return A + B * z + C * z**2
    
    # 执行曲线拟合
    popt, pcov = optimize.curve_fit(hyperbola, z_rel, d_sq_values)
    A, B, C = popt
    
    # 从拟合系数计算光束参数
    # 束腰位置
    z0 = z_ref - B / (2 * C)
    # 束腰宽度平方
    d0_sq = A - B**2 / (4 * C)
    # 远场发散角（全角）
    theta = np.sqrt(C)
    
    # 确保束腰宽度为正值
    d0 = np.sqrt(max(d0_sq, 0))
    
    result = {
        'fitted_coefficients': {'A': A, 'B': B, 'C': C},
        'waist_position': z0,
        'waist_diameter': d0,
        'divergence_angle': theta,
        'fit_covariance': pcov
    }
    
    return result

def calculate_m2_factor(wavelength: float, waist_diameter: float, divergence_angle: float) -> float:
    """
    根据测量的光束参数计算M²因子。
    
    M²因子是衡量激光光束质量的重要参数，表示实际光束与理想高斯光束的偏离程度。
    理想的高斯光束M²=1，实际光束的M²≥1。M²因子越大，表示光束质量越差，
    光束在传播过程中发散越快。
    
    计算公式: M² = (π/λ) * (d0 * θ) / 4
    
    其中λ是激光波长，d0是束腰直径，θ是远场发散角（全角）。

    Args:
        wavelength: 激光波长（单位：米）
        waist_diameter: 光束束腰直径（单位：米）
        divergence_angle: 远场发散角（全角，单位：弧度）
        
    Returns:
        float: M²因子，无量纲参数，理想值为1，实际值≥1
    """
    # 计算M²因子
    m2 = (np.pi / wavelength) * (waist_diameter * divergence_angle) / 4
    return m2

def calculate_strehl_ratio(actual_peak_intensity: float, ideal_peak_intensity: float) -> float:
    """
    计算斯特列尔比（Strehl Ratio）。
    
    斯特列尔比是评价光学系统成像质量的重要参数，定义为实际点扩散函数(PSF)的峰值强度
    与理想衍射极限PSF峰值强度的比值。斯特列尔比的取值范围为0到1，其中1表示理想的衍射
    极限成像，0表示完全失焦或严重像差。
    
    计算公式: SR = I_peak_actual / I_peak_ideal
    
    在光束质量分析中，斯特列尔比可用于评估光束聚焦性能，反映光束的相干性和波前质量。

    Args:
        actual_peak_intensity: 实际PSF的峰值强度
        ideal_peak_intensity: 理想PSF的峰值强度
        
    Returns:
        float: 斯特列尔比，取值范围0到1，1表示理想成像质量
    """
    if ideal_peak_intensity <= 0:
        return 0.0
    
    sr = actual_peak_intensity / ideal_peak_intensity
    # 限制在有效范围内
    return np.clip(sr, 0.0, 1.0)

def encircled_energy(spot_image: SpotImage, max_radius: Optional[float] = None) -> tuple[SpotImage, tuple[np.ndarray, np.ndarray]]:
    """
    计算包围能量曲线。
    
    包围能量是光学系统性能评估的重要参数，表示以光斑中心为圆心，在不同半径内包含的总能量比例。
    该曲线反映了光束能量在空间上的分布情况，常用于评估激光光束的集中程度和能量利用率。
    
    对于理想的高斯光束，包围能量曲线具有特定的数学形式。通过对比实测曲线与理想曲线，
    可以评估光束质量并识别能量分散等问题。

    Args:
        spot_image (SpotImage): 输入的光斑图像对象，包含图像数据和坐标信息。
        max_radius (float, optional): 计算的最大半径。如果为None，则使用图像对角线长度。
        
    Returns:
        tuple[SpotImage, tuple[np.ndarray, np.ndarray]]: 包含原始图像数据和包围能量曲线数据的元组，
        其中包围能量曲线数据为 (radii, encircled_energy_values)：
            - radii: 半径数组
            - encircled_energy_values: 对应半径内的归一化能量值（0到1之间）
    """
    image_data = spot_image.image
    
    # 获取图像尺寸
    ny, nx = image_data.shape
    
    # 计算质心
    total_intensity = np.sum(image_data)
    xx, yy = _get_coordinate_grids(spot_image)
    x_centroid = np.sum(image_data * xx) / total_intensity
    y_centroid = np.sum(image_data * yy) / total_intensity
    
    # 创建相对于质心的坐标网格
    x_coords = xx - x_centroid
    y_coords = yy - y_centroid
    
    # 计算每个像素的径向距离
    radii_map = np.sqrt(x_coords**2 + y_coords**2)
    
    # 如果未提供最大半径，则确定最大半径
    if max_radius is None:
        max_radius = np.sqrt(np.max(x_coords)**2 + np.max(y_coords)**2)
    
    # 创建半径数组
    dx = np.diff(np.arange(nx)).mean() if nx > 1 else 1.0
    dy = np.diff(np.arange(ny)).mean() if ny > 1 else 1.0
    radius_steps = max(2, int(max_radius / np.min([dx, dy])))
    radius_array = np.linspace(0, max_radius, radius_steps)
    
    # 计算每个半径的包围能量
    encircled_energy_values = np.zeros_like(radius_array)
    for i, radius in enumerate(radius_array):
        mask = radii_map <= radius
        encircled_energy_values[i] = np.sum(image_data[mask])
    
    # 归一化到总能量
    encircled_energy_values /= total_intensity
    
    return image_data, (radius_array, encircled_energy_values)

def calculate_ellipticity(spot_image: SpotImage) -> tuple[SpotImage, dict]:
    """
    使用二阶矩法计算光斑椭圆度。
    
    该方法通过计算光强分布的二阶中心矩构建协方差矩阵，然后通过特征值分解得到光斑的主轴和次轴方向及长度。
    椭圆度定义为次轴直径与主轴直径的比值，用于表征光斑的圆形程度。完美的圆形光斑椭圆度为1，
    椭圆度越小表示光斑越扁平。
    
    除了椭圆度，该函数还返回光斑的方位角，即主轴相对于x轴的角度，可用于分析光束的指向稳定性。

    Args:
        spot_image (SpotImage): 输入的光斑图像对象，包含图像数据和坐标信息。
        
    Returns:
        tuple[SpotImage, dict]: 包含原始图像数据和椭圆度参数的字典，字典中包含：
            - major_diameter: 主轴直径（光斑最长方向）
            - minor_diameter: 次轴直径（光斑最短方向）
            - ellipticity: 椭圆度（次轴直径/主轴直径，0到1之间）
            - orientation: 方位角（主轴相对于x轴的角度，单位：弧度）
            - sigma_x_sq: X方向二阶中心矩
            - sigma_y_sq: Y方向二阶中心矩
            - sigma_xy_sq: XY方向二阶混合矩
    """
    image_data = spot_image.image
    
    # 计算总功率
    total_power = np.sum(image_data)
    
    # 获取坐标网格
    xv, yv = spot_image.xx, spot_image.yy
    
    # 计算质心
    x_centroid = np.sum(image_data * xv) / total_power
    y_centroid = np.sum(image_data * yv) / total_power
    
    # 计算二阶中心矩
    sigma_x_sq = np.sum(image_data * (xv - x_centroid)**2) / total_power
    sigma_y_sq = np.sum(image_data * (yv - y_centroid)**2) / total_power
    sigma_xy_sq = np.sum(image_data * (xv - x_centroid) * (yv - y_centroid)) / total_power
    
    # 构建协方差矩阵
    cov_matrix = np.array([[sigma_x_sq, sigma_xy_sq],
                           [sigma_xy_sq, sigma_y_sq]])
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 排序特征值（从大到小）
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 主轴和次轴长度（标准差）
    sigma_major = np.sqrt(eigenvalues[0])
    sigma_minor = np.sqrt(eigenvalues[1])
    
    # 转换为直径
    major_diameter = 4 * sigma_major
    minor_diameter = 4 * sigma_minor
    
    # 计算椭圆度
    ellipticity = minor_diameter / major_diameter if major_diameter > 0 else 0
    
    # 计算方位角（单位：弧度）
    orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    result = {
        'major_diameter': major_diameter,
        'minor_diameter': minor_diameter,
        'ellipticity': ellipticity,
        'orientation': orientation,
        'sigma_x_sq': sigma_x_sq,
        'sigma_y_sq': sigma_y_sq,
        'sigma_xy_sq': sigma_xy_sq
    }
    
    return image_data, result

def calculate_symmetry(spot_image: SpotImage) -> tuple[SpotImage, dict]:
    """
    通过比较四个象限的能量分布来计算光斑对称性。
    
    该方法以光斑质心为中心，将图像划分为四个象限，分别计算每个象限内的总能量。
    通过分析四个象限能量的差异来评估光斑的对称性。对称性越高，表示光束在各个方向
    上的能量分布越均匀，这对于许多应用（如激光加工、光学通信等）非常重要。
    
    函数提供了两种对称性度量：
    1. 变异系数：四个象限能量的标准差与平均值的比值
    2. 最大偏差：单个象限能量与平均值的最大偏差相对于总能量的比例

    Args:
        spot_image (SpotImage): 输入的光斑图像对象，包含图像数据和坐标信息。
        
    Returns:
        tuple[SpotImage, dict]: 包含原始图像数据和对称性度量的字典，字典中包含：
            - quadrant_energies: 四个象限的能量值列表 [Q1, Q2, Q3, Q4]
            - symmetry_coefficient_of_variation: 对称性变异系数（越小越对称）
            - symmetry_max_deviation: 对称性最大偏差（越小越对称）
    """
    image_data = spot_image.image
    
    # 计算总功率
    total_power = np.sum(image_data)
    
    # 获取坐标网格
    xv, yv = spot_image.xx, spot_image.yy
    
    # 计算质心
    x_centroid = np.sum(image_data * xv) / total_power
    y_centroid = np.sum(image_data * yv) / total_power
    
    # 为四个象限创建掩码
    x_mask = xv > x_centroid
    y_mask = yv > y_centroid
    
    # 定义四个象限
    q1_mask = x_mask & y_mask  # 第一象限（右上）
    q2_mask = ~x_mask & y_mask  # 第二象限（左上）
    q3_mask = ~x_mask & ~y_mask  # 第三象限（左下）
    q4_mask = x_mask & ~y_mask  # 第四象限（右下）
    
    # 计算每个象限的能量
    q1_energy = np.sum(image_data[q1_mask])
    q2_energy = np.sum(image_data[q2_mask])
    q3_energy = np.sum(image_data[q3_mask])
    q4_energy = np.sum(image_data[q4_mask])
    
    # 计算对称性度量
    quadrant_energies = np.array([q1_energy, q2_energy, q3_energy, q4_energy])
    mean_energy = np.mean(quadrant_energies)
    
    # 对称性作为象限能量的变异系数
    if mean_energy > 0:
        symmetry_cv = np.std(quadrant_energies) / mean_energy
    else:
        symmetry_cv = 0
    
    # 对称性作为与均值的最大差异
    max_deviation = np.max(np.abs(quadrant_energies - mean_energy))
    symmetry_deviation = max_deviation / total_power if total_power > 0 else 0
    
    result = {
        'quadrant_energies': quadrant_energies.tolist(),
        'symmetry_coefficient_of_variation': symmetry_cv,
        'symmetry_max_deviation': symmetry_deviation
    }
    
    return image_data, result

def calculate_uniformity(spot_image: SpotImage, threshold_percent: float = 80.0) -> tuple[SpotImage, dict]:
    """
    使用高强度区域的变异系数计算光斑均匀性。
    
    该方法通过分析光斑中心高强度区域的强度变化来评估光斑的均匀性。首先确定一个阈值，
    选择高于该阈值的区域作为"平坦区域"，然后计算该区域内强度的变异系数（标准差/均值）。
    变异系数越小，表示光斑在该区域内的强度分布越均匀。
    
    均匀性是激光光束质量的重要指标之一，特别是在激光加工应用中，不均匀的光强分布可能导致
    加工质量不一致。通过调整threshold_percent参数，可以选择不同的强度区域进行分析。

    Args:
        spot_image (SpotImage): 输入的光斑图像对象，包含图像数据和坐标信息。
        threshold_percent (float): 百分位数阈值，用于定义平坦区域（例如，80表示选择最强20%的区域）。
        
    Returns:
        tuple[SpotImage, dict]: 包含原始图像数据和均匀性度量的字典，字典中包含：
            - uniformity_cv: 均匀性变异系数（越小越均匀）
            - mean_intensity_in_flat_region: 平坦区域的平均强度
            - std_intensity_in_flat_region: 平坦区域的强度标准差
            - threshold_value: 用于定义平坦区域的阈值
            - peak_intensity: 光斑峰值强度
    """
    image_data = spot_image.image
    
    # 查找峰值强度
    peak_intensity = np.max(image_data)
    
    # 定义平坦区域（高于threshold_percent百分位数的区域）
    threshold_value = np.percentile(image_data[image_data > 0], threshold_percent)
    
    # 平坦区域的掩码
    flat_region_mask = image_data >= threshold_value
    
    # 计算平坦区域的统计信息
    if np.any(flat_region_mask):
        flat_intensities = image_data[flat_region_mask]
        mean_intensity = np.mean(flat_intensities)
        std_intensity = np.std(flat_intensities)
        
        # 变异系数
        cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
    else:
        cv = 0
        mean_intensity = 0
        std_intensity = 0
    
    result = {
        'uniformity_cv': cv,
        'mean_intensity_in_flat_region': mean_intensity,
        'std_intensity_in_flat_region': std_intensity,
        'threshold_value': threshold_value,
        'peak_intensity': peak_intensity
    }
    
    return image_data, result
