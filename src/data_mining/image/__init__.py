"""
data_mining.image 模块

提供图像处理相关功能，包括：
- 基础图像操作 (common.py)
- 图像降噪算法 (denoising.py)
- Zernike多项式拟合 (zernike.py)
- 图像处理流程 (process.py)
"""

# 从process模块导入光斑分析功能
from .process import (
    find_centroid,
    calculate_beam_width,
    encircled_energy,
    calculate_ellipticity,
    calculate_symmetry,
    calculate_uniformity,
    fit_hyperbola,
    calculate_m2_factor,
    calculate_strehl_ratio,
    iso11146_beam_diameter,
    iso11146_m2_fitting,
    multi_scale_edge_detection,
    calculate_beam_quality_product,
)
from .common import (
    # 类和注册器
    SpotImage,
    SpotImageCollection,
    Image2D,
    FunctionRegistry,
    img_process_func,
    
    # 图像读取和转换
    read_tiff_to_numpy,
    process_time_columns,
    convert_to_cv,
    normalize_data,
    
    # 坐标转换
    cartesian_to_polar,
    polar_to_cartesian,
    
    # 图像处理
    extract_radial_data,
    fourier_shift_to_center,
    get_profiles,
)

# 从denoising模块导入所有降噪功能
try:
    from .denoising import (
        # 主要降噪函数
        bm3d_denoise,
        gaussian_denoise,
        median_denoise,
        nlmeans_denoise,
        bilateral_denoise,
        wavelet_denoise,
        total_variation_denoise,
        wiener_denoise,
        
        # 通用接口
        denoise,
        batch_denoise,
        
        # 工具函数
        estimate_noise_sigma,
        evaluate_denoising,
        denoise_spot_image,
        
        # 类和枚举
        DenoiseMethod,
        DenoiseResult,
    )
    _DENOISING_AVAILABLE = True
except ImportError:
    _DENOISING_AVAILABLE = False

# 版本信息
__version__ = "0.1.0"

# 公开API
__all__ = [
    # ===== process模块 - 光斑分析 =====
    'find_centroid',
    'calculate_beam_width',
    'encircled_energy',
    'calculate_ellipticity',
    'calculate_symmetry',
    'calculate_uniformity',
    'fit_hyperbola',
    'calculate_m2_factor',
    'calculate_strehl_ratio',
    'iso11146_beam_diameter',
    'iso11146_m2_fitting',
    'multi_scale_edge_detection',
    'calculate_beam_quality_product',
    
    # ===== common模块 =====
    # 类
    'SpotImage',
    'SpotImageCollection', 
    'Image2D',
    'FunctionRegistry',
    'img_process_func',
    
    # 函数
    'read_tiff_to_numpy',
    'process_time_columns',
    'convert_to_cv',
    'normalize_data',
    'cartesian_to_polar',
    'polar_to_cartesian',
    'extract_radial_data',
    'fourier_shift_to_center',
    'get_profiles',
]

# 如果降噪模块可用，添加到公开API
if _DENOISING_AVAILABLE:
    __all__.extend([
        # ===== denoising模块 =====
        # 降噪函数
        'bm3d_denoise',
        'gaussian_denoise',
        'median_denoise',
        'nlmeans_denoise',
        'bilateral_denoise',
        'wavelet_denoise',
        'total_variation_denoise',
        'wiener_denoise',
        
        # 通用接口
        'denoise',
        'batch_denoise',
        
        # 工具函数
        'estimate_noise_sigma',
        'evaluate_denoising',
        'denoise_spot_image',
        
        # 类
        'DenoiseMethod',
        'DenoiseResult',
    ])
