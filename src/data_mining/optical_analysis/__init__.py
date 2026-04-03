"""
光学分析模块
============

提供光束质量分析的核心算法:
- 光束特征提取 (D4σ, PIB, 高斯拟合)
- 衍射计算 (菲涅尔衍射, FFT居中, 斯特列尔比)
- 图像处理工具

子模块:
    - beam_analysis: 光束质量分析算法
    - diffraction: 衍射计算算法
    - image_utils: 图像处理工具
"""

# 光束分析模块
from .beam_analysis import (
    gaussian,
    fitting_gaussian,
    d4sigma,
    pib_ratio,
    calculate_xy_diameters,
    calculate_bpp,
    calculate_m2,
    calculate_centroid,
    extract_beam_features,
)

# 衍射计算模块
from .diffraction import (
    crop_to_square,
    shift_to_center_fft,
    fnr3,
    calculate_strehl_ratio_with_energy_conservation,
    propagate_through_lens,
    angular_spectrum_propagation,
)

# 图像处理工具模块
from .image_utils import (
    normalize_image_for_display,
    read_image_to_numpy,
    load_image,
    subtract_dark_field,
    normalize_image,
    resize_image,
    pad_to_square,
    clip_negative_values,
    calculate_background_threshold,
)

__all__ = [
    # beam_analysis
    'gaussian',
    'fitting_gaussian',
    'd4sigma',
    'pib_ratio',
    'calculate_xy_diameters',
    'calculate_bpp',
    'calculate_m2',
    'calculate_centroid',
    'extract_beam_features',
    # diffraction
    'crop_to_square',
    'shift_to_center_fft',
    'fnr3',
    'calculate_strehl_ratio_with_energy_conservation',
    'propagate_through_lens',
    'angular_spectrum_propagation',
    # image_utils
    'normalize_image_for_display',
    'read_image_to_numpy',
    'load_image',
    'subtract_dark_field',
    'normalize_image',
    'resize_image',
    'pad_to_square',
    'clip_negative_values',
    'calculate_background_threshold',
]
