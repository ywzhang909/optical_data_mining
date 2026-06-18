"""
光学分析模块
============

提供光束质量分析的核心算法:
- 光束特征提取 (D4σ, PIB, 高斯拟合)
- 衍射计算 (菲涅尔衍射, FFT居中, 斯特列尔比)
- 图像处理工具
- 分析指标计算
- 可视化工具
- 历史记录管理

子模块:
    - beam_analysis: 光束质量分析算法
    - diffraction: 衍射计算算法
    - image_utils: 图像处理工具
    - beam_analysis_metrics: 光束分析指标计算
    - visualization: 可视化工具
    - history_manager: 历史记录管理
"""

# 光束分析模块
from .beam_analysis import (
    calculate_bpp,
    calculate_centroid,
    calculate_m2,
    calculate_xy_diameters,
    d4sigma,
    extract_beam_features,
    fitting_gaussian,
    gaussian,
    pib_ratio,
)

# 光束分析指标计算模块
from .beam_analysis_metrics import BeamAnalysisMetrics

# 衍射计算模块
from .diffraction import (
    angular_spectrum_propagation,
    calculate_strehl_ratio_with_energy_conservation,
    crop_to_square,
    fnr3,
    propagate_through_lens,
    shift_to_center_fft,
)

# 历史记录管理模块
from .history_manager import HistoryManager

# 图像处理工具模块
from .image_utils import (
    calculate_background_threshold,
    clip_negative_values,
    load_image,
    normalize_image,
    normalize_image_for_display,
    pad_to_square,
    read_image_to_numpy,
    resize_image,
    subtract_dark_field,
)

# 可视化模块
from .visualization.beam_visualization import (
    plot_3d_visualization,
    plot_beam_visualization,
    plot_multiple_beams_3d,
)

__all__ = [
    # beam_analysis
    "gaussian",
    "fitting_gaussian",
    "d4sigma",
    "pib_ratio",
    "calculate_xy_diameters",
    "calculate_bpp",
    "calculate_m2",
    "calculate_centroid",
    "extract_beam_features",
    # diffraction
    "crop_to_square",
    "shift_to_center_fft",
    "fnr3",
    "calculate_strehl_ratio_with_energy_conservation",
    "propagate_through_lens",
    "angular_spectrum_propagation",
    # image_utils
    "normalize_image_for_display",
    "read_image_to_numpy",
    "load_image",
    "subtract_dark_field",
    "normalize_image",
    "resize_image",
    "pad_to_square",
    "clip_negative_values",
    "calculate_background_threshold",
    # beam_analysis_metrics
    "BeamAnalysisMetrics",
    # history_manager
    "HistoryManager",
    # visualization
    "plot_beam_visualization",
    "plot_3d_visualization",
    "plot_multiple_beams_3d",
]
