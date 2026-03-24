"""
图像降噪算法模块

该模块提供多种先进的图像降噪算法，用于处理带噪声的图像数据。

主要功能:
    - BM3D: 基于块匹配和3D变换的先进降噪算法
    - 高斯滤波: 经典的线性平滑降噪
    - 中值滤波: 有效去除椒盐噪声
    - 非局部均值: 利用图像自相似性的降噪
    - 双边滤波: 边缘保持的降噪
    - 小波去噪: 基于多尺度分析的降噪

依赖:
    必选: numpy, opencv-python
    可选: bm3d, PyWavelets, scikit-image

安装:
    pip install numpy opencv-python
    pip install bm3d PyWavelets  # 可选
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple, Union, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
import logging
import time

# 可选依赖检查
try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# =============================================================================
# 日志配置
# =============================================================================

def _get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"data_mining.image.denoising.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger


# =============================================================================
# 枚举和数据类
# =============================================================================

class DenoiseMethod(Enum):
    """降噪方法枚举"""
    BM3D = "bm3d"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    NLMEANS = "nlmeans"
    BILATERAL = "bilateral"
    WAVELET = "wavelet"
    TOTAL_VARIATION = "total_variation"
    WIENER = "wiener"
    
    @classmethod
    def from_string(cls, method: str) -> "DenoiseMethod":
        method_lower = method.lower().strip()
        for m in cls:
            if m.value == method_lower:
                return m
        valid = [m.value for m in cls]
        raise ValueError(f"未知方法: '{method}'. 支持: {valid}")


class NoiseEstimationMethod(Enum):
    MAD = "mad"
    STD = "std"


@dataclass(frozen=True, slots=True)
class DenoiseResult:
    """降噪结果"""
    image: np.ndarray
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    noise_estimate: Optional[float] = None
    processing_time: float = 0.0
    
    def __array__(self, dtype=None, copy=False):
        img = self.image
        if dtype:
            img = img.astype(dtype)
        return img.copy() if copy else img
    
    @property
    def psnr(self) -> Optional[float]:
        return self.parameters.get('psnr')
    
    @property
    def ssim(self) -> Optional[float]:
        return self.parameters.get('ssim')


@dataclass
class QualityMetrics:
    """质量评估指标"""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    snr_improvement: Optional[float] = None
    
    def __str__(self) -> str:
        parts = []
        if self.psnr: parts.append(f"PSNR: {self.psnr:.2f} dB")
        if self.ssim: parts.append(f"SSIM: {self.ssim:.4f}")
        if self.snr_improvement: parts.append(f"SNR提升: {self.snr_improvement:.2f} dB")
        return ", ".join(parts) if parts else "无指标"


# =============================================================================
# 辅助函数
# =============================================================================

def _timed(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if isinstance(result, DenoiseResult):
            result = DenoiseResult(
                image=result.image,
                method=result.method,
                parameters={**result.parameters, '_processing_time': elapsed},
                noise_estimate=result.noise_estimate,
                processing_time=elapsed
            )
        return result
    return wrapper


# =============================================================================
# 降噪算法
# =============================================================================

@_timed
def bm3d_denoise(image: np.ndarray, sigma_psd: float = 25.0, 
                 stage_arg: Optional[str] = None, verbose: bool = False) -> DenoiseResult:
    """BM3D降噪 - 先进的块匹配3D滤波算法"""
    if not BM3D_AVAILABLE:
        raise ImportError("BM3D未安装: pip install bm3d")
    
    logger = _get_logger('bm3d')
    if verbose:
        logger.setLevel(logging.INFO)
    
    # 归一化
    img_min, img_max = float(image.min()), float(image.max())
    if img_max > 1.0:
        normalized = image.astype(np.float32) / 255.0
        sigma = sigma_psd / 255.0
    else:
        normalized = image.astype(np.float32)
        sigma = sigma_psd
    
    # 设置stage
    if stage_arg == 'hard':
        stage = bm3d.BM3DStages.HARD_THRESHOLDING
    else:
        stage = bm3d.BM3DStages.ALL_STAGES
    
    denoised = bm3d.bm3d(normalized, sigma_psd=sigma, stage_arg=stage)
    
    # 反归一化
    if img_max > 1.0:
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    
    return DenoiseResult(image=denoised, method="BM3D", 
                         parameters={"sigma_psd": sigma_psd, "stage": stage_arg or "all"},
                         noise_estimate=sigma_psd)


@_timed
def gaussian_denoise(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.5) -> DenoiseResult:
    """高斯滤波降噪"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return DenoiseResult(image=denoised, method="Gaussian",
                        parameters={"kernel_size": kernel_size, "sigma": sigma})


@_timed
def median_denoise(image: np.ndarray, kernel_size: int = 5) -> DenoiseResult:
    """中值滤波降噪 - 适合去除椒盐噪声"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    denoised = cv2.medianBlur(image, kernel_size)
    return DenoiseResult(image=denoised, method="Median",
                        parameters={"kernel_size": kernel_size})


@_timed
def nlmeans_denoise(image: np.ndarray, h: float = 10.0,
                    template_window_size: int = 7, 
                    search_window_size: int = 21) -> DenoiseResult:
    """非局部均值降噪"""
    if template_window_size % 2 == 0:
        template_window_size += 1
    if search_window_size % 2 == 0:
        search_window_size += 1
    
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        denoised = cv2.fastNlMeansDenoising(image, None, h=h,
                                            templateWindowSize=template_window_size,
                                            searchWindowSize=search_window_size)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h=h, hColor=h,
                                                   templateWindowSize=template_window_size,
                                                   searchWindowSize=search_window_size)
    
    return DenoiseResult(image=denoised, method="NLMeans",
                        parameters={"h": h, "template": template_window_size, 
                                  "search": search_window_size})


@_timed
def bilateral_denoise(image: np.ndarray, d: int = 9, 
                     sigma_color: float = 75.0, sigma_space: float = 75.0) -> DenoiseResult:
    """双边滤波降噪 - 边缘保持"""
    denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return DenoiseResult(image=denoised, method="Bilateral",
                        parameters={"d": d, "sigma_color": sigma_color, 
                                  "sigma_space": sigma_space})


@_timed
def wavelet_denoise(image: np.ndarray, wavelet: str = 'db1', 
                    level: int = 2, mode: str = 'soft',
                    sigma: Optional[float] = None) -> DenoiseResult:
    """小波降噪"""
    if not PYWAVELETS_AVAILABLE:
        raise ImportError("PyWavelets未安装: pip install PyWavelets")
    
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 分解
    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    
    # 估计噪声
    if sigma is None:
        sigma = float(np.median(np.abs(coeffs[-1][2])) / 0.6745)
    
    # 阈值
    threshold = sigma * np.sqrt(2 * np.log(gray.size))
    
    # 阈值处理
    def threshold_coeff(c, t, m):
        if m == 'soft':
            return np.sign(c) * np.maximum(np.abs(c) - t, 0)
        return c * (np.abs(c) > t)
    
    result = [coeffs[0]]
    for detail in coeffs[1:]:
        cH, cV, cD = detail
        result.append((threshold_coeff(cH, threshold, mode),
                      threshold_coeff(cV, threshold, mode),
                      threshold_coeff(cD, threshold, mode)))
    
    # 重建
    denoised = pywt.waverec2(result, wavelet)
    denoised = denoised[:gray.shape[0], :gray.shape[1]]
    
    if len(image.shape) == 3:
        denoised = cv2.cvtColor(denoised.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    return DenoiseResult(image=denoised.astype(image.dtype), method="Wavelet",
                        parameters={"wavelet": wavelet, "level": level, "mode": mode},
                        noise_estimate=sigma)


@_timed
def total_variation_denoise(image: np.ndarray, weight: float = 0.1,
                            max_iter: int = 200, eps: float = 1e-4) -> DenoiseResult:
    """全变分去噪"""
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    
    # Chambolle-Pock算法
    n_rows, n_cols = gray.shape
    p = np.zeros((n_rows, n_cols, 2))
    u = gray.copy()
    tau, sigma = 0.25, 0.25
    
    for _ in range(max_iter):
        u_old = u.copy()
        grad = np.stack([np.roll(u, -1, axis=0) - u, np.roll(u, -1, axis=1) - u], axis=2)
        p = p + sigma * grad
        p = p / np.maximum(1, np.sqrt(np.sum(p**2, axis=2, keepdims=True)))
        div = (p[:,:,0] - np.roll(p[:,:,0], 1, axis=0)) + (p[:,:,1] - np.roll(p[:,:,1], 1, axis=1))
        u = (u + tau * (gray / weight + div)) / (1 + tau / weight)
        if np.linalg.norm(u - u_old) / np.linalg.norm(u) < eps:
            break
    
    # 转回
    if len(image.shape) == 3:
        u = (u * 255).astype(np.uint8)
        u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    else:
        if image.max() > 1.0:
            u = (u * 255).astype(image.dtype)
    
    return DenoiseResult(image=u, method="TotalVariation",
                        parameters={"weight": weight, "max_iter": max_iter})


@_timed
def wiener_denoise(image: np.ndarray, kernel_size: int = 5,
                   noise_variance: Optional[float] = None) -> DenoiseResult:
    """维纳滤波"""
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    
    # 局部统计
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_mean_sq = cv2.blur(gray**2, (kernel_size, kernel_size))
    local_var = local_mean_sq - local_mean**2
    
    if noise_variance is None:
        noise_variance = float(np.mean(local_var))
    else:
        noise_variance = float(noise_variance)
    
    # 维纳滤波
    denoised = local_mean + (np.maximum(local_var - noise_variance, 0) / 
                             np.maximum(local_var, noise_variance)) * (gray - local_mean)
    
    # 转回
    if len(image.shape) == 3:
        denoised = (denoised * 255).astype(np.uint8)
        denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    else:
        if image.max() > 1.0:
            denoised = (denoised * 255).astype(image.dtype)
    
    return DenoiseResult(image=denoised, method="Wiener",
                        parameters={"kernel_size": kernel_size, "noise_variance": noise_variance},
                        noise_estimate=np.sqrt(noise_variance) if noise_variance > 0 else None)


# =============================================================================
# 通用接口
# =============================================================================

def denoise(image: np.ndarray, method: Union[str, DenoiseMethod] = "gaussian", **kwargs) -> DenoiseResult:
    """通用降噪接口"""
    if isinstance(method, str):
        method = DenoiseMethod.from_string(method)
    
    methods = {
        DenoiseMethod.BM3D: lambda: bm3d_denoise(image, **kwargs),
        DenoiseMethod.GAUSSIAN: lambda: gaussian_denoise(image, **kwargs),
        DenoiseMethod.MEDIAN: lambda: median_denoise(image, **kwargs),
        DenoiseMethod.NLMEANS: lambda: nlmeans_denoise(image, **kwargs),
        DenoiseMethod.BILATERAL: lambda: bilateral_denoise(image, **kwargs),
        DenoiseMethod.WAVELET: lambda: wavelet_denoise(image, **kwargs),
        DenoiseMethod.TOTAL_VARIATION: lambda: total_variation_denoise(image, **kwargs),
        DenoiseMethod.WIENER: lambda: wiener_denoise(image, **kwargs),
    }
    
    if method not in methods:
        raise ValueError(f"不支持: {method}")
    return methods[method]()


def batch_denoise(images: List[np.ndarray], method: Union[str, DenoiseMethod] = "gaussian", **kwargs) -> List[DenoiseResult]:
    """批量降噪"""
    return [denoise(img, method, **kwargs) for img in images]


def denoise_spot_image(spot_image, method: Union[str, DenoiseMethod] = "gaussian", **kwargs) -> DenoiseResult:
    """对SpotImage对象进行降噪"""
    result = denoise(spot_image.image_array, method, **kwargs)
    spot_image.meta_info['denoising'] = {
        'method': result.method,
        'parameters': result.parameters
    }
    return result


# =============================================================================
# 噪声估计
# =============================================================================

def estimate_noise_sigma(image: np.ndarray, method: str = "mad") -> float:
    """估计噪声标准差"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    if method == "mad":
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        mad = np.median(np.abs(laplacian - np.median(laplacian)))
        return float(mad) / 0.6745
    else:
        h, w = gray.shape
        margin = int(min(h, w) * 0.05)
        return float(np.std(gray[margin:h-margin, margin:w-margin]))


# =============================================================================
# 质量评估
# =============================================================================

def evaluate_denoising(original: Optional[np.ndarray], denoised: np.ndarray,
                       noisy: Optional[np.ndarray] = None) -> QualityMetrics:
    """评估降噪质量"""
    metrics = QualityMetrics()
    
    if original is not None:
        orig = original.astype(np.float32)
        deno = denoised.astype(np.float32)
        
        mse = np.mean((orig - deno) ** 2)
        if mse > 0:
            max_val = max(float(orig.max()), float(deno.max()))
            if max_val <= 1.0: max_val = 1.0
            metrics.psnr = float(20 * np.log10(max_val / np.sqrt(mse)))
        
        if SKIMAGE_AVAILABLE:
            if len(original.shape) == 3:
                metrics.ssim = float(ssim(original, denoised, channel_axis=2, data_range=255))
            else:
                metrics.ssim = float(ssim(original, denoised, data_range=255))
    
    return metrics


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'DenoiseMethod', 'NoiseEstimationMethod',
    'DenoiseResult', 'QualityMetrics',
    'bm3d_denoise', 'gaussian_denoise', 'median_denoise',
    'nlmeans_denoise', 'bilateral_denoise', 'wavelet_denoise',
    'total_variation_denoise', 'wiener_denoise',
    'denoise', 'batch_denoise',
    'estimate_noise_sigma', 'evaluate_denoising',
    'BM3D_AVAILABLE', 'PYWAVELETS_AVAILABLE', 'SKIMAGE_AVAILABLE',
]
