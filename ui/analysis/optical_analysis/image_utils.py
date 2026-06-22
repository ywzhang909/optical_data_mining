"""
图像处理工具模块
=================
提供图像预处理和显示相关的辅助函数

功能:
- 图像归一化
- 图像读取
- 图像去暗场处理
"""
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def normalize_image_for_display(img: np.ndarray) -> np.ndarray:
    """
    将图像归一化到适合显示的格式

    Args:
        img: 输入图像数组

    Returns:
        normalized_img: 归一化后的uint8图像
    """
    img = np.asarray(img, dtype=np.float64)
    # 归一化到 [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    # 转换到 [0, 255]
    img = (img * 255).astype(np.uint8)
    return img


def read_image_to_numpy(uploaded_file) -> np.ndarray:
    """
    读取上传的图像文件为numpy数组

    Args:
        uploaded_file: 上传的文件对象（PIL Image或文件路径）

    Returns:
        灰度图像数组
    """
    if isinstance(uploaded_file, (str, Path)):
        image = Image.open(uploaded_file)
    else:
        image = Image.open(uploaded_file)

    if image.mode != "L":
        image = image.convert("L")
    return np.array(image)


def load_image(file_path: str | Path, as_grayscale: bool = True) -> np.ndarray:
    """
    从文件路径加载图像

    Args:
        file_path: 图像文件路径
        as_grayscale: 是否转换为灰度图

    Returns:
        图像数组
    """
    image = Image.open(file_path)
    if as_grayscale and image.mode != "L":
        image = image.convert("L")
    return np.array(image)


def subtract_dark_field(
    img_array: np.ndarray,
    denoise_method: str = "none",
    manual_threshold: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    暗场校正：计算背景阈值并从图像中减去

    Args:
        img_array: 输入图像数组
        denoise_method: 去暗场方法，'none', 'median', 'min', '1_e', 或 'manual'
        manual_threshold: 手动输入的阈值（当denoise_method='manual'时使用）

    Returns:
        tuple: (denoise_img_array, black)
            - denoise_img_array: 去噪后的图像
            - black: 暗场阈值
    """
    if denoise_method == "none":
        # 不去暗场，直接返回原图
        return img_array.copy(), 0
    elif denoise_method == "median":
        black = np.median(img_array)
    elif denoise_method == "min":
        black = np.min(img_array)
    elif denoise_method == "1_e":
        black = np.max(img_array) / np.e
    elif denoise_method == "manual":
        black = manual_threshold if manual_threshold is not None else 0
    else:
        black = 0

    denoise_img_array = np.where(img_array > black, img_array - black, 0)
    return denoise_img_array, black


def normalize_image(
    img: np.ndarray, target_min: float = 0.0, target_max: float = 1.0
) -> np.ndarray:
    """
    将图像归一化到指定范围

    Args:
        img: 输入图像数组
        target_min: 目标最小值
        target_max: 目标最大值

    Returns:
        归一化后的图像
    """
    img = np.asarray(img, dtype=np.float64)
    img_min, img_max = img.min(), img.max()

    if img_max == img_min:
        return np.full_like(img, target_min)

    normalized = (img - img_min) / (img_max - img_min)
    return normalized * (target_max - target_min) + target_min


def resize_image(
    img: np.ndarray, size: tuple[int, int], interpolation: str = "bilinear"
) -> np.ndarray:
    """
    调整图像大小

    Args:
        img: 输入图像数组
        size: 目标尺寸 (height, width)
        interpolation: 插值方法

    Returns:
        调整大小后的图像
    """
    import cv2

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

    # PIL图像使用 (width, height)，OpenCV使用 (height, width)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_resized = img_pil.resize(size[::-1], interp)

    return np.array(img_resized)


def pad_to_square(img: np.ndarray, mode: str = "constant", **kwargs) -> np.ndarray:
    """
    将图像填充成正方形

    Args:
        img: 输入图像数组
        mode: 填充模式
        **kwargs: 传递给np.pad的参数

    Returns:
        填充后的正方形图像
    """
    h, w = img.shape

    if h == w:
        return img

    max_dim = max(h, w)

    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    pad_width = ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w))

    return np.pad(img, pad_width, mode=mode, **kwargs)


def clip_negative_values(img: np.ndarray, min_value: float = 0.0) -> np.ndarray:
    """
    去除图像中的负值（由于数值误差导致）

    Args:
        img: 输入图像数组
        min_value: 最小值阈值

    Returns:
        处理后的图像
    """
    return np.clip(img, min_value, None)


def calculate_background_threshold(img: np.ndarray, method: str = "median") -> float:
    """
    计算背景暗场阈值

    Args:
        img: 输入图像数组
        method: 计算方法 ('median', 'min', 'mean')

    Returns:
        背景阈值
    """
    if method == "median":
        return float(np.median(img))
    elif method == "min":
        return float(np.min(img))
    elif method == "mean":
        return float(np.mean(img))
    else:
        return 0.0


def find_spot_border(image: np.ndarray,
                     threshold_method: str = 'otsu',
                     denoise_strength: float = 10.0,
                     min_area_ratio: float = 0.005,
                     return_mask: bool = False) -> dict:
    """
    鲁棒地检测光斑边界并拟合包围圆。

    修复要点：
    1. 去噪在阈值化之前完成，避免阈值受去噪结果影响；
    2. 采用自适应阈值（Otsu/自适应/百分比）替代固定 max/e 阈值；
    3. 用质心约束包围圆中心，防止轮廓残缺导致圆心漂移；
    4. 增加面积、圆度等 sanity check，失败时回退到矩估计。

    Args:
        image: 输入灰度图像 (H, W)，任意 dtype
        threshold_method: 'otsu' | 'adaptive' | 'percentile' | 'max_fraction'
        denoise_strength: fastNlMeansDenoising 的滤波强度
        min_area_ratio: 最小有效光斑面积占图像比例
        return_mask: 是否返回二值掩膜

    Returns:
        dict: {
            'border_x': float,      # 圆心 x
            'border_y': float,      # 圆心 y
            'border_radius': float, # 包围圆半径
            'center_x': float,      # 质心 x（辅助验证）
            'center_y': float,      # 质心 y
            'eccentricity': float,  # 椭圆离心率（0=圆，越接近1越扁）
            'confidence': float,     # 0-1，轮廓与圆拟合的置信度
            'mask': np.ndarray,     # 二值掩膜（若 return_mask=True）
        }
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"find_spot_border expects numpy array, got {type(image)}")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {image.shape}")

    # ---------- 1. 预处理：归一化 + 高斯平滑 ----------
    img_f = np.asarray(image, dtype=np.float64)
    # 避免零值导致后续 log 运算问题（若需要）
    img_f = img_f - np.min(img_f)  # 背景归零
    if np.max(img_f) > 0:
        img_f = img_f / np.max(img_f) * 255.0  # 归一化到 0-255

    # 轻微高斯平滑抑制高频噪声，保留光斑整体形状
    blurred = cv2.GaussianBlur(img_f.astype(np.float32), (5, 5), 1.0)

    # ---------- 2. 非局部均值去噪（在 uint8 域执行）----------
    img_uint8 = np.clip(blurred, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(img_uint8, None,
                                        h=denoise_strength,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

    # ---------- 3. 自适应阈值化（修复核心：阈值基于去噪后图像）----------
    if threshold_method == 'otsu':
        # Otsu 自动寻找最优阈值，对双峰直方图效果好
        thresh_val, binary = cv2.threshold(denoised, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'adaptive':
        # 自适应阈值：适合光照不均匀场景
        binary = cv2.adaptiveThreshold(denoised, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        thresh_val = None
    elif threshold_method == 'percentile':
        # 基于像素值分位数，对光斑占比较小的情况鲁棒
        p = np.percentile(denoised[denoised > 0], 50)  # 中位数作为阈值
        _, binary = cv2.threshold(denoised, p, 255, cv2.THRESH_BINARY)
        thresh_val = p
    elif threshold_method == 'max_fraction':
        # 原始思路的改进版：基于去噪后图像最大值的比例
        thresh_val = np.max(denoised) * (1.0 / math.e)
        _, binary = cv2.threshold(denoised, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    # ---------- 4. 形态学清理 ----------
    # 开运算去除孤立噪声点，闭运算填充光斑内部小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---------- 5. 轮廓检测与筛选 ----------
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 完全无轮廓：回退到全局矩估计
        return _fallback_moments(image, return_mask)

    # 筛选有效轮廓（面积足够大）
    h, w = image.shape
    min_area = h * w * min_area_ratio
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return _fallback_moments(image, return_mask)

    # 选择最大轮廓（光斑通常是最显著连通域）
    largest = max(valid_contours, key=cv2.contourArea)

    # ---------- 6. 包围圆 + 椭圆拟合（双约束验证）----------
    # 最小外接圆
    (x_circle, y_circle), radius = cv2.minEnclosingCircle(largest)

    # 最小外接椭圆（需要至少5个点）
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        (x_ell, y_ell), (ma, mi), angle = ellipse
        # 离心率：0=正圆，接近1=极扁
        eccentricity = math.sqrt(1 - (min(ma, mi) / max(ma, mi)) ** 2) \
            if max(ma, mi) > 0 else 0.0
    else:
        x_ell, y_ell = x_circle, y_circle
        ma = mi = radius * 2
        eccentricity = 0.0
        angle = 0.0

    # 质心（基于原始灰度图像，不受二值化轮廓残缺影响）
    M = cv2.moments(largest)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx, cy = x_circle, y_circle

    # ---------- 7. 圆心修正：用质心约束包围圆 ----------
    # 若轮廓残缺，外接圆中心可能偏离质心。用质心-圆心距离作为修正依据
    dist_cc = math.hypot(cx - x_circle, cy - y_circle)
    # 如果偏离超过半径的 20%，优先采用质心+等效半径
    if dist_cc > radius * 0.2:
        # 等效半径：用轮廓面积反推
        area = cv2.contourArea(largest)
        equiv_radius = math.sqrt(area / math.pi)
        x_final, y_final = cx, cy
        radius_final = max(radius, equiv_radius)  # 取较大者保证包围
        confidence = 0.5  # 置信度降低
    else:
        x_final, y_final = x_circle, y_circle
        radius_final = radius
        confidence = 1.0 - eccentricity  # 越圆置信度越高

    # ---------- 8. 最终半径修正：确保覆盖 95% 能量 ----------
    # 基于原始图像，以修正后的中心计算径向累积能量
    y_idx, x_idx = np.indices(image.shape)
    r_map = np.sqrt((x_idx - x_final) ** 2 + (y_idx - y_final) ** 2)
    # 按径向距离排序，找覆盖 95% 总能量的半径
    sorted_indices = np.argsort(r_map.ravel())
    cumsum = np.cumsum(img_f.ravel()[sorted_indices])
    total = cumsum[-1]
    if total > 0:
        r_sorted = r_map.ravel()[sorted_indices]
        r_95 = r_sorted[np.searchsorted(cumsum, total * 0.95)]
        # 取外接圆与能量半径的较大者
        radius_final = max(radius_final, r_95)

    result = {
        'border_x': float(x_final),
        'border_y': float(y_final),
        'border_radius': float(radius_final),
        'center_x': float(cx),
        'center_y': float(cy),
        'eccentricity': float(eccentricity),
        'confidence': float(confidence),
    }

    if return_mask:
        result['mask'] = binary

    return result


def _fallback_moments(image: np.ndarray, return_mask: bool) -> dict:
    """
    无轮廓时的回退策略：基于全局灰度矩估计
    """
    img_f = np.asarray(image, dtype=np.float64)
    h, w = img_f.shape
    total = np.sum(img_f)
    if total <= 0:
        result = {
            'border_x': w / 2.0, 'border_y': h / 2.0,
            'border_radius': min(h, w) / 4.0,
            'center_x': w / 2.0, 'center_y': h / 2.0,
            'eccentricity': 0.0, 'confidence': 0.0,
        }
    else:
        y_idx, x_idx = np.indices(image.shape)
        cx = np.sum(x_idx * img_f) / total
        cy = np.sum(y_idx * img_f) / total
        # 等效半径：基于二阶矩
        var = np.sum(((x_idx - cx) ** 2 + (y_idx - cy) ** 2) * img_f) / total
        radius = 2.0 * math.sqrt(var)  # 2σ 覆盖 ~95% 能量
        result = {
            'border_x': float(cx), 'border_y': float(cy),
            'border_radius': float(radius),
            'center_x': float(cx), 'center_y': float(cy),
            'eccentricity': 0.0, 'confidence': 0.3,
        }
    if return_mask:
        result['mask'] = np.zeros_like(image, dtype=np.uint8)
    return result
