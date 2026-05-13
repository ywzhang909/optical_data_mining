"""
图像处理工具模块
=================
提供图像预处理和显示相关的辅助函数

功能:
- 图像归一化
- 图像读取
- 图像去暗场处理
"""

from pathlib import Path

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
    img_array: np.ndarray, denoise_method: str = "none", manual_threshold: float | None = None
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


def normalize_image(img: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
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


def resize_image(img: np.ndarray, size: tuple[int, int], interpolation: str = "bilinear") -> np.ndarray:
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
