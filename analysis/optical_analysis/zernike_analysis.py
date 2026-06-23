"""
Zernike 分解模块
================

基于包围圆区域对光瞳进行 Zernike 多项式展开：
- 将圆内光强归一化后投影到 Zernike 基函数
- 支持可配置最大阶数（Noll 归一化）
- 返回阶数映射表、各阶系数、波前重建图
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _zernike_norm_factor(n: int, m: int) -> float:
    """Zernike 归一化系数，使径向多项式的最大值（ρ=1处）为1。"""
    if m == 0:
        return n + 1
    return 2.0 * (n + 1)


def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """径向 Zernike 多项式 R_n^m(ρ)。"""
    if (n - m) % 2 != 0 or m > n:
        return np.zeros_like(rho)
    k = (n - m) // 2
    if m == 0:
        result = 0.0
        for s in range(k + 1):
            coeff = ((-1) ** s * math.factorial(n - s)) / (
                math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)
            )
            result += coeff * rho ** (n - 2 * s)
        return result
    result = 0.0
    for s in range(k):
        coeff = ((-1) ** s * (n - 2 * s)) * math.factorial(n - s) / (
            math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)
        )
        result += coeff * rho ** (n - 2 * s - 1)
    return result * rho


def zernike_polar(n: int, m: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """极坐标形式的 Zernike 单项式 Z_n^m(ρ, φ)。"""
    if m == 0:
        R = zernike_radial(n, 0, rho)
        return R * np.sqrt(n + 1)
    if m > 0:
        R = zernike_radial(n, m, rho)
        return R * np.sqrt(2.0 * (n + 1)) * np.cos(m * phi)
    R = zernike_radial(n, -m, rho)
    return R * np.sqrt(2.0 * (n + 1)) * np.sin(abs(m) * phi)


def make_zernike_grid(n: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 n 阶以内的 Zernike 基函数集合。

    Returns:
        rho: 归一化径向坐标 (0~1)，shape=(size, size)
        phi: 极角 (rad)
        mask: 单位圆内 bool 掩膜
        zernikes: list of ndarray，每个元素是一个 base
    """
    y, x = np.mgrid[:size, :size].astype(float)
    cx, cy = (size - 1.0) / 2.0, (size - 1.0) / 2.0
    rho = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2.0)
    phi = np.arctan2(y - cy, x - cx)
    mask = rho <= 1.0

    zernikes = []
    n_idx = 0
    for ni in range(n + 1):
        for mi in range(-ni, ni + 1, 2):
            zernikes.append(zernike_polar(ni, mi, rho, phi))
            n_idx += 1
    return rho, phi, mask, zernikes


def fit_zernike(
    img: np.ndarray,
    max_order: int = 4,
    pupil_radius_px: Optional[float] = None,
) -> dict:
    """
    对归一化光强图（已在包围圆内）拟合 Zernike 多项式。

    Args:
        img: 去暗场后的光强图 (2D array)
        max_order: 最大径向阶数 n（0: piston, 1: tilt/tip, 2: defocus, ...）
        pupil_radius_px: 包围圆半径（像素）。若为 None 则自动取短边/2 为中心裁剪。

    Returns:
        {
            "coeffs": np.ndarray,      # 各 Zernike 基的线性系数
            "noll_indices": list[int], # 对应 Noll 序号
            "reconstructed": np.ndarray, # 波前重建图
            "residual": np.ndarray,    # 残差
            "rmse": float,
            "mask": np.ndarray,        # 使用的 bool 掩膜
            "n_terms": int,
        }
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("img must be 2D")

    h, w = img.shape
    if pupil_radius_px is None:
        side = min(h, w)
        pupil_radius_px = side / 2.0

    # 居中裁剪到 square，边长 = 2 * radius
    side = int(round(2.0 * pupil_radius_px))
    if side < 4:
        raise ValueError(f"pupil_radius_px too small: {pupil_radius_px}")
    y0 = int(max(0, (h - side) / 2))
    x0 = int(max(0, (w - side) / 2))
    y1, x1 = y0 + side, x0 + side
    patch = img[y0:y1, x0:x1]

    flat = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    sum_intensity = flat.sum()
    if sum_intensity <= 0:
        raise ValueError("Pupil region contains no signal")

    pupil = flat / sum_intensity  # 归一化到面积和=1

    rho, phi, mask, zernikes = make_zernike_grid(max(4, max_order) + 1, side)
    valid = mask

    y_arr = pupil[valid].copy()
    n_terms = len(zernikes)
    A = np.column_stack([Z[valid].ravel() for Z in zernikes])

    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_arr, rcond=None)
    reconstructed = np.zeros((side, side), dtype=np.float64)
    for i, Z in enumerate(zernikes):
        reconstructed += coeffs[i] * Z
    reconstructed_inside = reconstructed[valid]

    rmse = float(np.sqrt(np.mean((y_arr - np.dot(A, coeffs)) ** 2)))
    noll_map = []
    n_idx = 0
    for ni in range(max_order + 1):
        for mi in range(-ni, ni + 1, 2):
            j = n_idx if n_idx < len(coeffs) else None
            noll_map.append(j)
            n_idx += 1

    return {
        "coeffs": coeffs,
        "noll_map": noll_map,
        "reconstructed": reconstructed,
        "residual": pupil - reconstructed,
        "rmse": rmse,
        "mask": mask,
        "n_terms": n_terms,
        "side": side,
        "y0": y0, "x0": x0, "y1": y1, "x1": x1,
    }


def zernike_order_label(order: int) -> str:
    """返回 Zernike 单项式的可读标签（按照 Noll 归一化常见标签）。"""
    labels = {
        0: "Piston (n=0)",
        1: "Tilt X (n=1)",
        2: "Tilt Y (n=1)",
        3: "Defocus (n=2)",
        4: "Astigmatism 45° (n=2)",
        5: "Astigmatism 0° (n=2)",
        6: "Coma X (n=3)",
        7: "Coma Y (n=3)",
        8: "Trefoil 0° (n=3)",
        9: "Spherical (n=4)",
    }
    if order in labels:
        return labels[order]
    n = ((order - 1 - ((order - 2) % 2)) // 2) + 1
    return f"n={n}, term {order}"


def recommend_zernike_order(pupil_diameter_px: float) -> int:
    """根据光瞳直径推荐最大 Zernike 阶数。经验公式：n ≈ 直径/24 + 3，上限 10。"""
    return int(np.clip(round(pupil_diameter_px / 24.0 + 3), 4, 10))
