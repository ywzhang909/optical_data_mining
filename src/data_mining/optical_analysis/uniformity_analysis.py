"""
光斑均匀度分析模块
===================

基于包围圆物理区域，计算光斑均匀度相关指标：
- 圆内强度统计：均值、标准差、RMS、变异系数
- XY 截面 1D 轮廓与圆内均值线对比
- 平顶函数（Flat-top）一维拟合，返回顶部坪台强度、边缘强度、Σ残差等
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import cv2
import numpy as np
from scipy.optimize import curve_fit


UniformityMetrics = Dict[str, Any]


def calculate_uniformity_metrics(
    image: np.ndarray,
    border: Dict[str, float],
) -> UniformityMetrics:
    """
    基于包围圆物理区域，计算光斑均匀度指标。

    Args:
        image: 二维灰度图像数组
        border: 包围圆字典，包含以下 key：
            - border_x / border_y: 圆心坐标（像素）
            - border_radius: 半径（像素）

    Returns:
        字典键说明:
            - cx, cy, radius: 圆几何参数
            - mean_intensity, std_intensity: 圆内强度均值与标准差
            - rms_uniformity: 圆内 RMS（标准差 / 均值）
            - pv: (max - min) / mean
            - flat_top_fit_x / flat_top_fit_y: XY 一维平顶拟合参数字典
                - center, radius, top, edge, residuals_sum, popt
            - profile_x, profile_y: 过圆心的截线强度序列
            - mask: 圆内掩膜（与 image shape 相同，bool 类型）
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"image 必须是二维灰度图，当前 shape={img.shape}")

    def _to_float(name: str) -> float:
        value = border.get(name, np.nan)
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        raise TypeError(f"border['{name}'] 必须是数值，当前={value!r}")

    cx = _to_float("border_x")
    cy = _to_float("border_y")
    radius = _to_float("border_radius")

    h, w = img.shape
    invalid = math.isnan(cx) or math.isnan(cy) or math.isnan(radius) or radius <= 0

    if invalid:
        return {
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "mean_intensity": np.nan,
            "std_intensity": np.nan,
            "rms_uniformity": np.nan,
            "pv": np.nan,
            "flat_top_fit_x": None,
            "flat_top_fit_y": None,
            "profile_x": None,
            "profile_y": None,
            "mask": np.zeros((h, w), dtype=bool),
        }

    y_idx, x_idx = np.indices(img.shape)
    mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= radius**2

    values = img[mask]
    mean_intensity = float(np.mean(values)) if values.size > 0 else np.nan
    std_intensity = float(np.std(values)) if values.size > 0 else np.nan

    if mean_intensity not in (0, np.nan):
        rms_uniformity = std_intensity / mean_intensity
        pv = (float(np.max(values)) - float(np.min(values))) / mean_intensity
    else:
        rms_uniformity = np.nan
        pv = np.nan

    profile_x = img[int(round(cy)), :]
    profile_y = img[:, int(round(cx))]

    flat_top_fit_x = _fit_flatted_top_1d(np.arange(w), profile_x, cx, radius)
    flat_top_fit_y = _fit_flatted_top_1d(np.arange(h), profile_y, cy, radius)

    return {
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "rms_uniformity": rms_uniformity,
        "pv": pv,
        "flat_top_fit_x": flat_top_fit_x,
        "flat_top_fit_y": flat_top_fit_y,
        "profile_x": profile_x,
        "profile_y": profile_y,
        "mask": mask,
    }


def plot_uniformity_analysis(
    image: np.ndarray,
    border: Dict[str, float],
    metrics: UniformityMetrics,
    pixel_size_um: float,
):
    """
    生成 1×3 均匀度分析图（Matplotlib）：(1) 原图+圆 (2) XY截面 (3) 平顶拟合对比
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle

    img = np.asarray(image, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cx = metrics["cx"]
    cy = metrics["cy"]
    radius = metrics["radius"]

    # 1) origin + border
    ax = axes[0]
    im = ax.imshow(img, cmap="hot", interpolation="bilinear")
    if not (math.isnan(cx) or math.isnan(cy) or math.isnan(radius) or radius <= 0):
        c = MplCircle(
            (cx, cy), radius, fill=False, color="yellow", linewidth=2, linestyle="-."
        )
        ax.add_patch(c)
    ax.set_title("Bounding circle")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 2) XY profile
    ax = axes[1]
    profile_x = metrics["profile_x"]
    profile_y = metrics["profile_y"]
    mean_intensity = metrics.get("mean_intensity", np.nan)
    if isinstance(profile_x, np.ndarray):
        ax.plot(
            np.arange(profile_x.shape[0]),
            profile_x,
            label="X profile",
            color="tab:blue",
        )
    if isinstance(profile_y, np.ndarray):
        ax.plot(
            np.arange(profile_y.shape[0]),
            profile_y,
            label="Y profile",
            color="tab:green",
        )
    if isinstance(mean_intensity, (int, float, np.floating)) and not math.isnan(
        mean_intensity
    ):
        ax.axhline(
            float(mean_intensity), color="gray", linestyle="--", label="Circle mean"
        )
    ax.set_title("XY cross-section")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) flat-top fit
    ax = axes[2]
    flat_top_fit_x = metrics.get("flat_top_fit_x")
    flat_top_fit_y = metrics.get("flat_top_fit_y")
    used = False
    for info, color, label in (
        (flat_top_fit_x, "tab:blue", "X fit"),
        (flat_top_fit_y, "tab:green", "Y fit"),
    ):
        if info is None:
            continue
        x = np.arange(info["length"])
        y_fit = np.where(
            np.abs(x - info["center"]) <= info["radius"], info["top"], info["edge"]
        )
        ax.scatter(x, info["y"], s=4, color=color, alpha=0.35)
        ax.plot(x, y_fit, color=color, linewidth=1.2, label=label)
        used = True
    if not used:
        ax.text(
            0.5,
            0.5,
            "No valid flat-top fit",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set_title("Flat-top fit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatted_top_func(
    x: np.ndarray, center: float, radius: float, top: float, edge: float
) -> np.ndarray:
    z = np.abs(x - center)
    return np.where(z <= radius, top, edge)


def _fit_flatted_top_1d(
    x: np.ndarray,
    y: np.ndarray,
    center: float,
    radius: float,
) -> Optional[Dict[str, Any]]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    mask = np.isfinite(y)
    if mask.sum() < 4:
        return None

    top_est = float(np.mean(y[mask]))
    edge_est = float(np.min(y))

    def loss(p: np.ndarray) -> float:
        r = _flatted_top_func(x, center, radius, p[0], p[1]) - y
        return float(np.sum(r[mask] ** 2))

    try:
        grid = np.linspace(float(np.min(y)) * 0.5, float(np.max(y)) * 1.5, 81)
        best: Optional[np.ndarray] = None
        best_loss = math.inf
        for top in grid:
            for edge in grid:
                l = loss(np.array([top, edge]))
                if l < best_loss:
                    best_loss = l
                    best = np.array([top, edge])
        if best is None:
            raise RuntimeError("no valid initial guess in grid search")

        popt, _ = curve_fit(
            lambda xv, top, edge: _flatted_top_func(xv, center, radius, top, edge),
            x[mask],
            y[mask],
            p0=best,
            maxfev=2000,
        )
        y_pred = _flatted_top_func(x, center, radius, float(popt[0]), float(popt[1]))
        return {
            "center": float(center),
            "radius": float(radius),
            "top": float(popt[0]),
            "edge": float(popt[1]),
            "residuals_sum": float(np.sum((y - y_pred) ** 2)),
            "popt": popt,
            "x": x,
            "y": y,
            "length": int(x.shape[0]),
        }
    except Exception:
        return None
