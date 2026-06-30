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
from typing import Any

import numpy as np

UniformityMetrics = dict[str, Any]


def calculate_uniformity_metrics(
    image: np.ndarray,
    border: dict[str, float],
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
            "energy_uniformity": np.nan,
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
        # 能量均匀度 γ = 1 - RMS / mean  (基于论文定义)
        energy_uniformity = 1.0 - std_intensity / mean_intensity
    else:
        rms_uniformity = np.nan
        pv = np.nan
        energy_uniformity = np.nan

    # X profile (horizontal through center, only within bounding circle)
    x_start = max(0, int(np.floor(cx - radius)))
    x_end = min(w, int(np.ceil(cx + radius)) + 1)
    profile_x = img[int(round(cy)), x_start:x_end]
    profile_x_idx = np.arange(x_start, x_end)

    # Y profile (vertical through center, only within bounding circle)
    y_start = max(0, int(np.floor(cy - radius)))
    y_end = min(h, int(np.ceil(cy + radius)) + 1)
    profile_y = img[y_start:y_end, int(round(cx))]
    profile_y_idx = np.arange(y_start, y_end)

    # RMS non-uniformity for each cross-section
    profile_x = np.asarray(profile_x, dtype=np.float64)
    profile_y = np.asarray(profile_y, dtype=np.float64)
    mu_x = float(np.mean(profile_x)) if profile_x.size > 0 else np.nan
    mu_y = float(np.mean(profile_y)) if profile_y.size > 0 else np.nan
    rms_x = float(np.std(profile_x)) / mu_x if (profile_x.size > 0 and mu_x > 0) else np.nan
    rms_y = float(np.std(profile_y)) / mu_y if (profile_y.size > 0 and mu_y > 0) else np.nan

    return {
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "rms_uniformity": rms_uniformity,
        "rms_x": rms_x,
        "rms_y": rms_y,
        "pv": pv,
        "energy_uniformity": energy_uniformity,
        "profile_x": profile_x,
        "profile_y": profile_y,
        "profile_x_idx": profile_x_idx,
        "profile_y_idx": profile_y_idx,
        "mask": mask,
    }


def plot_uniformity_analysis(
    image: np.ndarray,
    border: dict[str, float],
    metrics: UniformityMetrics,
    pixel_size_um: float,
):
    """
    生成 1×2 均匀度分析图（Matplotlib）：(1) 原图+包围圆 (2) XY截面
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle

    img = np.asarray(image, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cx = metrics["cx"]
    cy = metrics["cy"]
    radius = metrics["radius"]

    # 1) origin + border (含圆心/半径标注)
    ax = axes[0]
    im = ax.imshow(img, cmap="hot", interpolation="bilinear")
    if not (math.isnan(cx) or math.isnan(cy) or math.isnan(radius) or radius <= 0):
        ax.add_patch(MplCircle(
            (cx, cy), radius, fill=False, color="yellow", linewidth=2.5, linestyle="-.",
        ))
        ax.plot(cx, cy, "w+", markersize=12, markeredgewidth=2, label=f"圆心 ({cx:.0f}, {cy:.0f})")
        ax.legend(fontsize=9, loc="upper right")
        # 在图内添加文字框显示圆心与半径
        ax.text(0.95, 0.05,
                f"Center: ({cx:.1f}, {cy:.1f})\nRadius: {radius:.1f} px",
                transform=ax.transAxes, fontsize=9,
                color="white", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
    ax.set_title(f"Bounding circle ({radius:.1f} px)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 2) XY profile (only within bounding circle)
    ax = axes[1]
    profile_x = metrics["profile_x"]
    profile_y = metrics["profile_y"]
    profile_x_idx = metrics.get("profile_x_idx")
    profile_y_idx = metrics.get("profile_y_idx")
    mean_intensity = metrics.get("mean_intensity", np.nan)
    if isinstance(profile_x, np.ndarray) and isinstance(profile_x_idx, np.ndarray):
        ax.plot(profile_x_idx, profile_x, label="X profile", color="tab:blue")
    elif isinstance(profile_x, np.ndarray):
        ax.plot(np.arange(profile_x.shape[0]), profile_x, label="X profile", color="tab:blue")
    if isinstance(profile_y, np.ndarray) and isinstance(profile_y_idx, np.ndarray):
        ax.plot(profile_y_idx, profile_y, label="Y profile", color="tab:green")
    elif isinstance(profile_y, np.ndarray):
        ax.plot(np.arange(profile_y.shape[0]), profile_y, label="Y profile", color="tab:green")
    if isinstance(mean_intensity, (int, float, np.floating)) and not math.isnan(
        mean_intensity
    ):
        ax.axhline(
            float(mean_intensity), color="gray", linestyle="--", label="Circle mean"
        )
    # Annotate cross-section RMS non-uniformity
    rms_x = metrics.get("rms_x", np.nan)
    rms_y = metrics.get("rms_y", np.nan)
    rms_label = []
    if isinstance(rms_x, (int, float)) and not math.isnan(rms_x):
        rms_label.append(f"X RMS: {rms_x:.4f}")
    if isinstance(rms_y, (int, float)) and not math.isnan(rms_y):
        rms_label.append(f"Y RMS: {rms_y:.4f}")
    if rms_label:
        ax.text(0.02, 0.98, "  ".join(rms_label), transform=ax.transAxes,
                fontsize=9, color="black", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_title("XY cross-section (within circle)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ===========================================================================
# 基于 FTL 模型的均匀度分析 (文档 §3–§7)
# ===========================================================================


def iterative_weighted_centroid(
    image: np.ndarray,
    max_iter: int = 5,
    tol: float = 0.01,
    initial_center: tuple[float, float] | None = None,
) -> dict[str, float]:
    """
    迭代加权质心法精确定位光斑中心 (§3.1)。

    每次迭代以当前中心为基准施加高斯权重，抑制边缘噪声对质心的影响。

    Args:
        image: 归一化光强图 (峰值=1)
        max_iter: 最大迭代次数
        tol: 收敛判据 (像素)
        initial_center: 初始中心 (cx, cy)，默认图像几何中心

    Returns:
        {"cx": float, "cy": float, "n_iter": int, "converged": bool}
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    y_idx, x_idx = np.indices(img.shape)

    if initial_center is not None:
        cx, cy = float(initial_center[0]), float(initial_center[1])
    else:
        cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0

    rmax = max(h, w) * 0.5  # 衰减尺度
    delta = math.inf

    for i in range(max_iter):
        r2 = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
        wgt = np.exp(-r2 / (0.5 * rmax) ** 2)
        weighted = img * wgt
        total = weighted.sum()
        if total <= 0:
            break
        cx_new = float(np.sum(x_idx * weighted) / total)
        cy_new = float(np.sum(y_idx * weighted) / total)
        delta = math.hypot(cx_new - cx, cy_new - cy)
        cx, cy = cx_new, cy_new
        if delta < tol:
            return {"cx": cx, "cy": cy, "n_iter": i + 1, "converged": True}

    return {"cx": cx, "cy": cy, "n_iter": max_iter, "converged": delta < tol}


def compute_uniformity_zones(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> dict[str, dict[str, float]]:
    """
    将包围圆划分为三个子区域并计算各区域统计量 (§5.2)。

    区域划分:
        - 中心区: R < 0.3 * radius
        - 平顶区: 0.3*radius ≤ R < 0.7*radius  (均匀度核心评估区)
        - 边缘区: R ≥ 0.7*radius

    Returns:
        {
            "center":  {"mean": ..., "std": ..., "count": ...},
            "flat":    {"mean": ..., "std": ..., "count": ...,
                        "cv": ..., "pv": ...},
            "edge":    {"mean": ..., "std": ..., "count": ...},
        }
    """
    img = np.asarray(image, dtype=np.float64)
    y_idx, x_idx = np.indices(img.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mask = r <= radius

    def _stats(m: np.ndarray) -> dict[str, float]:
        vals = img[m]
        n = vals.size
        if n == 0:
            return {"mean": np.nan, "std": np.nan, "count": 0}
        mu = float(np.mean(vals))
        s = float(np.std(vals))
        return {
            "mean": mu,
            "std": s,
            "cv": s / mu if mu > 0 else np.nan,
            "pv": (float(vals.max()) - float(vals.min())) / mu if mu > 0 else np.nan,
            "count": n,
        }

    zone_center = (r < 0.3 * radius) & mask
    zone_flat = (r >= 0.3 * radius) & (r < 0.7 * radius) & mask
    zone_edge = (r >= 0.7 * radius) & mask

    return {
        "center": _stats(zone_center),
        "flat": _stats(zone_flat),
        "edge": _stats(zone_edge),
    }


def compute_tolerance_pass_rates(
    image: np.ndarray,
    ftl_result: dict,
    cx: float,
    cy: float,
    radius: float,
) -> dict[str, float]:
    """
    基于 FTL 模型计算各容差阈值下的达标率 (§5.3)。

    对包围圆内的每个像素，计算相对残差 ε = (I - I_FTL) / I_FTL，
    统计 |ε| 低于 ±2%, ±5%, ±10% 的像素比例。

    Args:
        image: 去暗场后的原始光强图
        ftl_result: fit_flat_topped_lorentz 的返回结果
        cx, cy, radius: 包围圆参数

    Returns:
        {"P_2%": ..., "P_5%": ..., "P_10%": ..., "sigma_epsilon": ...,
         "mean_epsilon": ..., "max_epsilon": ...}
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    y_idx, x_idx = np.indices(img.shape)
    rho = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mask = rho <= radius

    I0 = ftl_result.get("I0", np.nan)
    R_FL = ftl_result.get("R_FL_pixels", np.nan)
    q = ftl_result.get("q", np.nan)

    if any(np.isnan(v) for v in (I0, R_FL, q)) or R_FL <= 0:
        return {
            "P_2%": np.nan,
            "P_5%": np.nan,
            "P_10%": np.nan,
            "sigma_epsilon": np.nan,
            "mean_epsilon": np.nan,
            "max_epsilon": np.nan,
        }

    # 逐像素重建 FTL 模型
    R_phys = rho[mask]
    I_meas = img[mask]
    # FTL 模型: I0 / (1 + (R/R_FL)^q)^(1 + 2/q)
    I_ftl = I0 / (1.0 + (R_phys / R_FL) ** q) ** (1.0 + 2.0 / q)

    # 相对残差 (加小量防除零)
    eps = (I_meas - I_ftl) / (I_ftl + 1e-12)

    n = eps.size
    p2 = float(np.mean(np.abs(eps) < 0.02))
    p5 = float(np.mean(np.abs(eps) < 0.05))
    p10 = float(np.mean(np.abs(eps) < 0.10))

    return {
        "P_2%": p2,
        "P_5%": p5,
        "P_10%": p10,
        "sigma_epsilon": float(np.std(eps)),
        "mean_epsilon": float(np.mean(eps)),
        "max_epsilon": float(np.max(np.abs(eps))),
    }


def compute_comprehensive_rating(p5: float, sigma_eps: float) -> str:
    """
    综合评级 (§5.3)。

    Args:
        p5: P_5% 容差达标率
        sigma_eps: 相对残差标准差

    Returns:
        "EXCELLENT" / "GOOD" / "FAIR" / "POOR" / "N/A"
    """
    if np.isnan(p5) or np.isnan(sigma_eps):
        return "N/A"
    if p5 > 0.95 and sigma_eps < 0.03:
        return "EXCELLENT"
    if p5 > 0.85 and sigma_eps < 0.05:
        return "GOOD"
    if p5 > 0.70 and sigma_eps < 0.08:
        return "FAIR"
    return "POOR"


def compute_spatial_uniformity_2d(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> dict[str, float]:
    """
    二维空间均匀度直接统计指标 (§7.1)，不依赖模型。

    Returns:
        {
            "CV_global": ...,    # 全局变异系数
            "CV_flat": ...,      # 平顶区变异系数
            "NU": ...,           # 非均匀度 (max-min)/(max+min)
            "modulation": ...,   # 平顶区调制深度
            "RMSD": ...,         # 均方根偏差
            "NRMSD": ...,        # 归一化 RMSD
        }
    """
    img = np.asarray(image, dtype=np.float64)
    y_idx, x_idx = np.indices(img.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mask = r <= radius
    flat_mask = (r >= 0.3 * radius) & (r < 0.7 * radius) & mask

    vals = img[mask]
    flat_vals = img[flat_mask]

    if vals.size == 0:
        return {k: np.nan for k in ("CV_global", "CV_flat", "NU", "modulation", "RMSD", "NRMSD")}

    mu = float(np.mean(vals))
    std = float(np.std(vals))
    cv_g = std / mu if mu > 0 else np.nan

    if flat_vals.size > 0:
        mu_f = float(np.mean(flat_vals))
        std_f = float(np.std(flat_vals))
        cv_f = std_f / mu_f if mu_f > 0 else np.nan
        mod = (float(flat_vals.max()) - float(flat_vals.min())) / (float(flat_vals.max()) + float(flat_vals.min()))
    else:
        mu_f, cv_f, mod = np.nan, np.nan, np.nan

    i_max, i_min = float(vals.max()), float(vals.min())
    nu = (i_max - i_min) / (i_max + i_min) if (i_max + i_min) > 0 else np.nan
    rmsd = float(np.sqrt(np.mean((vals - mu) ** 2)))
    nrmsd = rmsd / mu if mu > 0 else np.nan

    return {
        "CV_global": cv_g,
        "CV_flat": cv_f,
        "NU": nu,
        "modulation": mod,
        "RMSD": rmsd,
        "NRMSD": nrmsd,
    }


def detect_local_defects(
    residual: np.ndarray,
    mask: np.ndarray,
    n_sigma: float = 3.0,
    min_area_px: int = 5,
) -> dict[str, Any]:
    """
    局部热点/暗点检测 (§7.2)。

    基于残差图，通过连通域分析识别 |残差| > n_sigma * σ_ε 的异常区域。

    Args:
        residual: 残差图 (与 image 同 shape)
        mask: 分析区域布尔掩膜
        n_sigma: 阈值倍数 (默认 3σ)
        min_area_px: 最小连通域面积 (像素)

    Returns:
        {
            "hot_spots": [{"x": ..., "y": ..., "area": ..., "peak": ...}, ...],
            "cold_spots": [...],
            "n_hot": int, "n_cold": int,
            "max_hot_peak": float, "max_cold_peak": float,
        }
    """
    import scipy.ndimage as ndi

    res = np.asarray(residual, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)

    inside = res[m]
    if inside.size == 0:
        return {"hot_spots": [], "cold_spots": [], "n_hot": 0, "n_cold": 0,
                "max_hot_peak": np.nan, "max_cold_peak": np.nan}

    rms = float(np.std(inside))
    threshold = n_sigma * rms

    # 热点: residual > +threshold
    hot_mask = (res > threshold) & m
    hot_labels_arr, n_hot_int = ndi.label(hot_mask)  # type: ignore[assignment]
    n_hot = int(n_hot_int)
    hot_spots = []
    max_hot_peak = 0.0
    for i in range(1, n_hot + 1):
        comp = hot_labels_arr == i
        area = int(np.sum(comp))
        if area < min_area_px:
            continue
        comp_y, comp_x = np.where(comp)
        peak = float(np.max(res[comp]))
        max_hot_peak = max(max_hot_peak, peak)
        hot_spots.append({
            "x": float(np.mean(comp_x)),
            "y": float(np.mean(comp_y)),
            "area": area,
            "peak": peak,
        })

    # 冷点: residual < -threshold
    cold_mask = (res < -threshold) & m
    cold_labels_arr, n_cold_int = ndi.label(cold_mask)  # type: ignore[assignment]
    n_cold = int(n_cold_int)
    cold_spots = []
    max_cold_peak = 0.0
    for i in range(1, n_cold + 1):
        comp = cold_labels_arr == i
        area = int(np.sum(comp))
        if area < min_area_px:
            continue
        comp_y, comp_x = np.where(comp)
        peak = float(np.min(res[comp]))
        max_cold_peak = min(max_cold_peak, peak)
        cold_spots.append({
            "x": float(np.mean(comp_x)),
            "y": float(np.mean(comp_y)),
            "area": area,
            "peak": peak,
        })

    return {
        "hot_spots": hot_spots,
        "cold_spots": cold_spots,
        "n_hot": len(hot_spots),
        "n_cold": len(cold_spots),
        "max_hot_peak": max_hot_peak,
        "max_cold_peak": max_cold_peak if cold_spots else np.nan,
    }


def compute_azimuthal_fft(
    theta_deg: list[float] | np.ndarray,
    r_fl_pixels: list[float] | np.ndarray,
) -> dict[str, Any]:
    """
    方位角周期性分析 (§6.1) — 对 R_FL(θ) 做 FFT 提取周期成分。

    Args:
        theta_deg: 角度序列 (度)，如 [0, 10, 20, ..., 350]
        r_fl_pixels: 各角度对应的 R_FL (像素)

    Returns:
        {
            "freqs": np.ndarray,         # 周期数 (n=1,2,3,...)
            "amplitudes": np.ndarray,    # 各周期幅度 (直流除外)
            "dominant_n": int,           # 主导周期
            "dominant_amplitude": float, # 主导幅度
            "n2_amp": float,             # n=2 (像散) 幅度
            "n3_amp": float,             # n=3 (彗差) 幅度
            "ellipticity": float,        # max/min R_FL
        }
    """
    theta = np.asarray(theta_deg, dtype=np.float64)
    rfl = np.asarray(r_fl_pixels, dtype=np.float64)

    ok = np.isfinite(rfl)
    if ok.sum() < 4:
        return {"dominant_n": 0, "dominant_amplitude": np.nan,
                "n2_amp": np.nan, "n3_amp": np.nan, "ellipticity": np.nan,
                "freqs": np.array([]), "amplitudes": np.array([])}

    rfl = rfl[ok]
    theta = theta[ok]

    # 确保等角度间隔（插值到均匀网格）
    n = len(rfl)
    uniform_theta = np.linspace(0, 360, n, endpoint=False)
    rfl_uniform = np.interp(uniform_theta, theta % 360, rfl)

    # FFT
    fft_vals = np.fft.rfft(rfl_uniform - np.mean(rfl_uniform))
    amplitudes = np.abs(fft_vals) / n * 2
    freqs = np.arange(len(amplitudes))  # 周期数

    dominant_idx = int(np.argmax(amplitudes[1:])) + 1 if len(amplitudes) > 1 else 0
    dominant_n = int(freqs[dominant_idx])
    dominant_amp = float(amplitudes[dominant_idx])

    n2_amp = float(amplitudes[2]) if len(amplitudes) > 2 else np.nan
    n3_amp = float(amplitudes[3]) if len(amplitudes) > 3 else np.nan

    ellipticity = float(np.nanmax(rfl) / np.nanmin(rfl)) if np.nanmin(rfl) > 0 else np.nan

    return {
        "freqs": freqs,
        "amplitudes": amplitudes,
        "dominant_n": dominant_n,
        "dominant_amplitude": dominant_amp,
        "n2_amp": n2_amp,
        "n3_amp": n3_amp,
        "ellipticity": ellipticity,
    }


def compute_gradient_metrics(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> dict[str, float]:
    """
    梯度分析 (§7.4)。

    用 Sobel 算子计算梯度幅值，在平顶区统计梯度均匀度。

    Returns:
        {
            "mean_grad": float,       # 圆内平均梯度幅值
            "max_grad": float,        # 圆内最大梯度幅值
            "CV_grad_flat": float,    # 平顶区梯度变异系数 (梯度均匀度)
            "laplacian_energy": float,# 圆内拉普拉斯能量
        }
    """
    from scipy.ndimage import sobel

    img = np.asarray(image, dtype=np.float64)
    y_idx, x_idx = np.indices(img.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mask = r <= radius
    flat_mask = (r >= 0.3 * radius) & (r < 0.7 * radius) & mask

    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    grad_mag = np.hypot(gx, gy)

    grad_inside = grad_mag[mask]
    grad_flat = grad_mag[flat_mask]

    if grad_inside.size == 0:
        return {"mean_grad": np.nan, "max_grad": np.nan,
                "CV_grad_flat": np.nan, "laplacian_energy": np.nan}

    mean_g = float(np.mean(grad_inside))
    max_g = float(np.max(grad_inside))

    if grad_flat.size > 0 and mean_g > 0:
        cv_gf = float(np.std(grad_flat) / np.mean(grad_flat))
    else:
        cv_gf = np.nan

    # 拉普拉斯能量 (二阶)
    from scipy.ndimage import laplace
    lap = laplace(img)
    lap_energy = float(np.sum(lap[mask] ** 2))

    return {
        "mean_grad": mean_g,
        "max_grad": max_g,
        "CV_grad_flat": cv_gf,
        "laplacian_energy": lap_energy,
    }


def compute_frequency_analysis(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """
    频域分析 (§7.3) — 功率谱密度及能量比。

    对圆内区域计算 2D FFT，将频谱划分为低/中/高频带并计算能量占比。

    Returns:
        {
            "low_frac": float,       # 低频能量比 (f < 0.1 * f_nyquist)
            "mid_frac": float,       # 中频能量比
            "high_frac": float,      # 高频能量比
            "peak_frequency": float, # 主导空间频率 (周期/图像)
        }
    """
    img = np.asarray(image, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)

    # 将圆外置零后计算 FFT
    fft_data = img * m
    fft_2d = np.fft.fftshift(np.fft.fft2(fft_data))
    psd = np.abs(fft_2d) ** 2

    h, w = psd.shape
    cy_f, cx_f = (h - 1) / 2.0, (w - 1) / 2.0
    y_f, x_f = np.indices(psd.shape)
    r_f = np.sqrt((x_f - cx_f) ** 2 + (y_f - cy_f) ** 2)
    r_max = max(cy_f, cx_f)

    if r_max == 0:
        return {"low_frac": np.nan, "mid_frac": np.nan,
                "high_frac": np.nan, "peak_frequency": np.nan}

    # 频率带: 低 < 0.1, 中 < 0.35, 高 >= 0.35 (归一化到 nyquist)
    low = r_f / r_max < 0.1
    mid = (r_f / r_max >= 0.1) & (r_f / r_max < 0.35)
    high = r_f / r_max >= 0.35

    total_power = float(psd.sum())
    if total_power <= 0:
        return {"low_frac": np.nan, "mid_frac": np.nan,
                "high_frac": np.nan, "peak_frequency": np.nan}

    low_frac = float(psd[low].sum() / total_power)
    mid_frac = float(psd[mid].sum() / total_power)
    high_frac = float(psd[high].sum() / total_power)

    # 主导频率 (排除直流)
    psd_center = psd.copy()
    psd_center[int(round(cy_f)), int(round(cx_f))] = 0
    peak_idx = np.unravel_index(np.argmax(psd_center), psd_center.shape)
    peak_r = float(r_f[peak_idx])
    peak_freq = peak_r / r_max

    return {
        "low_frac": low_frac,
        "mid_frac": mid_frac,
        "high_frac": high_frac,
        "peak_frequency": peak_freq,
    }


def compute_uniformity_report(
    image: np.ndarray,
    border: dict[str, float],
    ftl_result: dict | None = None,
    angular_theta_deg: list[float] | None = None,
    angular_r_fl_pixels: list[float] | None = None,
) -> dict[str, Any]:
    """
    完整均匀度分析报告 — 聚合所有只基于光瞳图片的分析方法。

    包含:
        - 区域统计 (基础 + 三区划分)
        - 二维空间均匀度 (CV, NU, NRMSD)
        - FTL 残差达标率 (P2%, P5%, P10%)
        - 综合评级
        - 方位角 FFT 周期分析
        - 局部热点/暗点检测
        - 梯度分析
        - 频域分析

    Args:
        image: 去暗场后的光强图
        border: 包围圆字典，需含 border_x, border_y, border_radius
        ftl_result: fit_flat_topped_lorentz 结果 (可选，用于残差分析)
        angular_theta_deg: 各角度序列 (度)
        angular_r_fl_pixels: 各角度 R_FL (像素)

    Returns:
        包含以上所有指标的嵌套字典。
    """
    img = np.asarray(image, dtype=np.float64)

    def _to_float(name: str) -> float:
        v = border.get(name, np.nan)
        return float(v) if isinstance(v, (int, float, np.floating)) else np.nan

    cx = _to_float("border_x")
    cy = _to_float("border_y")
    radius = _to_float("border_radius")

    invalid = math.isnan(cx) or math.isnan(cy) or math.isnan(radius) or radius <= 0
    if invalid:
        return {"success": False, "message": "无效的包围圆参数"}

    # 基础掩膜
    h, w = img.shape
    y_idx, x_idx = np.indices(img.shape)
    rho = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mask = rho <= radius

    report: dict[str, Any] = {"success": True}

    # 1. 区域统计
    zones = compute_uniformity_zones(img, cx, cy, radius)
    report["zones"] = zones
    report["flat_cv"] = zones["flat"]["cv"]
    report["flat_mean"] = zones["flat"]["mean"]

    # 2. 二维空间均匀度
    spatial = compute_spatial_uniformity_2d(img, cx, cy, radius)
    report["spatial_2d"] = spatial

    # 3. FTL 残差达标率 & 评级
    if ftl_result is not None and ftl_result.get("success", False):
        tol = compute_tolerance_pass_rates(img, ftl_result, cx, cy, radius)
        report["tolerance"] = tol
        report["rating"] = compute_comprehensive_rating(
            tol.get("P_5%", np.nan), tol.get("sigma_epsilon", np.nan)
        )
    else:
        report["tolerance"] = None
        report["rating"] = "N/A"

    # 4. 方位角 FFT
    if angular_theta_deg is not None and angular_r_fl_pixels is not None:
        az = compute_azimuthal_fft(angular_theta_deg, angular_r_fl_pixels)
        report["azimuthal_fft"] = az
    else:
        report["azimuthal_fft"] = None

    # 5. 局部缺陷检测 (基于 FTL 残差或原始图像)
    if ftl_result is not None and ftl_result.get("success", False):
        I0 = ftl_result.get("I0", np.nan)
        R_FL = ftl_result.get("R_FL_pixels", np.nan)
        q_val = ftl_result.get("q", np.nan)
        if all(not np.isnan(v) for v in (I0, R_FL, q_val)) and R_FL > 0:
            I_ftl = I0 / (1.0 + (rho / R_FL) ** q_val) ** (1.0 + 2.0 / q_val)
            residual = img - I_ftl
            defects = detect_local_defects(residual, mask)
            report["defects"] = defects
        else:
            report["defects"] = None
    else:
        report["defects"] = None

    # 6. 梯度分析
    gradient = compute_gradient_metrics(img, cx, cy, radius)
    report["gradient"] = gradient

    # 7. 频域分析
    freq = compute_frequency_analysis(img, mask)
    report["frequency"] = freq

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
