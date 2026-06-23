"""
光瞳 + 光轴联合分析模块
=======================

基于 `docs/联合分析.md` 实现的分析算法：

1. **鲁棒预处理流水线** — 坏点检测/修复 → 背景估计/扣除 → 去噪 → 归一化 → 有效区域掩膜 → 亚像素质心精化
2. **单张光瞳综合指标** — 等效直径、FWHM、能量约束半径 r95、圆度、能量集中度曲线
3. **TIE 相位恢复** — 基于强度传输方程的快速波前还原（光瞳面 + 焦面强度）
4. **泽尼克相位拟合** — 将恢复的波前分解为泽尼克模式系数

依赖:
    - numpy, scipy, cv2, matplotlib
    - zernike (用于相位分解)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel, laplace
from scipy.fft import fft2, ifft2, fftfreq
from scipy.stats import binned_statistic


# ===========================================================================
# §2 — 鲁棒光瞳图像预处理流水线
# ===========================================================================


def robust_pupil_preprocessing(
    image: np.ndarray,
    pixel_size_um: float = 1.0,
    bad_pixel_threshold: float = 5.0,
    denoise_sigma: float = 1.0,
    bg_method: str = "auto",
) -> dict[str, Any]:
    """
    鲁棒光瞳图像预处理流水线 (§2)。

    处理流程:
        1. 坏点/热像素检测与修复 (中值滤波 + MAD)
        2. 背景估计与扣除 (四角平均 / 形态学开运算)
        3. 高斯滤波去噪
        4. 归一化到 [0, 1]
        5. 有效区域掩膜 (强度 + 梯度约束)
        6. 亚像素质心精化

    Args:
        image: 输入光瞳强度图像 (H, W)
        pixel_size_um: 像素尺寸 (μm), 默认 1.0
        bad_pixel_threshold: 坏点检测阈值 (MAD 倍数), 默认 5.0
        denoise_sigma: 高斯滤波 sigma, 默认 1.0
        bg_method: 背景估计方法 ('auto' | 'corner' | 'morphology')

    Returns:
        dict 包含以下键:
            - original: 原始图像副本
            - despiked: 坏点修复后的图像
            - background: 估计的背景值
            - background_removed: 背景扣除后的图像
            - denoised: 高斯滤波去噪后的图像
            - normalized: 归一化到 [0,1] 的图像
            - valid_mask: 有效区域布尔掩膜
            - centroid: 亚像素质心 (cx, cy)
            - radial_distance: 各像素到质心的径向距离矩阵
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    results: dict[str, Any] = {"original": img.copy(), "pixel_size_um": pixel_size_um}

    # ---------- Step 1: 坏点/热像素检测与修复 ----------
    median_img = median_filter(img, size=3)
    diff = np.abs(img - median_img)
    mad = float(np.median(diff))
    bad_mask: np.ndarray | None = None
    if mad > 0:
        bad_mask = diff > bad_pixel_threshold * mad
        img_despike = img.copy()
        img_despike[bad_mask] = median_img[bad_mask]
    else:
        img_despike = img
    results["despiked"] = img_despike
    results["bad_pixel_count"] = int(np.sum(bad_mask)) if bad_mask is not None else 0

    # ---------- Step 2: 背景估计与扣除 ----------
    corner_size = max(h, w) // 10
    corners = [
        img_despike[:corner_size, :corner_size],
        img_despike[:corner_size, -corner_size:],
        img_despike[-corner_size:, :corner_size],
        img_despike[-corner_size:, -corner_size:],
    ]
    bg_corner = float(np.median([np.median(c) for c in corners]))

    # 形态学开运算估计背景（适合背景渐变）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(h // 20, 5), max(w // 20, 5)),
    )
    bg_morph = cv2.morphologyEx(img_despike.astype(np.float32), cv2.MORPH_OPEN, kernel)
    bg_morph_val = float(np.median(bg_morph))

    if bg_method == "corner":
        bg = bg_corner
    elif bg_method == "morphology":
        bg = bg_morph_val
    else:  # 'auto' — 保守估计，取较小值
        bg = min(bg_corner, bg_morph_val)

    img_bg = np.clip(img_despike - bg, 0, None)
    results["background"] = bg
    results["background_removed"] = img_bg

    # ---------- Step 3: 去噪 ----------
    img_gauss = gaussian_filter(img_bg, sigma=denoise_sigma)
    results["denoised"] = img_gauss

    # ---------- Step 4: 归一化 ----------
    img_max = float(np.max(img_gauss))
    if img_max > 0:
        img_norm = img_gauss / img_max
    else:
        img_norm = img_gauss
    results["normalized"] = img_norm

    # ---------- Step 5: 有效区域掩膜 ----------
    grad_y, grad_x = np.gradient(img_norm)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    threshold = float(np.percentile(img_norm[img_norm > 0], 5)) if np.any(img_norm > 0) else 0.0
    valid_mask = (img_norm > threshold) & (
        grad_mag < np.percentile(grad_mag, 95)
    )
    results["valid_mask"] = valid_mask

    # ---------- Step 6: 亚像素质心精化 ----------
    y_idx, x_idx = np.indices((h, w))
    total = float(np.sum(img_norm * valid_mask))
    if total > 0:
        cx = float(np.sum(x_idx * img_norm * valid_mask) / total)
        cy = float(np.sum(y_idx * img_norm * valid_mask) / total)
    else:
        cx, cy = w / 2.0, h / 2.0
    results["centroid"] = (cx, cy)

    # ---------- Step 7: 径向坐标 ----------
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    results["radial_distance"] = r

    return results


# ===========================================================================
# §1 — 单张光瞳图片综合指标
# ===========================================================================


def calculate_energy_confinement_radius(
    image: np.ndarray,
    cx: float,
    cy: float,
    energy_fractions: list[float] | None = None,
    n_bins: int = 500,
) -> dict[str, Any]:
    """
    计算能量约束半径曲线 — 径向累积能量达到指定比例时的半径 (§1.1)。

    Args:
        image: 去暗场后的强度图像 (2D array)
        cx, cy: 光斑中心坐标 (像素)
        energy_fractions: 需要计算的能量比例列表，默认 [0.50, 0.80, 0.90, 0.95, 0.99]
        n_bins: 径向分箱数

    Returns:
        {
            "r_xx": float,          # 各能量比例对应的半径 (像素)
            "radius_curve": ndarray,# 半径序列
            "energy_curve": ndarray,# 对应的径向累积能量比例
            "total_energy": float,  # 总能量
        }
    """
    if energy_fractions is None:
        energy_fractions = [0.50, 0.80, 0.90, 0.95, 0.99]
    energy_fractions = sorted(energy_fractions)

    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    r_max = float(max(h, w))
    bin_centers = np.linspace(0, r_max, n_bins, endpoint=False) + r_max / (2.0 * n_bins)

    mean_I, _, _ = binned_statistic(r.ravel(), img.ravel(), statistic="mean", bins=n_bins, range=(0, r_max))
    mean_I = np.nan_to_num(mean_I, nan=0.0)

    dr = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    ring_areas = 2 * np.pi * bin_centers * dr
    ring_energy = mean_I * ring_areas
    cum_energy = np.cumsum(ring_energy)
    total = float(cum_energy[-1]) if cum_energy[-1] > 0 else 1.0
    cum_frac = cum_energy / total

    result: dict[str, Any] = {
        "total_energy": cum_energy[-1] if len(cum_energy) > 0 else 0.0,
        "radius_curve": bin_centers,
        "energy_curve": cum_frac,
    }
    for ef in energy_fractions:
        idx = int(np.searchsorted(cum_frac, ef))
        idx = min(idx, len(bin_centers) - 1)
        result[f"r_{ef*100:.0f}"] = float(bin_centers[idx])

    return result


def calculate_roundness(
    image: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    计算光斑圆度 C = 4πA / P² (§1.1 — 形状指标)。

    基于轮廓检测: A = 轮廓面积, P = 轮廓周长。
    正圆的圆度 = 1.0, 越不规则越接近 0。

    Args:
        image: 输入灰度图像 (H, W)
        threshold: 二值化阈值。None 时使用 Otsu 自动阈值。

    Returns:
        {
            "roundness": float,     # 圆度 C
            "area": float,          # 轮廓面积 (像素²)
            "perimeter": float,     # 轮廓周长 (像素)
            "equivalent_diameter": float,  # 等效圆直径 Deq = 2*sqrt(A/π) (像素)
        }
    """
    img = np.asarray(image, dtype=np.float64)
    if np.max(img) > 0:
        img_u8 = (img / np.max(img) * 255).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    # 高斯平滑后二值化
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 1.0)
    if threshold is None:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "roundness": np.nan,
            "area": 0.0,
            "perimeter": 0.0,
            "equivalent_diameter": np.nan,
        }

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, closed=True))

    if perimeter <= 0 or area <= 0:
        return {
            "roundness": np.nan,
            "area": area,
            "perimeter": perimeter,
            "equivalent_diameter": np.nan,
        }

    roundness = float(4.0 * math.pi * area / (perimeter * perimeter))
    equiv_diameter = float(2.0 * math.sqrt(area / math.pi))

    return {
        "roundness": roundness,
        "area": area,
        "perimeter": perimeter,
        "equivalent_diameter": equiv_diameter,
    }


def calculate_pupil_beam_metrics(
    image: np.ndarray,
    pixel_size_um: float = 1.0,
    cx: float | None = None,
    cy: float | None = None,
) -> dict[str, Any]:
    """
    综合计算单张光瞳图片的全部可用指标 (§1.1)。

    涵盖:
        - 几何: 质心、等效圆直径 Deq、圆度
        - 尺寸: FWHM、D4σ、能量约束半径 r_90/r_95
        - 能量: 总能量、峰值强度、能量集中度曲线
        - 形状: 椭圆长短轴、离心率

    Args:
        image: 去暗场后的光强图 (2D array)
        pixel_size_um: 像素尺寸 (μm)
        cx, cy: 光斑中心 (像素); None 时自动用质心

    Returns:
        嵌套字典包含所有指标。
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    # ---- 自动中心定位 ----
    if cx is None or cy is None:
        total = float(np.sum(img))
        if total > 0:
            y_idx, x_idx = np.indices(img.shape)
            cx_f = float(np.sum(x_idx * img) / total)
            cy_f = float(np.sum(y_idx * img) / total)
        else:
            cx_f, cy_f = w / 2.0, h / 2.0
    else:
        cx_f, cy_f = float(cx), float(cy)

    # ---- 基础统计 ----
    total_energy = float(np.sum(img))
    peak_intensity = float(np.max(img))
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))

    # ---- D4σ (二阶矩直径) ----
    xv, yv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    mu_xx = float(np.sum((xv - cx_f) ** 2 * img)) / total_energy if total_energy > 0 else 0.0
    mu_yy = float(np.sum((yv - cy_f) ** 2 * img)) / total_energy if total_energy > 0 else 0.0
    sigma_x = math.sqrt(max(mu_xx, 0.0))
    sigma_y = math.sqrt(max(mu_yy, 0.0))
    d4sigma_x = 4.0 * sigma_x * pixel_size_um
    d4sigma_y = 4.0 * sigma_y * pixel_size_um
    d4sigma_avg = math.sqrt(d4sigma_x * d4sigma_y)

    # ---- 等效圆直径 ----
    # 基于总面积等效: Deq = 2 * sqrt(A/π), 其中 A = 有效像素数
    # 有效像素: 强度 > 均值 * 0.1
    valid_pixels = img > mean_intensity * 0.1
    pixel_area = float(np.sum(valid_pixels))
    equiv_diameter = 2.0 * math.sqrt(pixel_area / math.pi) * pixel_size_um if pixel_area > 0 else 0.0

    # ---- FWHM 等效半径 ----
    half_max = peak_intensity / 2.0
    r_grid = np.sqrt((xv - cx_f) ** 2 + (yv - cy_f) ** 2)
    mask_fwhm = img >= half_max
    if np.any(mask_fwhm):
        r_fwhm_max = float(np.max(r_grid[mask_fwhm]))
        fwhm_diameter = 2.0 * r_fwhm_max * pixel_size_um
    else:
        r_fwhm_max = 0.0
        fwhm_diameter = 0.0

    # ---- 能量约束半径 ----
    energy_metrics = calculate_energy_confinement_radius(img, cx_f, cy_f)

    # ---- 圆度 ----
    roundness_metrics = calculate_roundness(img)

    # ---- 椭圆拟合 (离心率) ----
    img_u8 = np.clip(
        (img - np.min(img)) / max(np.max(img) - np.min(img), 1e-12) * 255.0,
        0, 255,
    ).astype(np.uint8)
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            (_, _), (short, long_), _angle = cv2.fitEllipse(largest)
            ellipticity = float(long_ / short) if short > 0 else np.nan
            eccentricity = math.sqrt(1.0 - (short / long_) ** 2) if long_ > 0 else 0.0
        else:
            ellipticity = np.nan
            eccentricity = 0.0
    else:
        ellipticity = np.nan
        eccentricity = 0.0

    return {
        "centroid": (cx_f, cy_f),
        "total_energy": total_energy,
        "peak_intensity": peak_intensity,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        # 尺寸
        "equivalent_diameter_um": equiv_diameter,
        "d4sigma_diameter_x_um": d4sigma_x,
        "d4sigma_diameter_y_um": d4sigma_y,
        "d4sigma_diameter_avg_um": d4sigma_avg,
        "fwhm_diameter_um": fwhm_diameter,
        # 能量约束
        "energy_confinement": energy_metrics,
        # 形状
        "roundness": roundness_metrics["roundness"],
        "equivalent_circle_area_px": roundness_metrics["area"],
        "perimeter_px": roundness_metrics["perimeter"],
        "equivalent_diameter_from_area_px": roundness_metrics["equivalent_diameter"],
        "ellipticity": ellipticity,
        "eccentricity": eccentricity,
        "sigma_x_px": sigma_x,
        "sigma_y_px": sigma_y,
    }


# ===========================================================================
# §3.2 — TIE 相位恢复 + 泽尼克分解
# ===========================================================================


def tie_phase_retrieval(
    pupil_intensity: np.ndarray,
    focal_intensity: np.ndarray,
    focal_distance: float,
    wavelength: float,
    pixel_size: float,
    regularization: float = 1e-6,
) -> np.ndarray:
    """
    基于 TIE (Transport of Intensity Equation) 的相位恢复 (§3.2)。

    假设光瞳面强度近似均匀，主要相位信息来自焦面变化。
    通过求解泊松方程从光瞳面 + 焦面强度恢复波前相位。

    实现步骤:
        1. 归一化两平面强度
        2. 有限差分近似轴向强度导数 dI/dz
        3. 构建泊松方程右侧
        4. 频域 (FFT) 求解泊松方程
        5. 去除 piston 分量

    Args:
        pupil_intensity: 光瞳面强度 (H, W)
        focal_intensity: 焦面强度 (H, W)
        focal_distance: 光瞳到焦面的距离 [m]
        wavelength: 波长 [m]
        pixel_size: 像素物理尺寸 [m]
        regularization: 正则化参数，防止低频噪声放大

    Returns:
        phase: 恢复的波前相位 [rad], 形状同输入, piston 已去除
    """
    I_pupil = np.asarray(pupil_intensity, dtype=np.float64)
    I_focal = np.asarray(focal_intensity, dtype=np.float64)

    if I_pupil.shape != I_focal.shape:
        raise ValueError(f"两平面图像尺寸不一致: {I_pupil.shape} vs {I_focal.shape}")

    # 归一化总能量一致
    I_pupil = I_pupil / max(np.sum(I_pupil), 1e-12)
    I_focal = I_focal / max(np.sum(I_focal), 1e-12)

    # 计算轴向强度导数 (有限差分)
    dIdz = (I_focal - I_pupil) / max(focal_distance, 1e-12)

    # 平均强度 (TIE 中的 I)
    I_avg = (I_pupil + I_focal) / 2.0
    I_avg = np.clip(I_avg, np.max(I_avg) * 0.01, None)

    # 泊松方程右侧: -2π/λ * (dI/dz) / I
    wl_safe = wavelength if wavelength > 1e-12 else 1e-12
    rhs = -(2.0 * np.pi / wl_safe) * dIdz / I_avg

    # 频域求解泊松方程 ∇²φ = rhs
    H, W = rhs.shape
    fx = fftfreq(W, pixel_size)
    fy = fftfreq(H, pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    # 拉普拉斯频域表示: -4π²(fx² + fy²)
    laplacian_freq = -(2.0 * np.pi) ** 2 * (FX**2 + FY**2)
    laplacian_freq[0, 0] = 1.0  # 避免 DC 除零

    # 正则化: 对接近零的频率分量做截断
    laplacian_freq = np.where(
        np.abs(laplacian_freq) < regularization,
        regularization * np.sign(laplacian_freq + 1e-12),
        laplacian_freq,
    )

    # 求解: φ̂ = rhŝ / ∇²̂
    rhs_fft = fft2(rhs)
    phase_fft = rhs_fft / laplacian_freq
    phase = np.real(ifft2(phase_fft))  # type: ignore[arg-type]

    # 去除 piston (整体平均相位)
    phase = phase - np.mean(phase)

    return phase


def zernike_fit_phase(
    phase: np.ndarray,
    n_modes: int = 15,
    aperture_radius: float | None = None,
) -> dict[str, Any]:
    """
    将 TIE 恢复的波前相位拟合为泽尼克模式系数 (§3.2)。

    与 `zernike_analysis.fit_zernike()` 的区别:
        前者拟合光瞳强度分布得到"强度-泽尼克"系数；
        后者拟合已恢复的波前相位得到"相位-泽尼克"系数，
        物理意义更明确 (直接对应波前像差)。

    Args:
        phase: TIE 恢复的波前相位 (H, W), 单位 [rad]
        n_modes: 拟合的 Zernike 模式数 (Noll 序号 1..n_modes)
        aperture_radius: 孔径半径 (像素)。None 时取 min(H,W)/2

    Returns:
        {
            "coeffs": np.ndarray,          # Zernike 系数 (Noll 顺序), [n_modes]
            "noll_map": list[int],         # 1-based Noll 序号
            "reconstructed": np.ndarray,   # 重建相位 (side x side)
            "residual": np.ndarray,        # 残差相位 (side x side)
            "rmse": float,                 # 单位圆内重建 RMSE [rad]
            "mask": np.ndarray,            # 单位圆内布尔掩膜
            "side": int,                   # 裁剪后正方形边长
            "piston": float,               # piston 分量 (系数[0])
            "defocus": float,              # defocus 分量 (系数[3] 若存在)
            "astigmatism_0": float,        # 0° 像散 (系数[4] 若存在)
            "astigmatism_45": float,       # 45° 像散 (系数[5] 若存在)
            "coma_x": float,               # X 彗差 (系数[6] 若存在)
            "coma_y": float,               # Y 彗差 (系数[7] 若存在)
            "spherical": float,            # 球差 (系数[10] 若存在)
        }
    """
    from zernike import RZern

    phase_arr = np.asarray(phase, dtype=np.float64)
    H, W = phase_arr.shape

    _aperture_radius = aperture_radius if aperture_radius is not None else min(H, W) / 2.0

    side = int(round(2.0 * _aperture_radius))
    side = max(side, 4)
    if side % 2 != 0:
        side += 1

    # 裁剪到正方形区域
    y0 = int(max(0, round((H - side) / 2.0)))
    x0 = int(max(0, round((W - side) / 2.0)))
    patch = phase_arr[y0 : y0 + side, x0 : x0 + side]

    # 构建 [-1, 1] 网格
    ddx = np.linspace(-1.0, 1.0, side)
    ddy = np.linspace(-1.0, 1.0, side)
    xv, yv = np.meshgrid(ddx, ddy)

    cart = RZern(n_modes)
    cart.make_cart_grid(xv, yv)

    coeffs, *_ = cart.fit_cart_grid(patch)

    # 重建与残差
    reconstructed = cart.eval_grid(coeffs, matrix=True)
    rho = np.sqrt(xv**2 + yv**2)
    mask = rho <= 1.0

    valid_orig = patch[mask]
    valid_recon = reconstructed[mask]
    rmse = float(np.sqrt(np.mean((valid_orig - valid_recon) ** 2)))
    residual = patch - reconstructed

    noll_map = list(range(1, cart.nk + 1))

    # 提取关键像差系数 (Noll 1-based)
    def _coeff(noll_idx: int) -> float:
        idx = noll_idx - 1
        return float(coeffs[idx]) if idx < len(coeffs) else 0.0

    return {
        "coeffs": coeffs,
        "noll_map": noll_map,
        "reconstructed": reconstructed,
        "residual": residual,
        "rmse": rmse,
        "mask": mask,
        "side": side,
        "y0": y0,
        "x0": x0,
        "ntab": cart.ntab.copy(),
        "mtab": cart.mtab.copy(),
        "piston": _coeff(1),
        "defocus": _coeff(4),
        "astigmatism_0": _coeff(5),
        "astigmatism_45": _coeff(6),
        "coma_x": _coeff(7),
        "coma_y": _coeff(8),
        "spherical": _coeff(11),
        "total_aberration_rms": rmse,
    }
