"""温度脉动统计：方差、结构函数、功率谱与 C_T²/Cn² 反演。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import signal

from .config import AnalysisConfig, QualityConfig

# Gladstone 关系中的折射率常数（P 单位 hPa，T 单位 K）
GLADSTONE_CONST = 79e-6


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    """去除分块内的线性趋势（温度日变化在 10 min 尺度上接近线性）。"""
    if values.size < 3:
        return values - values.mean()
    x = np.arange(values.size, dtype="float64")
    coef = np.polyfit(x, values, 1)
    return values - np.polyval(coef, x)


def fit_spectral_slope(freq: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> tuple[float, float]:
    """在给定频段内对 log-log 功率谱做线性拟合，返回 (斜率, R²)。"""
    low, high = band
    mask = (freq >= low) & (freq <= high) & (power > 0)
    if mask.sum() < 5:
        return math.nan, math.nan
    x = np.log10(freq[mask])
    y = np.log10(power[mask])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), r2


def psd_curve(values: np.ndarray, dt: float, cfg: AnalysisConfig) -> tuple[np.ndarray, np.ndarray]:
    """Welch 功率谱密度（f, P）。"""
    fs = 1.0 / dt
    nperseg = int(min(cfg.welch_nperseg, values.size))
    return signal.welch(values, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")


def noise_floor_band(
    freq: np.ndarray, power: np.ndarray, band: tuple[float, float], noise_psd: float, factor: float = 2.0
) -> tuple[float, float] | None:
    """把拟合上限收到"信号仍高于量化噪声"的频率处。

    0.1 °C 量化在 7 Hz 采样下会产生一条白噪声底，其功率谱是平的；
    若拟合区间越过噪声底，斜率会被拉向 0（明显浅于 -5/3）。
    """
    if not math.isfinite(noise_psd) or noise_psd <= 0:
        return None
    low, high = band
    inside = (freq >= low) & (freq <= high) & (power > factor * noise_psd)
    if not inside.any():
        return None
    upper = float(np.max(freq[inside]))
    if upper <= low * 1.5:
        return None
    return low, min(high, upper)


def resolve_band(
    freq: np.ndarray, power: np.ndarray, cfg: AnalysisConfig, noise_psd: float = math.nan
) -> tuple[float, float]:
    """确定实际使用的拟合频段（auto 模式下收窄到量化噪声底以上）。"""
    band = cfg.spectrum_band_hz
    if cfg.spectrum_auto_band:
        narrowed = noise_floor_band(freq, power, band, noise_psd)
        if narrowed is not None:
            band = narrowed
    return band


def psd_slope(values: np.ndarray, dt: float, cfg: AnalysisConfig, noise_psd: float = math.nan) -> tuple[float, float]:
    """残差序列的谱斜率与拟合优度。

    Args:
        values: 去趋势后的脉动序列。
        dt: 采样间隔（秒）。
        cfg: 分析参数（拟合频段、是否自动收窄）。
        noise_psd: 量化噪声的单边功率谱密度估计，用于自动收窄频段。
    """
    if dt <= 0 or values.size < 64:
        return math.nan, math.nan
    freq, power = psd_curve(values, dt, cfg)
    return fit_spectral_slope(freq, power, resolve_band(freq, power, cfg, noise_psd))


def structure_function(values: np.ndarray, dt: float, lags_s: tuple[float, ...]) -> dict[float, float]:
    """温度结构函数 D_T(τ) = <[T(t+τ) - T(t)]²>。"""
    out: dict[float, float] = {}
    for lag in lags_s:
        lag_samples = int(round(lag / dt)) if dt > 0 else 0
        if lag_samples < 1 or lag_samples >= values.size:
            out[float(lag)] = math.nan
            continue
        diff = values[lag_samples:] - values[:-lag_samples]
        out[float(lag)] = float(np.mean(diff**2))
    return out


def structure_noise_floor(d_tau: dict[float, float], sigma_q: float) -> float:
    """估计结构函数里的白噪声偏置（°C²），即 τ→0 的截距。

    理论上白噪声贡献 2σ_q²，但实测量化步长常被有效位数稀释（本站 σ 只有 0.03–0.06 °C），
    直接减 2σ_q² 会把短滞后整段压成负数，斜率随之失去意义。因此用最小的两个滞后按
    D(τ) = a + b·τ^{2/3} 外推到 τ→0，取截距 a 作为经验噪声底，再与理论值取小者：
    经验值不会高于实测最短滞后的读数（噪声不可能超过它），理论值只作为上界。

    Args:
        d_tau: 结构函数（滞后秒 → °C²），允许为负或 NaN。
        sigma_q: 量化噪声标准差（°C）。

    Returns:
        噪声底（°C²），非负；数据不足时退化为理论值 2σ_q²。
    """
    theoretical = 2.0 * sigma_q**2 if math.isfinite(sigma_q) else math.nan
    points = sorted((float(tau), float(val)) for tau, val in d_tau.items() if math.isfinite(val))
    if len(points) < 2:
        return theoretical if math.isfinite(theoretical) else 0.0
    (tau1, d1), (tau2, d2) = points[0], points[1]
    if tau1 <= 0 or tau2 <= tau1:
        return theoretical if math.isfinite(theoretical) else 0.0
    # 由 d2 - d1 = b·τ1^{2/3}·((τ2/τ1)^{2/3} - 1) 解出 b·τ1^{2/3}，再从 d1 中扣除
    ratio = (tau2 / tau1) ** (2.0 / 3.0) - 1.0
    empirical = d1 - (d2 - d1) / ratio if ratio > 0 else d1
    empirical = min(d1, max(0.0, empirical))
    return min(theoretical, empirical) if math.isfinite(theoretical) else empirical


def _corrected(d_tau: dict[float, float], noise_floor: float) -> dict[float, float]:
    """扣除白噪声底；校正后 ≤0 的点置为 NaN（该滞后被噪声主导，不可用）。"""
    floor = noise_floor if math.isfinite(noise_floor) else 0.0
    out: dict[float, float] = {}
    for tau, value in d_tau.items():
        corrected = value - floor if math.isfinite(value) else math.nan
        out[float(tau)] = corrected if corrected > 0 else math.nan
    return out


def structure_slope(d_tau: dict[float, float], fit_range: tuple[float, float], noise_floor: float = 0.0) -> float:
    """结构函数幂律指数（Kolmogorov 惯性子区理论值 2/3）。

    Args:
        d_tau: 原始结构函数（含白噪声偏置）。
        fit_range: 拟合滞后区间（秒）。
        noise_floor: 白噪声底（°C²），先扣除再拟合。
    """
    low, high = fit_range
    corrected = _corrected(d_tau, noise_floor)
    points = [(tau, val) for tau, val in corrected.items() if low <= tau <= high and math.isfinite(val)]
    if len(points) < 2:
        return math.nan
    x = np.log10([point[0] for point in points])
    y = np.log10([point[1] for point in points])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def ct2_from_structure(
    d_tau: dict[float, float],
    wind_speed: float,
    fit_range: tuple[float, float],
    noise_floor: float = 0.0,
) -> float:
    """由结构函数反演温度结构常数 C_T² = D_T(τ) / (U·τ)^{2/3}（Taylor 冻结假设）。"""
    if not math.isfinite(wind_speed) or wind_speed <= 0:
        return math.nan
    low, high = fit_range
    corrected = _corrected(d_tau, noise_floor)
    estimates = [
        val / ((wind_speed * tau) ** (2.0 / 3.0))
        for tau, val in corrected.items()
        if low <= tau <= high and math.isfinite(val)
    ]
    if not estimates:
        return math.nan
    return float(np.median(estimates))


def cn2_from_ct2(ct2: float, temperature_c: float, pressure_hpa: float) -> float:
    """由 C_T² 反演折射率结构常数 Cn²（Gladstone 关系，忽略湿度项）。"""
    if not math.isfinite(ct2) or ct2 <= 0 or not math.isfinite(temperature_c):
        return math.nan
    temperature_k = temperature_c + 273.15
    if temperature_k <= 0:
        return math.nan
    index_term = GLADSTONE_CONST * pressure_hpa / temperature_k**2
    return float(index_term**2 * ct2)


def quantization_noise(step_c: float) -> float:
    """均匀量化噪声标准差估计 σ_q = Δ/√12。"""
    if not math.isfinite(step_c) or step_c <= 0:
        return math.nan
    return step_c / math.sqrt(12.0)


def compute_block_metrics(
    wide: pd.DataFrame,
    cfg: AnalysisConfig,
    quality_cfg: QualityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按分块窗口计算每个探针的脉动统计量，并给出各探针的代表性功率谱。

    风速与气压相关的反演（C_T²、Cn²）在 `aggregate` 阶段与辅助数据合并后计算，
    使本函数只依赖温度宽表，便于缓存。

    Args:
        wide: 以时间为索引、探针为列的清洗后宽表（无效值为 NaN）。
        cfg: 分析参数。
        quality_cfg: 质量参数（用于量化噪声与信噪比估计）。

    Returns:
        (分块长表, 谱曲线长表)。分块长表每行对应一个 (分块起点, 探针)；
        谱曲线长表为各探针在分块间的功率谱中位数，列为 探针 / 频率 / 功率。
    """
    rows: list[dict[str, object]] = []
    spectra: dict[str, list[pd.Series]] = {}
    empty = pd.DataFrame(columns=["探针", "频率", "功率"])
    if wide.empty:
        return pd.DataFrame(rows), empty

    dt = float(np.median(np.diff(wide.index.to_numpy(dtype="datetime64[ns]").astype("int64")))) / 1e9
    if not math.isfinite(dt) or dt <= 0:
        return pd.DataFrame(rows), empty

    sigma_q = quantization_noise(quality_cfg.quantization_step_c)
    # 单边功率谱密度中的白噪声底：σ²=f_s/2·N0 → N0=2·σ²·dt
    noise_psd = sigma_q**2 * 2.0 * dt if math.isfinite(sigma_q) else math.nan
    lags = cfg.structure_lags_s

    for block_start, frame in wide.groupby(pd.Grouper(freq=cfg.block_rule)):
        if frame.empty:
            continue
        for probe in frame.columns:
            values = frame[probe].to_numpy(dtype="float64", na_value=math.nan)
            valid = values[np.isfinite(values)]
            n_valid = int(valid.size)
            row: dict[str, object] = {
                "分块起点": block_start,
                "探针": probe,
                "有效采样": n_valid,
                "覆盖率": n_valid / values.size if values.size else 0.0,
                "均值": float(valid.mean()) if n_valid else math.nan,
            }
            if n_valid < 64 or row["覆盖率"] < cfg.min_block_coverage:
                row.update({"标准差": math.nan, "谱斜率": math.nan, "结构函数斜率": math.nan})
                rows.append(row)
                continue

            residual = _linear_detrend(valid)
            sigma = float(residual.std(ddof=1))
            freq, power = psd_curve(residual, dt, cfg)
            band = resolve_band(freq, power, cfg, noise_psd)
            slope, r2 = fit_spectral_slope(freq, power, band)
            d_tau = structure_function(residual, dt, lags)
            # 原始 D_T 含量化白噪声偏置：缓存原始值，斜率与 C_T² 反演时按噪声底扣除
            floor = structure_noise_floor(d_tau, sigma_q)
            sf_slope = structure_slope(d_tau, cfg.structure_fit_lags_s, noise_floor=floor)

            row.update(
                {
                    "标准差": sigma,
                    "标准差_原始": float(valid.std(ddof=1)),
                    "谱斜率": slope,
                    "谱拟合优度": r2,
                    "谱拟合下限": band[0],
                    "谱拟合上限": band[1],
                    "结构函数斜率": sf_slope,
                    "噪声底": floor,
                    "波动信噪比": sigma / sigma_q if sigma_q and math.isfinite(sigma_q) else math.nan,
                }
            )
            for tau, value in d_tau.items():
                row[f"D_{tau:g}s"] = value
            rows.append(row)

            spectra.setdefault(str(probe), []).append(pd.Series(power, index=freq))

    spectrum_frame = _median_spectra(spectra)
    return pd.DataFrame(rows), spectrum_frame


def _median_spectra(spectra: dict[str, list[pd.Series]]) -> pd.DataFrame:
    """把每个探针在多个分块上的功率谱取中位数，得到代表性谱曲线。"""
    frames: list[pd.DataFrame] = []
    for probe, curves in spectra.items():
        if not curves:
            continue
        stacked = pd.concat(curves, axis=1)
        median = stacked.median(axis=1)
        frames.append(pd.DataFrame({"探针": probe, "频率": median.index.to_numpy(), "功率": median.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["探针", "频率", "功率"])
    return pd.concat(frames, ignore_index=True)
