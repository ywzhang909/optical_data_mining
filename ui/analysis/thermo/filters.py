"""噪声治理：抖动异常剔除（样本级）与产品平滑（分钟级）。

温度原始数据量化到 0.1 °C、脉动标准差只有 0.03–0.06 °C，信噪比 1–2，噪声敏感。
这里把"降噪"拆成两件互不替代的事：

1. **抖动/离群剔除**：掉线、瞬时跳变、传感器错误读数不属于温度脉动，先剔除再统计，
   否则会直接抬高方差、结构函数与谱的低频段；
2. **产品平滑**：趋势图与时间片导出另给一列滑动均值，方便看走势。谱、结构函数、
   C_T² 一律用**未平滑**的去趋势序列——低通会抹掉惯性子区，平滑后再算斜率是错的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FilterConfig


def jitter_mask(
    values: pd.Series | np.ndarray,
    *,
    dt: float,
    window_s: float,
    threshold: float,
) -> pd.Series | np.ndarray:
    """标记偏离局部中值超过阈值的采样（抖动/跳变）。

    判据取**绝对阈值**而不是 MAD 倍数：0.1 °C 量化下 MAD 本身就接近量化噪声，
    按倍数判会把正常的量化跳动全部当成异常；本站真实跳变在 1 °C 量级，
    因此默认阈值取 1.0 °C，既能抓住掉线/跳变，又不动真实脉动。

    Args:
        values: 采样序列（可含 NaN）。
        dt: 采样间隔（秒）。
        window_s: 局部中值窗口（秒），换算为采样点数。
        threshold: 判定阈值（与原量纲一致，温度即 °C）。

    Returns:
        与输入同索引的布尔掩码，True 表示判为抖动。
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if threshold <= 0 or series.empty:
        return pd.Series(False, index=series.index)
    window = max(3, int(round(window_s / dt)) | 1) if dt > 0 else 11
    median = series.rolling(window, center=True, min_periods=3).median()
    deviation = (series - median).abs()
    mask = deviation > threshold
    return mask.fillna(False).astype(bool)


def moving_average(series: pd.Series, window: int, *, center: bool = True) -> pd.Series:
    """居中滑动均值（不足窗口的前后段保留原值）。"""
    if window <= 1 or len(series) < 2:
        return series
    smoothed = series.rolling(window, center=center, min_periods=max(1, window // 2)).mean()
    return smoothed.fillna(series)


def smooth_minute_frame(minute: pd.DataFrame, window: int) -> pd.DataFrame:
    """给分钟级长表补一列按探针分组的滑动均值。

    Args:
        minute: 长表，含 探针 / ts / 均值（以及 数值 通道表）。
        window: 滑动窗口（分钟），≤1 时原样返回。

    Returns:
        新的长表，追加 ``平滑值`` 列（窗口不足时等于原值）。
    """
    if window <= 1 or minute.empty:
        return minute
    value_column = "均值" if "均值" in minute.columns else "数值"
    if value_column not in minute.columns:
        return minute
    frame = minute.sort_values(["探针", "ts"]).copy()
    frame["平滑值"] = (
        frame.groupby("探针", observed=True)[value_column]
        .transform(lambda values: moving_average(values, window))
        .to_numpy()
    )
    return frame.sort_index()


def robust_sigma(values: pd.Series | np.ndarray) -> float:
    """用 1.4826×MAD 估计标准差，抗离群值（用于平滑后的噪声水平评估）。"""
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return float(1.4826 * mad)


def filter_frame(values: pd.DataFrame, cfg: FilterConfig, dt: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对宽表做抖动剔除，返回（清洗后宽表, 抖动掩码）。

    抖动样本置为 NaN（与无效值、跨探针离群同样处理），掩码留给质量统计计数。
    """
    mask = jitter_frame(values, cfg, dt)
    if not cfg.despike_enabled or values.empty:
        return values, mask
    return values.mask(mask), mask


def jitter_frame(values: pd.DataFrame, cfg: FilterConfig, dt: float) -> pd.DataFrame:
    """逐探针抖动掩码（宽表，True = 抖动）。未启用时返回全 False。"""
    if not cfg.despike_enabled or values.empty or dt <= 0:
        return pd.DataFrame(False, index=values.index, columns=values.columns)
    masks = {
        column: jitter_mask(
            values[column],
            dt=dt,
            window_s=cfg.despike_window_s,
            threshold=cfg.despike_threshold_c,
        )
        for column in values.columns
    }
    return pd.DataFrame(masks, index=values.index)
