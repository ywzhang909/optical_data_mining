"""数据质量评估：异常值、卡值、尖峰、采样节奏与覆盖率。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import QualityConfig


def invalid_mask(values: pd.DataFrame | pd.Series, cfg: QualityConfig) -> pd.DataFrame | pd.Series:
    """标记无效采样：非数值、超出物理范围。

    采集软件在无有效读数时写入固定异常码（如 1220.8），其值必然超出物理范围，
    因此统一用范围判据即可覆盖；`sentinel_values` 仅用于质量统计中的归类。
    """
    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else values
    outside = (numeric < cfg.valid_min_c) | (numeric > cfg.valid_max_c)
    return numeric.isna() | outside


def cross_probe_outliers(values: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    """标记与同一时刻跨探针中值相差过大的采样（掉线/错误读数）。

    同址多探针在同一时刻读数相近（本站瞬时离散度中位数 < 1 °C），出现 5 °C 以上的
    偏离只可能是传感器故障——例如 tp-16 曾连续若干个采样点返回 0.0 °C。这类错误值
    在单探针序列内看是「一段平稳值」，滑动中值判据（`spike_mask`）抓不到，必须借助
    其他探针做横向比对。
    """
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if cfg.cross_probe_max_dev_c <= 0 or numeric.shape[1] < 3:
        return pd.DataFrame(False, index=numeric.index, columns=numeric.columns)
    baseline = numeric.mask(invalid_mask(numeric, cfg)).median(axis=1, skipna=True)
    return numeric.sub(baseline, axis=0).abs() > cfg.cross_probe_max_dev_c


def clean(values: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
    """把无效采样与跨探针离群采样置为 NaN，返回浮点宽表。"""
    numeric = values.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.mask(invalid_mask(numeric, cfg))
    return numeric.mask(cross_probe_outliers(numeric, cfg))


@dataclass(frozen=True)
class CadenceStats:
    """采样节奏统计。"""

    n_samples: int
    median_dt_s: float
    max_gap_s: float
    gap_count: int
    span_s: float
    coverage: float
    expected_n: float

    def as_row(self) -> dict[str, float]:
        return {
            "采样点数": self.n_samples,
            "中位采样间隔 (s)": round(self.median_dt_s, 4),
            "标称采样率 (Hz)": round(1.0 / self.median_dt_s, 3) if self.median_dt_s > 0 else float("nan"),
            "最长间隔 (s)": round(self.max_gap_s, 2),
            "断记次数": self.gap_count,
            "覆盖率": round(self.coverage, 4),
        }


def cadence_stats(index: pd.DatetimeIndex, gap_factor: float = 5.0) -> CadenceStats:
    """由时间索引统计采样间隔、断记与覆盖率。"""
    stamps = np.asarray(index.to_numpy(dtype="datetime64[ns]")).astype("int64")
    if stamps.size < 2:
        return CadenceStats(len(stamps), 0.0, 0.0, 0, 0.0, 0.0, 0.0)
    diffs = np.diff(stamps) / 1e9
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_dt = float(np.median(diffs)) if diffs.size else 0.0
    max_gap = float(diffs.max()) if diffs.size else 0.0
    gap_count = int((diffs > gap_factor * median_dt).sum()) if median_dt > 0 else 0
    span = float(stamps[-1] - stamps[0]) / 1e9
    expected = span / median_dt + 1 if median_dt > 0 else 0.0
    coverage = len(stamps) / expected if expected > 0 else 0.0
    return CadenceStats(
        n_samples=int(stamps.size),
        median_dt_s=median_dt,
        max_gap_s=max_gap,
        gap_count=gap_count,
        span_s=span,
        coverage=min(coverage, 1.0),
        expected_n=expected,
    )


def stuck_mask(series: pd.Series, min_run: int) -> pd.Series:
    """标记长时间保持同一数值的采样（传感器卡死或读数冻结）。"""
    valid = series.notna()
    run_id = series.ne(series.shift()).cumsum()
    run_length = series.groupby(run_id).transform("size")
    return (run_length >= min_run) & valid


def longest_stuck_run(series: pd.Series) -> int:
    """最长连续相同读数长度（采样点）。"""
    if series.dropna().empty:
        return 0
    run_id = series.ne(series.shift()).cumsum()
    return int(series.groupby(run_id).transform("size").max())


def spike_mask(series: pd.Series, window: int, threshold: float) -> pd.Series:
    """标记相对滑动中值跳变超过阈值的采样（尖峰/毛刺）。"""
    min_periods = max(3, window // 3)
    baseline = series.rolling(window=window, center=True, min_periods=min_periods).median()
    return ((series - baseline).abs() > threshold) & series.notna()


def estimate_quantization_step(series: pd.Series, max_step: float = 1.0) -> float:
    """估计量化步长：取相邻不同取值之差的最小公约（众数）近似。"""
    values = np.sort(pd.unique(series.dropna().to_numpy()))
    if values.size < 2:
        return float("nan")
    diffs = np.diff(values)
    diffs = diffs[(diffs > 1e-9) & (diffs <= max_step)]
    if diffs.size == 0:
        return float("nan")
    # 取最小正差的多重化步长，抗少量跨越多个量级的跳变
    base = float(diffs.min())
    ratio = np.round(diffs / base)
    return round(base * float(np.median(ratio)), 6)


def probe_quality(
    series: pd.Series,
    cfg: QualityConfig,
    nominal_dt_s: float | None = None,
    exclude: pd.Series | None = None,
    jitter: pd.Series | None = None,
) -> dict[str, float]:
    """单探针质量指标。

    Args:
        series: 单探针原始读数（含异常码）。
        cfg: 质量判定阈值。
        nominal_dt_s: 标称采样间隔（秒）。缺省时由时间索引的中位间隔推断，
            用于把 `stuck_min_seconds` 换算成采样点数。
        exclude: 额外剔除掩码（跨探针离群），仅影响最小值/最大值与有效率统计口径。
        jitter: 抖动掩码（偏离局部中值超过阈值），同样计入剔除后的统计口径。
    """
    values = pd.to_numeric(series, errors="coerce")
    n_total = int(values.size)
    n_nan = int(values.isna().sum())
    sentinel = np.isin(values.to_numpy(dtype="float64", na_value=np.nan), np.asarray(cfg.sentinel_values))
    n_sentinel = int(sentinel.sum())
    outside = ((values < cfg.valid_min_c) | (values > cfg.valid_max_c)) & values.notna() & ~sentinel
    n_outside = int(outside.sum())
    valid = values.mask(invalid_mask(values, cfg))
    n_outlier = 0
    if exclude is not None and len(exclude):
        mask = exclude.reindex(valid.index).fillna(False).astype(bool)
        n_outlier = int((mask & valid.notna()).sum())
        valid = valid.mask(mask)
    n_jitter = 0
    if jitter is not None and len(jitter):
        jmask = jitter.reindex(valid.index).fillna(False).astype(bool)
        n_jitter = int((jmask & valid.notna()).sum())
        valid = valid.mask(jmask)

    dt = nominal_dt_s
    if dt is None and isinstance(series.index, pd.DatetimeIndex) and series.index.size > 1:
        gaps = np.diff(series.index.to_numpy(dtype="datetime64[ns]").astype("int64")) / 1e9
        dt = float(np.median(gaps)) if gaps.size else None
    stuck_samples = int(round((cfg.stuck_min_seconds or 0.0) / dt)) if dt and dt > 0 else 0
    stuck_samples = max(stuck_samples, 2)
    longest_run = longest_stuck_run(valid)

    n_stuck = int(stuck_mask(valid, stuck_samples).sum())
    n_spike = int(spike_mask(valid, cfg.spike_window, cfg.spike_threshold_c).sum())
    step = estimate_quantization_step(valid)
    n_valid = int(valid.notna().sum())

    return {
        "总采样数": n_total,
        "缺失采样": n_nan,
        "异常码采样": n_sentinel,
        "超范围采样": n_outside,
        "有效采样": n_valid,
        "有效率": n_valid / n_total if n_total else 0.0,
        "卡值采样": n_stuck,
        "最长卡值 (min)": round(longest_run * dt / 60.0, 2) if (dt and dt > 0 and n_valid) else float("nan"),
        "尖峰采样": n_spike,
        "离群采样": n_outlier,
        "抖动采样": n_jitter,
        "量化步长 (C)": step,
        "最小值 (C)": float(valid.min()) if n_valid else float("nan"),
        "最大值 (C)": float(valid.max()) if n_valid else float("nan"),
    }
