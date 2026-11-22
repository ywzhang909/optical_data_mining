"""跨文件聚合：逐日统计、日变化、探针一致性、湍流参数与 Cn2 反演。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from .config import AnalysisConfig
from .metrics import cn2_from_ct2, ct2_from_structure


def concat_minute(results: list[pd.DataFrame]) -> pd.DataFrame:
    """拼接分钟表并补上日期列。"""
    frames = [frame for frame in results if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["ts", "探针", "均值", "标准差", "最小", "最大", "采样数", "覆盖率", "日期"])
    minute = pd.concat(frames, ignore_index=True)
    minute["ts"] = pd.to_datetime(minute["ts"])
    minute["日期"] = minute["ts"].dt.strftime("%Y-%m-%d")
    return minute.sort_values(["ts", "探针"]).reset_index(drop=True)


def concat_block(results: list[pd.DataFrame]) -> pd.DataFrame:
    """拼接分块脉动表并补上日期列。"""
    frames = [frame for frame in results if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["分块起点", "探针", "日期"])
    block = pd.concat(frames, ignore_index=True)
    block["分块起点"] = pd.to_datetime(block["分块起点"])
    block["日期"] = block["分块起点"].dt.strftime("%Y-%m-%d")
    return block.sort_values(["分块起点", "探针"]).reset_index(drop=True)


def probe_day_summary(minute: pd.DataFrame) -> pd.DataFrame:
    """逐日逐探针的温度统计。"""
    if minute.empty:
        return pd.DataFrame()
    grouped = minute.groupby(["日期", "探针"], observed=True)
    summary = grouped.agg(
        有效分钟=("均值", "count"),
        平均温度=("均值", "mean"),
        逐分钟标准差=("均值", "std"),
        日内最低=("最小", "min"),
        日内最高=("最大", "max"),
        分钟覆盖率=("覆盖率", "mean"),
    ).reset_index()
    summary["日内温差"] = summary["日内最高"] - summary["日内最低"]
    quantiles = grouped["均值"].quantile([0.05, 0.95]).unstack()
    quantiles.columns = ["P05", "P95"]
    summary = summary.merge(quantiles.reset_index(), on=["日期", "探针"], how="left")
    return summary


def probe_overall_summary(minute: pd.DataFrame) -> pd.DataFrame:
    """全时段逐探针汇总（报告主表）。"""
    if minute.empty:
        return pd.DataFrame()
    grouped = minute.groupby("探针", observed=True)
    summary = grouped.agg(
        有效分钟=("均值", "count"),
        平均温度=("均值", "mean"),
        温度标准差=("均值", "std"),
        最低温度=("最小", "min"),
        最高温度=("最大", "max"),
        平均覆盖率=("覆盖率", "mean"),
    ).reset_index()
    spread = grouped["均值"].quantile([0.01, 0.99]).unstack()
    spread.columns = ["P01", "P99"]
    summary = summary.merge(spread.reset_index(), on="探针", how="left")
    return summary.sort_values("探针").reset_index(drop=True)


def diurnal_profile(minute: pd.DataFrame, min_span_hours: float = 6.0) -> pd.DataFrame:
    """平均日变化：按小时统计各探针温度均值。

    时段过短（不足 `min_span_hours`）时小时均值不足以代表日内节律，直接返回空表，
    避免报告里出现"日变化幅度 0.00 °C"这类误导性结论。
    """
    if minute.empty:
        return pd.DataFrame()
    span_hours = (minute["ts"].max() - minute["ts"].min()).total_seconds() / 3600.0
    if span_hours < min_span_hours:
        return pd.DataFrame()
    frame = minute.copy()
    frame["小时"] = frame["ts"].dt.hour
    return frame.groupby(["探针", "小时"], observed=True)["均值"].mean().reset_index(name="平均温度")


def daily_mean_series(minute: pd.DataFrame) -> pd.DataFrame:
    """逐日平均温度（探针 × 日期）。"""
    if minute.empty:
        return pd.DataFrame()
    frame = minute.copy()
    frame["日期"] = frame["ts"].dt.strftime("%m-%d")
    return frame.groupby(["探针", "日期"], observed=True)["均值"].mean().reset_index(name="平均温度")


def hourly_heatmap(minute: pd.DataFrame, probes: tuple[str, ...]) -> pd.DataFrame:
    """日期 × 小时 的温度矩阵（用于热图）。"""
    if minute.empty:
        return pd.DataFrame()
    frame = minute[minute["探针"].isin(probes)].copy()
    frame["小时"] = frame["ts"].dt.floor("h")
    return frame.groupby(["探针", "小时"], observed=True)["均值"].mean().unstack("探针")


def ensemble_spread(minute: pd.DataFrame) -> pd.DataFrame:
    """探针间瞬时离散度：每分钟跨探针标准差，反映仪器间随机不一致。"""
    if minute.empty:
        return pd.DataFrame()
    pivot = minute.pivot_table(index="ts", columns="探针", values="均值", observed=True)
    spread = pivot.std(axis=1, ddof=1).rename("探针间标准差")
    return spread.to_frame().reset_index()


def anomaly_correlation(minute: pd.DataFrame, window_minutes: int = 60) -> tuple[pd.DataFrame, list[str]]:
    """去趋势后的探针间相关矩阵与聚类排序。

    温度日变化幅度远大于脉动幅度，直接求相关会被共同趋势主导，
    因此先用滑动中值去趋势，再做相关分析。
    """
    if minute.empty:
        return pd.DataFrame(), []
    pivot = minute.pivot_table(index="ts", columns="探针", values="均值", observed=True)
    pivot = pivot.interpolate(limit=5, limit_area="inside")
    span_minutes = (pivot.index.max() - pivot.index.min()).total_seconds() / 60.0 if pivot.index.size else 0.0
    # 去趋势窗口必须显著短于分析时段，否则连脉动一起被削掉，只剩噪声相关的假象
    window = int(min(window_minutes, span_minutes // 4))
    if span_minutes < 30 or window < 5:
        return pd.DataFrame(), []
    baseline = pivot.rolling(window=window, center=True, min_periods=max(5, window // 10)).median()
    residual = pivot - baseline
    corr = residual.corr(min_periods=max(10, window // 4))
    order = cluster_order(corr)
    return corr, order


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """基于 1-corr 距离的层次聚类排序，让相近探针在热图上相邻。"""
    if corr.empty or corr.shape[0] < 3:
        return list(corr.columns) if not corr.empty else []
    # pandas 3 的 copy-on-write 可能返回只读视图，聚类需要可写数组
    filled = np.array(corr.fillna(0.0).to_numpy(dtype="float64"), dtype="float64", copy=True)
    np.fill_diagonal(filled, 1.0)
    distance = np.clip(1.0 - filled, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    try:
        linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
        leaves = hierarchy.leaves_list(linkage)
    except ValueError:
        return list(corr.columns)
    return [str(corr.columns[idx]) for idx in leaves]


def attach_wind(block: pd.DataFrame, wind_minute: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把风速分钟均值聚合到分块起点，并计算 C_T² 与 Cn²。"""
    if block.empty:
        return block
    frame = block.copy()
    if wind_minute.empty:
        frame["风速"] = math.nan
        return frame
    wind = wind_minute.copy()
    wind["分块起点"] = pd.to_datetime(wind["ts"]).dt.floor(rule)
    wind_block = wind.groupby("分块起点", observed=True)["数值"].mean().rename("风速").reset_index()
    return frame.merge(wind_block, on="分块起点", how="left")


def turbulence_parameters(
    block: pd.DataFrame,
    cfg: AnalysisConfig,
    pressure_hpa: float,
) -> pd.DataFrame:
    """由结构函数得到 C_T²，再由 Gladstone 关系反演 Cn²。"""
    if block.empty:
        return block
    frame = block.copy()
    d_columns = {name: name for name in frame.columns if str(name).startswith("D_")}
    ct2_values = []
    for _, row in frame.iterrows():
        d_tau = {}
        for name in d_columns:
            try:
                lag = float(str(name)[2:].rstrip("s"))
            except ValueError:
                continue
            d_tau[lag] = row[name]
        floor = row.get("噪声底", 0.0)
        floor = 0.0 if pd.isna(floor) else float(floor)
        ct2_values.append(
            ct2_from_structure(d_tau, row.get("风速", math.nan), cfg.structure_fit_lags_s, noise_floor=floor)
        )
    frame["C_T2"] = ct2_values
    frame["Cn2"] = [
        cn2_from_ct2(value, temp, pressure_hpa) for value, temp in zip(frame["C_T2"], frame["均值"], strict=False)
    ]
    return frame


def turbulence_summary(block: pd.DataFrame) -> pd.DataFrame:
    """逐日逐探针的湍流参数汇总（取分块中位数，抗短时离群）。"""
    if block.empty:
        return pd.DataFrame()
    grouped = block.groupby(["日期", "探针"], observed=True)
    summary = grouped.agg(
        有效分块=("标准差", "count"),
        标准差中位数=("标准差", "median"),
        标准差P90=("标准差", lambda s: s.quantile(0.9)),
        谱斜率中位数=("谱斜率", "median"),
        结构函数斜率中位数=("结构函数斜率", "median"),
        C_T2中位数=("C_T2", "median"),
        Cn2中位数=("Cn2", "median"),
        波动信噪比中位数=("波动信噪比", "median"),
    ).reset_index()
    return summary


def structure_curve(block: pd.DataFrame, probes: tuple[str, ...]) -> pd.DataFrame:
    """结构函数 D_T(τ) 的平均曲线（探针 × 滞后），附各探针的噪声底中位数。

    纵轴保留**原始** D_T（含量化白噪声偏置），报告与图表同时给出噪声底，
    读者可以直接看出哪一段滞后已被噪声主导。
    """
    if block.empty:
        return pd.DataFrame()
    d_columns = [name for name in block.columns if str(name).startswith("D_")]
    frame = block[block["探针"].isin(probes)][["探针", *d_columns]]
    long = frame.melt(id_vars="探针", var_name="滞后", value_name="D_T").dropna()
    long["滞后秒"] = long["滞后"].str.removeprefix("D_").str.removesuffix("s").astype(float)
    curve = long.groupby(["探针", "滞后秒"], observed=True)["D_T"].median().reset_index()
    if "噪声底" in block:
        floors = block[block["探针"].isin(probes)].groupby("探针", observed=True)["噪声底"].median()
        curve["噪声底"] = curve["探针"].map(floors)
    return curve


def merge_spectra(spectrum: pd.DataFrame, probes: tuple[str, ...] = ()) -> pd.DataFrame:
    """把逐文件的中位功率谱合成全时段中位谱（探针 × 频率）。

    逐文件的谱曲线直接拼起来会得到 159 条 × 952 点的锯齿（每个频率上散落着不同小时的
    谱值），对数坐标下看起来像一片噪声带。这里按 (探针, 频率) 取中位数，得到每个探针
    一条平滑的代表谱。
    """
    if spectrum.empty:
        return spectrum
    subset = spectrum[spectrum["探针"].isin(probes)] if probes else spectrum
    if subset.empty:
        return subset
    return (
        subset.groupby(["探针", "频率"], observed=True)["功率"]
        .median()
        .reset_index()
        .sort_values(["探针", "频率"], ignore_index=True)
    )


def spectrum_curves(block: pd.DataFrame, probes: tuple[str, ...]) -> pd.DataFrame:
    """谱斜率相对 -5/3 的偏离汇总（用于表格展示）。"""
    if block.empty or "谱斜率" not in block:
        return pd.DataFrame()
    subset = block[block["探针"].isin(probes)]
    grouped = subset.groupby("探针", observed=True)["谱斜率"]
    summary = pd.DataFrame(
        {
            "谱斜率中位数": grouped.median(),
            "谱斜率P10": grouped.quantile(0.1),
            "谱斜率P90": grouped.quantile(0.9),
            "有效分块": grouped.count(),
        }
    ).reset_index()
    if "谱拟合优度" in subset:
        r2 = subset.groupby("探针", observed=True)["谱拟合优度"].median().rename("谱拟合R²中位数")
        summary = summary.merge(r2.reset_index(), on="探针", how="left")
    if "谱拟合上限" in subset:
        band = subset.groupby("探针", observed=True).agg(
            谱拟合下限中位数=("谱拟合下限", "median"),
            谱拟合上限中位数=("谱拟合上限", "median"),
        )
        summary = summary.merge(band.reset_index(), on="探针", how="left")
    return summary


def merge_device_cn2(block: pd.DataFrame, device_cn2: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把设备输出的 Cn2 聚合到分块，与反演值逐分块对齐。"""
    if block.empty or device_cn2.empty:
        return pd.DataFrame()
    device = device_cn2.copy()
    device["分块起点"] = pd.to_datetime(device["ts"]).dt.floor(rule)
    aggregated = device.groupby(["分块起点", "探针"], observed=True)["数值"].median().rename("设备Cn2").reset_index()
    merged = block[["分块起点", "探针", "Cn2", "均值"]].merge(aggregated, on=["分块起点", "探针"], how="inner")
    return merged[(merged["Cn2"] > 0) & (merged["设备Cn2"] > 0)].reset_index(drop=True)


def summarize_cn2_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    """反演 Cn² 与设备 Cn² 的一致性指标。"""
    if merged.empty:
        return pd.DataFrame()
    ratio = merged["Cn2"] / merged["设备Cn2"]
    log_corr = np.corrcoef(np.log10(merged["Cn2"]), np.log10(merged["设备Cn2"]))[0, 1]
    return pd.DataFrame(
        {
            "指标": ["样本数", "比值中位数", "比值P10", "比值P90", "对数相关系数"],
            "数值": [
                len(merged),
                float(ratio.median()),
                float(ratio.quantile(0.1)),
                float(ratio.quantile(0.9)),
                float(log_corr),
            ],
        }
    )
