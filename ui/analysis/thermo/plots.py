"""报告图表：统一样式的中文 matplotlib 图。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"]


def setup_style(dpi: int = 130) -> None:
    """统一图表样式与中文字体。"""
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.sans-serif": _CJK_FONTS,
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.edgecolor": "#666666",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.autolayout": True,
        }
    )


def _palette(count: int) -> list[tuple[float, float, float]]:
    base = plt.get_cmap("tab20")
    return [base(idx % 20) for idx in range(count)]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def coverage_heatmap(quality_daily: pd.DataFrame, path: Path) -> Path | None:
    """日期 × 探针 的有效率热图。"""
    if quality_daily.empty:
        return None
    pivot = quality_daily.pivot_table(index="探针", columns="日期", values="有效率", observed=True)
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * pivot.shape[1]), 0.42 * pivot.shape[0] + 2.0))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]), [str(col)[5:] for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=7, color="#222222")
    ax.set_title("逐日逐探针有效采样率")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="有效率")
    return _save(fig, path)


def temperature_timeseries(minute: pd.DataFrame, probes: tuple[str, ...], path: Path) -> Path | None:
    """全时段温度分钟均值序列。"""
    if minute.empty:
        return None
    subset = minute[minute["探针"].isin(probes)]
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for color, (probe, frame) in zip(_palette(len(probes)), subset.groupby("探针", observed=True), strict=False):
        ax.plot(frame["ts"], frame["均值"], lw=0.8, color=color, label=probe)
    ax.set_ylabel("温度 (°C)")
    ax.set_title("温度时间序列（分钟均值）")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(ncol=min(8, len(probes)), loc="upper right")
    return _save(fig, path)


def daily_mean_lines(daily: pd.DataFrame, path: Path) -> Path | None:
    """逐日平均温度趋势。"""
    if daily.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for color, (probe, frame) in zip(
        _palette(daily["探针"].nunique()), daily.groupby("探针", observed=True), strict=False
    ):
        frame = frame.sort_values("日期")
        ax.plot(frame["日期"], frame["平均温度"], marker="o", ms=3, lw=1.2, color=color, label=probe)
    ax.set_ylabel("日平均温度 (°C)")
    ax.set_title("逐日平均温度")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=4, fontsize=7)
    return _save(fig, path)


def diurnal_cycle(diurnal: pd.DataFrame, probes: tuple[str, ...], path: Path) -> Path | None:
    """平均日变化曲线。"""
    if diurnal.empty:
        return None
    subset = diurnal[diurnal["探针"].isin(probes)]
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for color, (probe, frame) in zip(_palette(len(probes)), subset.groupby("探针", observed=True), strict=False):
        frame = frame.sort_values("小时")
        ax.plot(frame["小时"], frame["平均温度"], marker="o", ms=3, lw=1.2, color=color, label=probe)
    ensemble = subset.groupby("小时", observed=True)["平均温度"].mean()
    ax.plot(ensemble.index, ensemble.to_numpy(), lw=2.6, color="#333333", alpha=0.65, label="探针均值")
    ax.set_xlabel("小时 (本地时)")
    ax.set_ylabel("平均温度 (°C)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("平均日变化（全时段按小时合成）")
    ax.legend(ncol=4, fontsize=7)
    return _save(fig, path)


def distribution_box(minute: pd.DataFrame, probes: tuple[str, ...], path: Path) -> Path | None:
    """探针温度分布箱线图。"""
    if minute.empty:
        return None
    subset = minute[minute["探针"].isin(probes)]
    if subset.empty:
        return None
    ordered = [probe for probe in probes if probe in set(subset["探针"])]
    data = [subset.loc[subset["探针"] == probe, "均值"].dropna().to_numpy() for probe in ordered]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(ordered) + 3), 4.0))
    ax.boxplot(data, tick_labels=ordered, showfliers=False)
    ax.set_ylabel("分钟均值温度 (°C)")
    ax.set_title("探针温度分布（箱线：四分位，须：1.5 IQR）")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def correlation_heatmap(corr: pd.DataFrame, order: list[str], path: Path) -> Path | None:
    """去趋势后探针间相关矩阵热图。"""
    if corr.empty:
        return None
    ordered = [probe for probe in order if probe in corr.columns] or list(corr.columns)
    matrix = corr.loc[ordered, ordered].to_numpy()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(ordered) + 3), max(5.0, 0.45 * len(ordered) + 2.5)))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ordered)), ordered, fontsize=7)
    ax.set_title("探针间相关系数（去趋势后）")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    return _save(fig, path)


def psd_overlay(spectrum: pd.DataFrame, probes: tuple[str, ...], band: tuple[float, float], path: Path) -> Path | None:
    """代表性功率谱与 -5/3 参考线。

    `spectrum` 已是全时段中位谱（每探针一条曲线，由 `aggregate.merge_spectra` 合成）。
    纵轴按功率分位数裁剪，避免个别分块的尖峰把整幅图压扁。
    """
    if spectrum.empty:
        return None
    subset = spectrum[spectrum["探针"].isin(probes)]
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for color, (probe, frame) in zip(_palette(len(probes)), subset.groupby("探针", observed=True), strict=False):
        frame = frame[(frame["频率"] > 0) & (frame["功率"] > 0)].sort_values("频率")
        if frame.empty:
            continue
        ax.loglog(frame["频率"], frame["功率"], lw=1.1, color=color, label=str(probe))
    reference = subset[
        (subset["频率"] >= band[0]) & (subset["频率"] <= band[1]) & (subset["功率"] > 0) & (subset["频率"] > 0)
    ]
    if not reference.empty:
        freq = np.array(sorted(reference["频率"].unique()))
        anchor = float(np.median(reference["功率"]))
        center = float(np.median(freq))
        if anchor > 0 and center > 0:
            ax.loglog(freq, anchor * (freq / center) ** (-5 / 3), "k--", lw=1.2, label="-5/3 参考")
    ax.axvspan(band[0], band[1], color="#999999", alpha=0.12, label="拟合频段")

    positive = subset.loc[subset["功率"] > 0, "功率"].to_numpy(dtype="float64")
    if positive.size:
        low, high = np.percentile(positive, [0.5, 99.5])
        if low > 0 and high > low:
            ax.set_ylim(low * 0.5, high * 2.0)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("功率谱密度 (°C²/Hz)")
    ax.set_title("去趋势温度脉动功率谱（全时段中位数）")
    ax.legend(ncol=2, fontsize=7)
    return _save(fig, path)


def structure_plot(curve: pd.DataFrame, path: Path) -> Path | None:
    """结构函数 D_T(τ) 与 2/3 参考线，并标出量化噪声底。

    纵轴用原始 D_T：白噪声会在 τ→0 处把曲线抬成一个平台，噪声底以上的部分才是
    惯性子区信息。图中横虚线为各探针噪声底的中位数。
    """
    if curve.empty:
        return None
    positive = curve[curve["D_T"] > 0]
    if positive.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for color, (probe, frame) in zip(
        _palette(curve["探针"].nunique()), positive.groupby("探针", observed=True), strict=False
    ):
        frame = frame.sort_values("滞后秒")
        ax.loglog(frame["滞后秒"], frame["D_T"], marker="o", ms=3, lw=1.1, color=color, label=str(probe))
    taus = np.array(sorted(positive["滞后秒"].unique()), dtype="float64")
    if taus.size >= 2 and taus.min() > 0:
        anchor = float(np.median(positive["D_T"]))
        center = float(np.sqrt(taus.min() * taus.max()))
        if anchor > 0:
            ax.loglog(taus, anchor * (taus / center) ** (2 / 3), "k--", lw=1.2, label="τ^(2/3) 参考")
    if "噪声底" in positive:
        floor = float(np.nanmedian(positive["噪声底"].to_numpy(dtype="float64")))
        if math.isfinite(floor) and floor > 0:
            ax.axhline(floor, color="#c0392b", lw=1.0, ls=":", label=f"噪声底≈{floor:.1e} °C²")
    ax.set_xlabel("滞后 τ (s)")
    ax.set_ylabel("D_T(τ) (°C²，含噪声)")
    ax.set_title("温度结构函数（分块中位数）")
    ax.legend(ncol=2, fontsize=7)
    return _save(fig, path)


def environment_panel(
    channels: dict[str, Any],
    wind: pd.DataFrame | None,
    path: Path,
) -> Path | None:
    """环境要素总览：湿度、能见度、设备 Cn²、风速。

    各通道采样率与量纲不同，因此每个子图独立纵轴；曲线为各探针分钟均值，
    半透明细线是单探针，粗线是探针中位数，避免个别掉线探针主导观感。
    """
    series: list[tuple[str, pd.DataFrame, str, bool]] = []
    if channels:
        for key in ("humidity", "visibility", "cn2"):
            channel = channels.get(key)
            if channel is not None and not channel.minute.empty:
                series.append((channel.label, channel.minute, channel.unit, key == "cn2"))
    if wind is not None and not wind.empty:
        series.append(("风速", wind, "m/s", False))
    if not series:
        return None

    columns = 2
    rows = (len(series) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(11.0, 2.6 * rows), sharex=True, squeeze=False)
    for ax, (label, minute, unit, log_scale) in zip(axes.ravel(), series, strict=False):
        frame = minute.copy()
        frame["ts"] = pd.to_datetime(frame["ts"])
        for _, probe_frame in frame.groupby("探针", observed=True):
            probe_frame = probe_frame.sort_values("ts")
            ax.plot(probe_frame["ts"], probe_frame["数值"], lw=0.5, alpha=0.35, color="#4c72b0")
        median = frame.groupby("ts", observed=True)["数值"].median().sort_index()
        ax.plot(median.index, median.to_numpy(), lw=1.2, color="#c0392b", label="探针中位数")
        if log_scale and (median > 0).any():
            ax.set_yscale("log")
        ax.set_title(f"{label}（{unit}）" if unit else label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes.ravel()[len(series) :]:
        ax.set_visible(False)
    axes.ravel()[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes.ravel()[0].xaxis.set_major_locator(mdates.DayLocator())
    fig.suptitle("环境要素时间序列（分钟均值）", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def sigma_timeseries(block: pd.DataFrame, probes: tuple[str, ...], path: Path) -> Path | None:
    """10 min 尺度温度脉动标准差时间序列。"""
    if block.empty:
        return None
    subset = block[block["探针"].isin(probes)]
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for color, (probe, frame) in zip(_palette(len(probes)), subset.groupby("探针", observed=True), strict=False):
        frame = frame.sort_values("分块起点")
        ax.plot(frame["分块起点"], frame["标准差"], lw=0.9, color=color, label=probe)
    ax.set_ylabel("去趋势标准差 σ_T (°C)")
    ax.set_title("10 min 尺度温度脉动强度")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(ncol=min(8, len(probes)), fontsize=7)
    return _save(fig, path)


def probe_spread(block: pd.DataFrame, path: Path) -> Path | None:
    """分块尺度探针间温度差异。"""
    if block.empty:
        return None
    pivot = block.pivot_table(index="分块起点", columns="探针", values="均值", observed=True)
    if pivot.shape[1] < 2:
        return None
    spread = pivot.max(axis=1) - pivot.min(axis=1)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(spread.index, spread.to_numpy(), lw=0.8, color="#2c7fb8")
    ax.set_ylabel("探针间极差 (°C)")
    ax.set_title("分块探针间温度极差（一致性检查）")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    return _save(fig, path)


def cn2_comparison(merged: pd.DataFrame, path: Path) -> Path | None:
    """反演 Cn² 与设备输出 Cn² 的对比散点。"""
    if merged.empty:
        return None
    valid = merged[(merged["Cn2"] > 0) & (merged["设备Cn2"] > 0)]
    if valid.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.loglog(valid["设备Cn2"], valid["Cn2"], ".", ms=4, alpha=0.5, color="#1f77b4")
    low = max(float(valid["设备Cn2"].min()), float(valid["Cn2"].min()))
    high = min(float(valid["设备Cn2"].max()), float(valid["Cn2"].max()))
    if high > low:
        ax.loglog([low, high], [low, high], "k--", lw=1.0, label="1:1")
    ax.set_xlabel("设备输出 Cn² (m^-2/3)")
    ax.set_ylabel("本报告反演 Cn² (m^-2/3)")
    ax.set_title(f"Cn² 交叉验证（{len(valid)} 个分块样本）")
    ax.legend()
    return _save(fig, path)
