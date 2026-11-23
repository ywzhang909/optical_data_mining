"""分析流水线：发现数据 → 解析缓存 → 聚合统计 → 生成图表与报告。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from . import aggregate, channels, slices, spatial
from .config import ThermoReportConfig
from .discovery import Inventory, discover
from .ingest import FileResult, load_aux_minute_means, process_source
from .report import write_report

TEMPERATURE_CHANNEL = "Temperature"


@dataclass
class AnalysisResult:
    """一次完整分析的全部中间结果。"""

    cfg: ThermoReportConfig
    inventory: Inventory
    minute: pd.DataFrame
    block: pd.DataFrame
    spectrum: pd.DataFrame
    quality_daily: pd.DataFrame
    quality_overall: pd.DataFrame
    cadence: pd.DataFrame
    daily_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    diurnal: pd.DataFrame
    daily_means: pd.DataFrame
    correlation: pd.DataFrame
    correlation_order: list[str]
    spread: pd.DataFrame
    structure: pd.DataFrame
    spectral_summary: pd.DataFrame
    turbulence_summary: pd.DataFrame
    cn2_merged: pd.DataFrame
    cn2_summary: pd.DataFrame
    wind: pd.DataFrame
    channels: dict[str, channels.ChannelResult] = field(default_factory=dict)
    slices: pd.DataFrame = field(default_factory=pd.DataFrame)
    spatial: pd.DataFrame = field(default_factory=pd.DataFrame)
    spatial_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    facts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    probes: tuple[str, ...] = ()
    valid_probes: tuple[str, ...] = ()
    failed_files: tuple[str, ...] = ()


def _parse_files(
    cfg: ThermoReportConfig,
    sources: list,
    use_cache: bool,
) -> tuple[list[FileResult], list[str]]:
    """逐个解析文件；单个文件失败只跳过并记录，不中断整批分析。"""
    results: list[FileResult] = []
    failed: list[str] = []
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        logger.debug("解析 {}/{} {}", index, total, source.path.name)
        try:
            results.append(process_source(source, cfg, use_cache=use_cache))
        except Exception as exc:  # 单个坏文件不应让整周分析失败
            logger.warning("跳过 {}: {}: {}", source.path, type(exc).__name__, exc)
            failed.append(f"{source.day:%Y-%m-%d} {source.path.name}")
    return results, failed


def _quality_tables(results: list[FileResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for result in results:
        if result.quality.empty:
            continue
        frame = result.quality.copy()
        frame["日期"] = result.source.day.strftime("%Y-%m-%d")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    daily = pd.concat(frames, ignore_index=True)
    count_columns = [
        "总采样数",
        "缺失采样",
        "异常码采样",
        "超范围采样",
        "有效采样",
        "卡值采样",
        "尖峰采样",
        "离群采样",
        "抖动采样",
    ]
    grouped = daily.groupby(["日期", "探针"], observed=True)[count_columns].sum()
    grouped["有效率"] = grouped["有效采样"] / grouped["总采样数"].replace(0, np.nan)
    grouped = grouped.reset_index()

    overall_group = daily.groupby("探针", observed=True)[count_columns].sum()
    overall_group["有效率"] = overall_group["有效采样"] / overall_group["总采样数"].replace(0, np.nan)
    overall = overall_group.reset_index()
    # 极值跨文件合并：最小值取各文件最小值的最小，最大值取最大；量化步长取各文件最细步长
    extrema = daily.groupby("探针", observed=True).agg(
        最小值_C=("最小值 (C)", "min"),
        最大值_C=("最大值 (C)", "max"),
        量化步长_C=("量化步长 (C)", "min"),
        最长卡值_min=("最长卡值 (min)", "max"),
    )
    overall = overall.merge(
        extrema.rename(
            columns={
                "最小值_C": "最小值 (C)",
                "最大值_C": "最大值 (C)",
                "量化步长_C": "量化步长 (C)",
                "最长卡值_min": "最长卡值 (min)",
            }
        ).reset_index(),
        on="探针",
        how="left",
    )
    return grouped, overall


def _cadence_table(results: list[FileResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"日期": result.source.day.strftime("%Y-%m-%d"), "文件": result.source.path.name}
        row.update(result.cadence)
        row["首条时间"] = result.first_ts.isoformat() if result.first_ts is not None else ""
        row["末条时间"] = result.last_ts.isoformat() if result.last_ts is not None else ""
        row["备注"] = "；".join(result.notes)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_aux_frames(
    cfg: ThermoReportConfig,
    sources: list,
    *,
    value_min: float = float("-inf"),
    value_max: float = 1e6,
) -> pd.DataFrame:
    """逐个加载辅助通道文件；单个文件异常只跳过，不影响整批。"""
    frames = []
    for source in sources:
        try:
            frame = load_aux_minute_means(
                source,
                cfg.analysis.minute_rule,
                cache_dir=cfg.paths.cache_dir,
                value_min=value_min,
                value_max=value_max,
            )
        except Exception as exc:  # 辅助通道缺失/损坏不应中断主分析
            logger.warning("跳过辅助文件 {}: {}: {}", source.path, type(exc).__name__, exc)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["ts", "探针", "数值"])
    return pd.concat(frames, ignore_index=True)


def _load_wind(cfg: ThermoReportConfig, inventory: Inventory) -> pd.DataFrame:
    """加载风速辅助数据，自动选取有效率最高的探针作为站点风速。"""
    sources = inventory.by_channel(cfg.paths.wind_subdir)
    if not sources:
        return pd.DataFrame(columns=["ts", "探针", "数值"])
    wind = _load_aux_frames(cfg, sources, value_min=0.0, value_max=60.0)
    if wind.empty:
        return wind
    counts = wind.groupby("探针", observed=True)["数值"].count()
    if counts.empty:
        return pd.DataFrame(columns=["ts", "探针", "数值"])
    probe = str(counts.idxmax())
    logger.info("风速采用探针 {}（有效分钟 {}）", probe, int(counts.max()))
    selected = wind[wind["探针"] == probe].copy()
    selected["探针"] = probe
    return selected


def _spatial_tables(
    cfg: ThermoReportConfig,
    minute: pd.DataFrame,
    channel_results: dict[str, channels.ChannelResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """温度与环境要素各通道的跨探针空间平均（逐分钟 + 逐日）。

    温度用分钟均值 ``均值`` 列，环境要素用 ``数值`` 列；探针高度取自
    ``cfg.probes.heights_m``，缺失时按 ``cfg.spatial.method`` 退化为等权聚合。
    单路异常只记日志，不中断整批分析。
    """
    empty = pd.DataFrame(columns=spatial.SPATIAL_COLUMNS)
    empty_daily = pd.DataFrame(columns=spatial.SPATIAL_DAILY_COLUMNS)
    if not cfg.spatial.enabled:
        return empty, empty_daily

    jobs: list[tuple[str, str, str, str, pd.DataFrame]] = [
        ("温度", "°C", "均值", "temperature", minute),
    ]
    for key, channel in (channel_results or {}).items():
        if channel.minute is None or channel.minute.empty:
            continue
        jobs.append((channel.label, channel.unit, "数值", key, channel.minute))

    series_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    for label, unit, value_col, key, frame in jobs:
        try:
            series = spatial.spatial_series(
                frame,
                value_col,
                cfg.probes.heights_m,
                method=cfg.spatial.method,
                min_probes=cfg.spatial.min_probes,
            )
            if series.empty:
                continue
            series.insert(0, "单位", unit)
            series.insert(0, "通道", label)
            series["通道键"] = key
            series_parts.append(series)
            daily_parts.append(spatial.daily_spatial_summary(series, label, unit))
        except Exception as exc:  # 单通道异常不应中断整批分析
            logger.warning("空间平均 {} 计算失败: {}: {}", label, type(exc).__name__, exc)
            continue

    if not series_parts:
        return empty, empty_daily
    merged = pd.concat(series_parts, ignore_index=True)
    merged = merged[["通道", "通道键", "单位", "ts", "空间平均", "探针数", "方式"]]
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else empty_daily
    return merged, daily


def _facts(result: AnalysisResult) -> dict[str, Any]:
    """提取报告结论所需的关键数字。"""
    facts: dict[str, Any] = {}
    minute = result.minute
    block = result.block

    facts["探针数"] = len(result.probes)
    facts["有效探针数"] = len(result.valid_probes)
    facts["有效探针"] = ", ".join(result.valid_probes) if result.valid_probes else "无"
    facts["无效探针"] = ", ".join(p for p in result.probes if p not in result.valid_probes) or "无"
    facts["天数"] = int(minute["日期"].nunique()) if not minute.empty else 0
    facts["起止时间"] = (
        f"{minute['ts'].min():%Y-%m-%d %H:%M} – {minute['ts'].max():%Y-%m-%d %H:%M}" if not minute.empty else ""
    )
    facts["文件数"] = len(result.inventory.by_channel(TEMPERATURE_CHANNEL))
    facts["原始数据量MB"] = round(result.inventory.total_bytes(TEMPERATURE_CHANNEL) / 1024 / 1024, 1)
    facts["日变化门槛小时"] = result.cfg.analysis.min_span_hours_for_diurnal

    if not minute.empty:
        values = minute["均值"].to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        facts["平均温度"] = float(np.mean(values)) if values.size else float("nan")
        facts["温度范围"] = (float(np.min(values)), float(np.max(values))) if values.size else (np.nan, np.nan)
        span_hours = (minute["ts"].max() - minute["ts"].min()).total_seconds() / 3600.0
        facts["时长小时"] = float(span_hours)
        if span_hours >= result.cfg.analysis.min_span_hours_for_diurnal:
            hourly = minute.assign(小时=minute["ts"].dt.hour).groupby("小时", observed=True)["均值"].mean()
            facts["日变化幅度"] = float(hourly.max() - hourly.min()) if not hourly.empty else float("nan")
            facts["最热时次"] = int(hourly.idxmax()) if not hourly.empty else -1
            facts["最冷时次"] = int(hourly.idxmin()) if not hourly.empty else -1

    if not result.cadence.empty and "中位采样间隔 (s)" in result.cadence:
        dt = float(pd.to_numeric(result.cadence["中位采样间隔 (s)"], errors="coerce").median())
        facts["采样间隔"] = dt
        facts["采样率"] = 1.0 / dt if dt > 0 else float("nan")
    if not result.quality_overall.empty:
        steps = (
            result.quality_overall["量化步长 (C)"]
            if "量化步长 (C)" in result.quality_overall
            else pd.Series(dtype=float)
        )
        steps = pd.to_numeric(steps, errors="coerce").dropna()
        step = float(steps.median()) if not steps.empty else float("nan")
        facts["量化步长"] = step
        facts["量化噪声"] = step / np.sqrt(12.0) if np.isfinite(step) and step > 0 else float("nan")

    if not block.empty and "标准差" in block:
        sigma = block.groupby("探针", observed=True)["标准差"].median().dropna()
        facts["脉动标准差中位数"] = float(sigma.median()) if not sigma.empty else float("nan")
        snr = block.groupby("探针", observed=True)["波动信噪比"].median().dropna()
        facts["波动信噪比"] = float(snr.median()) if not snr.empty else float("nan")
        slope = block["谱斜率"].dropna()
        facts["谱斜率中位数"] = float(slope.median()) if not slope.empty else float("nan")
        sf = block["结构函数斜率"].dropna()
        facts["结构函数斜率中位数"] = float(sf.median()) if not sf.empty else float("nan")
        if "谱拟合优度" in block:
            r2 = block["谱拟合优度"].dropna()
            facts["谱拟合优度"] = float(r2.median()) if not r2.empty else float("nan")
        low, high = result.cfg.analysis.spectrum_band_hz
        facts["谱拟合频段"] = f"{low:g}–{high:g} Hz"
        if "谱拟合下限" in block:
            uses = block[["谱拟合下限", "谱拟合上限"]].dropna()
            if not uses.empty:
                facts["谱拟合频段"] = (
                    f"{uses['谱拟合下限'].median():.3g}–{uses['谱拟合上限'].median():.3g} Hz（分块中位数）"
                )

    if not result.spread.empty:
        facts["探针间极差中位数"] = float(result.spread["探针间标准差"].median())

    facts.update(_channel_facts(result))

    if not result.overall_summary.empty:
        best = result.overall_summary.sort_values("有效分钟", ascending=False)
        facts["数据最完整探针"] = str(best.iloc[0]["探针"]) if not best.empty else ""
        facts["有效率中位数"] = float(result.overall_summary["平均覆盖率"].median())

    if not result.spatial.empty:
        modes = result.spatial["方式"].dropna()
        facts["空间平均方式"] = str(modes.mode().iloc[0]) if not modes.empty else ""
        facts["空间平均通道数"] = int(result.spatial["通道"].nunique())
    return facts


def _channels(cfg: ThermoReportConfig, with_auxiliary: bool) -> tuple[str, ...]:
    """需要扫描的通道目录：温度必需，其余环境要素仅在需要辅助分析时纳入。"""
    subdirs = [cfg.paths.temperature_subdir]
    if with_auxiliary:
        subdirs.extend([cfg.paths.wind_subdir, cfg.paths.cn2_subdir])
        subdirs.extend(channel.subdir for channel in cfg.channels if channel.enabled)
    return tuple(dict.fromkeys(subdirs))


def _channel_facts(result: AnalysisResult) -> dict[str, Any]:
    """环境要素与湍流参数的关键数字（供结论摘要直接引用）。"""
    facts: dict[str, Any] = {}
    for channel in (result.channels or {}).values():
        if channel.minute.empty:
            continue
        label = channel.label
        # 全程为 0 的探针通常是未接入/失效的传感器，不能拉低通道级中位数
        probe_medians = channel.minute.groupby("探针", observed=True)["数值"].median()
        usable = probe_medians[probe_medians != 0].index
        minute = channel.minute[channel.minute["探针"].isin(usable)] if len(usable) else channel.minute
        values = pd.to_numeric(minute["数值"], errors="coerce").dropna()
        if values.empty:
            continue
        facts[f"{label}均值"] = float(values.mean())
        facts[f"{label}中位数"] = float(values.median())
        facts[f"{label}范围"] = (float(values.min()), float(values.max()))
        facts[f"{label}单位"] = channel.unit
        facts[f"{label}探针数"] = int(minute["探针"].nunique())

    if not result.wind.empty:
        wind = pd.to_numeric(result.wind["数值"], errors="coerce").dropna()
        if not wind.empty:
            facts["风速均值"] = float(wind.mean())
            facts["风速中位数"] = float(wind.median())
            facts["风速P95"] = float(wind.quantile(0.95))
            facts["风速探针"] = str(result.wind["探针"].iloc[0])

    block = result.block
    if not block.empty and "C_T2" in block:
        ct2 = pd.to_numeric(block["C_T2"], errors="coerce").dropna()
        cn2 = pd.to_numeric(block.get("Cn2", pd.Series(dtype=float)), errors="coerce").dropna()
        if not ct2.empty:
            facts["C_T2中位数"] = float(ct2.median())
            facts["C_T2范围"] = (float(ct2.min()), float(ct2.max()))
        if not cn2.empty:
            facts["Cn2中位数"] = float(cn2.median())
            facts["湍流有效分块"] = int(cn2.size)
    return facts


def analyze(
    cfg: ThermoReportConfig,
    *,
    days: tuple[str, ...] = (),
    max_files_per_day: int | None = None,
    use_cache: bool = True,
    with_auxiliary: bool = True,
) -> AnalysisResult:
    """执行完整分析（不含报告落盘）。"""
    started = time.perf_counter()
    inventory = discover(
        cfg.paths.data_root,
        channel_dirs=_channels(cfg, with_auxiliary),
        days=days,
        max_files_per_day=max_files_per_day,
    )
    sources = list(inventory.by_channel(cfg.paths.temperature_subdir))
    logger.info(
        "发现 {} 个温度文件，共 {:.1f} MB",
        len(sources),
        inventory.total_bytes(cfg.paths.temperature_subdir) / 1024 / 1024,
    )

    results, failed = _parse_files(cfg, sources, use_cache)
    if not results:
        raise RuntimeError("没有成功解析任何温度文件")

    probes = tuple(sorted({probe for result in results for probe in result.probes}, key=_probe_sort_key))
    if cfg.probes.include:
        probes = tuple(probe for probe in probes if probe in cfg.probes.include)
    if cfg.probes.exclude:
        probes = tuple(probe for probe in probes if probe not in cfg.probes.exclude)
    if not probes:
        raise RuntimeError("探针筛选后没有剩余探针，请检查 probes.include / probes.exclude")

    quality_daily, quality_overall = _quality_tables(results)
    cadence = _cadence_table(results)

    valid_probes = probes
    if not quality_overall.empty and cfg.quality.min_probe_valid_ratio > 0:
        ratios = dict(zip(quality_overall["探针"], quality_overall["有效率"], strict=False))
        valid_probes = tuple(p for p in probes if ratios.get(p, 0.0) >= cfg.quality.min_probe_valid_ratio)

    minute = aggregate.concat_minute([result.minute for result in results])
    block = aggregate.concat_block([result.block for result in results])
    spectrum = aggregate.merge_spectra(
        pd.concat([f for f in (result.spectrum for result in results) if not f.empty], ignore_index=True)
        if any(not result.spectrum.empty for result in results)
        else pd.DataFrame(columns=["探针", "频率", "功率"]),
        valid_probes,
    )

    wind = _load_wind(cfg, inventory) if with_auxiliary else pd.DataFrame(columns=["ts", "探针", "数值"])
    channel_results = channels.load_channels(cfg, inventory) if with_auxiliary else {}
    block = aggregate.attach_wind(block, wind, cfg.analysis.block_rule)
    if not block.empty:
        block = aggregate.turbulence_parameters(block, cfg.analysis, cfg.analysis.site_pressure_hpa)

    device_cn2 = (
        channel_results["cn2"].minute
        if "cn2" in channel_results and not channel_results["cn2"].minute.empty
        else pd.DataFrame(columns=["ts", "探针", "数值"])
    )
    cn2_merged = aggregate.merge_device_cn2(block, device_cn2, cfg.analysis.block_rule)
    cn2_summary = aggregate.summarize_cn2_agreement(cn2_merged)

    slice_frame = (
        slices.build_slice_frame(cfg, minute, block, channel_results, wind) if cfg.export.enabled else pd.DataFrame()
    )

    correlation, order = aggregate.anomaly_correlation(
        minute[minute["探针"].isin(valid_probes)] if valid_probes else minute,
        window_minutes=int(cfg.analysis.detrend_window_s // 60),
    )

    spatial_frame, spatial_daily = _spatial_tables(cfg, minute, channel_results)

    result = AnalysisResult(
        cfg=cfg,
        inventory=inventory,
        minute=minute,
        block=block,
        spectrum=spectrum,
        quality_daily=quality_daily,
        quality_overall=quality_overall,
        cadence=cadence,
        daily_summary=aggregate.probe_day_summary(minute),
        overall_summary=aggregate.probe_overall_summary(minute),
        diurnal=aggregate.diurnal_profile(minute, cfg.analysis.min_span_hours_for_diurnal),
        daily_means=aggregate.daily_mean_series(minute),
        correlation=correlation,
        correlation_order=order,
        spread=aggregate.ensemble_spread(minute).rename(columns={"探针间标准差": "探针间标准差"}),
        structure=aggregate.structure_curve(block, valid_probes),
        spectral_summary=aggregate.spectrum_curves(block, valid_probes),
        turbulence_summary=aggregate.turbulence_summary(block),
        cn2_merged=cn2_merged,
        cn2_summary=cn2_summary,
        wind=wind,
        channels=channel_results,
        slices=slice_frame,
        spatial=spatial_frame,
        spatial_daily=spatial_daily,
        probes=probes,
        valid_probes=valid_probes,
        failed_files=tuple(failed),
        runtime_s=time.perf_counter() - started,
    )
    result.facts = _facts(result)
    result.facts["缓存文件数"] = sum(1 for item in results if item.from_cache)
    result.facts["解析文件数"] = sum(1 for item in results if not item.from_cache)
    result.facts["解析失败数"] = len(failed)
    result.notes = _collect_notes(result)
    logger.info("分析完成，用时 {:.1f} s", result.runtime_s)
    return result


def _probe_sort_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = str(name).partition("-")
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, 0


def _collect_notes(result: AnalysisResult) -> list[str]:
    """汇总数据质量提示，供报告与日志使用。"""
    notes: list[str] = []
    if result.failed_files:
        shown = "、".join(result.failed_files[:3])
        more = f" 等 {len(result.failed_files)} 个" if len(result.failed_files) > 3 else ""
        notes.append(f"以下文件解析失败已跳过：{shown}{more}，其时段不参与统计")
    missing = [note for note in result.cadence.get("备注", pd.Series(dtype=str)).tolist() if note]
    notes.extend(sorted(set(missing))[:5])
    quality = result.quality_overall
    if not quality.empty and "离群采样" in quality:
        affected = quality[quality["离群采样"] > 0].sort_values("离群采样", ascending=False)
        if not affected.empty:
            names = "、".join(affected["探针"].astype(str).head(3))
            total = int(affected["离群采样"].sum())
            notes.append(f"探针 {names} 共 {total} 个采样与同刻跨探针中值相差超过 5 °C（掉线/错误读数），已剔除")
    if not quality.empty and "抖动采样" in quality:
        affected = quality[quality["抖动采样"] > 0].sort_values("抖动采样", ascending=False)
        if not affected.empty:
            threshold = result.cfg.filtering.despike_threshold_c
            names = "、".join(affected["探针"].astype(str).head(3))
            total = int(affected["抖动采样"].sum())
            notes.append(
                f"探针 {names} 共 {total} 个采样偏离局部中值超过 {threshold:g} °C（抖动/跳变），已从统计中剔除"
            )
    if result.valid_probes and len(result.valid_probes) < len(result.probes):
        invalid = [p for p in result.probes if p not in result.valid_probes]
        notes.append(f"探针 {', '.join(invalid)} 有效率低于阈值，未纳入统计")
    snr = result.facts.get("波动信噪比")
    if snr is not None and np.isfinite(snr) and snr < 3:
        notes.append(f"脉动幅度仅为量化噪声的 {snr:.1f} 倍，脉动统计受量化限制")
    if result.wind is None or result.wind.empty:
        notes.append("未接入风速数据，C_T²/Cn² 未反演（仅给出温度脉动统计）")
    elif "C_T2" in result.block and result.block["C_T2"].isna().all():
        notes.append("风速全部缺失或为 0，C_T²/Cn² 无法按 Taylor 冻结假设反演")
    else:
        u_median = result.facts.get("风速中位数")
        if u_median is not None and np.isfinite(u_median) and u_median < 0.5:
            notes.append(
                f"风速中位数仅 {u_median:.2f} m/s，接近静风；Taylor 冻结假设不成立，C_T²/Cn² 的量级会偏大，只能作参考"
            )
    if result.cn2_merged is None or result.cn2_merged.empty:
        notes.append("设备 Cn² 通道没有有效数据，未做 Cn² 交叉验证")
    if result.diurnal.empty:
        notes.append(f"分析时段不足 {result.cfg.analysis.min_span_hours_for_diurnal:.0f} h，未计算平均日变化")
    r2 = result.facts.get("谱拟合优度")
    if r2 is not None and isinstance(r2, (int, float)) and np.isfinite(r2) and r2 < 0.7:
        notes.append(f"功率谱拟合 R² 仅 {r2:.2f}，谱形不符单一幂律，谱斜率仅作参考")
    for channel in (result.channels or {}).values():
        notes.extend(channel.notes)
    if not result.spatial.empty:
        heights = result.cfg.probes.heights_m
        if not heights:
            method_name = "等权算术平均" if result.cfg.spatial.method == "mean" else "等权中位数"
            notes.append(
                "未配置探针安装高度（probes.heights_m / --probe-heights），空间平均按跨探针"
                f"{method_name}计算；notebook 的位置插值平均需要 tp → 高度映射"
            )
        else:
            uncovered = [probe for probe in result.probes if probe not in heights]
            if uncovered:
                notes.append(f"探针 {'、'.join(uncovered)} 未配置安装高度，未参与位置插值平均")
            if "位置插值平均" not in set(result.spatial["方式"].dropna()):
                notes.append(f"可定位探针不足 {result.cfg.spatial.min_probes} 个或高度重复，空间平均退化为等权聚合")
    return notes


def build(
    cfg: ThermoReportConfig,
    *,
    days: tuple[str, ...] = (),
    max_files_per_day: int | None = None,
    use_cache: bool = True,
    with_auxiliary: bool = True,
    run_name: str | None = None,
) -> Path:
    """执行分析并生成 HTML/Markdown 报告，返回报告目录。"""
    result = analyze(
        cfg,
        days=days,
        max_files_per_day=max_files_per_day,
        use_cache=use_cache,
        with_auxiliary=with_auxiliary,
    )

    stamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cfg.paths.output_dir / f"thermo_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_files = spatial.write_spatial(result.spatial, result.spatial_daily, output_dir)
    if spatial_files.get("spatial_mean"):
        result.facts["空间平均文件"] = spatial_files["spatial_mean"]
    if spatial_files.get("spatial_daily"):
        result.facts["空间平均日统计文件"] = spatial_files["spatial_daily"]
    figures_dir = output_dir / "figures"
    figures = _render_figures(result, figures_dir)
    if cfg.export.enabled and not result.slices.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = slices.write_slices(result.slices, output_dir, cfg)
        result.facts.update({f"时间片{key}": value for key, value in manifest.items() if value not in ("", None)})
    return write_report(result, output_dir, figures)


def _render_figures(result: AnalysisResult, figures_dir: Path) -> dict[str, Path]:
    """生成全部图表，返回 {标题: 路径}。"""
    from . import plots

    cfg = result.cfg
    plots.setup_style(cfg.report.figure_dpi)
    probes = result.valid_probes or result.probes
    max_series = max(1, cfg.report.max_series_probes)
    series_probes = probes[:max_series]
    figures: dict[str, Path] = {}

    candidates = {
        "coverage": (plots.coverage_heatmap, (result.quality_daily, figures_dir / "coverage.png")),
        "timeseries": (
            plots.temperature_timeseries,
            (result.minute, series_probes, figures_dir / "timeseries.png"),
        ),
        "daily": (plots.daily_mean_lines, (result.daily_means, figures_dir / "daily_means.png")),
        "diurnal": (plots.diurnal_cycle, (result.diurnal, series_probes, figures_dir / "diurnal.png")),
        "distribution": (plots.distribution_box, (result.minute, probes, figures_dir / "distribution.png")),
        "correlation": (
            plots.correlation_heatmap,
            (result.correlation, result.correlation_order, figures_dir / "correlation.png"),
        ),
        "psd": (
            plots.psd_overlay,
            (result.spectrum, series_probes, cfg.analysis.spectrum_band_hz, figures_dir / "psd.png"),
        ),
        "structure": (plots.structure_plot, (result.structure, figures_dir / "structure.png")),
        "sigma": (plots.sigma_timeseries, (result.block, series_probes, figures_dir / "sigma.png")),
        "environment": (plots.environment_panel, (result.channels, result.wind, figures_dir / "environment.png")),
        "spread": (plots.probe_spread, (result.block, figures_dir / "spread.png")),
        "cn2": (plots.cn2_comparison, (result.cn2_merged, figures_dir / "cn2.png")),
    }
    for name, (function, args) in candidates.items():
        target = Path(args[-1])
        # 图表按文件缓存：同一 --run-name 重跑时跳过已生成的图，
        # 既省时间，也让偶发崩溃后的重试能逐张推进。
        if target.exists() and target.stat().st_size > 0:
            figures[name] = target
            continue
        try:
            path = function(*args)
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("绘图 {} 失败: {}", name, exc)
            continue
        if path is not None:
            figures[name] = path
    return figures
