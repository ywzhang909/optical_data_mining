"""温度以外的环境要素通道：湿度、能见度、设备 Cn²（以及可扩展的任意同类通道）。

所有通道都是同一长表结构（``ID, 数值, Timestamp``），因此解析复用 `ingest` 的流式读取与
缓存，这里只负责：按物理范围剔除异常码 → 分钟均值 → 逐探针统计 → 时间片聚合。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from .config import ChannelConfig, ThermoReportConfig
from .discovery import Inventory
from .filters import smooth_minute_frame
from .ingest import load_aux_minute_means


@dataclass
class ChannelResult:
    """一路通道的全部产物。"""

    key: str
    label: str
    unit: str
    minute: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ts", "探针", "数值"]))
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    slices: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)
    is_reference: bool = False


def _probe_summary(minute: pd.DataFrame, label: str, unit: str) -> pd.DataFrame:
    """逐探针统计（分钟级）。"""
    if minute.empty:
        return pd.DataFrame()
    grouped = minute.groupby("探针", observed=True)["数值"]
    summary = pd.DataFrame(
        {
            "有效分钟": grouped.count(),
            "均值": grouped.mean(),
            "标准差": grouped.std(),
            "最小": grouped.min(),
            "P05": grouped.quantile(0.05),
            "中位": grouped.median(),
            "P95": grouped.quantile(0.95),
            "最大": grouped.max(),
        }
    ).reset_index()
    span_minutes = max(1, int((minute["ts"].max() - minute["ts"].min()).total_seconds() // 60) + 1)
    summary["覆盖率"] = (summary["有效分钟"] / span_minutes).round(4)
    summary.insert(0, "通道", label)
    summary.insert(1, "单位", unit)
    return summary.sort_values("探针").reset_index(drop=True)


def _daily_summary(minute: pd.DataFrame, label: str) -> pd.DataFrame:
    """逐日逐探针日统计（日均值 / 日极值）。"""
    if minute.empty:
        return pd.DataFrame()
    frame = minute.copy()
    frame["日期"] = pd.to_datetime(frame["ts"]).dt.strftime("%Y-%m-%d")
    grouped = frame.groupby(["日期", "探针"], observed=True)["数值"]
    daily = pd.DataFrame(
        {
            "日均值": grouped.mean(),
            "日最低": grouped.min(),
            "日最高": grouped.max(),
            "有效分钟": grouped.count(),
        }
    ).reset_index()
    daily.insert(1, "通道", label)
    return daily


def slice_table(minute: pd.DataFrame, rule: str, label: str, unit: str, smooth_window: int) -> pd.DataFrame:
    """把分钟级长表聚合为时间片长表（时间片 × 探针）。

    这是"随时间片的中间数据"的核心形态：每行是一个（时间片, 探针）组合，
    给出原始统计量与平滑值，可被后续分析直接消费。
    """
    if minute.empty:
        return pd.DataFrame()
    frame = smooth_minute_frame(minute, smooth_window)
    frame = frame.copy()
    frame["时间片"] = pd.to_datetime(frame["ts"]).dt.floor(rule)
    grouped = frame.groupby(["时间片", "探针"], observed=True)
    table = pd.DataFrame(
        {
            "均值": grouped["数值"].mean(),
            "标准差": grouped["数值"].std(),
            "最小": grouped["数值"].min(),
            "最大": grouped["数值"].max(),
            "有效采样": grouped["数值"].count(),
        }
    )
    if "平滑值" in frame:
        table["平滑均值"] = grouped["平滑值"].mean()
    table = table.reset_index()
    table.insert(0, "通道", label)
    table.insert(1, "单位", unit)
    ordered = [
        "通道",
        "单位",
        "探针",
        "时间片",
        *[c for c in table.columns if c not in {"通道", "单位", "探针", "时间片"}],
    ]
    table = table[ordered]
    return table.sort_values(["时间片", "探针"]).reset_index(drop=True)


def load_channel(cfg: ThermoReportConfig, inventory: Inventory, channel: ChannelConfig) -> ChannelResult:
    """读取一路环境要素通道；失败只记录，不中断整批分析。"""
    result = ChannelResult(key=channel.key, label=channel.label, unit=channel.unit, is_reference=channel.is_reference)
    sources = inventory.by_channel(channel.subdir)
    if not sources:
        result.notes.append(f"数据中没有 {channel.subdir}/ 目录，{channel.label}未参与分析")
        return result

    frames = []
    failed = 0
    for source in sources:
        try:
            frame = load_aux_minute_means(
                source,
                cfg.analysis.minute_rule,
                value_min=channel.valid_min,
                value_max=channel.valid_max,
                cache_dir=cfg.paths.cache_dir,
            )
        except Exception as exc:  # 单文件损坏不应中断整批
            failed += 1
            logger.warning("跳过 {} 的 {}: {}: {}", channel.subdir, source.path.name, type(exc).__name__, exc)
            continue
        if not frame.empty:
            frames.append(frame)

    if failed:
        result.notes.append(f"{channel.label}有 {failed} 个文件解析失败已跳过")
    if not frames:
        result.notes.append(f"{channel.label} 通道没有解析出有效数据（文件为空或全部为异常码）")
        return result

    minute = pd.concat(frames, ignore_index=True).sort_values(["ts", "探针"]).reset_index(drop=True)
    result.minute = minute
    result.summary = _probe_summary(minute, channel.label, channel.unit)
    result.daily = _daily_summary(minute, channel.label)
    result.slices = slice_table(
        minute, cfg.export.slice_rule, channel.label, channel.unit, cfg.filtering.smooth_window_min
    )
    result.notes.extend(_quality_notes(minute, channel, cfg))
    return result


def _quality_notes(minute: pd.DataFrame, channel: ChannelConfig, cfg: ThermoReportConfig) -> list[str]:
    """覆盖率与量程饱和提示。"""
    notes: list[str] = []
    span = pd.to_datetime(minute["ts"])
    if span.empty:
        return notes
    minutes = int((span.max() - span.min()).total_seconds() // 60) + 1
    counts = minute.groupby("探针", observed=True)["数值"].count()
    if minutes > 0 and len(counts):
        coverage = float(counts.max()) / minutes
        if coverage < 0.95:
            notes.append(
                f"{channel.label}最完整探针的分钟覆盖率 {coverage:.0%}（{int(counts.max())}/{minutes} min），存在断记"
            )
        weak = counts[counts < 0.5 * counts.max()]
        if len(weak):
            notes.append(f"{channel.label}探针 {', '.join(map(str, weak.index))} 有效分钟不足最完整探针的一半")
    # 量程封顶：按探针自身观测上限判，而不是配置上限。能见度实际封在 75 km，
    # 配置写的是 100 km，用配置判永远抓不到；探针观测值中大量贴在自身最大值即视为封顶。
    # 只在覆盖率足够的探针上判：少数有效分钟的探针（多为缺测）最大值只是残留读数，会误报。
    for probe, group in minute.groupby("探针", observed=True):
        values = pd.to_numeric(group["数值"], errors="coerce").dropna()
        if len(values) < 100:
            continue
        if minutes > 0 and len(values) / minutes < 0.5:
            continue
        max_value = float(values.max())
        if max_value <= 0 or not np.isfinite(max_value):
            continue
        pinned = float((values >= max_value * 0.999).mean())
        if pinned >= 0.05:
            notes.append(
                f"{channel.label}探针 {probe} 有 {pinned:.0%} 的分钟均值贴在其自身上限 {max_value:g} {channel.unit}，"
                "疑似达到量程封顶"
            )
    # 大量为 0：传感器未接入或间歇失效，不要把 0 当成真实观测
    for probe, group in minute.groupby("探针", observed=True):
        values = pd.to_numeric(group["数值"], errors="coerce").dropna()
        if values.empty:
            continue
        if minutes > 0 and len(values) / minutes < 0.5:
            continue
        share = float((values == 0).mean())
        if share >= 0.3:
            note = f"{channel.label}探针 {probe} 有 {share:.0%} 的分钟均值为 0"
            if share == 1.0:
                note += "（全程为 0，疑似未接入或失效）"
            else:
                note += "，疑似未接入或间歇失效"
            notes.append(note + "，已保留在表中但不参与通道级中位数")
    return notes


def load_channels(cfg: ThermoReportConfig, inventory: Inventory) -> dict[str, ChannelResult]:
    """按配置并行无关地逐个加载已启用的通道。"""
    results: dict[str, ChannelResult] = {}
    for channel in cfg.channels:
        if not channel.enabled:
            continue
        results[channel.key] = load_channel(cfg, inventory, channel)
    return results
