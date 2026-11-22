"""随时间片的中间数据导出。

产物是自动化流水线与下游分析之间的接口：每个时间片一份 CSV，行是（通道, 探针），
列是原始统计量与湍流参数，时间片长度、是否合并 Parquet 都可配置。

示例::

    slices/
    ├── 20261014/
    │   ├── 0000.csv … 2350.csv     # 每片一份，便于增量消费
    │   └── …
    └── all_slices.parquet          # 合并视图，便于批量分析

为什么单独导出：报告 HTML 只适合人看；下游拟合、画图、入库需要的是规整的中间表，
重新从 2 GB 原始 CSV 里算一遍既慢又容易口径不一致。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from .channels import ChannelResult, slice_table
from .config import ThermoReportConfig
from .filters import smooth_minute_frame

SLICE_COLUMNS = [
    "通道",
    "单位",
    "探针",
    "时间片",
    "均值",
    "标准差",
    "最小",
    "最大",
    "有效采样",
    "平滑均值",
    "C_T2",
    "Cn2",
    "谱斜率",
    "结构函数斜率",
    "噪声底",
]


def _temperature_slices(minute: pd.DataFrame, block: pd.DataFrame, cfg: ThermoReportConfig) -> pd.DataFrame:
    """温度分钟表 + 分块湍流参数 → 时间片长表。"""
    if minute.empty:
        return pd.DataFrame(columns=SLICE_COLUMNS)
    frame = smooth_minute_frame(minute, cfg.filtering.smooth_window_min)
    frame = frame.copy()
    frame["时间片"] = pd.to_datetime(frame["ts"]).dt.floor(cfg.export.slice_rule)
    grouped = frame.groupby(["时间片", "探针"], observed=True)
    table = pd.DataFrame(
        {
            "均值": grouped["均值"].mean(),
            "标准差": grouped["标准差"].mean(),
            "最小": grouped["最小"].min(),
            "最大": grouped["最大"].max(),
            "有效采样": grouped["采样数"].sum(),
            "平滑均值": grouped["平滑值"].mean(),
        }
    ).reset_index()
    table.insert(0, "通道", "温度")
    table.insert(1, "单位", "°C")

    if not block.empty:
        metrics = block.copy()
        metrics["时间片"] = pd.to_datetime(metrics["分块起点"]).dt.floor(cfg.export.slice_rule)
        keep = [column for column in ("C_T2", "Cn2", "谱斜率", "结构函数斜率", "噪声底") if column in metrics.columns]
        if keep:
            metrics = metrics.groupby(["时间片", "探针"], observed=True)[keep].median().reset_index()
            table = table.merge(metrics, on=["时间片", "探针"], how="left")
    return table


def _wind_slices(wind: pd.DataFrame, cfg: ThermoReportConfig) -> pd.DataFrame:
    """风速分钟表 → 时间片长表（与通道表同构）。"""
    if wind is None or wind.empty:
        return pd.DataFrame(columns=SLICE_COLUMNS)
    return slice_table(wind, cfg.export.slice_rule, "风速", "m/s", cfg.filtering.smooth_window_min)


def build_slice_frame(
    cfg: ThermoReportConfig,
    temperature_minute: pd.DataFrame,
    block: pd.DataFrame,
    channels: dict[str, ChannelResult],
    wind: pd.DataFrame | None,
) -> pd.DataFrame:
    """汇总所有通道的时间片长表。"""
    frames = [_temperature_slices(temperature_minute, block, cfg)]
    frames.extend(result.slices for result in channels.values() if not result.slices.empty)
    frames.append(_wind_slices(wind if wind is not None else pd.DataFrame(), cfg))
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=SLICE_COLUMNS)
    merged = pd.concat(non_empty, ignore_index=True)
    for column in SLICE_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
    merged = merged[SLICE_COLUMNS]
    return merged.sort_values(["时间片", "通道", "探针"], ignore_index=True)


def write_slices(frame: pd.DataFrame, output_dir: Path, cfg: ThermoReportConfig) -> dict[str, object]:
    """把时间片长表落盘为 ``slices/`` 目录。

    Args:
        frame: `build_slice_frame` 的结果。
        output_dir: 报告目录（新建 ``slices/`` 子目录）。
        cfg: 导出配置（时间片长度、是否写单文件 / Parquet）。

    Returns:
        写入清单（文件数、时间片数、合并文件路径），写入报告附录。
    """
    target = output_dir / "slices"
    target.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return {"文件数": 0, "Parquet": ""}

    files = 0
    if cfg.export.per_slice_files:
        stamps = pd.to_datetime(frame["时间片"])
        for (day, hhmm), group in frame.groupby([stamps.dt.strftime("%Y%m%d"), stamps.dt.strftime("%H%M")], sort=True):
            day_dir = target / str(day)
            day_dir.mkdir(parents=True, exist_ok=True)
            path = day_dir / f"{hhmm}.csv"
            group.to_csv(path, index=False, encoding="utf-8-sig")
            files += 1

    parquet_note = ""
    if cfg.export.combined_parquet:
        path = target / "all_slices.parquet"
        try:
            frame.to_parquet(path, index=False)
            parquet_note = str(path.relative_to(output_dir))
        except (OSError, ValueError) as exc:  # pyarrow 偶发异常时不让整批报告失败
            logger.warning("时间片 Parquet 写入失败（忽略）: {}", exc)

    logger.info("时间片中间数据已导出：{} 个文件，{} 行", files, len(frame))
    return {
        "文件数": files,
        "行数": int(len(frame)),
        "片数": int(frame["时间片"].nunique()),
        "Parquet": parquet_note,
    }
