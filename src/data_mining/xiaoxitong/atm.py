#!/usr/bin/env python
"""小系统试验的大气环境窗口聚合。

把 ``data/daily_data/<YYYYMMDD>/<通道>`` 下的长表环境数据（``ID, 数值, Timestamp``）
按分钟聚合成跨探针中位序列，并进一步折算成：

* 每次试验窗口内的统计量（中位/均值/覆盖分钟数）；
* 同日"非试验时段"的背景值（扣除试验窗口并留出保护间隔）；
* 相对试验开始时刻的滞后剖面（-60 ~ +30 min）。

数据解析直接复用 ``ui/analysis/thermo`` 的 ``discover`` / ``load_aux_minute_means``，
因此异常码（温度 1220.8、风速 122.08、Cn² 的 0 值）按各自物理范围剔除，
分钟级中间结果落盘缓存，重复运行不再重读 GB 级 CSV。

Usage:
    uv run python -m data_mining.xiaoxitong.atm --days 20261009 20261010 20261012
    uv run python -m data_mining.xiaoxitong.atm --out data/小系统/试验报告/大气窗口统计.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ui.analysis.thermo.discovery import discover
from ui.analysis.thermo.ingest import load_aux_minute_means

DEFAULT_ATM_ROOT = Path("data/daily_data")
DEFAULT_CACHE_DIR = Path("data/小系统/试验报告/.cache/atm")
MINUTE_RULE = "1min"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Channel:
    """一路大气通道的读取与有效性配置。"""

    key: str
    subdir: str
    label: str
    unit: str
    vmin: float
    vmax: float
    digits: int = 2
    note: str = ""
    # 背景值低于该量级时，相对变化率会被“近零分母”放大到无意义，改为不报百分比。
    rel_floor: float = 0.0

    @property
    def bounds_tag(self) -> str:
        return f"{self.vmin:g}_{self.vmax:g}".replace("-", "m").replace(".", "p")


# 只接入三天试验期确认有数据、且文件命名符合 thermo 发现规则的通道。
# 未接入：r0_computed / aveCn2_computed / Absorption（本批数据只有表头）、
# PM2.5（文件名含小数点，不符合小时分片命名）、Particle（多列宽表）。
#
# 能见度的有效区间被收到 0.05–70 km：原始序列里 0 与 75 分别对应设备的零点与
# 满量程码（20261009 有 822 个 0、609 个 75），不剔除会把中位数压到 0。
CHANNELS: tuple[Channel, ...] = (
    Channel("wind", "Wind", "风速", "m/s", 0.0, 60.0, 2, rel_floor=0.2),
    Channel("temperature", "Temperature", "气温", "°C", -40.0, 60.0, 1, rel_floor=1.0),
    Channel("humidity", "Humidity", "相对湿度", "%", 0.0, 100.0, 1, rel_floor=5.0),
    Channel(
        "visibility",
        "Visibility",
        "能见度",
        "km",
        0.05,
        70.0,
        1,
        note="原始序列含设备零点（0）与满量程码（75），已按 0.05–70 km 剔除",
        rel_floor=2.0,
    ),
    Channel("cn2", "Cn2", "设备 Cn²", "m^(-2/3)", 1e-22, 1e-10, 4, rel_floor=1e-18),
)


def channel_by_key(key: str) -> Channel:
    for channel in CHANNELS:
        if channel.key == key:
            return channel
    raise KeyError(f"未知通道: {key}")


# --------------------------------------------------------------------------- #
# 分钟级序列
# --------------------------------------------------------------------------- #
def _day_cache_paths(cache_dir: Path, channel: Channel, day: str) -> tuple[Path, Path]:
    stem = cache_dir / f"v{SCHEMA_VERSION}_{channel.key}_{day}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def _source_fingerprint(files: tuple, channel: Channel) -> str:
    raw = "|".join(f"{Path(item.path).name}:{item.size}:{item.mtime_ns}" for item in files)
    return f"{SCHEMA_VERSION}|{channel.key}|{channel.bounds_tag}|{MINUTE_RULE}|{raw}"


def minute_median_table(root: Path, channel: Channel, day: str) -> pd.DataFrame:
    """读取单日单通道，返回 ``ts`` 索引的跨探针中位序列（列：中位、探针数）。"""
    inventory = discover(root, channel_dirs=(channel.subdir,), days=(day,))
    files = inventory.by_channel(channel.subdir)
    if not files:
        return pd.DataFrame(columns=["中位", "探针数"])

    frames = []
    for source in files:
        try:
            minute = load_aux_minute_means(
                source,
                MINUTE_RULE,
                value_min=channel.vmin,
                value_max=channel.vmax,
                cache_dir=None,
            )
        except Exception as exc:  # 单个损坏的 CSV 不应让整天整通道失效
            logger.warning("跳过异常大气文件 {}（{}）", Path(source.path).name, exc)
            continue
        if not minute.empty:
            frames.append(minute)
    if not frames:
        return pd.DataFrame(columns=["中位", "探针数"])

    stacked = pd.concat(frames, ignore_index=True)
    stacked["数值"] = pd.to_numeric(stacked["数值"], errors="coerce")
    stacked = stacked.replace([np.inf, -np.inf], np.nan).dropna(subset=["数值"])
    if stacked.empty:
        return pd.DataFrame(columns=["中位", "探针数"])
    grouped = stacked.groupby("ts", observed=True)["数值"]
    table = pd.DataFrame({"中位": grouped.median(), "探针数": grouped.count()})
    table.index = pd.to_datetime(table.index)
    return table.sort_index()


def load_day_channel(
    root: Path,
    channel: Channel,
    day: str,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """带缓存的单日单通道读取（缓存键包含文件指纹与物理范围）。

    某一天缺少该通道目录时返回空表，不抛异常——三日试验里的辅助通道
    （如 PM2.5）并非每天都有。
    """
    try:
        inventory = discover(root, channel_dirs=(channel.subdir,), days=(day,))
    except FileNotFoundError:
        logger.debug("通道 {} 在 {} 无数据", channel.label, day)
        return pd.DataFrame(columns=["中位", "探针数"])
    files = inventory.by_channel(channel.subdir)
    if cache_dir is None:
        return minute_median_table(root, channel, day)

    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _day_cache_paths(cache_dir, channel, day)
    fingerprint = _source_fingerprint(files, channel)
    if not refresh and data_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if meta.get("fingerprint") == fingerprint:
            try:
                cached = pd.read_parquet(data_path)
                if not {"中位", "探针数"}.issubset(cached.columns):
                    raise ValueError("缓存列不完整")
                logger.debug("命中大气缓存 {} {}", channel.key, day)
                return cached
            except Exception as exc:  # 缓存损坏时回退原始文件
                logger.warning("大气缓存不可用（{} {}：{}），回退原始文件", channel.key, day, exc)

    table = minute_median_table(root, channel, day)
    try:
        table.to_parquet(data_path)
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "rows": int(len(table))}), encoding="utf-8")
    except (OSError, ValueError, ImportError) as exc:  # 缓存只加速，不应影响分析
        logger.warning("大气缓存写入失败（{} {}：{}）", channel.key, day, exc)
    return table


def load_channels(
    root: Path,
    days: list[str],
    channels: tuple[Channel, ...] = CHANNELS,
    *,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """把多天多通道拼成 ``{key: DataFrame}``，索引为分钟时间戳。"""
    result: dict[str, pd.DataFrame] = {}
    for channel in channels:
        parts = []
        for day in days:
            try:
                table = load_day_channel(root, channel, day, cache_dir, refresh=refresh)
            except Exception as exc:  # 单日失败不拖累其它日期
                logger.warning("大气通道 {} 在 {} 读取失败，已跳过：{}", channel.key, day, exc)
                continue
            if not table.empty:
                parts.append(table)
        merged = pd.concat(parts) if parts else pd.DataFrame(columns=["中位", "探针数"])
        if not merged.empty:
            merged = merged[~merged.index.duplicated(keep="first")].sort_index()
        result[channel.key] = merged
        logger.info("大气通道 {}（{}）覆盖 {} 分钟", channel.label, channel.subdir, len(merged))
    return result


# --------------------------------------------------------------------------- #
# 窗口统计
# --------------------------------------------------------------------------- #
def minute_bounds(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """把秒级时间窗扩到分钟边界（分钟级序列的收录口径）。"""
    low = start.replace(second=0, microsecond=0)
    high = end.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return low, high


def window_stats(series: pd.Series, start: datetime, end: datetime) -> dict[str, float]:
    """窗口 ``[start, end]`` 内的基本统计；无覆盖时返回空字典。

    序列是分钟级的，窗口往往是几十秒，因此先把窗口扩到分钟边界再取值，
    否则 30 秒的窗口大多取不到任何分钟点。
    """
    if series.empty:
        return {}
    low, high = minute_bounds(start, end)
    window = series.loc[(series.index >= low) & (series.index < high)].dropna()
    if window.empty:
        return {}
    return {
        "n": float(window.size),
        "median": float(window.median()),
        "mean": float(window.mean()),
        "min": float(window.min()),
        "max": float(window.max()),
    }


def background_stats(
    series: pd.Series,
    windows: list[tuple[datetime, datetime]],
    *,
    guard_minutes: int = 15,
) -> dict[str, float]:
    """同日"非试验时段"背景：扣除窗口并按 ``guard_minutes`` 前后各留一段保护间隔。"""
    if series.empty:
        return {}
    mask = pd.Series(True, index=series.index)
    guard = timedelta(minutes=guard_minutes)
    for start, end in windows:
        mask &= ~((series.index >= start - guard) & (series.index <= end + guard))
    background = series.loc[mask].dropna()
    if background.empty:
        return {}
    return {
        "n": float(background.size),
        "median": float(background.median()),
        "mean": float(background.mean()),
        "min": float(background.min()),
        "max": float(background.max()),
    }


def relative_change(
    window_value: float | None,
    background_value: float | None,
    *,
    floor: float = 0.0,
) -> float | None:
    """窗口相对背景的变化率；背景无值或低于 ``floor`` 时返回 ``None``。

    ``floor`` 用于挡掉“背景接近零”的分母放大：例如夜间静风时风速背景中位只有
    0.04 m/s，风机一开窗口就到 0.87 m/s，百分比会算成 +2290%，没有解释意义。
    """
    if window_value is None or background_value is None:
        return None
    if not np.isfinite(background_value) or abs(background_value) <= max(floor, 0.0):
        return None
    return window_value / background_value - 1.0


def ratio_orders(window_value: float | None, background_value: float | None) -> float | None:
    """窗口与背景的倍率（以 10 为底的对数）；跨数量级的通道用倍率而不是百分比。"""
    if window_value is None or background_value is None:
        return None
    if not np.isfinite(window_value) or not np.isfinite(background_value):
        return None
    if window_value <= 0 or background_value <= 0:
        return None
    return float(np.log10(window_value / background_value))


def lag_profile(
    series: pd.Series,
    events: list[datetime],
    *,
    before_minutes: int = 60,
    after_minutes: int = 30,
    bin_minutes: int = 10,
) -> pd.DataFrame:
    """相对事件时刻的滞后剖面：每个事件在每个时间桶内取中位，再对事件取中位。"""
    edges = list(range(-before_minutes, after_minutes + bin_minutes, bin_minutes))
    if series.empty or not events:
        return pd.DataFrame(columns=["offset_min", "中位", "事件数"])
    rows = []
    for event in events:
        for offset in edges:
            window = series.loc[
                (series.index >= event + timedelta(minutes=offset))
                & (series.index < event + timedelta(minutes=offset + bin_minutes))
            ]
            if not window.empty:
                rows.append({"事件": event, "offset_min": offset, "值": float(window.median())})
    if not rows:
        return pd.DataFrame(columns=["offset_min", "中位", "事件数"])
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("offset_min", observed=True)["值"]
    return pd.DataFrame({"中位": grouped.median(), "事件数": grouped.count()}).reset_index()


def hourly_medians(series: pd.Series) -> pd.DataFrame:
    """逐小时中位序列，索引为带日期的整点时间。"""
    if series.empty:
        return pd.DataFrame(columns=["中位", "样本数"])
    grouped = series.groupby(pd.Grouper(freq="1h"))
    table = pd.DataFrame({"中位": grouped.median(), "样本数": grouped.count()})
    return table.dropna(subset=["中位"])


def parse_day(value: str) -> str:
    """校验 ``YYYYMMDD`` 并原样返回。"""
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYYMMDD: {value}") from exc
    return value


def day_of(day: str) -> date:
    return datetime.strptime(day, "%Y%m%d").date()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_writer(  # pragma: no cover - CLI 辅助
    experiments: list,
    channels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """把每次试验的窗口统计铺成一张宽表，供 CSV 落盘与报告复用。"""
    rows = []
    for exp in experiments:
        if not (exp.capture_start and exp.capture_end):
            continue
        row = {
            "试验": exp.exp_id,
            "开始": exp.capture_start,
            "结束": exp.capture_end,
        }
        for key, series in channels.items():
            channel = channel_by_key(key)
            stats = window_stats(series, exp.capture_start, exp.capture_end)
            row[f"{channel.label}_中位"] = stats.get("median")
            row[f"{channel.label}_覆盖分钟"] = stats.get("n")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI 入口
    parser = argparse.ArgumentParser(description="小系统试验的大气环境窗口统计")
    parser.add_argument("--root", type=Path, default=DEFAULT_ATM_ROOT, help="大气数据根目录（含 YYYYMMDD 子目录）")
    parser.add_argument("--days", nargs="*", default=[], help="只处理这些日期（YYYYMMDD），默认全部")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="分钟级缓存目录")
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重新解析")
    parser.add_argument("--out", type=Path, default=None, help="窗口统计 CSV 输出路径")
    args = parser.parse_args(argv)

    days = args.days or sorted(p.name for p in args.root.iterdir() if p.is_dir() and p.name.isdigit())
    channels = load_channels(args.root, days, cache_dir=args.cache_dir, refresh=args.refresh)
    for key, table in channels.items():
        channel = channel_by_key(key)
        if table.empty:
            logger.warning("通道 {} 无数据", channel.label)
            continue
        logger.info(
            "{} 中位 {:.4g} {}，覆盖 {} 分钟",
            channel.label,
            table["中位"].median(),
            channel.unit,
            len(table),
        )
    if args.out and channels:
        summary = pd.DataFrame(
            [
                {
                    "通道": channel_by_key(key).label,
                    "单位": channel_by_key(key).unit,
                    "覆盖分钟": len(table),
                    "中位": table["中位"].median() if not table.empty else None,
                    "P05": table["中位"].quantile(0.05) if not table.empty else None,
                    "P95": table["中位"].quantile(0.95) if not table.empty else None,
                }
                for key, table in channels.items()
            ]
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False, encoding="utf-8-sig")
        logger.info("已写出 {}", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
