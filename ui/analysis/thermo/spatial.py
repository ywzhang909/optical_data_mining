"""跨探针空间平均（融合 ``notebooks/atmosphere_analysis.ipynb`` 的方法）。

旧导出的宽表把测点高度写进列名（形如 ``20m``），notebook 因此可以直接
``int(列名[:-1])`` 得到位置；当前长表导出（``ID, 数值, Timestamp``）只保留
``tp-1…tp-16``，高度信息在数据侧已经丢失，仓库与文档里也没有 tp → 高度的映射。

所以这里把位置显式化：

* 配置里给出 ``probes.heights_m``（或命令行 ``--probe-heights``）时，按高度做
  **线性插值到等间隔网格后取平均**，与 notebook 的 ``interp1d`` + ``np.mean`` 等价；
* 没有高度（或可用高度不足）时退化为**跨探针等权聚合**，并在结果里用 ``方式``
  列写明，报告与提示同步说明，避免把等权平均当成位置平均来解读。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SPATIAL_COLUMNS = ["ts", "空间平均", "探针数", "方式"]
SPATIAL_DAILY_COLUMNS = ["通道", "单位", "日期", "日均值", "最低", "最高", "有效分钟", "平均探针数", "方式"]

METHOD_INTERP = "位置插值平均"
METHOD_MEDIAN = "等权中位数"
METHOD_MEAN = "等权算术平均"


def equal_interval_mean(values: Sequence[float], positions: Sequence[float]) -> float:
    """复刻 notebook 的等间隔插值平均。

    实现要点（保持与 notebook 一致，不做“改进”）：

    * 按 ``positions`` 升序重排（notebook 假设列已按高度升序，这里显式排序更安全）；
    * ``gap = min(相邻位置差)``，网格 = ``np.arange(min(positions), max(positions), gap)``；
    * 线性插值到网格后取算术平均；
    * 网格不含 ``max``（``arange`` 右开），这是 notebook 的原行为，保留；
    * 成对剔除 NaN 后再插值；
    * 退化：有效点少于 2 个、位置全部相同或 ``gap <= 0`` 时，返回 ``values`` 的算术平均。
    """
    pairs: list[tuple[float, float]] = []
    for position, value in zip(positions, values, strict=False):
        try:
            pos = float(position)
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pos) and np.isfinite(val):
            pairs.append((pos, val))
    if len(pairs) < 2:
        clean = [val for _, val in pairs]
        return float(np.mean(clean)) if clean else float("nan")

    pairs.sort(key=lambda item: item[0])
    pos = np.asarray([item[0] for item in pairs], dtype=float)
    val = np.asarray([item[1] for item in pairs], dtype=float)
    gaps = np.diff(pos)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return float(np.mean(val))
    grid = np.arange(pos.min(), pos.max(), float(positive.min()))
    if grid.size == 0:
        return float(np.mean(val))
    return float(np.mean(np.interp(grid, pos, val)))


def aggregate_across_probes(values: Sequence[float], method: str = "median") -> float:
    """跨探针等权聚合：``median`` 或 ``mean``，忽略 NaN；空序列返回 NaN。"""
    clean = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(clean))
    return float(np.median(clean))


def fallback_label(method: str) -> str:
    """等权聚合在报告里的显示名。"""
    return METHOD_MEAN if method == "mean" else METHOD_MEDIAN


def spatial_series(
    minute: pd.DataFrame,
    value_col: str,
    positions: Mapping[str, float] | None = None,
    *,
    method: str = "median",
    min_probes: int = 3,
) -> pd.DataFrame:
    """逐分钟把多个探针合成一条通道序列。

    ``minute`` 是长表，必须含列 ``ts``、``探针``、``value_col``。
    ``positions`` 为探针名 → 安装高度（米）：

    * 与数据探针的交集 ``>= min_probes`` 且高度互不相同 → 每分钟对交集探针做
      `equal_interval_mean`，方式记为 ``位置插值平均``；
    * 否则 → 每分钟 `aggregate_across_probes`，方式记为 ``等权中位数`` / ``等权算术平均``。

    返回列固定为 ``ts``、``空间平均``、``探针数``、``方式``，按 ``ts`` 排序并 reset_index。
    输入为空或缺少列时返回带上述列的空 DataFrame。
    """
    if minute is None or minute.empty or value_col not in getattr(minute, "columns", ()):
        return pd.DataFrame(columns=SPATIAL_COLUMNS)

    frame = minute[["ts", "探针", value_col]].copy()
    frame["ts"] = pd.to_datetime(frame["ts"])
    numeric = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame[np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))]
    if frame.empty:
        return pd.DataFrame(columns=SPATIAL_COLUMNS)

    heights = {str(key): float(val) for key, val in (positions or {}).items()}
    fallback = fallback_label(method)
    rows: list[tuple[object, float, int, str]] = []
    for ts, group in frame.groupby("ts", observed=True, sort=True):
        measured: dict[str, float] = {}
        for probe, value in zip(group["探针"], group[value_col], strict=False):
            measured.setdefault(str(probe), float(value))
        located = {name: heights[name] for name in measured if name in heights}
        if len(located) >= max(2, int(min_probes)) and len(set(located.values())) >= 2:
            value = equal_interval_mean(
                [measured[name] for name in located],
                [located[name] for name in located],
            )
            label = METHOD_INTERP
        else:
            value = aggregate_across_probes(list(measured.values()), method)
            label = fallback
        rows.append((ts, value, len(measured), label))
    if not rows:
        return pd.DataFrame(columns=SPATIAL_COLUMNS)
    return pd.DataFrame(rows, columns=SPATIAL_COLUMNS).sort_values("ts").reset_index(drop=True)


def spatial_summary(spatial: pd.DataFrame) -> pd.DataFrame:
    """把逐分钟空间平均表压成「每通道一行」的概况表。

    返回列：``通道``、``单位``、``有效分钟``、``平均探针数``、``方式``、``均值``、
    ``最低``、``最高``。表里没有 ``通道`` 列时返回空表。
    """
    columns = ["通道", "单位", "有效分钟", "平均探针数", "方式", "均值", "最低", "最高"]
    if spatial is None or spatial.empty or "通道" not in spatial.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (label, unit), group in spatial.groupby(["通道", "单位"], observed=True, sort=False):
        values = pd.to_numeric(group["空间平均"], errors="coerce").dropna()
        if values.empty:
            continue
        modes = group["方式"].mode()
        rows.append(
            {
                "通道": label,
                "单位": unit,
                "有效分钟": int(values.size),
                "平均探针数": float(pd.to_numeric(group["探针数"], errors="coerce").mean()),
                "方式": str(modes.iloc[0]) if not modes.empty else "",
                "均值": float(values.mean()),
                "最低": float(values.min()),
                "最高": float(values.max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def daily_spatial_summary(series: pd.DataFrame, label: str, unit: str) -> pd.DataFrame:
    """把 `spatial_series` 结果按日汇总。

    返回列：``通道``、``单位``、``日期``、``日均值``、``最低``、``最高``、``有效分钟``、
    ``平均探针数``、``方式``。空输入返回空 DataFrame。
    """
    if series is None or series.empty:
        return pd.DataFrame(columns=SPATIAL_DAILY_COLUMNS)

    frame = series.copy()
    frame["ts"] = pd.to_datetime(frame["ts"])
    values = pd.to_numeric(frame["空间平均"], errors="coerce")
    frame = frame[values.notna() & np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))]
    if frame.empty:
        return pd.DataFrame(columns=SPATIAL_DAILY_COLUMNS)

    frame["日期"] = frame["ts"].dt.strftime("%Y-%m-%d")
    grouped = frame.groupby("日期", observed=True)
    daily = pd.DataFrame(
        {
            "日均值": grouped["空间平均"].mean(),
            "最低": grouped["空间平均"].min(),
            "最高": grouped["空间平均"].max(),
            "有效分钟": grouped["空间平均"].count(),
            "平均探针数": grouped["探针数"].mean(),
            "方式": grouped["方式"].agg(lambda s: str(s.mode().iloc[0]) if not s.mode().empty else ""),
        }
    ).reset_index()
    daily.insert(0, "单位", unit)
    daily.insert(0, "通道", label)
    return daily[SPATIAL_DAILY_COLUMNS].sort_values("日期").reset_index(drop=True)


def write_spatial(spatial: pd.DataFrame, daily: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """写出 ``spatial/spatial_mean.csv`` 与 ``spatial/spatial_daily.csv``。

    只写非空的表（utf-8-sig，便于 Excel 直接打开）；返回**相对 output_dir** 的路径，
    空表对应空字符串。目录不存在则创建。
    """
    written = {"spatial_mean": "", "spatial_daily": ""}
    target = Path(output_dir) / "spatial"
    target.mkdir(parents=True, exist_ok=True)
    if spatial is not None and not spatial.empty:
        spatial.to_csv(target / "spatial_mean.csv", index=False, encoding="utf-8-sig")
        written["spatial_mean"] = "spatial/spatial_mean.csv"
    if daily is not None and not daily.empty:
        daily.to_csv(target / "spatial_daily.csv", index=False, encoding="utf-8-sig")
        written["spatial_daily"] = "spatial/spatial_daily.csv"
    return written


def heights_from_text(text: str) -> dict[str, float]:
    """解析 CLI 文本 ``tp-1=1.5,tp-2=3.0`` → ``{"tp-1": 1.5, "tp-2": 3.0}``。

    空串返回 ``{}``；非法片段抛 `ValueError` 并说明片段内容。
    """
    heights: dict[str, float] = {}
    for chunk in str(text or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"无法解析高度片段 {item!r}，应为 tp-1=1.5 形式")
        try:
            heights[name] = float(raw.strip())
        except ValueError as exc:
            raise ValueError(f"高度片段 {item!r} 的数值无法解析") from exc
    return heights
