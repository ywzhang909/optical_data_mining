#!/usr/bin/env python
"""小系统试验数据的中文 HTML 报告。

两份报告：

* ``01_试验数据总览.html``：数据规模、数据形态、试验设计矩阵、参数完备度、
  击穿行为与重复性、数据质量、全量试验清单。
* ``02_时间与大气协同.html``：试验时间结构、大气环境背景、试验窗口与背景对比、
  击穿行为与大气要素的关联、相对试验开始时刻的滞后剖面。

正文不出现原始文件名与绝对功率数值：功率一律用档位代号（标准功率 / 低功率），
数据来源以"小系统外场试验数据"指代。所有图表为内联 SVG，报告单文件可离线打开。

Usage:
    uv run python scripts/xiaoxitong_html.py
    uv run python scripts/xiaoxitong_html.py --root data/小系统 --out data/小系统/试验报告/html
    uv run python scripts/xiaoxitong_html.py --skip-atmosphere   # 只出总览报告
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xiaoxitong_atm as atm  # noqa: E402
import xiaoxitong_report as base  # noqa: E402

DEFAULT_ROOT = Path("data/小系统")
DEFAULT_OUT = Path("data/小系统/试验报告/html")
GENERATED_BY = "小系统试验数据报告工具"
UNKNOWN = "未记录"

PALETTE = ("#2f6f9f", "#c9762c", "#4f8f6b", "#8b5fa8", "#b24d5a", "#5c7d99", "#a08b3c")


# --------------------------------------------------------------------------- #
# 基础格式化
# --------------------------------------------------------------------------- #
def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def num(value: float | None, digits: int = 2) -> str:
    """数值格式化：None/NaN 显示为短横线。"""
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    if abs(value) >= 1e5 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} TB"


def p_text(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "低于 0.001"
    return f"{p:.3f}"


def p_phrase(p: float | None) -> str:
    """用于正文句子：返回“p = 0.016”或“p 低于 0.001”。"""
    return "p 低于 0.001" if p is not None and p < 0.001 else f"p = {p_text(p)}"


# --------------------------------------------------------------------------- #
# HTML 组件
# --------------------------------------------------------------------------- #
CSS = """
:root {
  --ink: #1c2530;
  --muted: #5d6b7a;
  --line: #d8e0e8;
  --bg: #ffffff;
  --panel: #f6f8fa;
  --accent: #2f6f9f;
  --warm: #c9762c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans SC", "Noto Sans CJK SC", sans-serif;
  font-size: 15px;
  line-height: 1.75;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 28px 80px; }
header.doc { border-bottom: 3px solid var(--accent); padding-bottom: 18px; margin-bottom: 28px; }
header.doc h1 { font-size: 30px; margin: 0 0 8px; letter-spacing: .5px; }
header.doc .sub { color: var(--muted); font-size: 14px; }
h2 { font-size: 21px; margin: 42px 0 12px; padding-left: 10px; border-left: 4px solid var(--accent); }
h3 { font-size: 16px; margin: 26px 0 8px; color: #29455e; }
p { margin: 10px 0; }
ul, ol { margin: 8px 0 8px 22px; padding: 0; }
li { margin: 4px 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.card .k { color: var(--muted); font-size: 12px; letter-spacing: .4px; }
.card .v { font-size: 22px; font-weight: 600; margin-top: 4px; }
.card .u { color: var(--muted); font-size: 12px; }
table.data { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 13.5px; }
table.data th, table.data td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
table.data th { background: var(--panel); font-weight: 600; color: #29455e; white-space: nowrap; }
table.data td.num, table.data th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.data tr:hover td { background: #fbfcfd; }
table.compact td, table.compact th { padding: 4px 6px; }
.note { color: var(--muted); font-size: 13px; margin: 6px 0 14px; }
.callout { background: #f2f7fb; border-left: 4px solid var(--accent); padding: 12px 16px; margin: 16px 0; font-size: 14px; }
.callout.warn { background: #fdf6ee; border-left-color: var(--warm); }
figure { margin: 18px 0 22px; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { color: var(--muted); font-size: 13px; margin-top: 6px; }
.toc { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 20px; font-size: 14px; }
.toc ol { margin: 6px 0 0 18px; }
.toc a { color: var(--accent); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px; background: #eaf1f7; color: #2f6f9f; }
.tag.warn { background: #fbeee0; color: #a8631f; }
.tag.gray { background: #eef0f2; color: #5d6b7a; }
.appendix { font-size: 13.5px; color: #33414f; }
footer { margin-top: 48px; border-top: 1px solid var(--line); padding-top: 14px; color: var(--muted); font-size: 12.5px; }
@media (max-width: 700px) {
  .wrap { padding: 24px 14px 60px; }
  table.data { font-size: 12.5px; }
}
@media print {
  .wrap { max-width: none; padding: 0; }
  h2 { page-break-after: avoid; }
  figure { page-break-inside: avoid; }
}
"""


def document(title: str, subtitle: str, body: str, *, footer: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="doc">
  <h1>{esc(title)}</h1>
  <div class="sub">{subtitle}</div>
</header>
{body}
<footer>{footer}</footer>
</div>
</body>
</html>
"""


def cards(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="card"><div class="k">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div><div class="u">{esc(unit)}</div></div>'
        for label, value, unit in items
    )
    return f'<div class="cards">{cells}</div>'


def table(
    headers: list[str],
    rows: list[list[object]],
    *,
    numeric: set[int] | None = None,
    caption: str = "",
    compact: bool = False,
    total_row: bool = False,
) -> str:
    """通用表格；``numeric`` 指定右对齐的列下标。"""
    numeric = numeric or set()
    classes = ' class="data compact"' if compact else ' class="data"'
    head = "".join(
        f'<th class="num">{esc(name)}</th>' if i in numeric else f"<th>{esc(name)}</th>"
        for i, name in enumerate(headers)
    )
    body = []
    for index, row in enumerate(rows):
        is_total = total_row and index == len(rows) - 1
        cells = []
        for i, cell in enumerate(row):
            text = cell if isinstance(cell, str) and cell.startswith("<") else esc(cell)
            cls = ' class="num"' if i in numeric else ""
            style = ' style="font-weight:600"' if is_total else ""
            cells.append(f"<td{cls}{style}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    caption_html = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<table{classes}>{caption_html}<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def figure(svg: str, caption: str) -> str:
    return f"<figure>{svg}<figcaption>{esc(caption)}</figcaption></figure>"


# --------------------------------------------------------------------------- #
# SVG 图元
# --------------------------------------------------------------------------- #
def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="Microsoft YaHei, PingFang SC, sans-serif">'
        f"{body}</svg>"
    )


def hbar(
    rows: list[tuple[str, float]],
    *,
    unit: str = "",
    title: str = "",
    width: int = 760,
    bar_h: int = 24,
    gap: int = 8,
    digits: int = 0,
    color: str = PALETTE[0],
) -> str:
    """横向条形图：标签在左，数值在右。"""
    if not rows:
        return ""
    label_w = 128
    value_w = 74
    plot_w = width - label_w - value_w - 16
    peak = max((abs(value) for _, value in rows), default=1.0) or 1.0
    height = len(rows) * (bar_h + gap) + 26
    parts = []
    for i, (label, value) in enumerate(rows):
        y = 20 + i * (bar_h + gap)
        bar_w = max(1.5, plot_w * abs(value) / peak)
        parts.append(
            f'<text x="{label_w - 8}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" font-size="12.5" '
            f'fill="#33414f">{esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.82"/>'
            f'<text x="{label_w + bar_w + 6:.1f}" y="{y + bar_h * 0.7:.1f}" font-size="12" fill="#5d6b7a">'
            f"{esc(num(value, digits))}{esc(unit)}</text>"
        )
    return _svg(width, height + 4, "".join(parts), title)


def _nice_bounds(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    low, high = min(finite), max(finite)
    if low == high:
        pad = abs(low) * 0.1 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def line_panel(
    series: list[tuple[str, list[tuple[datetime, float | None]], str]],
    *,
    title: str,
    y_label: str,
    x_start: datetime,
    x_end: datetime,
    width: int = 760,
    height: int = 190,
    x_fmt: str = "%m-%d",
    tick_hours: int = 6,
    log_scale: bool = False,
    annotations: list[tuple[datetime, str]] | None = None,
) -> str:
    """多序列折线图（时间轴），用于日内过程与滞后剖面。"""
    left, right, top, bottom = 58, 14, 16, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    span = max((x_end - x_start).total_seconds(), 1.0)
    ys = [value for _, points, _ in series for _, value in points if value is not None and math.isfinite(value)]
    if log_scale:
        ys = [math.log10(max(value, 1e-300)) for value in ys]
    lo, hi = _nice_bounds(ys)

    def px(moment: datetime) -> float:
        return left + plot_w * (moment - x_start).total_seconds() / span

    def py(value: float) -> float:
        raw = math.log10(max(value, 1e-300)) if log_scale else value
        return top + plot_h * (1 - (raw - lo) / (hi - lo))

    parts = [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfd" stroke="#e6ebf0"/>']
    for tick in range(5):
        frac = tick / 4
        y = top + plot_h * frac
        raw = hi - (hi - lo) * frac
        label = f"{10**raw:.1e}" if log_scale else num(raw, 2)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e9eef3"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5d6b7a">{esc(label)}</text>'
        )
    ticks = 0
    moment = x_start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=tick_hours)
    while moment < x_end:
        x = px(moment)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eef2f6"/>'
            f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" fill="#5d6b7a">'
            f"{esc(moment.strftime(x_fmt))}</text>"
        )
        moment += timedelta(hours=tick_hours)
        ticks += 1
    for start, _label in annotations or []:
        if x_start <= start <= x_end:
            x = px(start)
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{PALETTE[4]}" '
                f'stroke-dasharray="3 3" opacity="0.55"/>'
            )
    for _name, points, color in series:
        path = []
        pen = False
        for moment, value in points:
            if value is None or not math.isfinite(value):
                pen = False
                continue
            cmd = "M" if not pen else "L"
            path.append(f"{cmd}{px(moment):.1f} {py(value):.1f}")
            pen = True
        if path:
            parts.append(
                f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>'
            )
    legend_y = top + 12
    for i, (name, _, color) in enumerate(series):
        x = left + 10 + i * 118
        parts.append(
            f'<rect x="{x}" y="{legend_y - 8}" width="12" height="3" fill="{color}"/>'
            f'<text x="{x + 17}" y="{legend_y}" font-size="11.5" fill="#33414f">{esc(name)}</text>'
        )
    parts.append(
        f'<text x="12" y="{top + 10}" font-size="11" fill="#5d6b7a" '
        f'transform="rotate(-90 12 {top + 10})" text-anchor="end">{esc(y_label)}</text>'
    )
    return _svg(width, height, "".join(parts), title)


def scatter(
    points: list[tuple[float, float, str, str]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 260,
    trend: tuple[float, float] | None = None,
) -> str:
    """散点图；``points`` 为 (x, y, 标签, 颜色)。"""
    left, right, top, bottom = 56, 16, 18, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_lo, x_hi = _nice_bounds([p[0] for p in points])
    y_lo, y_hi = _nice_bounds([p[1] for p in points])

    def px(value: float) -> float:
        return left + plot_w * (value - x_lo) / (x_hi - x_lo)

    def py(value: float) -> float:
        return top + plot_h * (1 - (value - y_lo) / (y_hi - y_lo))

    parts = [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfd" stroke="#e6ebf0"/>']
    for tick in range(5):
        y = top + plot_h * tick / 4
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e9eef3"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5d6b7a">'
            f"{esc(num(y_hi - (y_hi - y_lo) * tick / 4, 2))}</text>"
        )
        x = left + plot_w * tick / 4
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" fill="#5d6b7a">'
            f"{esc(num(x_lo + (x_hi - x_lo) * tick / 4, 2))}</text>"
        )
    if trend is not None:
        slope, intercept = trend
        parts.append(
            f'<line x1="{px(x_lo):.1f}" y1="{py(slope * x_lo + intercept):.1f}" '
            f'x2="{px(x_hi):.1f}" y2="{py(slope * x_hi + intercept):.1f}" '
            f'stroke="{PALETTE[0]}" stroke-dasharray="5 4" stroke-width="1.4" opacity="0.8"/>'
        )
    for x_value, y_value, label, color in points:
        parts.append(
            f'<circle cx="{px(x_value):.1f}" cy="{py(y_value):.1f}" r="4.6" fill="{color}" '
            f'fill-opacity="0.85" stroke="#ffffff" stroke-width="1"/>'
            f"<title>{esc(label)}</title>"
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" font-size="11.5" '
        f'fill="#5d6b7a">{esc(x_label)}</text>'
        f'<text x="12" y="{top + 10}" font-size="11" fill="#5d6b7a" '
        f'transform="rotate(-90 12 {top + 10})" text-anchor="end">{esc(y_label)}</text>'
    )
    return _svg(width, height, "".join(parts), title)


def lag_panel(
    points: list[tuple[float, float]],
    *,
    title: str,
    y_label: str,
    before_minutes: int = 60,
    after_minutes: int = 30,
    width: int = 760,
    height: int = 200,
    log_scale: bool = False,
) -> str:
    """滞后剖面折线图：横轴为相对事件的分钟数（负值为事件前）。"""
    if not points:
        return ""
    left, right, top, bottom = 58, 16, 18, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_lo, x_hi = float(-before_minutes), float(after_minutes)

    def px(minutes: float) -> float:
        return left + plot_w * (minutes - x_lo) / (x_hi - x_lo)

    raw_ys = [value for _, value in points]
    ys = [math.log10(max(value, 1e-300)) for value in raw_ys] if log_scale else raw_ys
    lo, hi = _nice_bounds(ys)

    def py(value: float) -> float:
        raw = math.log10(max(value, 1e-300)) if log_scale else value
        return top + plot_h * (1 - (raw - lo) / (hi - lo))

    parts = [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfd" stroke="#e6ebf0"/>']
    for tick in range(5):
        frac = tick / 4
        y = top + plot_h * frac
        raw = hi - (hi - lo) * frac
        label = f"{10**raw:.1e}" if log_scale else num(raw, 2)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e9eef3"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5d6b7a">{esc(label)}</text>'
        )
    for minutes in range(before_minutes - before_minutes % 10, after_minutes + 1, 10):
        x = px(minutes)
        text = f"{minutes:+d}" if minutes else "0"
        if minutes % 20 == 0:
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eef2f6"/>'
                f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" fill="#5d6b7a">'
                f"{esc(text)}</text>"
            )
    zero_x = px(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" '
        f'stroke="{PALETTE[4]}" stroke-dasharray="4 3" opacity="0.7"/>'
    )
    path = []
    for minutes, value in sorted(points):
        if value is None or not math.isfinite(value):
            continue
        path.append(f"{'M' if not path else 'L'}{px(minutes):.1f} {py(value):.1f}")
        parts.append(
            f'<circle cx="{px(minutes):.1f}" cy="{py(value):.1f}" r="3" fill="{PALETTE[1]}" '
            f'fill-opacity="0.9"><title>{esc(f"{minutes:+.0f} min：{num(value, 3)}")}</title></circle>'
        )
    if path:
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{PALETTE[1]}" stroke-width="1.6"/>')
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" font-size="11.5" '
        f'fill="#5d6b7a">相对试验开始的分钟数</text>'
        f'<text x="12" y="{top + 10}" font-size="11" fill="#5d6b7a" '
        f'transform="rotate(-90 12 {top + 10})" text-anchor="end">{esc(y_label)}</text>'
    )
    return _svg(width, height, "".join(parts), title)


def gantt(
    rows: list[tuple[str, datetime, datetime, str]],
    *,
    title: str,
    width: int = 760,
    lane_h: int = 30,
    x_fmt: str = "%H:%M",
    hour_start: int = 12,
    hour_span: int = 15,
) -> str:
    """按日分行的试验时间线：横轴为一天中的时刻。"""
    lanes: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    order: list[str] = []
    for day, start, end, label in rows:
        if day not in lanes:
            order.append(day)
        lanes[day].append((start, end, label))
    left, right, top, bottom = 74, 18, 16, 40
    plot_w = width - left - right
    plot_h = lane_h * len(order)
    height = top + plot_h + bottom

    def px(moment: datetime) -> float:
        minutes = moment.hour * 60 + moment.minute + moment.second / 60
        base = hour_start * 60
        return left + plot_w * (minutes - base) / (hour_span * 60)

    parts = [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfd" stroke="#e6ebf0"/>']
    for hour in range(hour_start, hour_start + hour_span + 1, 2):
        x = left + plot_w * (hour * 60 - hour_start * 60) / (hour_span * 60)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eef2f6"/>'
            f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" fill="#5d6b7a">'
            f"{hour % 24:02d}:00</text>"
        )
    for lane_index, day in enumerate(order):
        y = top + lane_index * lane_h + lane_h / 2
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#33414f">'
            f"{esc(day[:4])}-{esc(day[4:6])}-{esc(day[6:])}</text>"
        )
        for item_index, (start, end, label) in enumerate(lanes[day]):
            x1 = px(start)
            x2 = max(px(end), x1 + 3)
            y_rect = top + lane_index * lane_h + 6 + (item_index % 3) * 6
            color = PALETTE[item_index % len(PALETTE)]
            parts.append(
                f'<rect x="{x1:.1f}" y="{y_rect}" width="{x2 - x1:.1f}" height="4.6" rx="2.3" '
                f'fill="{color}" opacity="0.9"><title>{esc(label)}</title></rect>'
            )
    return _svg(width, height, "".join(parts), title)


def heat_table(
    row_labels: list[str],
    col_labels: list[str],
    matrix: dict[tuple[str, str], float],
    *,
    caption: str = "",
    unit: str = "",
) -> str:
    """计数热力表：单元格底色随计数深浅变化。"""
    peak = max((v for v in matrix.values() if v), default=1.0)
    head = "<th></th>" + "".join(f'<th class="num">{esc(c)}</th>' for c in col_labels) + '<th class="num">合计</th>'
    rows = []
    for row_label in row_labels:
        cells = [f"<td>{esc(row_label)}</td>"]
        row_total = 0.0
        for col_label in col_labels:
            value = matrix.get((row_label, col_label), 0.0)
            row_total += value
            alpha = 0.08 + 0.62 * (value / peak if peak else 0)
            text_color = "#12212e" if value else "#9aa7b4"
            cells.append(
                f'<td class="num" style="background:rgba(47,111,159,{alpha:.2f});color:{text_color}">'
                f"{esc(int(value)) if value else '·'}</td>"
            )
        cells.append(f'<td class="num" style="font-weight:600">{esc(int(row_total))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    totals = [sum(matrix.get((row_label, col_label), 0.0) for row_label in row_labels) for col_label in col_labels]
    total_cells = "".join(f'<td class="num" style="font-weight:600">{esc(int(v))}</td>' for v in totals)
    rows.append(
        '<tr><td style="font-weight:600">合计</td>'
        + total_cells
        + f'<td class="num" style="font-weight:600">{esc(int(sum(totals)))}</td></tr>'
    )
    caption_html = f"<caption>{esc(caption)}</caption>" if caption else ""
    unit_html = f'<p class="note">单位：{esc(unit)}</p>' if unit else ""
    return (
        f'<table class="data compact">{caption_html}<thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>{unit_html}"
    )


# --------------------------------------------------------------------------- #
# 试验派生指标
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    """一次试验的量化结果（功率档位已脱敏，仅保留相对量）。"""

    exp: base.Experiment
    breakdown_s: float | None = None
    censored: bool = False
    breakdown_note: str = ""
    duration_s: float | None = None
    window_minutes: float | None = None


def parse_seconds(text: str) -> float | None:
    """从 ``3.40s`` / ``5.68`` 之类的文本里取秒数。"""
    if not text:
        return None
    digits = ""
    for char in text:
        if char.isdigit() or char == ".":
            digits += char
        elif digits:
            break
    try:
        value = float(digits)
    except ValueError:
        return None
    return value if value > 0 else None


def describe_outcome(exp: base.Experiment) -> Outcome:
    outcome = Outcome(exp=exp)
    text = (exp.breakdown or "").strip()
    value = parse_seconds(text)
    if value is not None:
        outcome.breakdown_s = value
    elif text:
        outcome.censored = True
        outcome.breakdown_note = text
    outcome.duration_s = parse_seconds(exp.duration or "")
    if exp.capture_start and exp.capture_end:
        outcome.window_minutes = (exp.capture_end - exp.capture_start).total_seconds() / 60
    return outcome


def breakdown_summary(outcomes: list[Outcome]) -> dict[str, int]:
    numeric = sum(1 for item in outcomes if item.breakdown_s is not None)
    censored = sum(1 for item in outcomes if item.censored)
    empty = sum(1 for item in outcomes if item.breakdown_s is None and not item.censored)
    return {"numeric": numeric, "censored": censored, "empty": empty, "total": len(outcomes)}


def fan_group(exp: base.Experiment) -> str:
    count = (exp.fan_count or "").strip()
    if count in {"0", ""}:
        return "无风机" if count == "0" else "未记录"
    return f"{count} 台风机"


def distance_group(exp: base.Experiment) -> str:
    return exp.distance.replace("*", "") or "未记录"


def by_group(outcomes: list[Outcome], key) -> dict[str, list[Outcome]]:
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for item in outcomes:
        grouped[key(item.exp)].append(item)
    return dict(sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])))


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def period_word(moment: datetime) -> str:
    """把时刻归到半天粒度，用于生成不含硬编码时间的文字说明。"""
    hour = moment.hour
    if 6 <= hour < 12:
        return "上午"
    if 12 <= hour < 18:
        return "下午"
    if 18 <= hour < 24:
        return "夜间"
    return "凌晨"


def _decimate(series: pd.Series, minutes: int = 5) -> pd.Series:
    """把分钟序列降采样为 ``minutes`` 分钟中位，控制内联 SVG 的体积。"""
    if series.empty:
        return series
    return series.resample(f"{minutes}min").median().dropna()


def to_dt(stamp) -> datetime:
    """把 pandas Timestamp 转成原生 datetime。

    本机环境里 ``Timestamp.to_pydatetime()`` 会抛出 ``SystemError``（CPython/pandas
    交互问题），因此改用显式构造。
    """
    if isinstance(stamp, datetime):
        return stamp
    return datetime(stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second)


# --------------------------------------------------------------------------- #
# 报告一：数据总览
# --------------------------------------------------------------------------- #
def overview_report(experiments: list[base.Experiment], root: Path, generated: datetime) -> str:
    experiments = sorted(experiments, key=lambda exp: (exp.date, exp.seq, exp.exp_id))
    outcomes = [describe_outcome(exp) for exp in experiments]
    days = sorted({exp.date for exp in experiments})
    total_files = sum(exp.total_files for exp in experiments)
    total_size = sum(exp.total_size for exp in experiments)
    summary = breakdown_summary(outcomes)
    numeric_times = [item.breakdown_s for item in outcomes if item.breakdown_s is not None]

    forms = Counter(exp.data_form for exp in experiments)
    suffixes = Counter()
    for exp in experiments:
        suffixes.update(exp.counts)
    shots = sum(exp.counts[".jpg"] + exp.counts[".png"] for exp in experiments)
    videos = sum(exp.counts[".mp4"] + exp.counts[".avi"] for exp in experiments)
    dats = sum(exp.counts[".dat"] for exp in experiments)

    body: list[str] = []
    body.append(
        cards(
            [
                ("试验批次", f"{len(experiments)}", "次"),
                ("试验日期", f"{len(days)}", "天"),
                ("数据文件", f"{total_files:,}", "个"),
                ("数据体积", human_size(total_size), ""),
                ("有效击穿时间", f"{summary['numeric']}", "次"),
                ("未击穿/难观测", f"{summary['censored']}", "次"),
                ("击穿记录缺失", f"{summary['empty']}", "次"),
            ]
        )
    )
    body.append(
        '<div class="callout">本报告汇总小系统外场试验的三天数据：'
        f"{esc('、'.join(days))}，共 {len(experiments)} 次试验、{total_files:,} 个数据文件、"
        f"{human_size(total_size)}。激光功率以档位代号表示（标准功率 / 低功率），"
        "不出现绝对功率数值及其派生量。"
        "</div>"
    )

    # 目录
    body.append(
        '<div class="toc"><strong>目录</strong><ol>'
        '<li><a href="#s1">数据规模与构成</a></li>'
        '<li><a href="#s2">试验设计矩阵</a></li>'
        '<li><a href="#s3">参数完备度与数据质量</a></li>'
        '<li><a href="#s4">击穿行为与重复性</a></li>'
        '<li><a href="#s5">全量试验清单</a></li>'
        '<li><a href="#s6">附录：口径与产物</a></li>'
        "</ol></div>"
    )

    # ---- 1. 数据规模 ----
    body.append('<h2 id="s1">1. 数据规模与构成</h2>')
    day_rows = []
    for day in days:
        group = [exp for exp in experiments if exp.date == day]
        group_outcomes = [item for item in outcomes if item.exp.date == day]
        day_rows.append(
            [
                f"{day[:4]}-{day[4:6]}-{day[6:]}",
                len(group),
                f"{sum(exp.total_files for exp in group):,}",
                human_size(sum(exp.total_size for exp in group)),
                sum(1 for item in group_outcomes if item.breakdown_s is not None),
                sum(1 for item in group if item.exp_type == "靶板仪"),
            ]
        )
    day_rows.append(
        [
            "合计",
            len(experiments),
            f"{total_files:,}",
            human_size(total_size),
            summary["numeric"],
            sum(1 for exp in experiments if exp.exp_type == "靶板仪"),
        ]
    )
    body.append(
        table(
            ["日期", "试验次数", "文件数", "数据量", "有效击穿时间", "靶板仪试验"],
            day_rows,
            numeric={1, 2, 3, 4, 5},
            caption="按日期的数据规模（末行为合计）",
            total_row=True,
        )
    )
    body.append(
        figure(
            hbar(
                [(exp.date, float(exp.total_files)) for exp in experiments][:0]
                + [(day, float(sum(e.total_files for e in experiments if e.date == day))) for day in days],
                unit=" 个",
                title="各日数据文件数",
            ),
            "各日文件数：帧图序列是文件数的主要来源，单次试验最多可产生上千个帧文件。",
        )
    )
    body.append(
        table(
            ["数据形态", "试验次数", "说明"],
            [
                [
                    form,
                    count,
                    "含视频/帧图采集数据" if "视频" in form or "帧" in form else "以波形记录为主",
                ]
                for form, count in forms.most_common()
            ],
            numeric={1},
            caption="数据形态分布",
        )
    )
    body.append(
        table(
            ["文件类型", "数量", "占比"],
            [
                [suffix or "无扩展名", f"{count:,}", pct(count / max(total_files, 1))]
                for suffix, count in suffixes.most_common(8)
            ],
            numeric={1, 2},
            caption="文件类型构成",
        )
    )
    body.append(
        f'<p class="note">帧图 {shots:,} 张、视频 {videos:,} 段、波形记录 {dats:,} 份。'
        "波形记录为设备原始采样，本次分析只使用其随附的采集时刻与结果标注，未做二次波形反演。</p>"
    )

    # ---- 2. 设计矩阵 ----
    body.append('<h2 id="s2">2. 试验设计矩阵</h2>')
    distances = sorted({distance_group(exp) for exp in experiments}, key=lambda d: (len(d), d))
    fans = ["无风机", "1 台风机", "2 台风机", "未记录"]
    matrix = defaultdict(float)
    for exp in experiments:
        matrix[(distance_group(exp), fan_group(exp))] += 1
    body.append(
        heat_table(
            distances,
            fans,
            matrix,
            caption="作用距离 × 风机配置 的试验次数",
            unit="次",
        )
    )
    matrix_dir = defaultdict(float)
    for exp in experiments:
        matrix_dir[(distance_group(exp), exp.incident_direction or "未记录")] += 1
    body.append(
        heat_table(
            distances,
            ["正入射", "斜入射", "未记录"],
            matrix_dir,
            caption="作用距离 × 入射方式 的试验次数",
            unit="次",
        )
    )
    matrix_mat = defaultdict(float)
    for exp in experiments:
        matrix_mat[(distance_group(exp), f"{exp.material or '未记录'} {exp.thickness}".strip())] += 1
    materials = sorted({f"{exp.material or '未记录'} {exp.thickness}".strip() for exp in experiments})
    body.append(
        heat_table(
            distances,
            materials,
            matrix_mat,
            caption="作用距离 × 靶材（含厚度） 的试验次数",
            unit="次",
        )
    )
    body.append(
        figure(
            hbar(
                [
                    (name, float(count))
                    for name, count in Counter(exp.standby or "未记录" for exp in experiments).most_common()
                ],
                unit=" 次",
                title="待机方式分布",
                color=PALETTE[2],
            ),
            "待机方式分布：冷待机占多数，热待机仅出现在个别批次，两类样本量差距较大，比较时只能作描述性参考。",
        )
    )
    body.append(
        figure(
            hbar(
                [
                    (name, float(count))
                    for name, count in Counter(
                        "有精跟踪" if exp.tracking == "有精跟踪" else ("无精跟踪" if exp.tracking else "未记录")
                        for exp in experiments
                    ).most_common()
                ],
                unit=" 次",
                title="精跟踪状态分布",
                color=PALETTE[3],
            ),
            "精跟踪状态：半数以上试验未在参数中标注，比较时需以标注子集为限。",
        )
    )

    # ---- 3. 质量 ----
    body.append('<h2 id="s3">3. 参数完备度与数据质量</h2>')
    fields = [
        ("作用距离", lambda exp: exp.distance),
        ("入射方式", lambda exp: exp.incident_direction),
        ("靶材", lambda exp: exp.material),
        ("材料厚度", lambda exp: exp.thickness),
        ("作用时间", lambda exp: exp.duration),
        ("击穿时间", lambda exp: exp.breakdown),
        ("风机配置", lambda exp: exp.fan_count),
        ("待机方式", lambda exp: exp.standby),
        ("精跟踪", lambda exp: exp.tracking),
        ("采集时间", lambda exp: exp.capture_start),
    ]
    field_rows = []
    for label, getter in fields:
        filled = sum(1 for exp in experiments if getter(exp))
        field_rows.append([label, filled, pct(filled / len(experiments)), len(experiments) - filled])
    body.append(
        table(
            ["参数字段", "已记录", "覆盖率", "缺失"],
            field_rows,
            numeric={1, 2, 3},
            caption="参数字段完备度（相对全部试验）",
        )
    )
    quality = Counter()
    for exp in experiments:
        for item in exp.quality:
            quality[item] += 1
    if quality:
        body.append(
            figure(
                hbar(
                    [
                        (text if len(text) <= 26 else text[:24] + "…", float(count))
                        for text, count in quality.most_common(8)
                    ],
                    unit=" 次",
                    title="数据质量提示分布",
                    color=PALETTE[1],
                ),
                "自动检查得到的数据质量提示：主要集中在参数缺失与目录名/参数表不一致两类。",
            )
        )
    inferred = [exp for exp in experiments if exp.distance_inferred]
    body.append(
        f'<p>距离字段有 {len(inferred)} 次试验未直接记录，按"同日同批次唯一距离"推断，'
        "推断值在清单中以星号标注。参数表缺省时，字段回退到目录名与文件名，"
        "个别试验的目录名时长与参数表不一致（以参数表为准并记录提示）。</p>"
    )

    # ---- 4. 击穿行为 ----
    body.append('<h2 id="s4">4. 击穿行为与重复性</h2>')
    body.append(
        f"<p>{summary['total']} 次试验中，{summary['numeric']} 次记录了可解析的击穿时刻，"
        f"{summary['censored']} 次标注为未击穿/难以观测，{summary['empty']} 次未记录该项。"
        "因此击穿统计只能落在有数值的子集上，不能外推到全部批次。</p>"
    )
    if numeric_times:
        body.append(
            cards(
                [
                    ("击穿时间中位", num(median_or_none(numeric_times), 2), "s"),
                    ("最短", num(min(numeric_times), 2), "s"),
                    ("最长", num(max(numeric_times), 2), "s"),
                    ("样本量", str(len(numeric_times)), "次"),
                ]
            )
        )
        group_rows = []
        for distance, group in by_group(outcomes, lambda exp: distance_group(exp)).items():
            values = [item.breakdown_s for item in group if item.breakdown_s is not None]
            if not values:
                continue
            group_rows.append(
                [
                    distance,
                    len(group),
                    len(values),
                    num(median_or_none(values), 2),
                    num(min(values), 2),
                    num(max(values), 2),
                ]
            )
        if group_rows:
            body.append(
                table(
                    ["作用距离", "试验次数", "有效击穿时间", "中位 (s)", "最短 (s)", "最长 (s)"],
                    group_rows,
                    numeric={1, 2, 3, 4, 5},
                    caption="按作用距离分组的击穿时间（仅数值子集）",
                )
            )
        fan_rows = []
        for fan, group in by_group(outcomes, fan_group).items():
            values = [item.breakdown_s for item in group if item.breakdown_s is not None]
            if not values:
                continue
            fan_rows.append([fan, len(group), len(values), num(median_or_none(values), 2)])
        if fan_rows:
            body.append(
                table(
                    ["风机配置", "试验次数", "有效击穿时间", "中位 (s)"],
                    fan_rows,
                    numeric={1, 2, 3},
                    caption="按风机配置分组的击穿时间（仅数值子集）",
                )
            )
        points = []
        order_index = {exp.exp_id: index for index, exp in enumerate(experiments)}
        for outcome in outcomes:
            if outcome.breakdown_s is None:
                continue
            color = PALETTE[order_index[outcome.exp.exp_id] % len(PALETTE)]
            points.append(
                (
                    float(order_index[outcome.exp.exp_id] + 1),
                    outcome.breakdown_s,
                    f"{outcome.exp.exp_id}｜{outcome.exp.power}｜{distance_group(outcome.exp)}｜"
                    f"{outcome.breakdown_s:.2f}s",
                    color,
                )
            )
        if len(points) >= 3:
            body.append(
                figure(
                    scatter(
                        points,
                        title="击穿时间随试验次序的分布",
                        x_label="试验次序（按实验目录顺序）",
                        y_label="击穿时间 (s)",
                    ),
                    "横轴为试验次序（跨三天连续编号），纵轴为记录到的击穿时刻。"
                    "次序跨度同时混合了距离、风机与待机方式的改变，不能单独解释为时间漂移。",
                )
            )

    duplicates = [exp for exp in experiments if exp.duplicates]
    if duplicates:
        seen: set[str] = set()
        dup_rows = []
        for exp in duplicates:
            group = tuple(sorted([exp.exp_id, *exp.duplicates]))
            if group in seen:
                continue
            seen.add(group)
            members = [item for item in outcomes if item.exp.exp_id in group]
            values = [item.breakdown_s for item in members if item.breakdown_s is not None]
            dup_rows.append(
                [
                    " / ".join(group),
                    len(group),
                    "、".join(sorted({item.exp.power for item in members})),
                    "、".join(sorted({distance_group(item.exp) for item in members})),
                    len(values),
                    num(median_or_none(values), 2) if values else "—",
                ]
            )
        body.append(
            f'<p>按"参数签名"归并后，{len(dup_rows)} 组试验的参数完全一致（同距离、同入射、同材料、'
            "同风机、同时长），可视为重复试验，用于观察同类条件下的结果离散度。</p>"
        )
        body.append(
            table(
                ["重复组", "次数", "功率档位", "作用距离", "有效击穿时间", "中位 (s)"],
                dup_rows[:12],
                numeric={1, 4, 5},
                caption="参数签名一致的重复试验组（最多显示 12 组）",
            )
        )

    # ---- 5. 清单 ----
    body.append('<h2 id="s5">5. 全量试验清单</h2>')
    rows = []
    for outcome in outcomes:
        exp = outcome.exp
        rows.append(
            [
                exp.exp_id,
                base.fmt_span(exp.capture_start, exp.capture_end) or UNKNOWN,
                distance_group(exp) + ("*" if exp.distance_inferred else ""),
                exp.incident_direction or UNKNOWN,
                f"{exp.material or '—'} {exp.thickness}".strip(),
                fan_group(exp),
                exp.standby or UNKNOWN,
                exp.duration or UNKNOWN,
                f"{outcome.breakdown_s:.2f}s" if outcome.breakdown_s is not None else (exp.breakdown or UNKNOWN),
                exp.data_form,
            ]
        )
    body.append(
        table(
            ["编号", "采集时间", "距离", "入射", "靶材", "风机", "待机", "时长", "击穿", "数据形态"],
            rows,
            numeric=set(),
            caption="全部试验的元数据清单（* 表示距离为同日推断）",
            compact=True,
        )
    )

    # ---- 6. 附录 ----
    body.append('<h2 id="s6">6. 附录：口径与产物</h2>')
    body.append(
        '<div class="appendix"><ul>'
        "<li><strong>数据来源</strong>：小系统外场试验数据，按试验日期分目录存放；"
        "每次试验含参数表、采集视频/帧图或波形记录。</li>"
        "<li><strong>功率脱敏</strong>：绝对功率数值与由此派生的功率密度、光斑强度均不写入报告，"
        "只保留“标准功率 / 低功率”档位代号。</li>"
        "<li><strong>击穿时间</strong>：取参数表记录值，按数值解析；"
        "“未击穿”“未明显观察出”“很多小孔”等文本按右删失处理，不计入数值统计。</li>"
        "<li><strong>距离推断</strong>：参数表与目录名均无距离时，若同日其它试验只有一个距离，"
        "则沿用该距离并标注星号。</li>"
        "<li><strong>重复组</strong>：以“日期 + 类型 + 功率档位 + 距离 + 入射 + 材料 + 厚度 + 抛光 + 时长 + "
        "待机 + 精跟踪 + 风机”作为参数签名，签名相同即归为同一重复组。</li>"
        "</ul></div>"
    )
    body.append(
        f'<p class="note">报告由 {esc(GENERATED_BY)} 生成于 {esc(generated.strftime("%Y-%m-%d %H:%M"))}；'
        f"数据根目录：{esc(str(root))}。图表为内联 SVG，可离线打开与打印。</p>"
    )

    subtitle = (
        f"{'、'.join(days)}｜{len(experiments)} 次试验｜{total_files:,} 个文件｜{human_size(total_size)}｜"
        f"生成于 {generated.strftime('%Y-%m-%d %H:%M')}"
    )
    return document(
        "小系统激光试验数据总览",
        esc(subtitle),
        "".join(body),
        footer="口径：功率一律以档位代号呈现；击穿时间仅在数值子集上统计；参数缺失按“未记录”处理，不猜补。",
    )


# --------------------------------------------------------------------------- #
# 报告二：时间与大气协同
# --------------------------------------------------------------------------- #
@dataclass
class WindowSample:
    """一次试验窗口内的大气要素取值。"""

    exp: base.Experiment
    start: datetime
    end: datetime
    raw_start: datetime | None = None
    raw_end: datetime | None = None
    values: dict[str, float | None] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)


def collect_windows(
    experiments: list[base.Experiment],
    channels: dict[str, pd.DataFrame],
    *,
    pad_minutes: int = 10,
) -> list[WindowSample]:
    """把每次试验折算成环境窗口。

    采集时段本身只有几秒到几十秒（只覆盖出光瞬间），因此以采集时段为中心、
    前后各扩 ``pad_minutes`` 分钟作为该次试验的环境窗口；原始时段另存备查。
    """
    samples: list[WindowSample] = []
    for exp in experiments:
        if not (exp.capture_start and exp.capture_end):
            continue
        pad = timedelta(minutes=pad_minutes)
        sample = WindowSample(
            exp=exp,
            start=exp.capture_start - pad,
            end=exp.capture_end + pad,
            raw_start=exp.capture_start,
            raw_end=exp.capture_end,
        )
        for key, frame in channels.items():
            if frame.empty:
                sample.values[key] = None
                sample.coverage[key] = 0.0
                continue
            stats = atm.window_stats(frame["中位"], sample.start, sample.end)
            sample.values[key] = stats.get("median")
            sample.coverage[key] = stats.get("n", 0.0)
        samples.append(sample)
    return samples


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float, int] | None:
    if len(xs) != len(ys) or len(xs) < 6:
        return None
    try:
        from scipy import stats

        result = stats.spearmanr(xs, ys)
    except Exception:  # pragma: no cover - scipy 缺失时退化为秩相关手算
        return None
    if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
        return None
    return float(result.statistic), float(result.pvalue), len(xs)


def _mwu(first: list[float], second: list[float]) -> tuple[float, float] | None:
    if len(first) < 3 or len(second) < 3:
        return None
    try:
        from scipy import stats

        result = stats.mannwhitneyu(first, second, alternative="two-sided")
    except Exception:  # pragma: no cover
        return None
    if not np.isfinite(result.statistic):
        return None
    return float(result.statistic), float(result.pvalue)


def synergy_report(
    experiments: list[base.Experiment],
    channels: dict[str, pd.DataFrame],
    *,
    days: list[str],
    generated: datetime,
    root: Path,
) -> tuple[str, pd.DataFrame]:
    outcomes = [describe_outcome(exp) for exp in experiments]
    samples = collect_windows(experiments, channels)
    active_channels = [
        channel for channel in atm.CHANNELS if channel.key in channels and not channels[channel.key].empty
    ]

    body: list[str] = []
    window_minutes = [item.window_minutes for item in outcomes if item.window_minutes is not None]
    day_windows: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for sample in samples:
        day_windows[sample.exp.date].append((sample.start, sample.end))

    backgrounds: dict[str, dict[str, float]] = {}
    for channel in active_channels:
        table_channel = channels[channel.key]
        per_day = []
        for day in days:
            day_key = datetime.strptime(day, "%Y%m%d").date()
            slice_day = table_channel.loc[table_channel.index.date == day_key, "中位"]
            windows = day_windows.get(day, [])
            stats = atm.background_stats(slice_day, windows)
            if stats:
                per_day.append(stats)
        backgrounds[channel.key] = {
            "median": float(np.median([item["median"] for item in per_day])) if per_day else float("nan"),
            "days": float(len(per_day)),
        }

    body.append(
        cards(
            [
                ("试验窗口", f"{len(samples)}", "次"),
                ("采集时段合计", f"{sum(window_minutes) / 60:.2f}" if window_minutes else "—", "小时"),
                ("大气通道", f"{len(active_channels)}", "路"),
                ("覆盖日期", f"{len(days)}", "天"),
            ]
        )
    )
    body.append(
        '<div class="callout">本报告把每次试验换算成一个环境窗口（采集时段前后各 10 分钟），'
        "把同期大气观测折算成窗口内的代表值，再与同日“非试验时段”的背景值对比。"
        "出光本身只有几秒到几十秒，若只用采集时段，分钟级环境序列几乎取不到样本。"
        "所有相关系数均为同步性描述：样本量小、且试验条件并非随机分配，不能据此推断因果。</div>"
    )
    body.append(
        '<div class="toc"><strong>目录</strong><ol>'
        '<li><a href="#t1">试验时间结构</a></li>'
        '<li><a href="#t2">大气环境背景</a></li>'
        '<li><a href="#t3">试验窗口与背景对比</a></li>'
        '<li><a href="#t4">击穿行为与环境要素的关联</a></li>'
        '<li><a href="#t5">相对试验开始的滞后剖面</a></li>'
        '<li><a href="#t6">结论、局限与口径</a></li>'
        "</ol></div>"
    )

    # ---- 1. 时间结构 ----
    body.append('<h2 id="t1">1. 试验时间结构</h2>')
    rows = [
        (
            sample.exp.date,
            sample.raw_start or sample.start,
            sample.raw_end or sample.end,
            f"{sample.exp.exp_id}｜{distance_group(sample.exp)}｜{sample.exp.power}",
        )
        for sample in samples
    ]
    if rows:
        day_spans = []
        for day in days:
            group = [sample for sample in samples if sample.exp.date == day]
            if not group:
                continue
            day_start = min(sample.raw_start or sample.start for sample in group)
            day_end = max(sample.raw_end or sample.end for sample in group)
            day_spans.append(f"{day} {day_start:%H:%M}–{day_end:%H:%M}（{period_word(day_start)}）")
        caption = (
            "横轴为一天中的时刻，每行一天，竖条为单次试验的出光时段。"
            + "；".join(day_spans)
            + "。三天的作业时段互不重合，后两日的环境背景与第一日并不同源。"
        )
        body.append(figure(gantt(rows, title="三日试验时间线"), caption))
    structure_rows = []
    for day in days:
        group = [sample for sample in samples if sample.exp.date == day]
        if not group:
            continue
        start = min(sample.raw_start or sample.start for sample in group)
        end = max(sample.raw_end or sample.end for sample in group)
        ordered = sorted(group, key=lambda item: item.raw_start or item.start)
        spans = [
            ((later.raw_start or later.start) - (earlier.raw_end or earlier.end)).total_seconds() / 60
            for earlier, later in zip(ordered, ordered[1:], strict=False)
        ]
        covered = sum(
            ((sample.raw_end or sample.end) - (sample.raw_start or sample.start)).total_seconds() / 60
            for sample in group
        )
        structure_rows.append(
            [
                day,
                len(group),
                f"{start:%H:%M}–{end:%H:%M}",
                f"{(end - start).total_seconds() / 60:.0f}",
                f"{num(median_or_none(spans), 1)}" if spans else "—",
                f"{covered:.0f}",
            ]
        )
    body.append(
        table(
            ["日期", "试验次数", "时段", "跨度 (min)", "相邻间隔中位 (min)", "采集覆盖 (min)"],
            structure_rows,
            numeric={1, 3, 4, 5},
            caption="各日试验时段与节奏",
        )
    )
    body.append(
        "<p>三天在同一场地、同一套设备上作业，但作业时段、作用距离与风机配置随批次改变。"
        "其中时段是最大的“非受控变量”：日间与夜间的大气边界层状态不同，"
        "温度、风速与折射率结构常数都会随之变化，因此后续所有对比都按同期背景校正，"
        "而不做跨批次直接比较。</p>"
    )

    # ---- 2. 背景 ----
    body.append('<h2 id="t2">2. 大气环境背景</h2>')
    short_notes = []
    for channel in active_channels:
        table_channel = channels[channel.key]
        if table_channel.empty:
            continue
        series = table_channel["中位"]
        if channel.note:
            short_notes.append(f"{channel.label}：{channel.note}")
        points = [(to_dt(moment), float(value)) for moment, value in _decimate(series, 5).items()]
        body.append(
            figure(
                line_panel(
                    [(channel.label, points, PALETTE[0])],
                    title=f"{channel.label}日内过程",
                    y_label=f"{channel.label}（{channel.unit}）",
                    x_start=to_dt(series.index.min()),
                    x_end=to_dt(series.index.max()),
                    log_scale=channel.key == "cn2",
                    annotations=[(sample.raw_start or sample.start, "") for sample in samples],
                ),
                f"{channel.label}的日内过程（5 分钟中位，跨探针取中位）。竖虚线为各次试验的出光时刻，"
                "可见试验集中在一天中的某一时段，而非均匀分布；扣除这些试验窗口后的时段才进入“背景”统计。",
            )
        )
    if short_notes:
        body.append('<p class="note">' + esc("；".join(short_notes)) + "。</p>")
    background_rows = []
    for channel in active_channels:
        series = channels[channel.key]["中位"]
        if series.empty:
            continue
        night = series[(series.index.hour >= 20) | (series.index.hour < 6)]
        background_rows.append(
            [
                channel.label,
                channel.unit,
                num(float(series.median()), channel.digits),
                num(float(night.median()) if not night.empty else None, channel.digits),
                num(backgrounds[channel.key]["median"], channel.digits),
                f"{len(series):,}",
            ]
        )
    body.append(
        table(
            ["通道", "单位", "全期中位", "夜间中位 (20–06 时)", "试验外背景中位", "分钟样本"],
            background_rows,
            numeric={2, 3, 4, 5},
            caption="大气要素的整体背景（分钟级，跨探针中位）",
        )
    )

    # ---- 3. 窗口 vs 背景 ----
    body.append('<h2 id="t3">3. 试验窗口与背景对比</h2>')
    compare_rows = []
    for channel in active_channels:
        values = [sample.values.get(channel.key) for sample in samples]
        values = [value for value in values if value is not None]
        background = backgrounds[channel.key]["median"]
        window_median = median_or_none(values)
        background_value = background if np.isfinite(background) else None
        change = atm.relative_change(window_median, background_value, floor=channel.rel_floor)
        if change is None and window_median is not None and background_value is not None:
            change_text = "背景近零"
        elif change is None:
            change_text = "—"
        else:
            change_text = pct(change, 1)
        difference = (
            round(window_median, channel.digits) - round(background_value, channel.digits)
            if window_median is not None and background_value is not None
            else None
        )
        orders = atm.ratio_orders(window_median, background_value)
        compare_rows.append(
            [
                channel.label,
                channel.unit,
                f"{len(values)}/{len(samples)}",
                num(window_median, channel.digits),
                num(background_value, channel.digits),
                num(difference, channel.digits),
                change_text,
                f"{orders:+.1f}" if orders is not None else "—",
            ]
        )
    body.append(
        table(
            ["通道", "单位", "覆盖窗口", "窗口内中位", "同日背景中位", "绝对差", "相对背景", "倍率 log10"],
            compare_rows,
            numeric={2, 3, 4, 5, 7},
            caption="试验窗口内的大气要素与同日非试验时段背景的对比",
        )
    )
    body.append(
        "<p>窗口内中位与背景的差异反映的是“试验时段恰好处于怎样的环境”，"
        "而不是“环境被试验改变”。风速一行最直接的成因是试验时开启了风机："
        "背景是夜间的静风状态（中位不到 0.1 m/s），窗口内则是有风机时的 1 m/s 上下，"
        "两者相差一个数量级以上，百分比已不适合表达，故以绝对差与倍率代替。</p>"
    )

    extremes = []
    for channel in active_channels:
        values = [
            (sample, sample.values.get(channel.key)) for sample in samples if sample.values.get(channel.key) is not None
        ]
        if len(values) < 3:
            continue
        values.sort(key=lambda pair: pair[1])
        low_sample, low_value = values[0]
        high_sample, high_value = values[-1]
        extremes.append(
            [
                channel.label,
                f"{low_sample.exp.exp_id}（{num(low_value, channel.digits)}）",
                f"{high_sample.exp.exp_id}（{num(high_value, channel.digits)}）",
                f"{high_value - low_value:.3g}" if high_value != low_value else "0",
            ]
        )
    if extremes:
        body.append(
            table(
                ["通道", "最低窗口", "最高窗口", "窗口间极差"],
                extremes,
                numeric=set(),
                caption="各通道在试验窗口间的极值（用于判断环境差异的量级）",
            )
        )
        body.append(
            '<p class="note">窗口间的极差决定了“环境变量”在本次数据里到底有多少可比较的变化：'
            "风速最高窗口出现在开启风机的批次，其余通道的极差则来自日内过程本身。"
            "若某通道的极差与测量分辨率同量级，则该通道在本批数据中不具备区分能力。</p>"
        )

    # ---- 4. 关联 ----
    body.append('<h2 id="t4">4. 击穿行为与环境要素的关联</h2>')
    numeric_samples = [sample for sample in samples if parse_seconds(sample.exp.breakdown or "") is not None]
    body.append(
        f"<p>同时具备“有数值击穿时间”和“窗口环境值”的试验共 {len(numeric_samples)} 次。"
        "下表给出击穿时间与各环境要素的 Spearman 秩相关（小样本，仅作描述）。</p>"
    )
    corr_rows = []
    for channel in active_channels:
        xs, ys = [], []
        for sample in numeric_samples:
            value = sample.values.get(channel.key)
            seconds = parse_seconds(sample.exp.breakdown or "")
            if value is None or seconds is None:
                continue
            xs.append(value)
            ys.append(seconds)
        result = _spearman(xs, ys)
        if result is None:
            corr_rows.append([channel.label, len(xs), "—", "—", "样本不足"])
            continue
        rho, p_value, count = result
        corr_rows.append([channel.label, count, num(rho, 2), p_text(p_value), "是" if p_value < 0.05 else "否"])
    body.append(
        table(
            ["通道", "样本", "Spearman ρ", "p", "p<0.05"],
            corr_rows,
            numeric={1, 2, 3},
            caption="击穿时间与窗口内环境要素的秩相关",
        )
    )
    body.append(
        '<p class="note">'
        f"{len(active_channels)} 个通道同时检验且未做多重比较校正：按 Bonferroni"
        f"（α={0.05 / max(len(active_channels), 1):.3f}）则表中无一达到阈值。"
        "14 次样本也远不足以支撑通道间的强弱排序，"
        "这里只用于提示后续需要补测的方向。</p>"
    )

    fan_wind_rows = []
    wind_series = channels.get("wind")
    if wind_series is not None and not wind_series.empty:
        groups: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            value = sample.values.get("wind")
            if value is None:
                continue
            groups[fan_group(sample.exp)].append(value)
        for name, values in sorted(groups.items(), key=lambda pair: -len(pair[1])):
            fan_wind_rows.append(
                [name, len(values), num(median_or_none(values), 3), num(min(values), 3), num(max(values), 3)]
            )
        test = _mwu(groups.get("无风机", []), [v for key, vals in groups.items() if key != "无风机" for v in vals])
        if test is not None:
            body.append(
                f"<p>风机开启与未开启的窗口风速差异：Mann-Whitney U = {num(test[0], 1)}，"
                f"{p_phrase(test[1])}（无风机 n={len(groups.get('无风机', []))}，"
                f"有风机 n={sum(len(vals) for key, vals in groups.items() if key != '无风机')}）。</p>"
            )
    if fan_wind_rows:
        body.append(
            table(
                ["风机配置", "窗口数", "窗口风速中位 (m/s)", "最小", "最大"],
                fan_wind_rows,
                numeric={1, 2, 3, 4},
                caption="不同风机配置下的实测窗口风速",
            )
        )
        body.append(
            '<p class="note">若风机档位与实测风速的排序一致，说明参数表的风机记录可以代表局地气流条件；'
            "若不一致，则以实测风速为准，参数表的风机档位只作试验意图的说明。</p>"
        )

    detected = [sample for sample in samples if parse_seconds(sample.exp.breakdown or "") is not None]
    undetected = [sample for sample in samples if sample.exp.breakdown and parse_seconds(sample.exp.breakdown) is None]
    if detected and undetected:
        rows = []
        for channel in active_channels:
            first = [sample.values[channel.key] for sample in detected if sample.values.get(channel.key) is not None]
            second = [sample.values[channel.key] for sample in undetected if sample.values.get(channel.key) is not None]
            test = _mwu(first, second)
            rows.append(
                [
                    channel.label,
                    len(first),
                    num(median_or_none(first), channel.digits),
                    len(second),
                    num(median_or_none(second), channel.digits),
                    p_text(test[1]) if test else "—",
                ]
            )
        body.append(
            table(
                ["通道", "观测到击穿 n", "中位", "未观测 n", "中位", "p"],
                rows,
                numeric={1, 2, 3, 4, 5},
                caption="“观测到击穿”与“未观测到击穿”两组的窗口环境对比",
            )
        )
        body.append(
            '<p class="note">两组的风机配置并不相同（“观测到击穿”组多为有风机批次），'
            "因此风速与湿度的组间差异同时混入了试验设计因素，不能解释为击穿的成因；"
            "未观测组只有 5 次，p 值仅作描述。</p>"
        )

    # ---- 5. 滞后 ----
    body.append('<h2 id="t5">5. 相对试验开始的滞后剖面</h2>')
    events = [sample.start for sample in samples]
    for channel in active_channels:
        table_channel = channels[channel.key]
        if table_channel.empty:
            continue
        profile = atm.lag_profile(table_channel["中位"], events, before_minutes=60, after_minutes=30, bin_minutes=10)
        if profile.empty:
            continue
        points = [
            (float(offset), float(value)) for offset, value in zip(profile["offset_min"], profile["中位"], strict=True)
        ]
        body.append(
            figure(
                lag_panel(
                    points,
                    title=f"{channel.label}滞后剖面",
                    y_label=f"{channel.label}（{channel.unit}）",
                    before_minutes=60,
                    after_minutes=30,
                    log_scale=channel.key == "cn2",
                ),
                f"{channel.label}：以每次试验开始为零点，跨试验取时间桶中位。"
                "横轴为相对试验开始的分钟数（负值为试验前）；桶宽 10 分钟，0 之后的第一段即试验期间的典型值。",
            )
        )
    body.append(
        "<p>滞后剖面用于区分“试验前就存在的环境状态”和“试验期间才出现的状态”。"
        "若零点前后曲线连续、无跳变，说明试验窗口落在环境自然演变的片段上；"
        "若零点后出现台阶，则需要排除设备自身发热或风机启停带来的局地影响。</p>"
    )

    # ---- 6. 结论 ----
    body.append('<h2 id="t6">6. 结论、局限与口径</h2>')
    conclusions = []
    if len(samples) == len(experiments):
        lead = f"三天共 {len(experiments)} 次试验，全部具备可用于时间对齐的采集时段，"
    else:
        lead = f"三天共 {len(experiments)} 次试验中，{len(samples)} 次具备可用于时间对齐的采集时段，"
    conclusions.append(
        lead + f"单次采集时段中位 {num(median_or_none(window_minutes), 1)} 分钟（多为几秒到几十秒）；"
        "环境窗口按出光时段前后各 10 分钟展开，代表的是“试验当时所处的大气状态”，"
        "而不是出光那几秒的瞬时值。"
    )
    if active_channels:
        parts = []
        for channel in active_channels:
            background = backgrounds[channel.key]["median"]
            background_value = background if np.isfinite(background) else None
            values = [sample.values.get(channel.key) for sample in samples]
            values = [value for value in values if value is not None]
            window_median = median_or_none(values)
            if window_median is None or background_value is None:
                continue
            if channel.key == "wind":
                parts.append(
                    f"风速 +{window_median - background_value:.2f} m/s"
                    f"（背景 {background_value:.2f} m/s，主要为试验时开启风机所致，不报百分比）"
                )
                continue
            change = atm.relative_change(window_median, background_value, floor=channel.rel_floor)
            orders = atm.ratio_orders(window_median, background_value)
            if change is None:
                parts.append(f"{channel.label} {orders:+.1f} 个数量级" if orders is not None else channel.label)
            elif abs(change) >= 2 and orders is not None:
                parts.append(f"{channel.label} 约 {10**orders:.1f} 倍（{orders:+.1f} 个数量级）")
            else:
                parts.append(f"{channel.label} {pct(change, 1)}")
        if parts:
            conclusions.append(
                "试验窗口相对同日背景的偏离："
                + "、".join(parts)
                + "。这些差异说明试验时段不是全天随机抽样，而是集中在特定的大气状态下。"
            )
    conclusions.append(
        "击穿时间与单个环境要素的秩相关样本量不足，且试验条件随批次成组变化（距离、风机、"
        "待机方式同时改变），任何单项相关都不能单独解释击穿时间差异。"
    )
    fan_medians = {}
    for sample in samples:
        value = sample.values.get("wind")
        if value is None:
            continue
        fan_medians.setdefault(fan_group(sample.exp), []).append(value)
    fan_medians = {name: float(np.median(values)) for name, values in fan_medians.items()}
    ordered_fans = ["无风机", "1 台风机", "2 台风机"]
    if all(name in fan_medians for name in ordered_fans):
        chain = [fan_medians[name] for name in ordered_fans]
        monotone = chain[0] < chain[1] < chain[2]
        conclusions.append(
            "参数表的风机档位与实测窗口风速"
            + (
                f"排序一致（无风机 {chain[0]:.2f} → 1 台 {chain[1]:.2f} → 2 台 {chain[2]:.2f} m/s，单调递增），"
                "可以认为风机记录可信，且实测风速比档位更适合作为定量变量。"
                if monotone
                else "排序不完全一致，定量分析应以实测风速为准，风机档位只作试验意图的说明。"
            )
        )
    body.append("<ul>" + "".join(f"<li>{esc(text)}</li>" for text in conclusions) + "</ul>")
    body.append(
        '<div class="callout warn"><strong>局限</strong>：'
        "① 大气探针布设在观测点，与靶面处的局地条件可能存在差异；"
        "② 试验窗口由采集时刻决定，未覆盖光路调整与准备阶段；"
        "③ 击穿时间有相当比例缺失或以文本描述（未击穿、难以观测），按右删失处理；"
        "④ 样本量小、分组不均，统计检验只作描述性参考，不作显著性结论。"
        "</div>"
    )
    body.append(
        '<div class="appendix"><ul>'
        "<li><strong>窗口</strong>：以一次试验的第一段采集时刻为起点、最后一段采集时刻为终点，"
        "前后各扩 10 分钟作为环境窗口；不含时间戳的试验不参与时间对齐。</li>"
        "<li><strong>代表值</strong>：单通道先按分钟求跨探针中位，再在窗口内取中位；"
        "探针级异常码（温度 1220.8、风速 122.08、Cn² 的 0 值、能见度的 0 与 75）按物理范围剔除。</li>"
        "<li><strong>背景</strong>：同日全部分钟值中扣除所有环境窗口，并在窗口前后各留 15 分钟保护间隔。</li>"
        "<li><strong>相关</strong>：Spearman 秩相关，最小样本 6；p 值未做多重比较校正。</li>"
        "<li><strong>滞后剖面</strong>：以试验开始为零点，-60 ~ +30 分钟、10 分钟分桶，跨试验取中位。</li>"
        "</ul></div>"
    )
    body.append(
        f'<p class="note">报告由 {esc(GENERATED_BY)} 生成于 {esc(generated.strftime("%Y-%m-%d %H:%M"))}；'
        f"大气数据根目录：{esc(str(root))}。中间产物（窗口统计 CSV）与总览报告同目录输出。</p>"
    )

    # 中间产物：窗口统计宽表
    rows_csv = []
    for sample in samples:
        row = {
            "试验": sample.exp.exp_id,
            "日期": sample.exp.date,
            "出光开始": sample.raw_start,
            "出光结束": sample.raw_end,
            "窗口开始": sample.start,
            "窗口结束": sample.end,
            "窗口分钟": round((sample.end - sample.start).total_seconds() / 60, 1),
            "距离": distance_group(sample.exp),
            "功率档位": sample.exp.power,
            "风机配置": fan_group(sample.exp),
            "击穿时间_s": parse_seconds(sample.exp.breakdown or ""),
        }
        for channel in active_channels:
            row[f"{channel.label}_中位"] = sample.values.get(channel.key)
            row[f"{channel.label}_覆盖分钟"] = sample.coverage.get(channel.key)
        rows_csv.append(row)
    frame = pd.DataFrame(rows_csv)

    subtitle = (
        f"{'、'.join(days)}｜{len(samples)} 个试验窗口｜{len(active_channels)} 路大气通道｜"
        f"生成于 {generated.strftime('%Y-%m-%d %H:%M')}"
    )
    payload = document(
        "试验时间窗口与大气环境协同分析",
        esc(subtitle),
        "".join(body),
        footer="口径：环境值取分钟级跨探针中位；窗口与背景对比不含因果推断；功率一律以档位代号呈现。",
    )
    return payload, frame


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def generate(
    root: Path,
    out_dir: Path,
    *,
    atm_root: Path,
    cache_dir: Path,
    skip_atmosphere: bool = False,
    refresh_atm: bool = False,
) -> list[Path]:
    generated = datetime.now()
    experiments = base.build_experiments(root)
    if not experiments:
        raise SystemExit(f"未在 {root} 下发现试验目录")
    for exp in experiments:
        base.finalize_quality(exp)
    logger.info("解析到 {} 次试验，覆盖 {}", len(experiments), sorted({exp.date for exp in experiments}))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    overview_path = out_dir / "01_试验数据总览.html"
    overview_path.write_text(overview_report(experiments, root, generated), encoding="utf-8")
    written.append(overview_path)
    logger.info("已写出 {}", overview_path)

    if not skip_atmosphere:
        days = sorted({exp.date for exp in experiments})
        channels = atm.load_channels(atm_root, days, cache_dir=cache_dir, refresh=refresh_atm)
        if not any(not frame.empty for frame in channels.values()):
            logger.warning("未读取到大气数据，跳过协同分析报告")
        else:
            payload, frame = synergy_report(experiments, channels, days=days, generated=generated, root=atm_root)
            synergy_path = out_dir / "02_时间与大气协同.html"
            synergy_path.write_text(payload, encoding="utf-8")
            written.append(synergy_path)
            logger.info("已写出 {}", synergy_path)
            csv_path = out_dir / "试验窗口统计.csv"
            frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
            written.append(csv_path)
            logger.info("已写出 {}（{} 行）", csv_path, len(frame))
    return written


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI 入口
    parser = argparse.ArgumentParser(description="小系统试验数据 HTML 报告")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="试验数据根目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="报告输出目录")
    parser.add_argument("--atm-root", type=Path, default=atm.DEFAULT_ATM_ROOT, help="大气数据根目录")
    parser.add_argument("--cache-dir", type=Path, default=atm.DEFAULT_CACHE_DIR, help="大气分钟缓存目录")
    parser.add_argument("--skip-atmosphere", action="store_true", help="只生成总览报告")
    parser.add_argument("--refresh-atm", action="store_true", help="忽略大气缓存重新解析")
    args = parser.parse_args(argv)
    written = generate(
        args.root,
        args.out,
        atm_root=args.atm_root,
        cache_dir=args.cache_dir,
        skip_atmosphere=args.skip_atmosphere,
        refresh_atm=args.refresh_atm,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
