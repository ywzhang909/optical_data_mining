"""报告生成：自包含 HTML、Markdown 摘要、指标 JSON、表格 CSV 与运行清单。

设计约束：

* **自包含** —— 图片以 base64 内嵌，报告目录可整体拷贝/归档，不依赖相对路径；
* **结论优先** —— 正文只写数字与判断，不堆砌产物文件名；产物清单单独放在附录；
* **可核查** —— 每个结论都能回指到随附统计表中的对应行；
* **可复现** —— 报告内嵌运行配置、软件版本与输入规模。
"""

from __future__ import annotations

import base64
import html
import json
import math
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .spatial import spatial_summary

if TYPE_CHECKING:  # 仅用于类型标注，避免与 pipeline 形成循环导入
    from .pipeline import AnalysisResult

REPORT_SCHEMA_VERSION = 1

# 谱斜率与结构函数斜率的 Kolmogorov 理论值
KOLMOGOROV_PSD_SLOPE = -5.0 / 3.0
KOLMOGOROV_STRUCTURE_SLOPE = 2.0 / 3.0

_CN2_COLUMNS = ("Cn2", "Cn2中位数", "设备Cn2", "反演Cn2", "Cn2比值")


# --------------------------------------------------------------------------- #
# 数值格式化
# --------------------------------------------------------------------------- #
def _fmt(value: Any, spec: str | int | None = None) -> str:
    """把单元格值格式化为报告中的字符串。

    Args:
        value: 原始值（可能是 NaN、numpy 标量、Timestamp）。
        spec: 格式约定 —— ``"pct"`` 百分比、``"int"`` 整数、``"sci"`` 科学计数、
            ``"ts"`` 时间戳，或整数表示小数位数；``None`` 表示按量级自适应。
    """
    if value is None:
        return "—"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if isinstance(value, str):
        return value if value else "—"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "—"

    if spec == "pct":
        return f"{number:.1%}"
    if spec == "pct1":
        return f"{number:.2%}"
    if spec == "int":
        return f"{number:,.0f}"
    if spec == "sci":
        return f"{number:.2e}"
    if isinstance(spec, int):
        return f"{number:,.{spec}f}"

    magnitude = abs(number)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-3 or magnitude >= 1e6:
        return f"{number:.2e}"
    if magnitude >= 1000:
        return f"{number:,.0f}"
    if magnitude >= 100:
        return f"{number:.2f}"
    if magnitude >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"


def _auto_formats(frame: pd.DataFrame) -> dict[str, str | int]:
    """按列名/取值推断合适的显示格式。"""
    formats: dict[str, str | int] = {}
    for column in frame.columns:
        name = str(column)
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if any(token in name for token in ("有效率", "覆盖率", "比例")):
            formats[name] = "pct"
        elif any(token in name for token in ("(s)", "(Hz)", "间隔", "时长")):
            pass  # 秒/赫兹等物理量，交给自动格式
        elif any(
            token in name
            for token in ("采样数", "采样", "分钟", "文件数", "分块", "样本", "小时", "次数", "数量", "行数")
        ):
            formats[name] = "int"
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            magnitude = float(np.max(np.abs(values))) if values.size else 0.0
            # 全零列是计数/缺失占位，不是物理小量，不能套用科学计数法
            if 0.0 < magnitude < 1e-6:
                formats[name] = "sci"
    return formats


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #
@dataclass
class Table:
    """一张待输出到 CSV/HTML/Markdown 的表格。"""

    key: str
    title: str
    data: pd.DataFrame
    note: str = ""
    formats: dict[str, str | int] = field(default_factory=dict)
    max_rows: int = 200

    def resolved_formats(self) -> dict[str, str | int]:
        merged = _auto_formats(self.data)
        merged.update(self.formats)
        return merged

    def to_html(self) -> str:
        if self.data.empty:
            return '<p class="empty">该表无数据（对应通道缺失或全部无效）。</p>'
        formats = self.resolved_formats()
        shown = self.data.head(self.max_rows)
        head = "".join(f"<th>{html.escape(str(col))}</th>" for col in shown.columns)
        body: list[str] = []
        for _, row in shown.iterrows():
            cells = "".join(f"<td>{_fmt(row[col], formats.get(str(col)))}</td>" for col in shown.columns)
            body.append(f"<tr>{cells}</tr>")
        tail = ""
        if len(self.data) > self.max_rows:
            tail = f'<p class="empty">仅显示前 {self.max_rows} 行，共 {len(self.data)} 行，完整数据见 CSV。</p>'
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{tail}"

    def to_markdown(self) -> str:
        if self.data.empty:
            return "_该表无数据。_"
        formats = self.resolved_formats()
        shown = self.data.head(self.max_rows)
        header = "| " + " | ".join(str(col) for col in shown.columns) + " |"
        divider = "| " + " | ".join("---" for _ in shown.columns) + " |"
        lines = [header, divider]
        for _, row in shown.iterrows():
            cells = [_fmt(row[col], formats.get(str(col))) for col in shown.columns]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _tables(result: AnalysisResult) -> list[Table]:
    """汇总报告中所有表格（顺序即报告顺序）。"""
    tables: list[Table] = []

    inventory = pd.DataFrame(result.inventory.summary())
    tables.append(Table("inventory", "数据通道清单", inventory, "按通道目录统计的文件数与数据量。", {"总大小 (MB)": 1}))

    quality = result.quality_overall.copy()
    if not quality.empty:
        quality = quality[
            [
                "探针",
                "总采样数",
                "有效采样",
                "有效率",
                "异常码采样",
                "超范围采样",
                "卡值采样",
                "最长卡值 (min)",
                "尖峰采样",
                "离群采样",
                "抖动采样",
                "量化步长 (C)",
                "最小值 (C)",
                "最大值 (C)",
            ]
        ]
    tables.append(
        Table(
            "quality_overall",
            "全时段逐探针数据质量",
            quality,
            "异常码采样为采集软件写入的固定无效值（如 1220.8）；卡值仅统计读数连续 10 min 不变的情形"
            "（0.1 °C 量化下，夜间数分钟读数不变属正常）。卡值与尖峰仅统计、不剔除；"
            "离群采样为与同刻跨探针中值相差超过 5 °C 的读数（掉线/错误值），已从统计中剔除；"
            "抖动采样为偏离局部中值超过阈值的跳变，同样已剔除。",
            {"量化步长 (C)": 2, "最小值 (C)": 2, "最大值 (C)": 2, "最长卡值 (min)": 2},
            max_rows=40,
        )
    )

    tables.append(
        Table(
            "cadence",
            "文件级采样节奏与完整性",
            result.cadence,
            "覆盖率 = 实际采样数 / 理论采样数（按中位采样间隔推算）；备注列记录疑似截断等异常。",
            max_rows=60,
        )
    )

    overall = result.overall_summary.copy()
    if not overall.empty:
        overall = overall[
            [
                "探针",
                "有效分钟",
                "平均温度",
                "温度标准差",
                "最低温度",
                "最高温度",
                "P01",
                "P99",
                "平均覆盖率",
            ]
        ]
    tables.append(
        Table(
            "temperature_overall",
            "全时段温度统计",
            overall,
            "统计基于分钟均值序列，温度标准差反映的是逐分钟均值的日际离散度，不是高频脉动。",
            {"平均温度": 3, "温度标准差": 3, "最低温度": 2, "最高温度": 2, "P01": 2, "P99": 2},
            max_rows=40,
        )
    )

    daily = result.daily_summary.copy()
    if not daily.empty:
        daily = daily[
            [
                "日期",
                "探针",
                "有效分钟",
                "平均温度",
                "日内最低",
                "日内最高",
                "日内温差",
                "分钟覆盖率",
            ]
        ]
    tables.append(
        Table(
            "temperature_daily",
            "逐日逐探针温度统计",
            daily,
            "日内温差 = 当日分钟均值最大值 − 最小值，用于刻画日变化幅度。",
            {"平均温度": 3, "日内最低": 2, "日内最高": 2, "日内温差": 2},
            max_rows=200,
        )
    )

    diurnal = (
        result.diurnal.pivot_table(index="小时", columns="探针", values="平均温度", observed=True).reset_index()
        if not result.diurnal.empty
        else pd.DataFrame()
    )
    tables.append(
        Table(
            "diurnal",
            "平均日变化（小时 × 探针，°C）",
            diurnal,
            "对全时段同一小时取平均，可看出站点温度的日内节律。",
            max_rows=30,
        )
    )

    tables.append(
        Table(
            "correlation",
            "探针间去趋势相关系数矩阵",
            result.correlation.reset_index().rename(columns={"index": "探针"}),
            "先按 1 h 滑动中值去趋势，再计算分钟均值的相关，避免共同日变化淹没脉动信号。",
            {str(col): 3 for col in result.correlation.columns},
            max_rows=40,
        )
    )

    tables.append(
        Table(
            "spectral",
            "脉动功率谱斜率（分块中位数）",
            result.spectral_summary,
            f"Kolmogorov 惯性子区理论斜率约 {KOLMOGOROV_PSD_SLOPE:.3f}；偏离说明频段内存在噪声或非湍流过程。",
            {"谱斜率中位数": 3, "谱斜率P10": 3, "谱斜率P90": 3, "谱拟合R²中位数": 3},
            max_rows=40,
        )
    )

    turbulence = result.turbulence_summary.copy()
    if not turbulence.empty:
        turbulence = (
            turbulence.rename(columns={"日期": "日期"})
            .groupby("探针", observed=True)
            .agg(
                有效分块=("有效分块", "sum"),
                标准差中位数=("标准差中位数", "median"),
                谱斜率中位数=("谱斜率中位数", "median"),
                结构函数斜率中位数=("结构函数斜率中位数", "median"),
                C_T2中位数=("C_T2中位数", "median"),
                Cn2中位数=("Cn2中位数", "median"),
                波动信噪比中位数=("波动信噪比中位数", "median"),
            )
            .reset_index()
        )
    tables.append(
        Table(
            "turbulence",
            "湍流参数（全时段逐探针中位数）",
            turbulence,
            "C_T² 由结构函数按 Taylor 冻结假设反演，Cn² 由 Gladstone 关系换算；中位数抗短时离群。",
            {"标准差中位数": 4, "谱斜率中位数": 3, "结构函数斜率中位数": 3, "波动信噪比中位数": 2},
            max_rows=40,
        )
    )

    cn2 = result.cn2_summary.copy()
    tables.append(
        Table(
            "cn2_agreement",
            "设备输出 Cn² 与反演 Cn² 的一致性",
            cn2,
            "设备 Cn² 来自 Cn2/ 通道（约 10 s 采样），反演 Cn² 来自温度脉动；两者独立，一致性用于交叉验证。",
            max_rows=40,
        )
    )

    spread = result.spread.rename(columns={"ts": "时间"}).copy()
    tables.append(
        Table(
            "spread", "探针间瞬时离散度（分钟）", spread, "同一时刻跨探针的标准差，反映仪器间随机不一致。", max_rows=30
        )
    )

    channel_frames = [channel.summary for channel in (result.channels or {}).values() if not channel.summary.empty]
    if channel_frames:
        merged = pd.concat(channel_frames, ignore_index=True)
        tables.append(
            Table(
                "channels",
                "环境要素逐探针统计（分钟均值）",
                merged,
                "各通道按物理范围剔除异常码后做分钟均值；覆盖率的分母是分析时段的总分钟数。",
                {"均值": 3, "标准差": 3, "中位": 3, "P05": 3, "P95": 3},
                max_rows=80,
            )
        )

    if result.channel_correlation is not None and not result.channel_correlation.empty:
        tables.append(
            Table(
                "channel_correlation",
                "多通道逐分钟相关系数",
                result.channel_correlation,
                "交叉协同：每分钟先对探针取中位数，再按时间对齐计算 Pearson 相关系数；"
                "未去趋势，数值含昼夜节律，解读见正文“多数据协同”。",
                max_rows=20,
            )
        )

    if result.spatial is not None and not result.spatial.empty:
        tables.append(
            Table(
                "spatial_summary",
                "跨探针空间平均概况",
                spatial_summary(result.spatial),
                "位置插值平均：按测点高度线性插值到等间隔网格后取均值；"
                "缺少高度映射时退化为跨探针等权聚合，具体见“方式”列。高度是否为实测见正文说明。",
                {"均值": 3, "最低": 3, "最高": 3, "平均探针数": 1},
                max_rows=40,
            )
        )
        if result.spatial_daily is not None and not result.spatial_daily.empty:
            tables.append(
                Table(
                    "spatial_daily",
                    "逐日跨探针空间平均",
                    result.spatial_daily,
                    "对逐分钟空间平均序列按日再统计，便于与站点级观测或模式结果对比。",
                    {"日均值": 3, "最低": 3, "最高": 3, "平均探针数": 1},
                    max_rows=200,
                )
            )
    return tables


# --------------------------------------------------------------------------- #
# 结论摘要
# --------------------------------------------------------------------------- #
def _paragraphs(result: AnalysisResult) -> list[tuple[str, str]]:
    """自动生成的中文结论段落：(标题, 正文 HTML)。"""
    facts = result.facts
    blocks: list[tuple[str, str]] = []

    coverage = (
        f"本次分析覆盖 {facts.get('天数', 0)} 天（{facts.get('起止时间', '—')}），"
        f"共 {facts.get('文件数', 0)} 个温度原始文件、{facts.get('原始数据量MB', 0)} MB，"
        f"解析出 {facts.get('探针数', 0)} 个探针，其中 {facts.get('有效探针数', 0)} 个通过有效率门槛。"
        f"标称采样率约 {_fmt(facts.get('采样率'), 2)} Hz（采样间隔 {_fmt(facts.get('采样间隔'), 4)} s）。"
        f"本次 {facts.get('缓存文件数', 0)} 个文件命中解析缓存、{facts.get('解析文件数', 0)} 个重新解析，"
        f"分析用时 {_fmt(result.runtime_s, 1)} s。"
    )
    blocks.append(("数据覆盖", coverage))

    diurnal_fact = facts.get("日变化幅度")
    temperature = (
        f"全时段平均温度 {_fmt(facts.get('平均温度'), 2)} °C，"
        f"分钟均值范围 {_fmt(_range_low(facts), 2)} – {_fmt(_range_high(facts), 2)} °C。"
    )
    if isinstance(diurnal_fact, (int, float)) and math.isfinite(float(diurnal_fact)):
        temperature += (
            f"平均日变化幅度约 {_fmt(diurnal_fact, 2)} °C，"
            f"最热时次 {_hour(facts.get('最热时次'))}、最冷时次 {_hour(facts.get('最冷时次'))}。"
        )
    else:
        temperature += (
            f"本次时段仅 {_fmt(facts.get('时长小时'), 1)} h，"
            f"不足 {_fmt(facts.get('日变化门槛小时'), 0)} h，未给出日内变化节律结论。"
        )
    temperature += f"探针间瞬时离散度中位数为 {_fmt(facts.get('探针间极差中位数'), 3)} °C。"
    blocks.append(("温度水平与日变化", temperature))

    sigma = facts.get("脉动标准差中位数")
    snr = facts.get("波动信噪比")
    noise = facts.get("量化噪声")
    noise_text = (
        f"原始数据量化为 {_fmt(facts.get('量化步长'), 2)} °C，对应量化噪声 σ_q≈{_fmt(noise, 3)} °C，"
        f"因此波动标准差 {_fmt(sigma, 3)} °C 的信噪比约 {_fmt(snr, 1)}"
    )
    if isinstance(snr, (int, float)) and math.isfinite(float(snr)) and float(snr) < 3:
        noise_text += "，脉动信号已接近量化噪声下限，湍流参数只能作量级参考。"
    else:
        noise_text += "，脉动信号高于量化噪声，可用于湍流参数估计。"
    blocks.append(("脉动强度与噪声水平", noise_text))

    psd = facts.get("谱斜率中位数")
    sfs = facts.get("结构函数斜率中位数")
    spectral = (
        f"分块功率谱斜率中位数 {_fmt(psd, 3)}（Kolmogorov 理论值 {KOLMOGOROV_PSD_SLOPE:.3f}，"
        f"拟合频段 {facts.get('谱拟合频段', '—')}，拟合 R² 中位数 {_fmt(facts.get('谱拟合优度'), 2)}），"
        f"结构函数幂律指数中位数 {_fmt(sfs, 3)}（理论值 {KOLMOGOROV_STRUCTURE_SLOPE:.3f}）。"
        f"两者同时接近理论值时，说明所选频段/滞后区间落在惯性子区内；"
        f"结构函数按每块的实测噪声底（τ→0 截距，取 2σ_q² 与实测量的较小者）扣除量化偏置，"
        f"校正后 ≤0 的滞后视为被噪声主导，不参与拟合。"
    )
    blocks.append(("谱特征", spectral))

    environment = _environment_paragraph(result)
    if environment:
        blocks.append(("环境要素与湍流", environment))

    if not result.spatial.empty:
        overview = spatial_summary(result.spatial)
        parts: list[str] = []
        for _, row in overview.iterrows():
            unit = str(row.get("单位", "")).strip()
            parts.append(f"{row['通道']} {_fmt(row.get('均值'))}" + (f" {unit}" if unit else ""))
        method = str(facts.get("空间平均方式", "—"))
        source = str(result.spatial_heights_source or "").strip()
        text = f"跨探针空间平均方式为{method}，覆盖 {facts.get('空间平均通道数', 0)} 个通道。"
        if parts:
            text += "全时段均值：" + "、".join(parts) + "。"
        if result.spatial_heights:
            if result.cfg.probes.heights_even_enabled() and not result.cfg.probes.heights_m:
                text += (
                    f"本次没有实测安装高度，位置插值所用高度按{source}生成，属假设而非现场测量；"
                    "空间平均与垂直梯度的物理解读应据此保守处理。"
                )
            else:
                text += f"位置插值所用的探针高度依据：{source or '显式配置'}。"
        else:
            text += "本次没有可用的探针安装高度，空间平均为跨探针等权聚合，不是按高度插值的剖面平均。"
        blocks.append(("跨探针空间平均", text))

    synergy = _synergy_paragraph(result)
    if synergy:
        blocks.append(("多数据协同", synergy))

    if not result.cn2_summary.empty:
        ratio = _median_column(result.cn2_summary, "Cn2比值")
        rho = _median_column(result.cn2_summary, "对数相关")
        cn2_text = (
            f"设备输出 Cn² 与温度反演 Cn² 的比值中位数为 {_fmt(ratio, 2)}，"
            f"二者对数相关 {_fmt(rho, 3)}。"
            f"反演使用假设站点气压 {_fmt(result.cfg.analysis.site_pressure_hpa, 2)} hPa，"
            f"应按实际站点气压修正后重算。"
        )
        blocks.append(("Cn² 交叉验证", cn2_text))
    return blocks


def _describe_pair(value: float) -> str:
    magnitude = abs(value)
    direction = "同向起伏" if value > 0 else "反相起伏"
    if magnitude >= 0.7:
        return f"强相关、{direction}"
    if magnitude >= 0.4:
        return f"中等相关、{direction}"
    return f"弱相关、{direction}"


def _synergy_paragraph(result: AnalysisResult) -> str:
    """多通道协同：分钟序列两两相关，点出同步起伏最强的通道对。"""
    matrix = getattr(result, "channel_correlation", None)
    if matrix is None or getattr(matrix, "empty", True) or "通道" not in matrix:
        return ""
    labels = [str(label) for label in matrix["通道"].tolist()]
    pairs: list[tuple[float, str, str]] = []
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            try:
                value = float(pd.to_numeric(matrix[right], errors="coerce").iloc[index])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                pairs.append((value, left, right))
    if not pairs:
        return ""
    pairs.sort(key=lambda item: abs(item[0]), reverse=True)
    ranked = "、".join(f"{left}与{right} r={value:+.2f}（{_describe_pair(value)}）" for value, left, right in pairs[:3])
    text = (
        f"分钟尺度上共比对 {len(labels)} 个通道，起伏最同步的通道对为：{ranked}。"
        "这些系数建立在未去趋势的分钟序列上，昼夜节律会让同受太阳辐射驱动的量"
        "（温度、湿度、Cn²）同时涨落并呈现高相关，符号主要由相位决定；"
        "因此它说明的是各要素“是否同步”，不能直接当作湍流尺度上的耦合或因果关系。"
    )
    if "温度" in labels and any("Cn²" in label for label in labels):
        text += "温度反演 Cn² 与设备 Cn² 的独立交叉验证另见下一节。"
    return text


def _spatial_height_note(result: AnalysisResult) -> str:
    """空间平均所依据的探针高度来历（实测 / 等间距假设 / 未配置）。"""
    source = str(result.spatial_heights_source or "").strip()
    if not result.spatial_heights:
        return "本次没有可用的探针安装高度，空间平均为跨探针等权聚合，不是按高度插值。"
    if result.cfg.probes.heights_even_enabled() and not result.cfg.probes.heights_m:
        return f"高度按{source}生成，属假设值（非现场测量）；剖面类结论需按此前提解读。"
    return f"空间平均使用的探针高度依据：{source or '显式配置（实测或手工整理）'}。"


def _environment_paragraph(result: AnalysisResult) -> str:
    """环境要素与湍流参数的一句话总结。"""
    facts = result.facts
    parts: list[str] = []
    if facts.get("风速均值") is not None:
        probe = facts.get("风速探针", "—")
        parts.append(
            f"站点风速均值 {_fmt(facts.get('风速均值'), 2)} m/s"
            f"（中位 {_fmt(facts.get('风速中位数'), 2)}，P95 {_fmt(facts.get('风速P95'), 2)}，取自 {probe}）"
        )
    for label in ("相对湿度", "能见度", "设备 Cn²"):
        value = facts.get(f"{label}中位数")
        if value is None:
            continue
        unit = facts.get(f"{label}单位", "")
        parts.append(f"{label}中位 {_fmt(value, 3)} {unit}".strip())
    ct2 = facts.get("C_T2中位数")
    cn2 = facts.get("Cn2中位数")
    if ct2 is not None:
        parts.append(f"温度结构常数 C_T² 中位 {_fmt(ct2, 3)} K²·m^(-2/3)")
    if cn2 is not None:
        parts.append(f"对应 Cn² 中位 {_fmt(cn2, 3)} m^(-2/3)")
    if not parts:
        return ""
    text = "；".join(parts) + "。"
    if ct2 is None:
        text += "风速缺失，未按 Taylor 冻结假设反演湍流参数。"
    else:
        text += "湍流参数只对纳入统计的分块取中位数，短时离群不会主导结论。"
    return text


def _range_low(facts: dict[str, Any]) -> Any:
    value = facts.get("温度范围")
    return value[0] if isinstance(value, (tuple, list)) and value else None


def _range_high(facts: dict[str, Any]) -> Any:
    value = facts.get("温度范围")
    return value[1] if isinstance(value, (tuple, list)) and len(value) > 1 else None


def _hour(value: Any) -> str:
    if not isinstance(value, (int, float, np.integer)) or value is None or int(value) < 0:
        return "—"
    return f"{int(value):02d}:00"


def _median_column(frame: pd.DataFrame, name: str) -> float:
    if name not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[name], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 40px 28px 72px; max-width: 1180px;
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", system-ui, sans-serif;
  font-size: 15px; line-height: 1.75; color: #1f2328; background: #ffffff;
}
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: .5px; }
h2 { font-size: 21px; margin: 44px 0 14px; padding-bottom: 8px; border-bottom: 2px solid #e6e8eb; }
h3 { font-size: 16px; margin: 26px 0 10px; color: #30363d; }
.sub { color: #6a737d; font-size: 13px; margin-bottom: 26px; }
.toc { background: #f6f8fa; border: 1px solid #e6e8eb; border-radius: 8px; padding: 14px 20px; }
.toc ol { margin: 0; padding-left: 20px; columns: 2; column-gap: 32px; }
.toc a { color: #0969da; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }
.card { border: 1px solid #e6e8eb; border-radius: 8px; padding: 12px 14px; background: #fcfcfd; }
.card .label { font-size: 12px; color: #6a737d; }
.card .value { font-size: 20px; font-weight: 600; margin-top: 2px; word-break: break-all; }
.callout { border-left: 4px solid #d4a72c; background: #fffbeb; padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 14px 0; }
.callout.info { border-color: #0969da; background: #f1f7ff; }
.callout ul { margin: 6px 0; padding-left: 22px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 6px; display: block; overflow-x: auto; }
thead th { background: #f2f4f7; text-align: left; font-weight: 600; white-space: nowrap; }
th, td { border: 1px solid #e6e8eb; padding: 6px 9px; white-space: nowrap; }
tbody tr:nth-child(even) { background: #fafbfc; }
figure { margin: 18px 0 26px; }
figure img { width: 100%; height: auto; border: 1px solid #e6e8eb; border-radius: 8px; }
figcaption { font-size: 12.5px; color: #6a737d; margin-top: 8px; }
.note { font-size: 13px; color: #57606a; margin: 4px 0 18px; }
code { background: #f2f4f7; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
pre { background: #f6f8fa; border: 1px solid #e6e8eb; border-radius: 8px; padding: 12px 16px; overflow-x: auto; font-size: 12.5px; }
.empty { color: #8b949e; font-size: 13px; }
footer { margin-top: 56px; padding-top: 16px; border-top: 1px solid #e6e8eb; font-size: 12px; color: #8b949e; }
@media print { body { max-width: none; padding: 0; } h2 { page-break-after: avoid; } figure { page-break-inside: avoid; } }
"""


def _img(figures: dict[str, Path], key: str, caption: str) -> str:
    """把 PNG 以 base64 内嵌为 <figure>；缺失时返回空串。"""
    path = figures.get(key)
    if path is None or not Path(path).exists():
        return ""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (
        f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _section(anchor: str, title: str, body: str) -> str:
    return f'<section id="{anchor}"><h2>{html.escape(title)}</h2>{body}</section>'


def _table_block(table: Table | None) -> str:
    if table is None:
        return ""
    note = f'<p class="note">{html.escape(table.note)}</p>' if table.note else ""
    return f"<h3>{html.escape(table.title)}</h3>{table.to_html()}{note}"


def _fact_cards(result: AnalysisResult) -> str:
    facts = result.facts
    cards = [
        ("分析天数", f"{facts.get('天数', 0)} 天"),
        ("探针（有效/总数）", f"{facts.get('有效探针数', 0)} / {facts.get('探针数', 0)}"),
        ("平均温度", f"{_fmt(facts.get('平均温度'), 2)} °C"),
        ("平均日变化幅度", f"{_fmt(facts.get('日变化幅度'), 2)} °C"),
        ("采样率", f"{_fmt(facts.get('采样率'), 2)} Hz"),
        ("脉动标准差中位数", f"{_fmt(facts.get('脉动标准差中位数'), 3)} °C"),
        ("波动信噪比", _fmt(facts.get("波动信噪比"), 1)),
        ("谱斜率中位数", _fmt(facts.get("谱斜率中位数"), 3)),
    ]
    cells = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    return f'<div class="cards">{cells}</div>'


def _find(tables: list[Table], key: str) -> Table | None:
    return next((table for table in tables if table.key == key), None)


def _html(result: AnalysisResult, tables: list[Table], figures: dict[str, Path]) -> str:
    cfg = result.cfg
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = cfg.report.title

    summary_body = _fact_cards(result)
    summary_body += "".join(f"<h3>{html.escape(heading)}</h3><p>{body}</p>" for heading, body in _paragraphs(result))
    if result.notes:
        items = "".join(f"<li>{html.escape(note)}</li>" for note in result.notes)
        summary_body += f'<div class="callout"><strong>数据提示</strong><ul>{items}</ul></div>'

    sections: list[tuple[str, str, str]] = [
        ("summary", "结论摘要", summary_body),
        (
            "inventory",
            "数据来源与完整性",
            _table_block(_find(tables, "inventory"))
            + _table_block(_find(tables, "cadence"))
            + '<div class="callout info"><strong>读取说明</strong>'
            "<ul><li>原始 CSV 为长表（<code>ID, 数值, Timestamp</code>），按时间透视成探针宽表后统一聚合。</li>"
            "<li>文件大小恰为 10 MiB 或 100 MiB 时判定为写入截断，已在备注列标注。</li>"
            "<li>覆盖率为实际采样数与理论采样数之比，可直接指示断记与丢帧。</li></ul></div>",
        ),
        (
            "quality",
            "数据质量",
            _table_block(_find(tables, "quality_overall"))
            + _img(figures, "coverage", "图 1　逐日逐探针有效采样率")
            + '<div class="callout"><strong>判据</strong>'
            "<ul><li>超出物理范围（默认 −80 ~ 80 °C）的取值判为无效；采集软件写入的异常码"
            "（1220.8 / 122.08 等）必然越界，同样落入无效。</li>"
            "<li>卡值：连续相同读数超过阈值；尖峰：偏离滑动中值超过阈值。二者仅统计，不剔除，"
            "以免削弱真实高频脉动。</li>"
            "<li>离群：与同一时刻跨探针中值相差超过 5 °C 的读数（如传感器掉线返回 0.0 °C）。"
            "这类值在单探针序列内是一段“平稳值”，滑动判据抓不到，只能靠同址探针横向比对；"
            "它们不是真实测量，已从统计中剔除。</li>"
            "<li>原始数据量化为 0.1 °C 量级，量化噪声 σ_q = Δ/√12 ≈ 0.029 °C，"
            "是高频脉动统计的固有噪声底。</li></ul></div>",
        ),
        (
            "temperature",
            "温度统计",
            _table_block(_find(tables, "temperature_overall"))
            + _img(figures, "timeseries", "图 2　温度分钟均值时间序列")
            + _img(figures, "distribution", "图 3　各探针温度分布")
            + _table_block(_find(tables, "temperature_daily")),
        ),
        (
            "diurnal",
            "日变化",
            _img(figures, "diurnal", "图 4　平均日变化（按小时）")
            + _img(figures, "daily", "图 5　逐日平均温度")
            + _table_block(_find(tables, "diurnal")),
        ),
        (
            "consistency",
            "探针一致性",
            _img(figures, "correlation", "图 6　去趋势后的探针间相关系数")
            + _img(figures, "spread", "图 7　探针间瞬时离散度")
            + _table_block(_find(tables, "correlation")),
        ),
        (
            "environment",
            "环境要素",
            _img(figures, "environment", "图 8　湿度、能见度、设备 Cn² 与风速")
            + _table_block(_find(tables, "channels"))
            + _table_block(_find(tables, "channel_correlation"))
            + (
                '<div class="callout info"><strong>说明</strong><ul>'
                "<li>各通道按物理范围剔除异常码后做分钟均值：湿度 0–100 %、风速 0–60 m/s、能见度 0–100 km、"
                "Cn² 1e-22–1e-10；温度另做抖动剔除与跨探针离群剔除。</li>"
                "<li>相关系数用于判断通道间是否同步起伏，未去趋势，含昼夜节律；"
                "不代表因果，也不等于湍流尺度上的耦合。</li>"
                "<li>报告与导出中的“平滑值”是滑窗均值，只用于看趋势；谱、结构函数与 C_T² 一律用未平滑序列，"
                "否则会抹掉惯性子区。</li></ul></div>"
                if _find(tables, "channels") is not None
                else '<p class="note">本次运行未读取环境要素通道（使用了 --no-aux 或数据中没有对应目录）。</p>'
            ),
        ),
        (
            "spatial",
            "跨探针空间平均",
            _table_block(_find(tables, "spatial_summary"))
            + _table_block(_find(tables, "spatial_daily"))
            + '<div class="callout info"><strong>方法说明</strong><ul>'
            "<li>每分钟取各测点值，按测点高度线性插值到等间隔网格（步长取最小相邻高差，网格右端开区间），再取算术平均。</li>"
            f"<li>{html.escape(_spatial_height_note(result))}</li>"
            "<li>未配置高度、可定位探针不足或高度重复时，退化为跨探针等权聚合，"
            "“方式”列会如实标注；两者不可混同解读。</li>"
            "</ul></div>"
            if _find(tables, "spatial_summary") is not None
            else '<p class="note">本次未生成空间平均（使用了 --no-spatial）。</p>',
        ),
        (
            "turbulence",
            "脉动与湍流参数",
            _img(figures, "sigma", "图 9　分块脉动标准差随时间变化")
            + _img(figures, "psd", "图 10　脉动功率谱")
            + _img(figures, "structure", "图 11　温度结构函数")
            + _table_block(_find(tables, "spectral"))
            + _table_block(_find(tables, "turbulence"))
            + '<div class="callout info"><strong>计算方法</strong><ul>'
            f"<li>分块长度 {html.escape(cfg.analysis.block_rule)}，块内先做线性去趋势，再计算标准差、Welch 功率谱与结构函数。</li>"
            "<li>结构函数 D_T(τ)=⟨[T(t+τ)−T(t)]²⟩（原始值，含量化噪声）；每块用最短两个滞后按 "
            "D(τ)=a+b·τ<sup>2/3</sup> 外推到 τ→0 估计噪声底 a（以 2σ_q² 为上界），"
            "扣除后 Taylor 冻结假设下 C_T² = D_T(τ)/(U·τ)<sup>2/3</sup>，取拟合区间内各滞后的中位数。</li>"
            "<li>Cn² = (79×10⁻⁶·P/T²)²·C_T²，P 取配置值（默认 1013.25 hPa），T 为摄氏温度 + 273.15。</li>"
            "</ul></div>",
        ),
        (
            "cn2",
            "Cn² 交叉验证",
            _img(figures, "cn2", "图 12　设备输出 Cn² 与温度反演 Cn² 对比")
            + _table_block(_find(tables, "cn2_agreement")),
        ),
        (
            "appendix",
            "附录：运行清单",
            _appendix(result, generated),
        ),
    ]

    toc = "".join(f'<li><a href="#{anchor}">{html.escape(heading)}</a></li>' for anchor, heading, _ in sections)
    body = "".join(_section(anchor, heading, content) for anchor, heading, content in sections)
    author = (
        f'<div class="sub">生成者：{html.escape(cfg.report.author)}　|　' if cfg.report.author else '<div class="sub">'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{author}生成时间：{generated}　|　分析用时：{result.runtime_s:.1f} s</div>
<nav class="toc"><ol>{toc}</ol></nav>
{body}
<footer>
本报告由 <code>ui.analysis.thermo</code> 自动生成（schema v{REPORT_SCHEMA_VERSION}）。
关键指标、统计表与图表随报告目录一并导出，便于复核。
</footer>
</body>
</html>
"""


def _appendix(result: AnalysisResult, generated: str) -> str:
    """附录：配置、版本与产物清单。"""
    cfg_table = Table(
        "config",
        "分析配置",
        _flatten_config(result.cfg.to_manifest()),
        "修改配置后重跑即可复现；影响结果的字段同时参与缓存键计算。",
        max_rows=80,
    )
    versions = _versions()
    version_table = Table(
        "versions",
        "软件版本",
        pd.DataFrame(list(versions.items()), columns=["组件", "版本"]),
        "",
    )
    lines = [
        _table_block(cfg_table),
        _table_block(version_table),
        f'<p class="note">生成时间：{html.escape(generated)}；'
        f"输入文件 {len(result.inventory.by_channel(result.cfg.paths.temperature_subdir))} 个；"
        f"缓存命中信息见运行日志。</p>",
        '<p class="note">产物：<code>report.html</code>（本文件，自包含）、<code>summary.md</code>'
        "（Markdown 摘要）、<code>metrics.json</code>（关键指标）、<code>manifest.json</code>"
        "（配置/版本/输入清单）、<code>tables/*.csv</code>、<code>figures/*.png</code>。</p>",
    ]
    return "".join(lines)


def _flatten_config(value: Any, prefix: str = "") -> pd.DataFrame:
    """把嵌套配置字典拍平为「配置项 / 取值」两列表。"""
    rows: list[tuple[str, str]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            rows.append((path, ", ".join(str(item) for item in node)))
        else:
            rows.append((path, str(node)))

    walk(value, prefix)
    return pd.DataFrame(rows, columns=["配置项", "取值"])


def _versions() -> dict[str, str]:
    versions = {"Python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except ImportError:  # 图表可选
        pass
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:
        pass
    return versions


# --------------------------------------------------------------------------- #
# Markdown / JSON / 清单
# --------------------------------------------------------------------------- #
def _markdown(result: AnalysisResult, tables: list[Table], figures: dict[str, Path]) -> str:
    cfg = result.cfg
    lines = [
        f"# {cfg.report.title}",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}　|　分析用时：{result.runtime_s:.1f} s",
        "",
        "## 结论摘要",
        "",
    ]
    for heading, body in _paragraphs(result):
        # 摘要是 HTML 片段，写入 Markdown 前去掉标签
        lines.extend([f"**{heading}**：{_strip_tags(body)}", ""])
    if result.notes:
        lines.append("### 数据提示")
        lines.extend(f"- {note}" for note in result.notes)
        lines.append("")

    for table in tables:
        lines.extend([f"## {table.title}", "", table.to_markdown(), ""])
        if table.note:
            lines.extend([f"> {table.note}", ""])

    lines.append("## 图表")
    for key, path in figures.items():
        path = Path(path)
        lines.extend([f"### {key}", "", f"![{key}](./figures/{path.name})", ""])

    lines.extend(
        [
            "> 关键指标、完整统计表与图表原图随报告目录一并导出，可对照正文结论复核。",
            "",
        ]
    )
    return "\n".join(lines)


def _strip_tags(text: str) -> str:
    """粗暴去除自动摘要中的少量 HTML 标签（仅用 sup/sub）。"""
    return text.replace("<sup>", "^").replace("</sup>", "").replace("<sub>", "_").replace("</sub>", "")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def _clean_json(value: Any) -> Any:
    """递归清洗：NaN/Inf → null，numpy 标量 → Python 标量。"""
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _clean_json(value.item())
    return value


def _manifest(result: AnalysisResult, figures: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    cfg = result.cfg
    sources = result.inventory.by_channel(cfg.paths.temperature_subdir)
    return _clean_json(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report": {
                "title": cfg.report.title,
                "author": cfg.report.author,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "output_dir": str(output_dir),
                "runtime_s": round(result.runtime_s, 3),
            },
            "versions": _versions(),
            "config": cfg.to_manifest(),
            "inputs": {
                "data_root": str(cfg.paths.data_root),
                "temperature_files": len(sources),
                "temperature_bytes": result.inventory.total_bytes(cfg.paths.temperature_subdir),
                "days": [day.isoformat() for day in result.inventory.days(cfg.paths.temperature_subdir)],
                "files": [
                    {
                        "path": source.path.as_posix(),
                        "size": source.size,
                        "mtime_ns": source.mtime_ns,
                        "truncated_hint": source.truncated_hint,
                    }
                    for source in sources
                ],
            },
            "channels": result.inventory.summary(),
            "results": {
                "probes": list(result.probes),
                "valid_probes": list(result.valid_probes),
                "rows": {
                    "minute": len(result.minute),
                    "block": len(result.block),
                    "quality_daily": len(result.quality_daily),
                },
            },
            "figures": {key: Path(path).name for key, path in figures.items()},
            "notes": result.notes,
        }
    )


def _metrics(result: AnalysisResult) -> dict[str, Any]:
    return _clean_json(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "facts": result.facts,
            "probes": list(result.probes),
            "valid_probes": list(result.valid_probes),
            "notes": result.notes,
            "runtime_s": round(result.runtime_s, 3),
        }
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def write_report(
    result: AnalysisResult,
    output_dir: str | Path,
    figures: dict[str, Path] | None = None,
    *,
    extra_writers: dict[str, Callable[[Path], None]] | None = None,
) -> Path:
    """把分析结果落盘为完整报告目录。

    Args:
        result: `pipeline.analyze` 的返回值。
        output_dir: 报告目录（不存在则创建）。
        figures: 图表名 → PNG 路径，由 `pipeline._render_figures` 生成。
        extra_writers: 额外产物钩子，键为产物名，值为 ``writer(报告目录)``。

    Returns:
        报告目录（内含 `report.html`、`summary.md`、`metrics.json`、`manifest.json`、`tables/`）。
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures = {key: Path(path) for key, path in (figures or {}).items() if Path(path).exists()}

    tables = [table for table in _tables(result) if table is not None]
    for table in tables:
        if table.data.empty:
            continue
        table.data.to_csv(tables_dir / f"{table.key}.csv", index=False, encoding="utf-8-sig")

    for writer in (extra_writers or {}).values():
        writer(output_dir)

    (output_dir / "metrics.json").write_text(
        json.dumps(_metrics(result), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_manifest(result, figures, output_dir), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown(result, tables, figures), encoding="utf-8")

    report_path = output_dir / "report.html"
    report_path.write_text(_html(result, tables, figures), encoding="utf-8")
    return output_dir
