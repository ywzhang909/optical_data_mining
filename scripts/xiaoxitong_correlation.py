#!/usr/bin/env python
"""小系统试验参数相关性分析工具。

基于 ``xiaoxitong_report.py`` 的解析结果，对每次试验的参数做统计关联分析：

  * 数值型字段：Spearman 秩相关（成对完整样本，给出 n 与 p）
  * 分类型字段：Cramér's V 关联强度 + 卡方检验
  * 击穿时间：按入射方向 / 风机数量 / 距离 / 厚度分组比较（Mann-Whitney、Kruskal-Wallis）
  * 击穿发生率：按分组统计击穿 / 未击穿计数

输出：
    <out>/参数相关性分析.md
    <out>/参数相关性_数值.csv

Usage:
    uv run python scripts/xiaoxitong_correlation.py
    uv run python scripts/xiaoxitong_correlation.py --root data/小系统 --out data/小系统/试验报告
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xiaoxitong_report as base  # noqa: E402

SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|S|秒)")
METERS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\b")

NUMERIC_FIELDS: list[tuple[str, str]] = [
    ("distance_m", "试验距离 (m)"),
    ("duration_s", "作用时间 (s)"),
    ("breakdown_s", "击穿时间 (s)"),
    ("fan_count", "风机数量"),
    ("capture_s", "单组采集时长 (s)"),
    ("jpg", "JPG 帧数"),
    ("size_mb", "数据量 (MB)"),
]

CATEGORICAL_FIELDS: list[str] = [
    "类型",
    "距离",
    "入射方向",
    "风机数量",
    "待机状态",
    "精跟踪",
    "厚度",
]

GROUP_SPEC: list[tuple[str, list[str]]] = [
    ("入射方向", ["正入射", "斜入射"]),
    ("风机数量", ["0", "1", "2"]),
    ("距离", ["200m", "600m", "1000m"]),
    ("厚度", ["1mm", "2mm"]),
]

STATUS_BREAK = "击穿"
STATUS_NO_BREAK = "未击穿"
STATUS_UNKNOWN = "无法判定"
MISSING = "未记录"


# --------------------------------------------------------------------------- #
# 字段标准化
# --------------------------------------------------------------------------- #
def first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text or "")
    return float(match.group(1)) if match else None


def normalize_categories(exp: base.Experiment) -> dict[str, str]:
    """把解析结果映射为统一的分析口径。"""
    standby = exp.standby
    if standby.startswith("冷待机"):
        standby = "冷待机"
    return {
        "类型": exp.exp_type,
        "距离": exp.distance.rstrip("*") or MISSING,
        "入射方向": exp.incident_direction or MISSING,
        "风机数量": exp.fan_count or MISSING,
        "待机状态": standby or MISSING,
        "精跟踪": exp.tracking or MISSING,
        "厚度": exp.thickness or MISSING,
        "材料": exp.material or MISSING,
        "风速": exp.wind or MISSING,
    }


def breakdown_status(exp: base.Experiment) -> str:
    raw = exp.breakdown.strip()
    if not raw:
        return ""
    if first_float(SECONDS_RE, raw) is not None:
        return STATUS_BREAK
    if "未击穿" in raw or "未明显" in raw or "未观察" in raw:
        return STATUS_NO_BREAK
    return STATUS_UNKNOWN


def numeric_record(exp: base.Experiment) -> dict[str, float | None]:
    fan = float(exp.fan_count) if exp.fan_count.strip().isdigit() else None
    capture = None
    if exp.capture_start and exp.capture_end:
        capture = (exp.capture_end - exp.capture_start).total_seconds()
    return {
        "distance_m": first_float(METERS_RE, exp.distance),
        "duration_s": first_float(SECONDS_RE, exp.duration),
        "breakdown_s": first_float(SECONDS_RE, exp.breakdown),
        "fan_count": fan,
        "capture_s": capture,
        "jpg": float(exp.counts[".jpg"] + exp.counts[".jpeg"]),
        "size_mb": round(exp.total_size / 1024 / 1024, 3),
    }


def is_laser(exp: base.Experiment) -> bool:
    return exp.exp_type == "激光"


# --------------------------------------------------------------------------- #
# 统计量
# --------------------------------------------------------------------------- #
def spearman_pair(xs: list[float | None], ys: list[float | None], min_n: int = 5) -> tuple[float, float, int] | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < min_n:
        return None
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    rho, p = stats.spearmanr(a, b)
    return float(rho), float(p), len(pairs)


def cramers_v(first: list[str], second: list[str], min_n: int = 8) -> tuple[float, float, int] | None:
    pairs = [(a, b) for a, b in zip(first, second, strict=True) if a != MISSING and b != MISSING]
    if len(pairs) < min_n:
        return None
    cats_a = sorted({p[0] for p in pairs})
    cats_b = sorted({p[1] for p in pairs})
    if len(cats_a) < 2 or len(cats_b) < 2:
        return None
    index_a = {c: i for i, c in enumerate(cats_a)}
    index_b = {c: i for i, c in enumerate(cats_b)}
    table = np.zeros((len(cats_a), len(cats_b)))
    for a, b in pairs:
        table[index_a[a], index_b[b]] += 1
    with contextlib.suppress(ValueError):
        chi2, p, _, _ = stats.chi2_contingency(table)
        denom = table.sum() * (min(table.shape) - 1)
        if denom <= 0:
            return None
        return float(math.sqrt(chi2 / denom)), float(p), len(pairs)
    return None


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def describe(values: list[float]) -> str:
    if not values:
        return "-"
    arr = np.array(values, dtype=float)
    median = float(np.median(arr))
    if len(arr) == 1:
        return f"{median:g}"
    return f"{median:g} [{arr.min():g}–{arr.max():g}]"


def mw_test(first: list[float], second: list[float]) -> tuple[float, float] | None:
    if len(first) < 3 or len(second) < 3:
        return None
    result = stats.mannwhitneyu(first, second, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def kw_test(groups: list[list[float]]) -> tuple[float, float] | None:
    groups = [group for group in groups if len(group) >= 2]
    if len(groups) < 2:
        return None
    result = stats.kruskal(*groups)
    return float(result.statistic), float(result.pvalue)


# --------------------------------------------------------------------------- #
# 各分析段落
# --------------------------------------------------------------------------- #
def numeric_section(numbers: list[dict[str, float | None]]) -> tuple[list[str], list[dict[str, object]]]:
    labels = dict(NUMERIC_FIELDS)
    keys = [key for key, _ in NUMERIC_FIELDS]
    lines = ["| 变量对 | n | Spearman ρ | p | 显著性 |", "|---|---:|---:|---:|---|"]
    records: list[dict[str, object]] = []
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            result = spearman_pair([row[key_a] for row in numbers], [row[key_b] for row in numbers])
            if result is None:
                continue
            rho, p, n = result
            lines.append(f"| {labels[key_a]} ↔ {labels[key_b]} | {n} | {rho:+.3f} | {fmt_p(p)} | {stars(p)} |")
            records.append(
                {
                    "变量1": labels[key_a],
                    "变量2": labels[key_b],
                    "样本数": n,
                    "Spearman_rho": round(rho, 4),
                    "p值": round(p, 6),
                    "显著性": stars(p),
                }
            )
    if not records:
        lines.append("| （样本不足以计算） | - | - | - | - |")
    return lines, records


def categorical_section(categories: list[dict[str, str]]) -> list[str]:
    lines = ["| 变量对 | n | Cramér's V | p | 显著性 |", "|---|---:|---:|---:|---|"]
    filled = False
    for i, name_a in enumerate(CATEGORICAL_FIELDS):
        for name_b in CATEGORICAL_FIELDS[i + 1 :]:
            result = cramers_v([row[name_a] for row in categories], [row[name_b] for row in categories])
            if result is None:
                continue
            v, p, n = result
            lines.append(f"| {name_a} ↔ {name_b} | {n} | {v:.3f} | {fmt_p(p)} | {stars(p)} |")
            filled = True
    if not filled:
        lines.append("| （样本不足以计算） | - | - | - | - |")
    return lines


def breakdown_buckets(
    experiments: list[base.Experiment],
    categories: list[dict[str, str]],
    numbers: list[dict[str, float | None]],
    field: str,
    order: list[str],
) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {}
    for label in order:
        values = [
            numbers[i]["breakdown_s"]
            for i, exp in enumerate(experiments)
            if is_laser(exp) and categories[i][field] == label and numbers[i]["breakdown_s"] is not None
        ]
        buckets[label] = [float(v) for v in values if v is not None]
    return buckets


def group_section(
    experiments: list[base.Experiment],
    categories: list[dict[str, str]],
    numbers: list[dict[str, float | None]],
) -> list[str]:
    lines: list[str] = []
    lines.append("仅统计“激光”组且击穿时间为数值的记录。")
    lines.append("")
    for field, order in GROUP_SPEC:
        buckets = breakdown_buckets(experiments, categories, numbers, field, order)
        nonempty = [values for values in buckets.values() if values]
        test_text = "样本不足，未检验"
        if len(nonempty) == 2:
            result = mw_test(nonempty[0], nonempty[1])
            if result:
                test_text = f"Mann-Whitney U={result[0]:.1f}，p={fmt_p(result[1])} {stars(result[1])}"
        elif len(nonempty) >= 3:
            result = kw_test(nonempty)
            if result:
                test_text = f"Kruskal-Wallis H={result[0]:.2f}，p={fmt_p(result[1])} {stars(result[1])}"

        lines.append(f"**按{field}分组**")
        lines.append("")
        lines.append("| 分组 | n | 击穿时间中位数 [范围] (s) |")
        lines.append("|---|---:|---|")
        for label in order:
            values = buckets[label]
            lines.append(f"| {label} | {len(values)} | {describe(values)} |")
        lines.append("")
        lines.append(f"检验：{test_text}")
        lines.append("")
    return lines


def rate_section(experiments: list[base.Experiment], categories: list[dict[str, str]], status: list[str]) -> list[str]:
    lines = [
        "| 分组字段 | 取值 | 击穿 | 未击穿/未观察到 | 无法判定 | 击穿率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    filled = False
    for field, order in GROUP_SPEC:
        for label in order:
            hits = misses = unknown = 0
            for exp, cat, st in zip(experiments, categories, status, strict=True):
                if not is_laser(exp) or cat[field] != label:
                    continue
                if st == STATUS_BREAK:
                    hits += 1
                elif st == STATUS_NO_BREAK:
                    misses += 1
                elif st == STATUS_UNKNOWN:
                    unknown += 1
            if hits + misses + unknown == 0:
                continue
            decided = hits + misses
            rate = f"{hits / decided:.0%}" if decided else "-"
            lines.append(f"| {field} | {label} | {hits} | {misses} | {unknown} | {rate} |")
            filled = True
    if not filled:
        lines.append("| - | - | - | - | - | - |")
    return lines


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def render(
    experiments: list[base.Experiment],
    categories: list[dict[str, str]],
    numbers: list[dict[str, float | None]],
    status: list[str],
    numeric_lines: list[str],
) -> str:
    lasers = [exp for exp in experiments if is_laser(exp)]
    breaks = sum(1 for s in status if s == STATUS_BREAK)
    no_breaks = sum(1 for s in status if s == STATUS_NO_BREAK)
    unknowns = sum(1 for s in status if s == STATUS_UNKNOWN)

    lines: list[str] = [
        "# 小系统试验参数相关性分析",
        "",
        f"- 样本：{len(experiments)} 组试验（激光 {len(lasers)} 组，靶板仪 {len(experiments) - len(lasers)} 组）",
        f"- 击穿判定：数值击穿 {breaks} 组，明确未击穿/未观察到 {no_breaks} 组，无法判定 {unknowns} 组",
        "- 数值关联用 Spearman 秩相关（不假设线性、对离群值稳健）；分类关联用 Cramér's V",
        "- 显著性：\\* p<0.05，\\*\\* p<0.01，\\*\\*\\* p<0.001",
        "",
        "## 1. 字段可用性",
        "",
        "| 字段 | 有记录 | 覆盖率 | 备注 |",
        "|---|---:|---:|---|",
    ]
    field_stats = {
        "试验距离": (sum(1 for exp in experiments if exp.distance), "部分为同日同批次推断"),
        "入射方向": (sum(1 for exp in experiments if exp.incident_direction), ""),
        "材料": (sum(1 for exp in experiments if exp.material), "已记录的全为不锈钢"),
        "材料厚度": (sum(1 for exp in experiments if exp.thickness), ""),
        "入射面抛光程度": (0, "原始数据未记录，需在试验元数据.csv 补充"),
        "作用时间": (sum(1 for exp in experiments if exp.duration), "20261010 缺记录"),
        "击穿时间": (sum(1 for s in status if s), "含定性描述，数值样本更少"),
        "待机状态": (sum(1 for exp in experiments if exp.standby), "20261010/20261012 部分记录"),
        "精跟踪": (sum(1 for exp in experiments if exp.tracking), "仅 20261012 记录"),
        "风机数量": (sum(1 for exp in experiments if exp.fan_count), ""),
        "风速": (sum(1 for exp in experiments if exp.wind), "仅“正常/高”两档定性"),
        "激光波长/脉宽/重频": (0, "原始数据未记录，视为同一台激光器常量"),
        "环境温度/湿度": (0, "原始数据未记录"),
    }
    for field, (count, note) in field_stats.items():
        lines.append(f"| {field} | {count} | {count / len(experiments):.0%} | {note} |")
    lines.append("")

    lines.extend(["## 2. 数值型参数相关（Spearman）", ""])
    lines.extend(numeric_lines)
    lines.append("")
    lines.append(
        "> 距离、作用时间、待机、精跟踪都由“试验批次（日期）”决定，"
        "它们之间的强相关属于**设计共线性**，不能当作独立物理效应解读。"
    )
    lines.append("")

    lines.extend(["## 3. 分类型参数关联（Cramér's V）", ""])
    lines.extend(categorical_section(categories))
    lines.append("")
    lines.append("> V<0.3 弱、0.3–0.5 中等、>0.5 强；只在两组均有记录的样本上配对计算。")
    lines.append("")

    lines.extend(["## 4. 击穿时间分组比较", ""])
    lines.extend(group_section(experiments, categories, numbers))
    lines.append("> 中位数为该组全部数值击穿时间；检验只用有数值的组，样本不足时不做检验。")
    lines.append("")

    lines.extend(["## 5. 击穿发生率", ""])
    lines.extend(rate_section(experiments, categories, status))
    lines.append("")
    lines.append("> 单组内可判定样本很少，本表只报计数与比率，不做显著性判定。")
    lines.append("")

    lines.extend(["## 6. 结论与局限", ""])
    lines.extend(
        [
            "1. **设计共线**：距离与日期完全绑定（200m=20261009、600m=20261010、1000m=20261012），"
            "作用时间、待机、精跟踪、风速也随批次变化，跨批次的相关系数不能解释为因果效应。",
            "2. **可横向比较的变量**：只有同批次内部因素（入射方向、风机数量、材料厚度）可以在批次内比较，"
            "跨批次只能作趋势参考。",
            "3. **击穿时间样本有限**：数值记录仅 14 组，集中在 200m/600m，统计功效低，结论需补充试验验证。",
            "4. **关键字段缺失**：入射面抛光程度、激光波长、脉宽、重频、温湿度在原始数据中完全缺失，"
            "而这些是最可能影响击穿阈值的因素，目前无法纳入相关性分析。",
            "5. **数据量不是物理量**：JPG 帧数与数据量取决于采集设置，仅作数据完整性参考，不参与物理结论。",
        ]
    )
    lines.append("")
    lines.extend(
        [
            "## 7. 建议补充的元数据",
            "",
            "| 字段 | 原因 |",
            "|---|---|",
            "| 入射面抛光程度（镜面/精磨/原始轧制面） | 直接影响吸收率与击穿阈值，是当前最大缺口 |",
            "| 激光波长、脉宽、重复频率 | 判断不同日期是否同一激光状态 |",
            "| 环境温度、湿度、气压 | 长距离传输与击穿的环境修正 |",
            "| 风速实测值（m/s） | 现有仅“正常/高”两档定性描述 |",
            "| 光斑尺寸/功率密度 | 把“标准功率”换算成可比较的强度量 |",
            "",
            "填写 `试验报告/试验元数据.csv` 后重新运行本工具，即可把上述字段纳入相关性分析。",
            "",
        ]
    )
    return "\n".join(lines)


def write_numeric_csv(records: list[dict[str, object]], path: Path) -> None:
    columns = ["变量1", "变量2", "样本数", "Spearman_rho", "p值", "显著性"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="小系统试验参数相关性分析")
    parser.add_argument("--root", type=Path, default=base.default_root(), help="数据根目录（默认 data/小系统）")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认 <root>/试验报告）")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"数据根目录不存在：{root}")
    out_dir = (args.out or (root / "试验报告")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = base.build_experiments(root)
    metadata = base.load_metadata(out_dir / "试验元数据.csv")
    for exp in experiments:
        base.apply_metadata(exp, metadata.get(exp.exp_id))

    categories = [normalize_categories(exp) for exp in experiments]
    numbers = [numeric_record(exp) for exp in experiments]
    status = [breakdown_status(exp) for exp in experiments]

    numeric_lines, numeric_records = numeric_section(numbers)
    report = render(experiments, categories, numbers, status, numeric_lines)
    report_path = out_dir / "参数相关性分析.md"
    report_path.write_text(report, encoding="utf-8")
    csv_path = out_dir / "参数相关性_数值.csv"
    write_numeric_csv(numeric_records, csv_path)

    print(f"experiments={len(experiments)}")
    print(f"numeric_pairs={len(numeric_records)}")
    print(f"report={report_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
