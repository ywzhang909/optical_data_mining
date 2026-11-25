#!/usr/bin/env python
"""小系统试验报告生成工具。

按“每次试验”生成 Markdown 报告、总表 CSV 和索引，字段来自：
  * 试验目录名
  * 组内 ``参数.txt``
  * 视频/帧图时间戳、``.dat`` 文件名时刻
  * 可选的 ``试验元数据.csv``（补充抛光程度、波长、温湿度等未记录项）

Usage:
    uv run python scripts/xiaoxitong_report.py
    uv run python scripts/xiaoxitong_report.py --root data/小系统 --out data/小系统/试验报告
    uv run python scripts/xiaoxitong_report.py --refresh-metadata

输出（默认 ``data/小系统/试验报告/``）：
    索引.md
    试验参数总表.csv
    试验元数据.csv（首次运行生成模板，可手工补充）
    <日期>/<序号>_<目录名>.md
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from xiaoxitong_config import env_float

DATE_RE = re.compile(r"^\d{8}$")
CAPTURE_RE = re.compile(r"(\d{8})_(\d{14})_(\d{14})_(\d{6})")
DAT_RE = re.compile(r"^(\d{6})\.dat$", re.IGNORECASE)
DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s(?![a-zA-Z])", re.IGNORECASE)
DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(km|m)(?![a-zA-Z])", re.IGNORECASE)
THICKNESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米)")
ANGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|度)")
FAN_RE = re.compile(r"([一二两三四五六七八九十\d]+)\s*个风机[^\s，,;；]*")
FAN_FREQ_RE = re.compile(r"(?:风机)?频率\s*[:：]?\s*(\d+(?:\.\d+)?\s*(?:Hz|hz))")
POWER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:kW|KW|kw|W|w)(?![a-zA-Z])")
POWER_STD_LABEL = f"{env_float('XXT_POWER_STD_W'):.0f}W"
POWER_LOW_LABEL = f"{env_float('XXT_POWER_LOW_W'):.0f}W"

CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

META_COLUMNS = [
    "试验编号",
    "目录",
    "入射面抛光程度",
    "激光波长",
    "脉宽",
    "重复频率",
    "风速",
    "环境温度",
    "环境湿度",
    "入射角度",
    "备注",
]

MISSING = "未记录"


# --------------------------------------------------------------------------- #
# 通用解析
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> str:
    """读取参数文件，兼容 UTF-8 / GBK。"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def sanitize_power(text: str) -> str:
    """隐藏任何具体功率数值。"""
    if not text:
        return ""
    text = text.replace(POWER_STD_LABEL, "标准功率").replace(POWER_LOW_LABEL, "低功率")
    return POWER_RE.sub("功率（已脱敏）", text)


def parse_params(text: str) -> tuple[dict[str, str], list[str]]:
    """拆分 ``键: 值`` 行与自由文本行。"""
    kv: dict[str, str] = {}
    free: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().replace("：", ":")
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if 0 < len(key) <= 8:
                kv[key] = value
                continue
        free.append(line)
    return kv, free


def normalize_distance(raw: str) -> str:
    """把 1km 统一成 1000m。"""
    value = raw.strip().replace(" ", "")
    match = DISTANCE_RE.search(value)
    if not match:
        return value
    number, unit = float(match.group(1)), match.group(2).lower()
    meters = number * 1000 if unit == "km" else number
    return f"{meters:g}m"


def split_material(raw: str) -> tuple[str, str]:
    """从 ``1mm不锈钢`` 中拆出材料与厚度。"""
    raw = raw.strip()
    if not raw:
        return "", ""
    match = THICKNESS_RE.search(raw)
    if not match:
        return raw, ""
    unit = "mm" if match.group(2) in ("mm", "毫米") else "cm"
    thickness = f"{match.group(1)}{unit}"
    material = raw.replace(match.group(0), "").strip(" ，,;；()（）")
    return material or raw, thickness


def chinese_number(token: str) -> str:
    if token.isdigit():
        return token
    if token in CN_NUM:
        return str(CN_NUM[token])
    return token


def fmt_dt(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else ""


def fmt_span(start: datetime | None, end: datetime | None) -> str:
    if not start:
        return ""
    if not end or start == end:
        return fmt_dt(start)
    if start.date() == end.date():
        return f"{fmt_dt(start)} ~ {end:%H:%M:%S}"
    return f"{fmt_dt(start)} ~ {fmt_dt(end)}"


def md_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " / ")


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    date: str
    seq: int
    name: str
    path: Path
    # 解析结果
    has_params: bool = False
    params_text: str = ""
    kv: dict[str, str] = field(default_factory=dict)
    free: list[str] = field(default_factory=list)
    exp_type: str = "激光"
    power: str = "标准功率"
    distance: str = ""
    distance_source: str = ""
    distance_inferred: bool = False
    incident_direction: str = ""
    incident_angle: str = ""
    material: str = ""
    thickness: str = ""
    polish: str = ""
    duration: str = ""
    duration_source: str = ""
    breakdown: str = ""
    standby: str = ""
    tracking: str = ""
    fan_count: str = ""
    fan_detail: str = ""
    wind: str = ""
    fan_freq: str = ""
    turbulence: str = ""
    # 时间
    capture_start: datetime | None = None
    capture_end: datetime | None = None
    capture_source: str = ""
    capture_segments: int = 0
    # 文件
    counts: Counter = field(default_factory=Counter)
    total_size: int = 0
    # 元数据补充
    laser_wavelength: str = ""
    pulse_width: str = ""
    repetition_rate: str = ""
    temperature: str = ""
    humidity: str = ""
    meta_note: str = ""
    # 派生
    quality: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def exp_id(self) -> str:
        return f"{self.date}-{self.seq:02d}"

    @property
    def display_name(self) -> str:
        return sanitize_power(self.name)

    @property
    def data_form(self) -> str:
        if self.counts[".jpg"] or self.counts[".jpeg"] or self.counts[".png"] or self.counts[".mp4"]:
            return "图像/视频"
        if self.counts[".dat"]:
            return "仅 DAT 原始数据"
        if self.has_params:
            return "仅参数文件"
        return "无数据文件"

    @property
    def total_files(self) -> int:
        return sum(self.counts.values())


def parse_fan(name: str, kv: dict[str, str], free: list[str]) -> tuple[str, str, str, str, str]:
    """返回 数量、说明、风速、频率、湍流。"""
    raw = (kv.get("风机情况") or kv.get("风机数量") or "").strip()
    blob = " ".join([raw, name, *free])
    count = ""
    match = FAN_RE.search(blob)
    if match:
        count = chinese_number(match.group(1))
    elif re.search(r"(无|没有|没)\s*风机", blob):
        count = "0"

    detail = raw
    if not detail:
        if match:
            detail = match.group(0)
        elif count == "0":
            detail = "无风机"

    if "正常风速" in blob:
        wind = "正常风速"
    elif "高风速" in blob:
        wind = "高风速"
    else:
        wind = ""

    freq_match = FAN_FREQ_RE.search(blob)
    freq = freq_match.group(1).replace(" ", "") if freq_match else ""
    turbulence = "有" if "湍流" in blob else ""
    return count, detail, wind, freq, turbulence


def parse_experiment(date: str, exp_dir: Path) -> Experiment:
    param_path = exp_dir / "参数.txt"
    text = read_text(param_path) if param_path.exists() else ""
    kv, free = parse_params(text)
    name = exp_dir.name

    match = re.match(r"\s*(\d+)\s*[.．]\s*(.*)", name)
    seq = int(match.group(1)) if match else 999

    exp_type = "靶板仪" if "靶板仪" in name or any("靶板仪" in line for line in free) else "激光"
    if "低功率" in name:
        power = "低功率"
    elif "标准功率" in name:
        power = "标准功率"
    elif POWER_RE.search(name):
        power = "其他（已脱敏）"
    else:
        power = "未标注"

    blob = " ".join([name, text, *free])

    # 距离
    distance, distance_source = "", ""
    for key in ("实验距离", "作用距离"):
        if kv.get(key):
            distance, distance_source = normalize_distance(kv[key]), f"参数表 {key}"
            break
    if not distance:
        match = DISTANCE_RE.search(" ".join([name, *free]))
        if match:
            distance = normalize_distance(match.group(0))
            distance_source = "目录名/参数表"

    # 入射
    if "斜入射" in blob:
        incident_direction = "斜入射"
    elif "正入射" in blob:
        incident_direction = "正入射"
    else:
        incident_direction = ""
    angle_match = ANGLE_RE.search(blob)
    incident_angle = f"{angle_match.group(1)}°" if angle_match else ""

    # 材料 / 厚度
    material_raw = (kv.get("材料") or kv.get("靶材") or "").strip()
    material, thickness = split_material(material_raw)
    if not thickness:
        match = THICKNESS_RE.search(" ".join([name, *free]))
        if match:
            unit = "mm" if match.group(2) in ("mm", "毫米") else "cm"
            thickness = f"{match.group(1)}{unit}"

    # 时间
    duration, duration_source = "", ""
    for key in ("作用时间", "激光作用时间", "出光时间"):
        if kv.get(key):
            duration, duration_source = kv[key].strip(), f"参数表 {key}"
            break
    if not duration:
        match = DURATION_RE.search(name)
        if match:
            duration, duration_source = f"{match.group(1)}s", "目录名"

    # 待机 / 精跟踪
    if "冷待机" in blob:
        standby = "冷待机"
    elif "热待机" in blob:
        standby = "热待机"
    elif re.search(r"\s冷$", name):
        standby = "冷待机（目录名简写“冷”）"
    else:
        standby = ""

    if "没有开精跟踪" in blob or "没有精跟踪" in blob or "未开精跟踪" in blob:
        tracking = "无精跟踪"
    elif "开精跟踪" in blob:
        tracking = "有精跟踪"
    else:
        tracking = ""

    fan_count, fan_detail, wind, fan_freq, turbulence = parse_fan(name, kv, free)

    # 文件统计
    counts: Counter = Counter()
    total_size = 0
    for path in exp_dir.rglob("*"):
        if path.is_file():
            counts[path.suffix.lower()] += 1
            total_size += path.stat().st_size

    # 采集时间：优先视频/帧图时间戳，其次 DAT 文件名 HHMMSS
    starts: list[datetime] = []
    ends: list[datetime] = []
    capture_ids: set[str] = set()
    bad_timestamps = 0
    for path in exp_dir.rglob("*"):
        match = CAPTURE_RE.search(path.name)
        if match:
            try:
                start = datetime.strptime(match.group(2), "%Y%m%d%H%M%S")
                end = datetime.strptime(match.group(3), "%Y%m%d%H%M%S")
            except ValueError:
                bad_timestamps += 1
                continue
            capture_ids.add(match.group(0))
            starts.append(start)
            ends.append(end)
    dat_times: list[datetime] = []
    for path in exp_dir.glob("*.dat"):
        match = DAT_RE.match(path.name)
        if match:
            with contextlib.suppress(ValueError):
                dat_times.append(datetime.strptime(date + match.group(1), "%Y%m%d%H%M%S"))

    if starts:
        capture_start, capture_end = min(starts), max(ends)
        capture_source = "视频/帧图文件名时间戳"
        capture_segments = len(capture_ids)
    elif dat_times:
        capture_start, capture_end = min(dat_times), max(dat_times)
        capture_source = "DAT 文件名时刻"
        capture_segments = len(dat_times)
    else:
        capture_start = capture_end = None
        capture_source, capture_segments = "", 0

    exp = Experiment(
        date=date,
        seq=seq,
        name=name,
        path=exp_dir,
        has_params=param_path.exists(),
        params_text=sanitize_power(text),
        kv={key: sanitize_power(value) for key, value in kv.items()},
        free=[sanitize_power(line) for line in free],
        exp_type=exp_type,
        power=power,
        distance=distance,
        distance_source=distance_source,
        incident_direction=incident_direction,
        incident_angle=incident_angle,
        material=material,
        thickness=thickness,
        duration=duration,
        duration_source=duration_source,
        breakdown=sanitize_power((kv.get("击穿时间") or "").strip()),
        standby=standby,
        tracking=tracking,
        fan_count=fan_count,
        fan_detail=sanitize_power(fan_detail),
        wind=wind,
        fan_freq=fan_freq,
        turbulence=turbulence,
        capture_start=capture_start,
        capture_end=capture_end,
        capture_source=capture_source,
        capture_segments=capture_segments,
        counts=counts,
        total_size=total_size,
    )

    if bad_timestamps:
        exp.quality.append(f"{bad_timestamps} 个采集文件名时间戳无效，已忽略")
    # 目录名与参数表时间不一致
    if exp.duration_source.startswith("参数表"):
        named = DURATION_RE.search(name)
        if named and f"{named.group(1)}s".lower() != exp.duration.lower():
            exp.quality.append(f"目录名标注 {named.group(1)}s，参数表为 {exp.duration}（采用参数表）")
    if not exp.has_params:
        exp.quality.append("无参数.txt，字段取自目录名/文件名")
    return exp


# --------------------------------------------------------------------------- #
# 元数据与派生字段
# --------------------------------------------------------------------------- #
def ensure_metadata(path: Path, experiments: list[Experiment], refresh: bool, root: Path) -> None:
    """首次生成元数据模板；refresh 时补齐缺失行，不覆盖已有值。"""
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                existing[row.get("试验编号", "")] = row

    rows = []
    for exp in experiments:
        row = existing.get(exp.exp_id)
        if row is None:
            row = {column: "" for column in META_COLUMNS}
            row["试验编号"] = exp.exp_id
            row["目录"] = str(exp.path.relative_to(root))
        else:
            row = {column: (row.get(column) or "") for column in META_COLUMNS}
        rows.append(row)

    if path.exists() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=META_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row.get("试验编号", ""): row for row in csv.DictReader(handle)}


def apply_metadata(exp: Experiment, row: dict[str, str] | None) -> None:
    if not row:
        return
    fields = {
        "入射面抛光程度": "polish",
        "激光波长": "laser_wavelength",
        "脉宽": "pulse_width",
        "重复频率": "repetition_rate",
        "环境温度": "temperature",
        "环境湿度": "humidity",
    }
    for column, attr in fields.items():
        value = (row.get(column) or "").strip()
        if value:
            setattr(exp, attr, sanitize_power(value))
    if not exp.wind and (row.get("风速") or "").strip():
        exp.wind = row["风速"].strip()
    if not exp.incident_angle and (row.get("入射角度") or "").strip():
        exp.incident_angle = row["入射角度"].strip()
    note = (row.get("备注") or "").strip()
    if note:
        exp.meta_note = sanitize_power(note)


def finalize_quality(exp: Experiment) -> None:
    if exp.distance_inferred:
        exp.quality.append("距离未直接记录，按同日同批次唯一距离推断")
    if not exp.material:
        exp.quality.append("材料未记录")
    if not exp.thickness:
        exp.quality.append("材料厚度未记录")
    if not exp.polish:
        exp.quality.append("入射面抛光程度未记录，可在“试验元数据.csv”补充")
    if exp.fan_count and exp.fan_count != "0" and not exp.wind and not exp.fan_freq:
        exp.quality.append("风速/风机频率未记录")
    if "风机存在问题" in exp.fan_detail or "没开起来" in exp.fan_detail:
        exp.quality.append("参数表注明风机异常、未实际开启")
    if not exp.capture_start:
        exp.quality.append("无采集时间戳")
    if not (exp.counts[".jpg"] or exp.counts[".jpeg"] or exp.counts[".png"] or exp.counts[".mp4"]):
        exp.quality.append("无图像/视频数据")
    if exp.meta_note:
        exp.quality.append(f"元数据备注：{exp.meta_note}")


def infer_distance(experiments: list[Experiment]) -> None:
    by_date: dict[str, set[str]] = defaultdict(set)
    for exp in experiments:
        if exp.distance:
            by_date[exp.date].add(exp.distance)
    for exp in experiments:
        if not exp.distance and len(by_date.get(exp.date, ())) == 1:
            exp.distance = next(iter(by_date[exp.date]))
            exp.distance_inferred = True
            exp.distance_source = "同日同批次推断"


def signature(exp: Experiment) -> tuple:
    return (
        exp.date,
        exp.exp_type,
        exp.power,
        exp.distance.rstrip("*"),
        exp.incident_direction,
        exp.material,
        exp.thickness,
        exp.polish,
        exp.duration,
        exp.standby,
        exp.tracking,
        exp.fan_count,
        exp.wind,
        exp.turbulence,
    )


def annotate_duplicates(experiments: list[Experiment]) -> None:
    groups: dict[tuple, list[Experiment]] = defaultdict(list)
    for exp in experiments:
        groups[signature(exp)].append(exp)
    for group in groups.values():
        if len(group) > 1:
            for exp in group:
                exp.duplicates = [other.exp_id for other in group if other is not exp]


def build_experiments(root: Path, issues: list[str] | None = None) -> list[Experiment]:
    """扫描全部试验目录；单个目录出错时记录到 ``issues`` 并继续。"""
    experiments: list[Experiment] = []
    for date_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not DATE_RE.fullmatch(date_dir.name):
            continue
        try:
            candidates = sorted(path for path in date_dir.iterdir() if path.is_dir())
        except OSError as exc:
            if issues is not None:
                issues.append(f"{date_dir.name}/: 目录不可读（{exc}）")
            continue
        for exp_dir in candidates:
            try:
                experiments.append(parse_experiment(date_dir.name, exp_dir))
            except Exception as exc:  # 单个坏目录不应中断整批报告
                if issues is not None:
                    issues.append(f"{date_dir.name}/{exp_dir.name}: 解析失败（{type(exc).__name__}: {exc}）")
    infer_distance(experiments)
    annotate_duplicates(experiments)
    return experiments


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def slugify(name: str, seq: int) -> str:
    text = re.sub(rf"^\s*{seq}\s*[.．]\s*", "", name)
    text = re.sub(r'[\\/:*?"<>|()（）]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80] or f"exp{seq:02d}"


def rows_table(rows: list[tuple[str, object]]) -> list[str]:
    lines = ["| 项目 | 值 |", "|---|---|"]
    lines.extend(
        f"| {md_cell(label)} | {md_cell(value) if value not in ('', None) else MISSING} |" for label, value in rows
    )
    return lines


def render_report(exp: Experiment) -> str:
    lines: list[str] = []
    lines.append(f"# 试验报告：{exp.exp_id}")
    lines.append("")
    lines.append("[返回索引](../索引.md)")
    lines.append("")

    lines.append("## 1. 试验概览")
    lines.append("")
    lines.extend(
        rows_table(
            [
                ("试验编号", exp.exp_id),
                ("试验日期", f"{exp.date[:4]}-{exp.date[4:6]}-{exp.date[6:]}"),
                ("试验序号", f"{exp.seq:02d}"),
                ("目录", exp.display_name),
                ("类型", exp.exp_type),
                ("功率等级", exp.power),
                ("试验时段", fmt_span(exp.capture_start, exp.capture_end)),
                ("时段来源", exp.capture_source),
                ("数据段数", exp.capture_segments or ""),
            ]
        )
    )
    lines.append("")

    lines.append("## 2. 激光与作用参量")
    lines.append("")
    lines.extend(
        rows_table(
            [
                ("功率等级", exp.power),
                ("参数表激光能量", exp.kv.get("激光能量", "")),
                ("作用时间", exp.duration),
                ("时间来源", exp.duration_source),
                ("击穿时间", exp.breakdown),
                ("待机状态", exp.standby),
                ("精跟踪", exp.tracking),
                ("激光波长", exp.laser_wavelength),
                ("脉宽", exp.pulse_width),
                ("重复频率", exp.repetition_rate),
            ]
        )
    )
    lines.append("")

    lines.append("## 3. 靶材与光路")
    lines.append("")
    lines.extend(
        rows_table(
            [
                ("材料", exp.material),
                ("厚度", exp.thickness),
                ("入射面抛光程度", exp.polish),
                ("入射方向", exp.incident_direction),
                ("入射角度", exp.incident_angle),
                ("试验距离", exp.distance),
                ("距离来源", exp.distance_source),
            ]
        )
    )
    lines.append("")

    lines.append("## 4. 环境参量（对应试验时段）")
    lines.append("")
    lines.extend(
        rows_table(
            [
                ("风机数量", exp.fan_count),
                ("风机说明", exp.fan_detail),
                ("风速", exp.wind),
                ("风机频率", exp.fan_freq),
                ("湍流", exp.turbulence),
                ("环境温度", exp.temperature),
                ("环境湿度", exp.humidity),
            ]
        )
    )
    lines.append("")

    lines.append("## 5. 数据资产")
    lines.append("")
    lines.extend(
        rows_table(
            [
                ("数据形态", exp.data_form),
                ("JPG", exp.counts[".jpg"] + exp.counts[".jpeg"]),
                ("PNG", exp.counts[".png"]),
                ("MP4", exp.counts[".mp4"]),
                ("DAT", exp.counts[".dat"]),
                ("DB", exp.counts[".db"]),
                (
                    "其他文件",
                    exp.total_files - sum(exp.counts[e] for e in (".jpg", ".jpeg", ".png", ".mp4", ".dat", ".db")),
                ),
                ("总文件数", exp.total_files),
                ("总大小", f"{exp.total_size / 1024 / 1024:.2f} MB"),
            ]
        )
    )
    lines.append("")

    lines.append("## 6. 原始参数表")
    lines.append("")
    if exp.params_text.strip():
        lines.extend(["```text", exp.params_text.strip(), "```"])
    else:
        lines.append("无 `参数.txt`。")
    lines.append("")

    lines.append("## 7. 数据质量与备注")
    lines.append("")
    notes = list(exp.quality)
    if exp.duplicates:
        notes.append("同条件重复组：" + "、".join(exp.duplicates))
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def render_index(experiments: list[Experiment], out_dir: Path) -> str:
    by_date: dict[str, list[Experiment]] = defaultdict(list)
    for exp in experiments:
        by_date[exp.date].append(exp)

    lines = [
        "# 小系统试验报告索引",
        "",
        f"- 数据根目录：`{experiments[0].path.parents[1]}`" if experiments else "- 无试验数据",
        f"- 试验组数：{len(experiments)}",
        f"- 试验日期：{len(by_date)} 天",
        "- 功率已脱敏：标准功率 / 低功率 / 其他（已脱敏）",
        "- 报告字段：激光参量、功率、类型、入射角度、材料、厚度、抛光程度、距离、时间、环境参量",
        "",
        "## 总览",
        "",
        "| 日期 | 组数 | 激光组 | 靶板仪组 | 数据文件 |",
        "|---|---:|---:|---:|---:|",
    ]
    for date, items in by_date.items():
        lines.append(
            f"| {date} | {len(items)} | {sum(e.exp_type == '激光' for e in items)} | "
            f"{sum(e.exp_type == '靶板仪' for e in items)} | {sum(e.total_files for e in items)} |"
        )
    lines.append(
        f"| **合计** | **{len(experiments)}** | **{sum(e.exp_type == '激光' for e in experiments)}** | "
        f"**{sum(e.exp_type == '靶板仪' for e in experiments)}** | "
        f"**{sum(e.total_files for e in experiments)}** |"
    )
    lines.append("")

    for date, items in by_date.items():
        lines.append(f"## {date}")
        lines.append("")
        lines.append("| 编号 | 报告 | 类型 | 功率 | 距离 | 入射 | 材料/厚度 | 抛光 | 作用时间 | 试验时段 | 环境 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for exp in items:
            filename = f"{exp.seq:02d}_{slugify(exp.name, exp.seq)}.md"
            material = "/".join(part for part in (exp.material, exp.thickness) if part) or MISSING
            env_parts = []
            if exp.fan_count:
                env_parts.append(f"{exp.fan_count}风机")
            if exp.wind:
                env_parts.append(exp.wind)
            if exp.standby:
                env_parts.append(exp.standby)
            if exp.tracking:
                env_parts.append(exp.tracking)
            lines.append(
                f"| {exp.exp_id} | [{exp.seq:02d}]({date}/{filename}) | {exp.exp_type} | {exp.power} | "
                f"{exp.distance or MISSING} | {exp.incident_direction or MISSING} | {material} | "
                f"{exp.polish or MISSING} | {exp.duration or MISSING} | "
                f"{fmt_span(exp.capture_start, exp.capture_end) or MISSING} | {'；'.join(env_parts) or MISSING} |"
            )
        lines.append("")

    lines.append("## 使用说明")
    lines.append("")
    lines.append(
        "- `试验元数据.csv` 可补充原始数据未记录的项目（抛光程度、波长、脉宽、重频、温湿度等），重新运行本工具即可写入报告。"
    )
    lines.append("- `--refresh-metadata` 只补齐新增试验行，不覆盖已填写内容。")
    lines.append("")
    return "\n".join(lines)


CSV_COLUMNS = [
    "试验编号",
    "日期",
    "序号",
    "目录",
    "类型",
    "功率等级",
    "试验距离",
    "距离来源",
    "入射方向",
    "入射角度",
    "材料",
    "厚度",
    "入射面抛光程度",
    "作用时间",
    "时间来源",
    "击穿时间",
    "待机状态",
    "精跟踪",
    "风机数量",
    "风机说明",
    "风速",
    "风机频率",
    "湍流",
    "试验开始",
    "试验结束",
    "时段来源",
    "数据段数",
    "JPG",
    "PNG",
    "MP4",
    "DAT",
    "DB",
    "其他文件",
    "总文件数",
    "总大小_MB",
    "激光波长",
    "脉宽",
    "重复频率",
    "环境温度",
    "环境湿度",
    "数据质量",
    "备注",
]


def write_csv(experiments: list[Experiment], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for exp in experiments:
            other = exp.total_files - sum(exp.counts[e] for e in (".jpg", ".jpeg", ".png", ".mp4", ".dat", ".db"))
            writer.writerow(
                {
                    "试验编号": exp.exp_id,
                    "日期": exp.date,
                    "序号": exp.seq,
                    "目录": exp.display_name,
                    "类型": exp.exp_type,
                    "功率等级": exp.power,
                    "试验距离": exp.distance,
                    "距离来源": exp.distance_source,
                    "入射方向": exp.incident_direction,
                    "入射角度": exp.incident_angle,
                    "材料": exp.material,
                    "厚度": exp.thickness,
                    "入射面抛光程度": exp.polish,
                    "作用时间": exp.duration,
                    "时间来源": exp.duration_source,
                    "击穿时间": exp.breakdown,
                    "待机状态": exp.standby,
                    "精跟踪": exp.tracking,
                    "风机数量": exp.fan_count,
                    "风机说明": exp.fan_detail,
                    "风速": exp.wind,
                    "风机频率": exp.fan_freq,
                    "湍流": exp.turbulence,
                    "试验开始": fmt_dt(exp.capture_start),
                    "试验结束": fmt_dt(exp.capture_end),
                    "时段来源": exp.capture_source,
                    "数据段数": exp.capture_segments,
                    "JPG": exp.counts[".jpg"] + exp.counts[".jpeg"],
                    "PNG": exp.counts[".png"],
                    "MP4": exp.counts[".mp4"],
                    "DAT": exp.counts[".dat"],
                    "DB": exp.counts[".db"],
                    "其他文件": other,
                    "总文件数": exp.total_files,
                    "总大小_MB": f"{exp.total_size / 1024 / 1024:.2f}",
                    "激光波长": exp.laser_wavelength,
                    "脉宽": exp.pulse_width,
                    "重复频率": exp.repetition_rate,
                    "环境温度": exp.temperature,
                    "环境湿度": exp.humidity,
                    "数据质量": "；".join(exp.quality),
                    "备注": exp.meta_note,
                }
            )


def generate(root: Path, out_dir: Path, metadata_path: Path, refresh_metadata: bool, clean: bool) -> list[Experiment]:
    experiments = build_experiments(root)
    ensure_metadata(metadata_path, experiments, refresh_metadata, root)
    metadata = load_metadata(metadata_path)
    for exp in experiments:
        apply_metadata(exp, metadata.get(exp.exp_id))
        finalize_quality(exp)

    out_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for stale in out_dir.glob("*/*.md"):
            stale.unlink()

    for exp in experiments:
        date_dir = out_dir / exp.date
        date_dir.mkdir(parents=True, exist_ok=True)
        report_path = date_dir / f"{exp.seq:02d}_{slugify(exp.name, exp.seq)}.md"
        report_path.write_text(render_report(exp), encoding="utf-8")

    (out_dir / "索引.md").write_text(render_index(experiments, out_dir), encoding="utf-8")
    write_csv(experiments, out_dir / "试验参数总表.csv")
    return experiments


def default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "小系统"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="小系统试验报告生成工具")
    parser.add_argument("--root", type=Path, default=default_root(), help="数据根目录（默认 data/小系统）")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认 <root>/试验报告）")
    parser.add_argument("--metadata", type=Path, default=None, help="试验元数据 CSV（默认 <out>/试验元数据.csv）")
    parser.add_argument("--refresh-metadata", action="store_true", help="补齐新增试验行，不覆盖已有值")
    parser.add_argument("--clean", action="store_true", help="生成前删除输出目录中的旧报告")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"数据根目录不存在：{root}")
    out_dir = (args.out or (root / "试验报告")).resolve()
    metadata_path = (args.metadata or (out_dir / "试验元数据.csv")).resolve()

    experiments = generate(root, out_dir, metadata_path, args.refresh_metadata, args.clean)
    print(f"experiments={len(experiments)}")
    print(f"reports={out_dir}")
    print(f"index={out_dir / '索引.md'}")
    print(f"csv={out_dir / '试验参数总表.csv'}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
