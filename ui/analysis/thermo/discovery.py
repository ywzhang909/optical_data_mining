"""原始数据目录扫描：发现日期目录、小时文件，并给出完整性线索。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DAY_PATTERN = re.compile(r"^(?P<stamp>\d{8})$")
HOUR_PATTERN = re.compile(r"^(?P<stem>[A-Za-z0-9_]+?)_(?P<index>\d+)\.csv$")

SUSPICIOUS_SIZES = {0, 10 * 1024 * 1024, 100 * 1024 * 1024}


@dataclass(frozen=True)
class SourceFile:
    """一个原始 CSV 文件的元信息（不读取内容）。"""

    path: Path
    day: date
    channel_dir: str
    index: int
    size: int
    mtime_ns: int
    truncated_hint: str | None = None

    @property
    def fingerprint(self) -> str:
        """内容指纹：路径 + 大小 + 修改时间，用于缓存失效判断。"""
        raw = f"{self.path.as_posix()}|{self.size}|{self.mtime_ns}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    @property
    def day_key(self) -> str:
        return self.day.strftime("%Y%m%d")


@dataclass(frozen=True)
class Inventory:
    """一次扫描得到的完整数据清单。"""

    data_root: Path
    files: tuple[SourceFile, ...]

    def by_channel(self, channel_dir: str) -> tuple[SourceFile, ...]:
        return tuple(item for item in self.files if item.channel_dir == channel_dir)

    def days(self, channel_dir: str | None = None) -> tuple[date, ...]:
        items = self.files if channel_dir is None else self.by_channel(channel_dir)
        return tuple(sorted({item.day for item in items}))

    def total_bytes(self, channel_dir: str | None = None) -> int:
        items = self.files if channel_dir is None else self.by_channel(channel_dir)
        return sum(item.size for item in items)

    def summary(self) -> list[dict[str, object]]:
        """按通道目录汇总一行统计，供报告表格使用。"""
        rows: list[dict[str, object]] = []
        channels = sorted({item.channel_dir for item in self.files})
        for channel in channels:
            items = self.by_channel(channel)
            rows.append(
                {
                    "通道目录": channel,
                    "文件数": len(items),
                    "覆盖天数": len({item.day for item in items}),
                    "总大小 (MB)": round(sum(item.size for item in items) / 1024 / 1024, 1),
                    "起始日期": min(item.day for item in items).isoformat(),
                    "结束日期": max(item.day for item in items).isoformat(),
                }
            )
        return rows


def _parse_day(name: str) -> date | None:
    matched = DAY_PATTERN.match(name)
    if not matched:
        return None
    try:
        return datetime.strptime(matched.group("stamp"), "%Y%m%d").date()
    except ValueError:
        return None


def _truncation_hint(size: int) -> str | None:
    if size in SUSPICIOUS_SIZES:
        return f"文件大小恰为 {size} 字节，疑似写入被截断"
    return None


def discover(
    data_root: Path,
    *,
    channel_dirs: tuple[str, ...] = ("Temperature", "Wind", "Cn2"),
    days: tuple[str, ...] = (),
    max_files_per_day: int | None = None,
) -> Inventory:
    """扫描 `data_root` 下的日期目录。

    Args:
        data_root: 形如 ``data/daily_data`` 的根目录，其下为 ``YYYYMMDD`` 子目录。
        channel_dirs: 需要发现的通道子目录名。
        days: 仅保留这些日期（``YYYYMMDD``）；为空表示全部。
        max_files_per_day: 每个日期每个通道最多取多少个文件（用于快速试跑）。
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")

    wanted = set(days)
    files: list[SourceFile] = []
    for day_dir in sorted(data_root.iterdir()):
        if not day_dir.is_dir():
            continue
        day = _parse_day(day_dir.name)
        if day is None:
            continue
        if wanted and day_dir.name not in wanted:
            continue
        for channel in channel_dirs:
            channel_dir = day_dir / channel
            if not channel_dir.is_dir():
                continue
            found: list[SourceFile] = []
            for path in sorted(channel_dir.glob("*.csv")):
                matched = HOUR_PATTERN.match(path.name)
                if not matched:
                    continue
                stat = path.stat()
                found.append(
                    SourceFile(
                        path=path,
                        day=day,
                        channel_dir=channel,
                        index=int(matched.group("index")),
                        size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        truncated_hint=_truncation_hint(stat.st_size),
                    )
                )
            found.sort(key=lambda item: item.index)
            if max_files_per_day is not None:
                found = found[:max_files_per_day]
            files.extend(found)

    if not files:
        raise FileNotFoundError(f"{data_root} 下未发现可用的 CSV 文件")
    return Inventory(data_root=data_root, files=tuple(files))
