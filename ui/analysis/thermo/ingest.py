"""原始 CSV 解析：分块读取、宽表重构、按文件指纹缓存。

原始数据量约为每通道 2 GB/周，因此本模块：

* 以固定块大小流式读取，避免整周数据同时驻留内存；
* 每个文件只解析一次，结果（分钟级与分块级聚合）落入 Parquet 缓存；
* 缓存键包含源文件指纹与影响结果的配置，配置变更自动失效。
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

import pandas as pd
from loguru import logger

from .config import ThermoReportConfig
from .discovery import SourceFile
from .filters import jitter_frame
from .metrics import compute_block_metrics
from .quality import CadenceStats, cadence_stats, clean, cross_probe_outliers, probe_quality

DEFAULT_CHUNK_ROWS = 800_000
# 缓存结构版本：修改缓存内容的结构（列、单位、含义）时必须递增，
# 否则旧缓存会以“命中”的方式悄悄污染结果。
CACHE_SCHEMA = 5


@dataclass
class FileResult:
    """单个原始文件的解析结果（全部为长表，便于直接拼接）。"""

    source: SourceFile
    probes: tuple[str, ...]
    minute: pd.DataFrame
    block: pd.DataFrame
    spectrum: pd.DataFrame
    quality: pd.DataFrame
    cadence: dict[str, float]
    first_ts: pd.Timestamp | None
    last_ts: pd.Timestamp | None
    n_rows: int
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)


def _validate_header(path: Path) -> None:
    head = pd.read_csv(path, nrows=0)
    columns = [str(name).strip() for name in head.columns]
    if len(columns) != 3 or columns[0] != "ID" or columns[-1] != "Timestamp":
        raise ValueError(f"{path.name} 列名不符合预期: {columns}，应为 ['ID', <数值列>, 'Timestamp']")


def iter_wide_frames(
    path: Path,
    *,
    wanted: tuple[str, ...] = (),
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Iterator[pd.DataFrame]:
    """流式读取长表 CSV，逐块转成以时间为索引、探针为列的宽表。

    探针名统一为小写去空白。这里用向量化 ``str`` 方法而不是 ``Series.map``：
    逐元素回调在 800 万行/天的数据量下既慢，也更容易触发底层崩溃。
    """
    probe_col = 0
    pending: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False, dtype={probe_col: "string"}):
        chunk.columns = ["probe", "value", "ts"]
        chunk["probe"] = chunk["probe"].str.strip().str.lower()
        if wanted:
            chunk = chunk[chunk["probe"].isin(wanted)]
            if chunk.empty:
                continue
        # utc=True 让「带偏移」与「不带偏移」的行都能解析（否则同一文件混排会抛
        # ValueError: Mixed timezones detected）；再丢弃时区，退回本地墙钟时间。
        chunk["ts"] = pd.to_datetime(chunk["ts"], format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["ts"])

        if pending is not None and not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = None

        # 只有表头（或整块时间戳都无法解析）时没有可切分的末尾组，
        # 直接跳过：辅助通道里 r0_computed / aveCn2_computed 大量是仅表头的空文件。
        if chunk.empty:
            continue

        # 末尾时间戳可能是被切断的不完整组，留给下一块拼接
        last_ts = chunk["ts"].iloc[-1]
        complete = chunk[chunk["ts"] != last_ts]
        tail = chunk[chunk["ts"] == last_ts]
        if not complete.empty:
            yield _pivot(complete)
        pending = tail

    if pending is not None and not pending.empty:
        yield _pivot(pending)


def _pivot(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        wide = frame.pivot(index="ts", columns="probe", values="value")
    except ValueError:
        wide = frame.groupby(["ts", "probe"])["value"].mean().unstack("probe")
    wide.index.name = "ts"
    return wide.sort_index()


def read_wide(
    source: SourceFile,
    probes: tuple[str, ...] = (),
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> pd.DataFrame:
    """读取单个文件为完整宽表。"""
    frames = list(iter_wide_frames(source.path, wanted=probes, chunk_rows=chunk_rows))
    if not frames:
        return pd.DataFrame()
    wide = frames[0] if len(frames) == 1 else pd.concat(frames)
    return wide[~wide.index.duplicated(keep="first")].sort_index()


def minute_table(wide: pd.DataFrame, rule: str, nominal_dt_s: float) -> pd.DataFrame:
    """把宽表聚合为分钟级长表（含覆盖率）。"""
    if wide.empty:
        return pd.DataFrame(columns=["ts", "探针", "均值", "标准差", "最小", "最大", "采样数", "覆盖率"])
    expected = 60.0 / nominal_dt_s if nominal_dt_s > 0 else math.nan
    grouped = wide.resample(rule).agg(["mean", "std", "min", "max", "count"])
    stacked = grouped.stack(level=0, future_stack=True).reset_index()
    stacked.columns = ["ts", "探针", "均值", "标准差", "最小", "最大", "采样数"]
    stacked["采样数"] = stacked["采样数"].fillna(0).astype("int64")
    stacked["覆盖率"] = (stacked["采样数"] / expected).clip(upper=1.0) if expected else math.nan
    return stacked


def _cache_key(source: SourceFile, cfg: ThermoReportConfig) -> str:
    payload = {
        "schema": CACHE_SCHEMA,
        "fingerprint": source.fingerprint,
        "quality": cfg.quality.model_dump(mode="json"),
        "analysis": cfg.analysis.model_dump(mode="json"),
        "probes": cfg.probes.model_dump(mode="json"),
    }
    return sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def _cache_paths(cache_dir: Path, key: str) -> dict[str, Path]:
    base = cache_dir / key
    return {
        "base": base,
        "minute": base / "minute.parquet",
        "block": base / "block.parquet",
        "spectrum": base / "spectrum.parquet",
        "quality": base / "quality.parquet",
        "meta": base / "meta.json",
    }


def _load_from_cache(source: SourceFile, cfg: ThermoReportConfig) -> FileResult | None:
    paths = _cache_paths(cfg.paths.cache_dir, _cache_key(source, cfg))
    if not paths["meta"].exists():
        return None
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        return FileResult(
            source=source,
            probes=tuple(meta["probes"]),
            minute=pd.read_parquet(paths["minute"]),
            block=pd.read_parquet(paths["block"]),
            spectrum=pd.read_parquet(paths["spectrum"]),
            quality=pd.read_parquet(paths["quality"]),
            cadence=meta["cadence"],
            first_ts=pd.Timestamp(meta["first_ts"]) if meta.get("first_ts") else None,
            last_ts=pd.Timestamp(meta["last_ts"]) if meta.get("last_ts") else None,
            n_rows=int(meta["n_rows"]),
            from_cache=True,
            notes=list(meta.get("notes", [])),
        )
    except (OSError, ValueError, KeyError) as exc:  # 缓存损坏时退回重新解析
        logger.warning("缓存读取失败，重新解析 {}: {}", source.path.name, exc)
        shutil.rmtree(paths["base"], ignore_errors=True)
        return None


def _store_cache(result: FileResult, cfg: ThermoReportConfig) -> None:
    paths = _cache_paths(cfg.paths.cache_dir, _cache_key(result.source, cfg))
    try:
        paths["base"].mkdir(parents=True, exist_ok=True)
        result.minute.to_parquet(paths["minute"], index=False)
        result.block.to_parquet(paths["block"], index=False)
        result.spectrum.to_parquet(paths["spectrum"], index=False)
        result.quality.to_parquet(paths["quality"], index=False)
        meta = {
            "schema": CACHE_SCHEMA,
            "source": str(result.source.path),
            "probes": list(result.probes),
            "cadence": result.cadence,
            "first_ts": result.first_ts.isoformat() if result.first_ts is not None else None,
            "last_ts": result.last_ts.isoformat() if result.last_ts is not None else None,
            "n_rows": result.n_rows,
            "notes": result.notes,
        }
        paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # 缓存不可写不应中断分析
        logger.warning("缓存写入失败（忽略）: {}", exc)


def process_source(
    source: SourceFile,
    cfg: ThermoReportConfig,
    *,
    use_cache: bool = True,
) -> FileResult:
    """解析单个原始文件，得到分钟表、分块脉动表、质量表与采样节奏。"""
    if use_cache:
        cached = _load_from_cache(source, cfg)
        if cached is not None:
            return cached

    _validate_header(source.path)
    notes: list[str] = []
    if source.truncated_hint:
        notes.append(source.truncated_hint)

    chunks: list[pd.DataFrame] = []
    for frame in iter_wide_frames(source.path):
        if frame.empty:
            continue
        chunks.append(frame)

    if not chunks:
        empty = pd.DataFrame()
        return FileResult(
            source,
            (),
            empty,
            empty,
            empty,
            empty,
            CadenceStats(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0).as_row(),
            None,
            None,
            0,
            notes=notes,
        )

    wide = pd.concat(chunks).sort_index()
    wide = wide[~wide.index.duplicated(keep="first")]
    probes = tuple(str(col) for col in wide.columns)
    n_rows = int(len(wide))

    cadence = cadence_stats(wide.index)
    outliers = cross_probe_outliers(wide, cfg.quality)
    jitter = jitter_frame(wide, cfg.filtering, cadence.median_dt_s)
    cleaned = clean(wide, cfg.quality)
    if cfg.filtering.despike_enabled:
        cleaned = cleaned.mask(jitter)
    minute = minute_table(cleaned, cfg.analysis.minute_rule, cadence.median_dt_s)
    block, spectrum = compute_block_metrics(cleaned, cfg.analysis, cfg.quality)
    quality_rows = []
    for probe in probes:
        row = {"探针": probe}
        exclude = outliers.get(probe)
        probe_jitter = jitter.get(probe) if cfg.filtering.despike_enabled else None
        row.update(probe_quality(wide[probe], cfg.quality, cadence.median_dt_s, exclude=exclude, jitter=probe_jitter))
        quality_rows.append(row)
    quality = pd.DataFrame(quality_rows)

    if cadence.coverage < 0.95:
        notes.append(f"采样覆盖率 {cadence.coverage:.1%}，存在断记")

    result = FileResult(
        source=source,
        probes=probes,
        minute=minute,
        block=block,
        spectrum=spectrum,
        quality=quality,
        cadence=cadence.as_row(),
        first_ts=wide.index.min(),
        last_ts=wide.index.max(),
        n_rows=n_rows,
        notes=notes,
    )
    if use_cache:
        _store_cache(result, cfg)
    return result


def load_aux_minute_means(
    source: SourceFile,
    rule: str,
    *,
    probes: tuple[str, ...] = (),
    value_min: float = float("-inf"),
    value_max: float = 1e6,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """读取辅助通道（湿度、风速、Cn² 等）并聚合为分钟均值长表。

    辅助通道与温度共用同一长表结构；越出物理范围的值（异常码 1220.8、122.08，
    以及 Cn² 的 0 值）在聚合前剔除，避免污染均值。结果按文件指纹与取值范围缓存。
    """
    if cache_dir is not None:
        cached = _load_aux_cache(source, rule, probes, cache_dir, value_min, value_max)
        if cached is not None:
            return cached

    frames = []
    for frame in iter_wide_frames(source.path, wanted=probes):
        if frame.empty:
            continue
        frame = frame.where((frame >= value_min) & (frame <= value_max))
        frames.append(frame.resample(rule).mean())
    if not frames:
        return pd.DataFrame(columns=["ts", "探针", "数值"])
    merged = pd.concat(frames).groupby(level=0).mean()
    stacked = merged.stack(level=0, future_stack=True).reset_index()
    stacked.columns = ["ts", "探针", "数值"]
    if cache_dir is not None:
        _store_aux_cache(source, rule, probes, stacked, cache_dir, value_min, value_max)
    return stacked


def _bounds_tag(value_min: float, value_max: float) -> str:
    """把取值范围压成短标签，避免不同物理范围的通道共用缓存。"""
    return f"{value_min:g}_{value_max:g}".replace("-", "m").replace(".", "p").replace("+", "")


def _aux_cache_file(
    source: SourceFile,
    rule: str,
    probes: tuple[str, ...],
    cache_dir: Path,
    value_min: float = float("-inf"),
    value_max: float = 1e6,
) -> Path:
    # 目录名不能叫 aux：AUX 是 Windows 保留设备名，mkdir 会报 WinError 267。
    suffix = "all" if not probes else "-".join(sorted(probes))
    tag = _bounds_tag(value_min, value_max)
    return cache_dir / "auxiliary" / f"{source.fingerprint}_{rule}_{suffix}_{tag}.parquet"


def _load_aux_cache(
    source: SourceFile,
    rule: str,
    probes: tuple[str, ...],
    cache_dir: Path,
    value_min: float = float("-inf"),
    value_max: float = 1e6,
) -> pd.DataFrame | None:
    path = _aux_cache_file(source, rule, probes, cache_dir, value_min, value_max)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return None


def _store_aux_cache(
    source: SourceFile,
    rule: str,
    probes: tuple[str, ...],
    frame: pd.DataFrame,
    cache_dir: Path,
    value_min: float = float("-inf"),
    value_max: float = 1e6,
) -> None:
    path = _aux_cache_file(source, rule, probes, cache_dir, value_min, value_max)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    except OSError as exc:
        logger.warning("辅助数据缓存写入失败（忽略）: {}", exc)
