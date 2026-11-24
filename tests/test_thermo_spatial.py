"""跨探针空间平均（ui/analysis/thermo/spatial.py）测试。

覆盖：notebook 等间隔插值平均的数值等价性、退化路径、逐分钟序列与逐日汇总、
落盘、CLI/配置接线，以及从原始长表到 ``AnalysisResult.spatial`` 的管线集成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analysis.thermo.channels import ChannelResult
from analysis.thermo.cli import build_parser, config_from_args
from analysis.thermo.config import AnalysisConfig, PathsConfig, ProbeConfig, SpatialConfig, ThermoReportConfig
from analysis.thermo.pipeline import _spatial_tables, analyze
from analysis.thermo.spatial import (
    METHOD_INTERP,
    METHOD_MEAN,
    METHOD_MEDIAN,
    aggregate_across_probes,
    daily_spatial_summary,
    equal_interval_mean,
    even_heights,
    even_heights_from_text,
    heights_from_text,
    probe_number,
    resolve_heights,
    spatial_series,
    spatial_summary,
    write_spatial,
)

HZ = 10.0


# --------------------------------------------------------------------------- #
# 等间隔插值平均
# --------------------------------------------------------------------------- #
def test_equal_interval_mean_matches_notebook_formula():
    """positions 0/10/20、values 1/2/3：grid=arange(0,20,10)=[0,10]，均值 1.5。"""
    assert equal_interval_mean([1.0, 2.0, 3.0], [0.0, 10.0, 20.0]) == pytest.approx(1.5)


def test_equal_interval_mean_sorts_positions_and_skips_nan():
    values = [3.0, np.nan, 1.0, 2.0]
    positions = [20.0, 15.0, 0.0, 10.0]
    assert equal_interval_mean(values, positions) == pytest.approx(1.5)


def test_equal_interval_mean_degenerate_paths():
    assert equal_interval_mean([7.0], [1.0]) == pytest.approx(7.0)
    assert equal_interval_mean([], []) != equal_interval_mean([], [])  # 空输入为 NaN
    # 高度全部相同：退化为算术平均
    assert equal_interval_mean([1.0, 3.0], [5.0, 5.0]) == pytest.approx(2.0)


def test_aggregate_across_probes():
    assert aggregate_across_probes([1.0, 2.0, np.nan], "median") == pytest.approx(1.5)
    assert aggregate_across_probes([1.0, 2.0], "mean") == pytest.approx(1.5)
    assert np.isnan(aggregate_across_probes([], "median"))


# --------------------------------------------------------------------------- #
# 逐分钟序列与逐日汇总
# --------------------------------------------------------------------------- #
def _minute_frame() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-10-14 00:00", "2026-10-14 00:00", "2026-10-14 00:01", "2026-10-14 00:01"])
    return pd.DataFrame(
        {
            "ts": ts,
            "探针": ["tp-1", "tp-2", "tp-1", "tp-2"],
            "均值": [20.0, 21.0, 22.0, 23.0],
        }
    )


def test_spatial_series_uses_heights_when_enough_probes():
    frame = _minute_frame()
    frame = pd.concat(
        [frame, frame.assign(探针="tp-3", 均值=frame["均值"] + 2.0)],
        ignore_index=True,
    )
    series = spatial_series(
        frame,
        "均值",
        {"tp-1": 1.0, "tp-2": 2.0, "tp-3": 3.0},
        min_probes=3,
    )
    assert list(series.columns) == ["ts", "空间平均", "探针数", "方式"]
    assert set(series["方式"]) == {METHOD_INTERP}
    # gap = 1 → grid [1, 2]，取 tp-1 与 tp-2 的平均
    assert series["空间平均"].tolist() == pytest.approx([20.5, 22.5])
    assert series["探针数"].tolist() == [3, 3]


def test_spatial_series_falls_back_without_heights():
    frame = _minute_frame()
    series = spatial_series(frame, "均值", None, method="mean")
    assert set(series["方式"]) == {METHOD_MEAN}
    assert series["空间平均"].tolist() == pytest.approx([20.5, 22.5])

    median = spatial_series(frame, "均值", {}, method="median")
    assert set(median["方式"]) == {METHOD_MEDIAN}


def test_spatial_series_empty_and_missing_column():
    assert spatial_series(pd.DataFrame(), "均值").empty
    assert spatial_series(_minute_frame(), "不存在").empty


def test_daily_spatial_summary_groups_by_day():
    series = spatial_series(
        pd.concat(
            [_minute_frame(), _minute_frame().assign(ts=lambda df: df["ts"] + pd.Timedelta(days=1))],
            ignore_index=True,
        ),
        "均值",
        None,
    )
    daily = daily_spatial_summary(series, "温度", "°C")
    assert list(daily.columns) == ["通道", "单位", "日期", "日均值", "最低", "最高", "有效分钟", "平均探针数", "方式"]
    assert daily["日期"].tolist() == ["2026-10-14", "2026-10-15"]
    assert daily["有效分钟"].tolist() == [2, 2]
    assert daily["通道"].unique().tolist() == ["温度"]


def test_spatial_summary_and_write(tmp_path: Path):
    series = spatial_series(_minute_frame(), "均值", None)
    series.insert(0, "单位", "°C")
    series.insert(0, "通道", "温度")
    overview = spatial_summary(series)
    assert overview.loc[0, "通道"] == "温度"
    assert overview.loc[0, "有效分钟"] == 2

    written = write_spatial(series, daily_spatial_summary(series, "温度", "°C"), tmp_path)
    assert written["spatial_mean"] == "spatial/spatial_mean.csv"
    assert (tmp_path / written["spatial_mean"]).exists()
    assert (tmp_path / written["spatial_daily"]).exists()

    empty = write_spatial(pd.DataFrame(), pd.DataFrame(), tmp_path / "empty")
    assert empty == {"spatial_mean": "", "spatial_daily": ""}


def test_heights_from_text():
    assert heights_from_text("") == {}
    assert heights_from_text("tp-1=1.5, tp-2=3") == {"tp-1": 1.5, "tp-2": 3.0}
    with pytest.raises(ValueError, match="tp-1"):
        heights_from_text("tp-1")


# --------------------------------------------------------------------------- #
# 管线与 CLI 接线
# --------------------------------------------------------------------------- #
def _write_day(root: Path, day: str = "20261014", probes: tuple[str, ...] = ("tp-1", "tp-2", "tp-3")) -> None:
    channel = root / day / "Temperature"
    channel.mkdir(parents=True, exist_ok=True)
    n = int(4 * 60 * HZ)
    start = pd.Timestamp("2026-10-14 00:00:00")
    stamps = start + pd.to_timedelta(np.arange(n) / HZ, unit="s")
    rng = np.random.default_rng(7)
    frames = []
    for index, probe in enumerate(probes):
        values = np.round((20.0 + 0.5 * index + rng.normal(0.0, 0.05, n)) / 0.1) * 0.1
        frames.append(
            pd.DataFrame({"ID": probe, "Temperature": values, "Timestamp": stamps.strftime("%Y-%m-%d %H:%M:%S.%f")})
        )
    pd.concat(frames).sort_values("Timestamp", kind="stable").to_csv(channel / "temperature_00.csv", index=False)


def _config(
    root: Path, tmp_path: Path, heights: dict[str, float] | None = None, **spatial_kwargs
) -> ThermoReportConfig:
    return ThermoReportConfig(
        paths=PathsConfig(data_root=root, cache_dir=tmp_path / "cache", output_dir=tmp_path / "reports"),
        probes=ProbeConfig(heights_m=heights or {}),
        analysis=AnalysisConfig(),
        spatial=SpatialConfig(**spatial_kwargs) if spatial_kwargs else SpatialConfig(),
    )


def test_analyze_produces_spatial_with_heights(tmp_path: Path):
    root = tmp_path / "daily"
    _write_day(root)
    cfg = _config(root, tmp_path, heights={"tp-1": 1.0, "tp-2": 2.0, "tp-3": 3.0})
    result = analyze(cfg, with_auxiliary=False)

    assert not result.spatial.empty
    assert set(result.spatial["方式"]) == {METHOD_INTERP}
    assert not result.spatial_daily.empty
    assert result.facts["空间平均方式"] == METHOD_INTERP
    assert result.facts["空间平均通道数"] == 1
    assert not any("未配置探针安装高度" in note for note in result.notes)


def test_analyze_falls_back_without_heights(tmp_path: Path):
    root = tmp_path / "daily"
    _write_day(root)
    cfg = _config(root, tmp_path)
    result = analyze(cfg, with_auxiliary=False)

    assert set(result.spatial["方式"]) == {METHOD_MEDIAN}
    assert any("未配置探针安装高度" in note for note in result.notes)


def test_analyze_spatial_disabled(tmp_path: Path):
    root = tmp_path / "daily"
    _write_day(root)
    cfg = _config(root, tmp_path, heights={"tp-1": 1.0, "tp-2": 2.0, "tp-3": 3.0}, enabled=False)
    result = analyze(cfg, with_auxiliary=False)
    assert result.spatial.empty
    assert result.spatial_daily.empty


def test_spatial_tables_handles_channel_failure(monkeypatch):
    from analysis.thermo import spatial as spatial_module

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(spatial_module, "spatial_series", boom)
    cfg = ThermoReportConfig(spatial=SpatialConfig())
    minute = _minute_frame()
    channel = ChannelResult(key="humidity", label="相对湿度", unit="%", minute=minute.rename(columns={"均值": "数值"}))
    frame, daily = _spatial_tables(cfg, minute, {"humidity": channel})
    assert frame.empty and daily.empty


def test_cli_wires_spatial_options():
    parser = build_parser()
    args = parser.parse_args(["--probe-heights", "tp-1=1.5,tp-2=2.5", "--spatial-method", "mean"])
    cfg = config_from_args(args)
    assert cfg.probes.heights_m == {"tp-1": 1.5, "tp-2": 2.5}
    assert cfg.spatial.method == "mean"
    assert cfg.spatial.enabled

    off = config_from_args(parser.parse_args(["--no-spatial"]))
    assert not off.spatial.enabled
    assert off.probes.heights_m == {}


def test_probe_number_and_even_heights():
    assert probe_number("tp-7") == 7
    assert probe_number("tp-x") is None
    heights = even_heights(["tp-3", "tp-1", "tp-2"], 5.0, 2.0)
    assert heights == {"tp-1": 5.0, "tp-2": 7.0, "tp-3": 9.0}
    with pytest.raises(ValueError):
        even_heights(["tp-1"], 5.0, 0.0)


def test_even_heights_from_text_validates():
    assert even_heights_from_text("5, 2") == (5.0, 2.0)
    assert even_heights_from_text("5:2") == (5.0, 2.0)
    for text in ("", "5", "a,b", "5,0", "5,-2"):
        with pytest.raises(ValueError):
            even_heights_from_text(text)


def test_resolve_heights_priority():
    explicit, source = resolve_heights(["tp-1", "tp-2"], {"tp-1": 100.0}, base_m=5.0, step_m=2.0)
    assert explicit == {"tp-1": 100.0}
    assert "显式配置" in source

    generated, source = resolve_heights(["tp-1", "tp-2", "tp-3"], None, base_m=5.0, step_m=2.0)
    assert generated == {"tp-1": 5.0, "tp-2": 7.0, "tp-3": 9.0}
    assert "等间距假设" in source

    empty, source = resolve_heights(["tp-1"], None)
    assert empty == {} and source == ""

    custom, source = resolve_heights(["tp-1"], None, base_m=5.0, step_m=2.0, source="自定义假设")
    assert custom == {"tp-1": 5.0} and source == "自定义假设"


def test_analyze_even_heights_assumption(tmp_path: Path):
    root = tmp_path / "daily"
    _write_day(root)
    cfg = _config(root, tmp_path)
    cfg = cfg.model_copy(
        update={"probes": cfg.probes.model_copy(update={"heights_even_base_m": 5.0, "heights_even_step_m": 2.0})}
    )
    result = analyze(cfg, with_auxiliary=False)

    assert set(result.spatial["方式"]) == {METHOD_INTERP}
    assert result.spatial_heights == {"tp-1": 5.0, "tp-2": 7.0, "tp-3": 9.0}
    assert "等间距假设" in result.spatial_heights_source
    assert any("假设值而非现场测量" in note for note in result.notes)


def test_analyze_explicit_heights_override_even(tmp_path: Path):
    root = tmp_path / "daily"
    _write_day(root)
    cfg = _config(root, tmp_path, heights={"tp-1": 1.0, "tp-2": 2.0, "tp-3": 3.0})
    cfg = cfg.model_copy(
        update={"probes": cfg.probes.model_copy(update={"heights_even_base_m": 5.0, "heights_even_step_m": 2.0})}
    )
    result = analyze(cfg, with_auxiliary=False)

    assert result.spatial_heights == {"tp-1": 1.0, "tp-2": 2.0, "tp-3": 3.0}
    assert not any("假设值而非现场测量" in note for note in result.notes)


def test_cli_wires_even_heights():
    parser = build_parser()
    cfg = config_from_args(parser.parse_args(["--probe-heights-even", "5,2"]))
    assert cfg.probes.heights_even_base_m == 5.0
    assert cfg.probes.heights_even_step_m == 2.0
    assert cfg.probes.heights_m == {}
    assert "等间距假设" in cfg.probes.heights_source

    both = config_from_args(parser.parse_args(["--probe-heights", "tp-1=1.5", "--probe-heights-even", "5,2"]))
    assert both.probes.heights_m == {"tp-1": 1.5}
    assert both.probes.heights_even_base_m is None

    with pytest.raises(ValueError):
        config_from_args(parser.parse_args(["--probe-heights-even", "5"]))
