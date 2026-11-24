"""温度脉动仪分析链路测试（ui/analysis/thermo/）。

覆盖：无效值剔除、量化步长、卡值判据、采样节奏、缓存复用、谱斜率与结构函数
标定、短时段保护，以及从原始 CSV 到报告产物的端到端流程。

为避开本机 matplotlib 渲染偶发崩溃，端到端测试只调用 ``write_report``，
不触发绑图（图表由真实运行时生成，见 docs/温度脉动仪数据报告.md）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analysis.thermo.aggregate import (
    anomaly_correlation,
    channel_cross_correlation,
    diurnal_profile,
    strongest_correlation,
)
from analysis.thermo.channels import _quality_notes, slice_table
from analysis.thermo.config import (
    AnalysisConfig,
    ChannelConfig,
    FilterConfig,
    PathsConfig,
    QualityConfig,
    ThermoReportConfig,
)
from analysis.thermo.discovery import discover
from analysis.thermo.filters import filter_frame, jitter_frame, jitter_mask
from analysis.thermo.ingest import load_aux_minute_means, process_source, read_wide
from analysis.thermo.metrics import (
    cn2_from_ct2,
    ct2_from_structure,
    fit_spectral_slope,
    quantization_noise,
    structure_noise_floor,
    structure_slope,
)
from analysis.thermo.pipeline import analyze
from analysis.thermo.quality import (
    clean,
    cross_probe_outliers,
    estimate_quantization_step,
    invalid_mask,
    probe_quality,
)
from analysis.thermo.report import write_report
from analysis.thermo.slices import SLICE_COLUMNS, write_slices

SENTINEL = 1220.8
HZ = 10.0


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
def write_day(
    root: Path,
    day: str = "20261014",
    probes: tuple[str, ...] = ("tp-1", "tp-2"),
    minutes: int = 20,
    sentinel_probes: tuple[str, ...] = (),
) -> Path:
    """写一个与真实采集软件同格式的长表 CSV（ID,<通道>,Timestamp）。"""
    channel = root / day / "Temperature"
    channel.mkdir(parents=True, exist_ok=True)
    n = int(minutes * 60 * HZ)
    start = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]} 00:00:07.976783")
    stamps = start + pd.to_timedelta(np.arange(n) / HZ, unit="s")
    rng = np.random.default_rng(20261014)
    seconds = np.arange(n) / HZ

    frames = []
    for index, probe in enumerate(probes):
        if probe in sentinel_probes:
            values = np.full(n, SENTINEL)
        else:
            signal = 20.0 + 0.5 * index + 2.0 * np.sin(2 * np.pi * seconds / 3600.0) + rng.normal(0.0, 0.05, n)
            values = np.round(signal / 0.1) * 0.1
        frames.append(
            pd.DataFrame(
                {
                    "ID": probe,
                    "Temperature": values,
                    "Timestamp": stamps.strftime("%Y-%m-%d %H:%M:%S.%f"),
                }
            )
        )
    table = pd.concat(frames).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    path = channel / "temperature_00.csv"
    table.to_csv(path, index=False)
    return path


def make_config(root: Path, tmp_path: Path, **analysis_overrides) -> ThermoReportConfig:
    return ThermoReportConfig(
        paths=PathsConfig(
            data_root=root,
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "reports",
        ),
        analysis=AnalysisConfig(**analysis_overrides),
    )


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "daily"
    write_day(root, sentinel_probes=("tp-9",), probes=("tp-1", "tp-2", "tp-9"))
    return root


@pytest.fixture()
def cfg(data_root: Path, tmp_path: Path) -> ThermoReportConfig:
    return make_config(data_root, tmp_path)


# --------------------------------------------------------------------------- #
# 质量判定
# --------------------------------------------------------------------------- #
def test_invalid_mask_covers_sentinel_and_range():
    cfg = QualityConfig()
    values = pd.Series([20.0, SENTINEL, 122.08, -999.9, -80.5, 19.9, np.nan])
    mask = invalid_mask(values, cfg)
    assert mask.tolist() == [False, True, True, True, True, False, True]


def test_clean_returns_nan_for_invalid():
    frame = pd.DataFrame(
        {
            "tp-1": [20.0, SENTINEL, 21.0],
            "tp-2": [20.1, 20.0, 21.1],
            "tp-3": [19.9, 20.2, 20.9],
        }
    )
    cleaned = clean(frame, QualityConfig())
    assert cleaned["tp-1"].isna().tolist() == [False, True, False]


def test_cross_probe_outlier_masking():
    """同址探针出现 0 °C 掉线：单序列看是平稳段，必须靠横向比对抓出。"""
    index = pd.date_range("2026-10-14", periods=12, freq="100ms")
    frame = pd.DataFrame(
        {
            "tp-1": np.full(12, 19.6),
            "tp-2": np.full(12, 19.7),
            "tp-16": np.r_[np.full(3, 19.6), np.zeros(6), np.full(3, 19.6)],
        },
        index=index,
    )
    outliers = cross_probe_outliers(frame, QualityConfig())
    assert int(outliers["tp-16"].sum()) == 6
    assert not outliers[["tp-1", "tp-2"]].to_numpy().any()

    cleaned = clean(frame, QualityConfig())
    assert cleaned["tp-16"].isna().sum() == 6

    quality = probe_quality(frame["tp-16"], QualityConfig(), exclude=outliers["tp-16"])
    assert quality["离群采样"] == 6
    # 掉线值不再定义极端值
    assert quality["最小值 (C)"] == pytest.approx(19.6)
    # 关闭判据后恢复原样
    loose = QualityConfig(cross_probe_max_dev_c=0.0)
    assert int(cross_probe_outliers(frame, loose).to_numpy().sum()) == 0


def test_quantization_step_detected():
    values = pd.Series(np.round(np.arange(0, 50, 0.1) / 0.1) * 0.1)
    assert estimate_quantization_step(values) == pytest.approx(0.1, abs=1e-9)
    assert quantization_noise(0.1) == pytest.approx(0.0289, abs=1e-3)


def test_stuck_mask_uses_duration_not_sample_count():
    """0.1 °C 量化下短时间读数不变属正常，只有持续 ≥10 min 才算卡值。"""
    index = pd.date_range("2026-10-14", periods=12000, freq="100ms")  # 1200 s
    warm = pd.Series(20.0, index=index)  # 全程不变：1200 s > 600 s
    quality = probe_quality(warm, QualityConfig())
    assert quality["卡值采样"] > 0
    assert quality["最长卡值 (min)"] == pytest.approx(20.0, abs=0.05)

    # 只在前 60 s 保持不变（600 个采样点），不应判为卡值
    varied = pd.Series(np.r_[np.full(600, 20.0), np.round(20.0 + np.arange(11400) * 0.01, 2)], index=index)
    assert probe_quality(varied, QualityConfig())["卡值采样"] == 0


def test_probe_quality_counts_sentinel_separately():
    index = pd.date_range("2026-10-14", periods=100, freq="1s")
    series = pd.Series([SENTINEL] * 40 + [20.0] * 60, index=index)
    quality = probe_quality(series, QualityConfig(), nominal_dt_s=1.0)
    assert quality["异常码采样"] == 40
    assert quality["有效采样"] == 60
    assert quality["有效率"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# 湍流统计
# --------------------------------------------------------------------------- #
def test_spectral_slope_recovers_kolmogorov():
    """合成 PSD ∝ f^(-5/3) 的序列，拟合应回到 -5/3。"""
    rng = np.random.default_rng(7)
    n, dt = 8192, 0.1
    freq = np.fft.rfftfreq(n, d=dt)
    amplitude = np.ones_like(freq)
    amplitude[1:] = freq[1:] ** (-5.0 / 6.0)  # |A|² ∝ f^(-5/3)
    phase = rng.uniform(0.0, 2 * np.pi, freq.size)
    series = np.fft.irfft(amplitude * np.exp(1j * phase), n=n)
    series = series / series.std()

    slope, r2 = fit_spectral_slope(np.fft.rfftfreq(n, d=dt), np.abs(np.fft.rfft(series)) ** 2 / n, (0.05, 1.0))
    assert slope == pytest.approx(-5.0 / 3.0, abs=0.12)
    assert r2 > 0.95


def test_structure_slope_power_law():
    d_tau = {tau: 2.0 * tau ** (2.0 / 3.0) for tau in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)}
    assert structure_slope(d_tau, (2.0, 16.0)) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_structure_noise_floor_prevents_negative_correction():
    """噪声底必须取自实测（τ→0 截距），硬减 2σ_q² 会把整条曲线压成负数。"""
    sigma_q = quantization_noise(0.1)  # 理论 2σ_q² ≈ 1.67e-3 °C²
    floor_true = 6.0e-4
    d_tau = {tau: floor_true + 4.0e-4 * tau ** (2.0 / 3.0) for tau in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)}

    floor = structure_noise_floor(d_tau, sigma_q)
    assert floor == pytest.approx(floor_true, rel=0.05)
    assert floor < 2.0 * sigma_q**2

    # 用实测噪声底校正：斜率仍还原 2/3，且没有点被压成负数
    assert structure_slope(d_tau, (2.0, 16.0), noise_floor=floor) == pytest.approx(2.0 / 3.0, abs=1e-6)
    # 硬减理论 2σ_q² 会把 2 s、4 s 压成负数，只剩两点，斜率严重偏大（图上只剩两个点的原因）
    overfit = structure_slope(d_tau, (2.0, 16.0), noise_floor=2.0 * sigma_q**2)
    assert not np.isnan(overfit) and abs(overfit - 2.0 / 3.0) > 0.3

    def ct2(floor_value):
        return ct2_from_structure(d_tau, wind_speed=2.0, fit_range=(2.0, 16.0), noise_floor=floor_value)

    assert ct2(floor) == pytest.approx(4.0e-4 / 2.0 ** (2.0 / 3.0), rel=0.05)
    # 校正面 ≤0 的点被丢弃后仍能量级一致；过校正则明显偏小
    assert ct2(0.0) > ct2(floor) > ct2(2.0 * sigma_q**2)


def test_ct2_and_cn2_relations():
    ct2 = ct2_from_structure(
        {2.0: 1e-3 * 2.0 ** (2 / 3), 4.0: 1e-3 * 4.0 ** (2 / 3)}, wind_speed=1.0, fit_range=(2.0, 4.0)
    )
    assert ct2 == pytest.approx(1e-3, rel=1e-6)
    # 风速缺失时无法按 Taylor 冻结假设反演
    assert np.isnan(ct2_from_structure({2.0: 1e-3}, wind_speed=float("nan"), fit_range=(2.0, 4.0)))

    pressure, temp_c = 1013.25, 20.0
    index_term = 79e-6 * pressure / (temp_c + 273.15) ** 2
    assert cn2_from_ct2(1e-3, temp_c, pressure) == pytest.approx(index_term**2 * 1e-3, rel=1e-9)


def test_short_span_protections():
    """20 min 数据不应给出日变化节律，相关分析也应跳过。"""
    index = pd.date_range("2026-10-14", periods=20, freq="1min")
    minute = pd.DataFrame(
        {
            "ts": np.tile(index, 2),
            "探针": ["tp-1"] * 20 + ["tp-2"] * 20,
            "均值": np.r_[np.linspace(20, 21, 20), np.linspace(21, 22, 20)],
        }
    )
    assert diurnal_profile(minute).empty
    correlation, order = anomaly_correlation(minute, window_minutes=15)
    assert correlation.empty and order == []


# --------------------------------------------------------------------------- #
# 解析与缓存
# --------------------------------------------------------------------------- #
def test_process_source_and_cache_reuse(cfg: ThermoReportConfig, data_root: Path):
    source = discover(data_root, days=("20261014",)).by_channel("Temperature")[0]
    first = process_source(source, cfg, use_cache=True)
    assert not first.from_cache
    assert first.minute["探针"].nunique() == 3
    # 20 min 的数据从 00:00:07.98 起，按分钟向下取整得到 21 个分钟桶（首尾各为不完整桶）
    assert first.minute["ts"].nunique() == 21
    counts = first.minute.groupby("探针")["均值"].count()
    assert counts.loc[["tp-1", "tp-2"]].eq(21).all()
    # 全时段哨兵值的探针保留分钟行但不产生有效均值
    assert counts.loc["tp-9"] == 0
    assert first.quality.set_index("探针").loc["tp-9", "有效率"] == 0.0

    second = process_source(source, cfg, use_cache=True)
    assert second.from_cache
    pd.testing.assert_frame_equal(first.minute, second.minute)
    pd.testing.assert_frame_equal(first.quality, second.quality)


def test_read_wide_masks_nothing_but_keeps_sentinel(cfg: ThermoReportConfig, data_root: Path):
    """原始读取保留异常码，清洗交给 quality.clean（便于统计异常码比例）。"""
    source = discover(data_root, days=("20261014",)).by_channel("Temperature")[0]
    wide = read_wide(source)
    assert wide["tp-9"].eq(SENTINEL).all()
    assert not wide["tp-1"].isna().any()


def test_header_only_aux_file_yields_empty_frame(tmp_path: Path):
    """r0_computed / aveCn2_computed 大量只有表头：应安静返回空表，而不是抛 IndexError。"""
    channel = tmp_path / "20261014" / "r0_computed"
    channel.mkdir(parents=True)
    (channel / "r0_computed_00.csv").write_text("ID,r0,Timestamp\n", encoding="utf-8")
    source = discover(tmp_path, days=("20261014",), channel_dirs=("r0_computed",)).by_channel("r0_computed")[0]
    frame = load_aux_minute_means(source, "1min", value_min=0.0, value_max=5.0)
    assert frame.empty
    assert list(frame.columns) == ["ts", "探针", "数值"]


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def test_end_to_end_report(cfg: ThermoReportConfig, tmp_path: Path):
    result = analyze(cfg, days=("20261014",), use_cache=False, with_auxiliary=False)
    assert result.probes == ("tp-1", "tp-2", "tp-9")
    assert result.valid_probes == ("tp-1", "tp-2")
    assert result.facts["探针数"] == 3

    out = write_report(result, tmp_path / "report", {})
    for name in ("report.html", "summary.md", "metrics.json", "manifest.json"):
        assert (out / name).is_file(), name
    assert list((out / "tables").glob("*.csv"))

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "温度脉动仪数据报告" in summary
    assert "日变化幅度约" not in summary  # 20 min 时段不下日变化结论
    assert "tp-9" in summary

    html = (out / "report.html").read_text(encoding="utf-8")
    assert "温度脉动仪数据报告" in html
    assert "<img" not in html or "base64" in html  # 不引用外部图片

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["analysis"]["site_pressure_hpa"] == pytest.approx(1013.25)


def test_report_metrics_json_is_strict_json(cfg: ThermoReportConfig, tmp_path: Path):
    result = analyze(cfg, days=("20261014",), use_cache=False, with_auxiliary=False)
    out = write_report(result, tmp_path / "report2", {})
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert isinstance(metrics, dict) and metrics
    json.dumps(metrics, allow_nan=False)  # 不允许 NaN/Infinity 泄漏到 JSON


# --------------------------------------------------------------------------- #
# 抖动剔除、通道与时间片导出
# --------------------------------------------------------------------------- #
def write_channel_day(
    root: Path,
    subdir: str,
    column: str,
    *,
    day: str = "20261014",
    probes: tuple[str, ...] = ("tp-1", "tp-2"),
    minutes: int = 20,
    value: float = 55.0,
    spike: tuple[int, float] | None = None,
) -> Path:
    """写一路环境要素长表（与温度同结构）；spike=(行号, 值) 用于注入单点跳变。"""
    directory = root / day / subdir
    directory.mkdir(parents=True, exist_ok=True)
    n = int(minutes * 60 * HZ)
    start = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]} 00:00:07.976783")
    stamps = start + pd.to_timedelta(np.arange(n) / HZ, unit="s")
    rng = np.random.default_rng(7)
    frames = []
    for probe in probes:
        values = value + rng.normal(0.0, 0.2, n)
        if spike is not None:
            row, injected = spike
            values[row] = injected
        frames.append(
            pd.DataFrame(
                {
                    "ID": probe,
                    column: values,
                    "Timestamp": stamps.strftime("%Y-%m-%d %H:%M:%S.%f"),
                }
            )
        )
    table = pd.concat(frames).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    path = directory / f"{subdir.lower()}_00.csv"
    table.to_csv(path, index=False)
    return path


def test_jitter_mask_flags_only_the_spike():
    index = pd.date_range("2026-10-14", periods=300, freq="100ms")
    values = pd.Series(20.0 + np.random.default_rng(1).normal(0, 0.03, 300), index=index)
    values.iloc[150] = 27.5
    mask = jitter_mask(values, dt=0.1, window_s=5.0, threshold=1.0)
    assert bool(mask.iloc[150]) is True
    assert int(mask.sum()) == 1

    frame = pd.DataFrame({"tp-1": values, "tp-2": values.shift(1).bfill()})
    cleaned, masks = filter_frame(frame, FilterConfig(), dt=0.1)
    assert cleaned["tp-1"].isna().sum() >= 1
    assert bool(masks["tp-1"].iloc[150]) is True

    # 关闭抖动剔除后不产生任何掩码
    assert int(jitter_frame(frame, FilterConfig(despike_enabled=False), 0.1).to_numpy().sum()) == 0


def test_probe_quality_counts_jitter():
    index = pd.date_range("2026-10-14", periods=200, freq="100ms")
    values = pd.Series(20.0, index=index)
    values.iloc[100] = 24.0
    mask = jitter_mask(values, dt=0.1, window_s=5.0, threshold=1.0)
    quality = probe_quality(values, QualityConfig(), nominal_dt_s=0.1, jitter=mask)
    assert quality["抖动采样"] == 1
    assert quality["最大值 (C)"] == pytest.approx(20.0)


def test_slice_table_groups_by_rule():
    minute = pd.DataFrame(
        {
            "ts": list(pd.date_range("2026-10-14 00:00", periods=30, freq="1min")) * 2,
            "探针": ["tp-1"] * 30 + ["tp-2"] * 30,
            "数值": list(np.linspace(50, 60, 30)) * 2,
        }
    )
    table = slice_table(minute, "10min", "相对湿度", "%", smooth_window=3)
    assert list(table.columns) == [
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
    ]
    assert len(table) == 6  # 3 个时间片 × 2 个探针
    assert table["时间片"].nunique() == 3
    assert set(table["通道"]) == {"相对湿度"}
    assert table["有效采样"].max() == 10
    assert table["平滑均值"].notna().all()


def test_write_slices_creates_files(tmp_path: Path, cfg: ThermoReportConfig):
    stamps = pd.to_datetime(["2026-10-14 00:00", "2026-10-14 00:10", "2026-10-14 00:20"])
    frame = pd.DataFrame(
        {
            "通道": "温度",
            "单位": "°C",
            "探针": "tp-1",
            "时间片": stamps,
            "均值": [20.1, 20.2, 20.3],
        }
    )
    out = tmp_path / "report"
    manifest = write_slices(frame, out, cfg)
    assert manifest["文件数"] == 3
    files = sorted((out / "slices").glob("20261014/*.csv"))
    assert [path.stem for path in files] == ["0000", "0010", "0020"]
    assert (out / "slices" / "all_slices.parquet").is_file()
    assert list(pd.read_csv(files[0]).columns)[:5] == SLICE_COLUMNS[:5]


def test_channels_and_slice_export_end_to_end(tmp_path: Path):
    root = tmp_path / "daily"
    write_day(root, probes=("tp-1", "tp-2"))
    write_channel_day(root, "Humidity", "Humidity", value=55.0)
    write_channel_day(root, "Visibility", "Visibility", value=18.0)
    cfg = make_config(root, tmp_path)

    result = analyze(cfg, days=("20261014",), use_cache=False, with_auxiliary=True)
    assert "humidity" in result.channels and not result.channels["humidity"].minute.empty
    assert result.facts["相对湿度中位数"] == pytest.approx(55.0, abs=1.0)
    assert not result.slices.empty
    assert {"温度", "相对湿度", "能见度"} <= set(result.slices["通道"])
    assert set(SLICE_COLUMNS) <= set(result.slices.columns)

    out = tmp_path / "report3"
    manifest = write_slices(result.slices, out, cfg)
    assert manifest["文件数"] >= 2
    assert manifest["片数"] == result.slices["时间片"].nunique()
    sample = pd.read_csv(next(iter(sorted((out / "slices").glob("*/[0-9]*.csv")))))
    assert "C_T2" in sample.columns and "平滑均值" in sample.columns


def test_quality_notes_flags_ceiling_and_zero_share():
    """量程封顶按探针自身上限判、大量为 0 要提示，缺测探针不得误报封顶。"""
    minute = 200
    ts = list(pd.date_range("2026-10-14 00:00", periods=minute, freq="1min"))

    # tp-1：140 分钟正常，60 分钟贴在自身上限 75（封顶）
    tp1 = [18.0 + (i % 7) * 0.1 for i in range(140)] + [75.0] * 60
    # tp-2：94% 为 0（间歇失效）
    tp2 = [75.0] * 6 + [0.0] * (minute - 6)
    # tp-3：只有 30 个有效分钟（缺测），恰好贴在自身上限，不得报封顶
    tp3 = [40.0] * 30 + [float("nan")] * (minute - 30)

    frame = pd.DataFrame(
        {
            "探针": (["tp-1"] * minute) + (["tp-2"] * minute) + (["tp-3"] * minute),
            "数值": tp1 + tp2 + tp3,
            "ts": ts * 3,
        }
    )
    channel = ChannelConfig(
        key="visibility",
        subdir="visibility",
        label="能见度",
        unit="km",
        valid_min=0.0,
        valid_max=100.0,
    )
    notes = _quality_notes(frame, channel, ThermoReportConfig())

    joined = "\n".join(notes)
    assert "tp-1" in joined and "量程封顶" in joined and "75 km" in joined
    assert "tp-2" in joined and "为 0" in joined and "未接入" in joined
    # 缺测探针可以出现在覆盖率提示里，但不得出现在量程封顶提示中
    assert all("tp-3" not in note for note in notes if "量程封顶" in note)


def test_channel_cross_correlation_aligns_minutes():
    stamps = pd.date_range("2026-10-09", periods=60, freq="1min")
    rising = np.linspace(10.0, 20.0, stamps.size)
    idle = np.zeros(stamps.size)
    frames = {
        "温度": pd.DataFrame({"ts": stamps, "探针": "tp-1", "数值": rising}),
        "湿度": pd.DataFrame({"ts": stamps, "探针": "tp-1", "数值": 2 * rising}),
        "能见度": pd.DataFrame({"ts": stamps, "探针": "tp-1", "数值": rising[::-1]}),
        # 全程为 0 的失效探针不应参与通道中位数
        "风速": pd.concat(
            [
                pd.DataFrame({"ts": stamps, "探针": "tp-1", "数值": rising}),
                pd.DataFrame({"ts": stamps, "探针": "tp-2", "数值": idle}),
            ],
            ignore_index=True,
        ),
    }
    matrix = channel_cross_correlation(frames)

    assert list(matrix["通道"]) == ["温度", "湿度", "能见度", "风速"]
    assert matrix.loc[0, "湿度"] == pytest.approx(1.0)
    assert matrix.loc[0, "能见度"] == pytest.approx(-1.0)
    assert matrix.loc[0, "风速"] == pytest.approx(1.0)
    best = strongest_correlation(matrix)
    assert best is not None and best[0] == "温度" and abs(best[2]) == pytest.approx(1.0)


def test_channel_cross_correlation_needs_two_channels():
    stamps = pd.date_range("2026-10-09", periods=10, freq="1min")
    single = pd.DataFrame({"ts": stamps, "探针": "tp-1", "数值": np.arange(10.0)})
    assert channel_cross_correlation({"温度": single}).empty
    assert strongest_correlation(pd.DataFrame()) is None
