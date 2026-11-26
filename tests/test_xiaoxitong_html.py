"""小系统 HTML 报告与大气窗口统计的单元测试。

测试全部使用合成数据，不读取 ``data/`` 下的原始试验与大气文件。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_mining.xiaoxitong import atm, html
from data_mining.xiaoxitong import report as base


def make_exp(**overrides) -> base.Experiment:
    """构造一次合成试验。"""
    values = {
        "date": "20261009",
        "seq": 1,
        "name": "试验_500W",
        "path": Path("data/小系统/20261009/试验 01"),
        "power": "低功率",
        "distance": "600m",
        "counts": Counter({".jpg": 3}),
    }
    values.update(overrides)
    return base.Experiment(**values)


def minute_series(values: list[float], start: datetime) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="1min")
    return pd.Series(values, index=index, dtype=float)


# --------------------------------------------------------------------------- #
# 大气统计
# --------------------------------------------------------------------------- #
def test_minute_bounds_expands_seconds_to_minutes() -> None:
    start = datetime(2026, 10, 9, 10, 0, 40)
    end = datetime(2026, 10, 9, 10, 0, 55)
    assert atm.minute_bounds(start, end) == (
        datetime(2026, 10, 9, 10, 0),
        datetime(2026, 10, 9, 10, 1),
    )


def test_window_stats_covers_seconds_long_window() -> None:
    series = minute_series([1.0, 2.0, 9.0], datetime(2026, 10, 9, 10, 0))
    stats = atm.window_stats(series, datetime(2026, 10, 9, 10, 0, 20), datetime(2026, 10, 9, 10, 0, 40))
    assert stats["n"] == 1
    assert stats["median"] == 1.0
    assert atm.window_stats(series, datetime(2026, 10, 9, 11, 0), datetime(2026, 10, 9, 11, 0, 30)) == {}


def test_background_stats_excludes_guarded_window() -> None:
    series = minute_series(list(range(21)), datetime(2026, 10, 9, 10, 0))
    stats = atm.background_stats(
        series,
        [(datetime(2026, 10, 9, 10, 10), datetime(2026, 10, 9, 10, 11))],
        guard_minutes=2,
    )
    # 10:08–10:13 共 6 分钟被扣除，剩下 15 分钟。
    assert stats["n"] == 15
    assert stats["median"] == pytest.approx(float(np.median([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20])))


def test_relative_change_respects_near_zero_floor() -> None:
    assert atm.relative_change(0.1, 0.04) == pytest.approx(1.5)
    assert atm.relative_change(0.1, 0.04, floor=0.2) is None
    assert atm.relative_change(1.0, 0.0) is None
    assert atm.relative_change(None, 1.0) is None
    assert atm.ratio_orders(1e-16, 1e-17) == pytest.approx(1.0)
    assert atm.ratio_orders(1.0, -1.0) is None


def test_channel_floors_are_configured() -> None:
    floors = {item.key: item.rel_floor for item in atm.CHANNELS}
    assert floors["wind"] > 0.04  # 夜间静风背景不应进入百分比计算
    assert floors["cn2"] > 0
    assert atm.channel_by_key("visibility").note


# --------------------------------------------------------------------------- #
# 击穿分类和窗口对齐
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [("3.40s", 3.4), ("5", 5.0), ("未击穿", None), ("很多小孔，截至靶不易观测", None), ("", None)],
)
def test_parse_seconds(text: str, expected: float | None) -> None:
    assert html.parse_seconds(text) == expected


def test_describe_outcome_classifies_censored_and_numeric() -> None:
    numeric = html.describe_outcome(make_exp(breakdown="3.40s", duration="8s"))
    censored = html.describe_outcome(make_exp(breakdown="未明显观察出"))
    missing = html.describe_outcome(make_exp(breakdown=""))
    assert numeric.breakdown_s == 3.4
    assert numeric.duration_s == 8.0
    assert censored.censored and not censored.breakdown_s
    assert not missing.censored and missing.breakdown_s is None
    summary = html.breakdown_summary([numeric, censored, missing])
    assert summary == {"numeric": 1, "censored": 1, "empty": 1, "total": 3}


def test_collect_windows_pads_capture_span() -> None:
    exp = make_exp(
        capture_start=datetime(2026, 10, 9, 10, 0, 10),
        capture_end=datetime(2026, 10, 9, 10, 0, 40),
    )
    series = minute_series([0.5, 0.6, 0.7], datetime(2026, 10, 9, 10, 0))
    frame = pd.DataFrame({"中位": series, "探针数": 2})
    samples = html.collect_windows([exp], {"wind": frame}, pad_minutes=10)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.start == datetime(2026, 10, 9, 9, 50, 10)
    assert sample.end == datetime(2026, 10, 9, 10, 10, 40)
    assert sample.raw_start == exp.capture_start
    assert sample.values["wind"] == pytest.approx(0.6)
    assert sample.coverage["wind"] == 3


def test_decimate_uses_five_minute_median() -> None:
    series = minute_series([1, 3, 2, 4, 100, 8], datetime(2026, 10, 9, 10, 0))
    result = html._decimate(series, 5)
    assert len(result) == 2
    assert result.iloc[0] == 3.0
    assert result.iloc[1] == 8.0


# --------------------------------------------------------------------------- #
# HTML 产物
# --------------------------------------------------------------------------- #
def test_overview_report_escapes_markup_and_masks_power() -> None:
    exp = make_exp(name="试验_500W_<script>alert(1)</script>", breakdown="3.40s")
    report = html.overview_report([exp], Path("data/小系统"), datetime(2026, 11, 25, 19, 0))
    assert "500W" not in report
    assert "5000W" not in report
    assert "低功率" in report
    assert exp.name not in report  # 正文不写目录名/文件名
    assert base.sanitize_power("试验_500W") == "试验_低功率"
    assert html.esc("<script>") == "&lt;script&gt;"
    assert "有效击穿时间" in report
    assert "未击穿/难观测" in report


def test_overview_renders_trend_and_correlation_sections() -> None:
    experiments = []
    for index in range(8):
        day = "20261009" if index < 4 else "20261010"
        experiments.append(
            make_exp(
                date=day,
                seq=index + 1,
                breakdown=f"{3.0 + index * 0.2:.2f}s",
                distance=f"{600 + index * 50}m",
                duration=f"{5 + index}s",
                fan_count="0" if index < 4 else "2 台风机",
                power="标准功率" if index % 2 == 0 else "低功率",
                incident_direction="正入射" if index % 2 == 0 else "斜入射",
                standby="热待机" if index % 2 == 0 else "冷待机",
                capture_start=datetime(2026, 10, 9 + index // 4, 9, 0),
                capture_end=datetime(2026, 10, 9 + index // 4, 9, 0, 30),
            )
        )
    report = html.overview_report(experiments, Path("data/小系统"), datetime(2026, 11, 25, 19, 0))
    for anchor in ("s5", "s6", "s7", "s8"):
        assert f'id="{anchor}"' in report
    for chapter in ("参数与结果的趋势分析", "参数与结果的相关性分析", "各日击穿时间中位数走势", "Mann-Whitney U"):
        assert chapter in report
    assert "500W" not in report
    assert "5000W" not in report
    assert "p = <" not in report
    assert html.distance_m(experiments[0]) == 600.0
    assert html.fan_number(experiments[0]) == 0.0
    assert html.fan_number(experiments[4]) == 2.0


def test_trend_helpers_handle_short_and_degenerate_series() -> None:
    assert html.linear_trend([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx((2.0, 0.0))
    assert html.linear_trend([1.0, 1.0, 1.0], [2.0, 4.0, 6.0]) is None
    assert html.linear_trend([1.0, 2.0], [2.0, 4.0]) is None
    assert html._clean_pairs([(1.0, 2.0), (None, 3.0), (3.0, None)]) == ([1.0], [2.0])


def test_synergy_report_renders_sections_without_raw_less_than() -> None:
    start = datetime(2026, 10, 9, 9, 0)
    series = minute_series([0.05] * 180 + [0.9] * 60, start)
    channels = {"wind": pd.DataFrame({"中位": series, "探针数": 2})}
    exp = make_exp(
        capture_start=datetime(2026, 10, 9, 12, 0, 10),
        capture_end=datetime(2026, 10, 9, 12, 0, 40),
        breakdown="3.40s",
    )
    payload, frame = html.synergy_report(
        [exp], channels, days=["20261009"], generated=datetime(2026, 11, 25, 19, 0), root=Path("data/daily_data")
    )
    for chapter in ("试验时间结构", "大气环境背景", "试验窗口与背景对比", "击穿行为与环境要素的关联", "滞后剖面"):
        assert chapter in payload
    assert "背景近零" in payload
    assert "同步性描述" in payload
    assert "p = <" not in payload
    assert "500W" not in payload
    assert list(frame["试验"]) == [exp.exp_id]
    assert frame.loc[0, "窗口分钟"] == pytest.approx(20.5)


def test_lag_scan_handles_empty_series_and_short_input() -> None:
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
    series = minute_series([1.0, 3.0, 5.0], datetime(2026, 10, 9, 10, 0))
    assert html.lag_value(empty, datetime(2026, 10, 9, 10, 0), 0) is None
    assert html.lag_value(series, datetime(2026, 10, 9, 10, 0), 0) == pytest.approx(3.0)
    assert html.lag_value(series, datetime(2026, 10, 9, 10, 30), 0) is None
    exp = make_exp(
        capture_start=datetime(2026, 10, 9, 10, 0, 10),
        capture_end=datetime(2026, 10, 9, 10, 0, 40),
        breakdown="3.40s",
    )
    sample = html.collect_windows([exp], {"wind": pd.DataFrame({"中位": empty})})[0]
    assert html.lag_scan(empty, [sample], offsets=(0, 10)) == {0: None, 10: None}
    assert html.quartile_range([1.0, 2.0]) is None
    assert html.quartile_range([1.0, 3.0, 5.0]) == pytest.approx(2.0)
    assert html.quartile_range([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.5)


def test_synergy_report_renders_trend_matrix_group_and_lag_tables() -> None:
    start = datetime(2026, 10, 9, 8, 0)
    wind = [2.0 + 0.6 * float(np.sin(index / 7.0)) + 0.01 * index for index in range(210)]
    humidity = [60.0 - 0.05 * float(index) for index in range(210)]
    channels = {
        "wind": pd.DataFrame({"中位": minute_series(wind, start), "探针数": 2}),
        "humidity": pd.DataFrame({"中位": minute_series(humidity, start), "探针数": 2}),
    }
    experiments = []
    for index in range(8):
        moment = datetime(2026, 10, 9, 9, 0) + timedelta(minutes=15 * index)
        experiments.append(
            make_exp(
                seq=index + 1,
                breakdown=f"{3.0 + 0.5 * index:.2f}s",
                distance="600m" if index < 4 else "700m",
                fan_count="0" if index < 4 else "2 台风机",
                power="标准功率" if index % 2 == 0 else "低功率",
                capture_start=moment,
                capture_end=moment + timedelta(seconds=30),
            )
        )
    payload, _frame = html.synergy_report(
        experiments, channels, days=["20261009"], generated=datetime(2026, 11, 25, 19, 0), root=Path("data/daily_data")
    )
    for caption in (
        "击穿时间随试验次序的变化",
        "逐日击穿时间随试验次序的秩相关与线性斜率",
        "环境要素与击穿时间的秩相关矩阵",
        "击穿时间与各通道在不同滞后窗口上的秩相关（窗口宽 10 分钟）",
        "按风机配置、作用距离与功率档位分组的击穿时间中位数",
    ):
        assert caption in payload
    assert "四分位距" in payload
    assert "未做多重比较校正" in payload
    assert "p = <" not in payload
    assert "500W" not in payload
    assert "5000W" not in payload


def test_p_text_is_safe_in_html() -> None:
    assert html.p_text(0.0005) == "低于 0.001"
    assert html.p_phrase(0.0005) == "p 低于 0.001"
    assert html.p_phrase(0.016) == "p = 0.016"
    assert "<" not in html.p_text(0.0005)
