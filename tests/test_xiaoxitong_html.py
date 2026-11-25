"""小系统 HTML 报告与大气窗口统计的单元测试。

测试全部使用合成数据，不读取 ``data/`` 下的原始试验与大气文件。
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import xiaoxitong_atm as atm  # noqa: E402
import xiaoxitong_html as html  # noqa: E402
import xiaoxitong_report as base  # noqa: E402


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


def test_p_text_is_safe_in_html() -> None:
    assert html.p_text(0.0005) == "低于 0.001"
    assert html.p_phrase(0.0005) == "p 低于 0.001"
    assert html.p_phrase(0.016) == "p = 0.016"
    assert "<" not in html.p_text(0.0005)
