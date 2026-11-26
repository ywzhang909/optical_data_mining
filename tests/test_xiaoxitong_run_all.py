"""一键脚本与数据错误容忍的单元测试（全部使用合成数据）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_mining.xiaoxitong import atm, run_all
from data_mining.xiaoxitong import report as base


def make_root(tmp_path: Path, name: str = "01_试验") -> Path:
    root = tmp_path / "xxt"
    exp = root / "20261009" / name
    exp.mkdir(parents=True)
    (exp / "参数.txt").write_text("作用距离: 600m\n击穿时间: 3.40s\n", encoding="utf-8")
    return root


def test_invalid_capture_timestamp_is_ignored(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    exp_dir = root / "20261009" / "01_试验"
    valid = "20261009_20261009193223_20261009193320_000001.jpg"
    invalid = "20261009_20261340993223_20261009193320_000002.jpg"
    for name in (valid, invalid):
        (exp_dir / name).write_bytes(b"")

    experiments = base.build_experiments(root)
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.capture_start is not None
    assert exp.capture_start.strftime("%H:%M:%S") == "19:32:23"
    assert any("时间戳无效" in item for item in exp.quality)


def test_bad_experiment_directory_does_not_abort_batch(tmp_path: Path, monkeypatch) -> None:
    root = make_root(tmp_path)
    bad = root / "20261009" / "02_坏目录"
    bad.mkdir()
    original = base.parse_experiment

    def flaky(date: str, exp_dir: Path) -> base.Experiment:
        if exp_dir == bad:
            raise RuntimeError("模拟损坏")
        return original(date, exp_dir)

    monkeypatch.setattr(base, "parse_experiment", flaky)
    issues: list[str] = []
    experiments = base.build_experiments(root, issues=issues)
    assert len(experiments) == 1
    assert any("02_坏目录" in item for item in issues)


def test_corrupt_cache_falls_back_to_source(tmp_path: Path, monkeypatch) -> None:
    channel = atm.channel_by_key("wind")
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    stem = atm._day_cache_paths(cache, channel, "20261009")[0]
    stem.write_bytes(b"not a parquet")

    class EmptyInventory:
        @staticmethod
        def by_channel(_name: str) -> tuple:
            return ()

    monkeypatch.setattr(atm, "discover", lambda *_args, **_kwargs: EmptyInventory())
    (cache / stem.with_suffix(".json").name).write_text(
        '{"fingerprint": ' + repr(atm._source_fingerprint((), channel)).replace("'", '"') + "}",
        encoding="utf-8",
    )
    expected = pd.DataFrame(
        {"中位": [0.5], "探针数": [2]},
        index=pd.to_datetime(["2026-10-09 10:00"]),
    )
    monkeypatch.setattr(atm, "minute_median_table", lambda *_args, **_kwargs: expected)
    result = atm.load_day_channel(tmp_path, channel, "20261009", cache_dir=cache)
    pd.testing.assert_frame_equal(result, expected)


def test_load_channels_skips_failing_day(tmp_path: Path, monkeypatch) -> None:
    channel = atm.channel_by_key("wind")
    good = pd.DataFrame(
        {"中位": [1.0], "探针数": [1]},
        index=pd.to_datetime(["2026-10-10 10:00"]),
    )

    def fake_load(_root, _channel, day, _cache, **_kwargs):
        if day == "20261009":
            raise OSError("模拟读取失败")
        return good

    monkeypatch.setattr(atm, "load_day_channel", fake_load)
    result = atm.load_channels(
        tmp_path,
        ["20261009", "20261010"],
        channels=(channel,),
        cache_dir=None,
    )
    assert list(result["wind"]["中位"]) == [1.0]


def test_one_click_keeps_partial_results_and_logs(tmp_path: Path, monkeypatch) -> None:
    root = make_root(tmp_path, name="01_试验_500W")
    bad = root / "20261009" / "02_试验_500W"
    bad.mkdir()
    original = base.parse_experiment

    def flaky(date: str, exp_dir: Path) -> base.Experiment:
        if exp_dir == bad:
            raise RuntimeError("模拟损坏")
        return original(date, exp_dir)

    monkeypatch.setattr(base, "parse_experiment", flaky)
    out = tmp_path / "报告"
    code = run_all.main(["--root", str(root), "--out", str(out), "--skip-atmosphere"])
    assert code == 0
    overview = out / "01_试验数据总览.html"
    assert overview.exists()
    text = overview.read_text(encoding="utf-8")
    assert "500W" not in text
    issues = (out / "数据异常清单.csv").read_text(encoding="utf-8-sig")
    assert "02_试验_500W" not in issues
    assert "02_试验_低功率" in issues
    summary = (out / "运行说明.txt").read_text(encoding="utf-8")
    assert "解析试验数：1" in summary
    assert "数据异常数：1" in summary


def test_one_click_reports_no_experiments(tmp_path: Path) -> None:
    out = tmp_path / "空报告"
    code = run_all.main(["--root", str(tmp_path / "不存在"), "--out", str(out), "--skip-atmosphere"])
    assert code == 1
    summary = (out / "运行说明.txt").read_text(encoding="utf-8")
    assert "解析试验数：0" in summary
    assert "数据异常数：2" in summary
    assert "01_试验数据总览.html" not in summary


@pytest.mark.parametrize("strict,expected", [(False, 0), (True, 1)])
def test_strict_mode_return_code(tmp_path: Path, monkeypatch, strict: bool, expected: int) -> None:
    root = make_root(tmp_path)
    bad = root / "20261009" / "02_坏目录"
    bad.mkdir()
    original = base.parse_experiment

    def flaky(date: str, exp_dir: Path) -> base.Experiment:
        if exp_dir == bad:
            raise RuntimeError("模拟损坏")
        return original(date, exp_dir)

    monkeypatch.setattr(base, "parse_experiment", flaky)
    out = tmp_path / "严格"
    args = ["--root", str(root), "--out", str(out), "--skip-atmosphere"]
    if strict:
        args.append("--strict")
    assert run_all.main(args) == expected
    assert (out / "01_试验数据总览.html").exists()
