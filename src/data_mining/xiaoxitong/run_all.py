#!/usr/bin/env python
"""小系统报告一键生成脚本（兼容数据错误）。

运行一次即可生成两份自包含 HTML 报告、窗口统计 CSV 和运行日志。对数据错误的处理原则：

* 单个试验目录解析失败：跳过并记录，继续其余试验；
* 采集文件名时间戳无效：忽略该时间戳，其余字段照常解析；
* 单日/单通道大气数据损坏或缓存损坏：跳过或回退原始文件，继续其余通道；
* 大气整体不可用：仍生成总览报告，并在日志中说明协同报告缺席的原因；
* 默认以「尽力产出」为原则返回 0；``--strict`` 下有任一异常返回 1。

Usage:
    uv run python -m data_mining.xiaoxitong.run_all
    uv run python -m data_mining.xiaoxitong.run_all --refresh-atm --strict
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from loguru import logger

from . import atm, html
from . import report as base


def record(issues: list[tuple[str, str]], scope: str, detail: object) -> None:
    """记录一条数据异常；功率数值一律脱敏后再落盘。"""
    issues.append((scope, base.sanitize_power(str(detail))))


def write_issues(out_dir: Path, issues: list[tuple[str, str]]) -> Path:
    path = out_dir / "数据异常清单.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["范围", "问题"])
        writer.writerows(issues)
    return path


def write_summary(
    out_dir: Path,
    *,
    generated: datetime,
    root: Path,
    atm_root: Path,
    experiments: int,
    written: list[Path],
    issues: list[tuple[str, str]],
    strict: bool,
) -> Path:
    lines = [
        "小系统报告一键生成记录",
        f"生成时间：{generated:%Y-%m-%d %H:%M:%S}",
        f"试验数据根目录：{root}",
        f"大气数据根目录：{atm_root}",
        f"解析试验数：{experiments}",
        f"数据异常数：{len(issues)}",
        f"严格模式：{'是' if strict else '否'}",
        "",
        "已生成产物：",
    ]
    lines.extend(f"  - {path.name}" for path in written)
    if issues:
        lines.extend(["", "异常明细见 数据异常清单.csv；以下对象未参与或仅部分参与统计："])
        lines.extend(f"  - {scope}：{detail}" for scope, detail in issues)
    else:
        lines.extend(["", "本次运行未发现数据异常。"])
    path = out_dir / "运行说明.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    generated = datetime.now()
    issues: list[tuple[str, str]] = []
    out_dir: Path = args.out
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("无法创建输出目录 {}：{}", out_dir, exc)
        return 1

    raw_issues: list[str] = []
    try:
        experiments = base.build_experiments(args.root, issues=raw_issues)
    except Exception as exc:
        record(issues, "试验数据根目录", f"扫描失败（{type(exc).__name__}: {exc}）")
        experiments = []
    for text in raw_issues:
        record(issues, "试验目录", text)
    if not experiments:
        record(issues, "试验数据", "未解析到任何试验")
    for exp in experiments:
        try:
            base.finalize_quality(exp)
        except Exception as exc:
            record(issues, f"试验 {exp.exp_id}", f"质量补全失败（{type(exc).__name__}: {exc}）")

    written: list[Path] = []
    if experiments:
        try:
            path = out_dir / "01_试验数据总览.html"
            path.write_text(html.overview_report(experiments, args.root, generated), encoding="utf-8")
            written.append(path)
            logger.info("已写出 {}", path)
        except Exception as exc:
            record(issues, "总览报告", f"生成失败（{type(exc).__name__}: {exc}）")

    if not args.skip_atmosphere and experiments:
        days = sorted({exp.date for exp in experiments})
        channels: dict[str, object] = {}
        for channel in atm.CHANNELS:
            try:
                merged = atm.load_channels(
                    args.atm_root,
                    days,
                    channels=(channel,),
                    cache_dir=args.cache_dir,
                    refresh=args.refresh_atm,
                )[channel.key]
            except Exception as exc:
                record(issues, f"大气通道 {channel.label}", f"读取失败（{type(exc).__name__}: {exc}）")
                continue
            if not merged.empty:
                channels[channel.key] = merged
        if channels:
            try:
                payload, frame = html.synergy_report(
                    experiments, channels, days=days, generated=generated, root=args.atm_root
                )
                synergy = out_dir / "02_时间与大气协同.html"
                synergy.write_text(payload, encoding="utf-8")
                written.append(synergy)
                csv_path = out_dir / "试验窗口统计.csv"
                frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
                written.append(csv_path)
                logger.info("已写出 {}（{} 行窗口）", synergy, len(frame))
            except Exception as exc:
                record(issues, "协同报告", f"生成失败（{type(exc).__name__}: {exc}）")
        else:
            record(issues, "大气数据", "全部通道无有效数据，已跳过协同报告")

    write_issues(out_dir, issues)
    write_summary(
        out_dir,
        generated=generated,
        root=args.root,
        atm_root=args.atm_root,
        experiments=len(experiments),
        written=written,
        issues=issues,
        strict=args.strict,
    )
    for path in written:
        print(path)
    if not written:
        logger.error("未生成任何报告，详见 {}", out_dir / "运行说明.txt")
        return 1
    if args.strict and issues:
        logger.warning("严格模式：存在 {} 条数据异常", len(issues))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="小系统报告一键生成（兼容数据错误）")
    parser.add_argument("--root", type=Path, default=html.DEFAULT_ROOT, help="试验数据根目录")
    parser.add_argument("--out", type=Path, default=html.DEFAULT_OUT, help="报告输出目录")
    parser.add_argument("--atm-root", type=Path, default=atm.DEFAULT_ATM_ROOT, help="大气数据根目录")
    parser.add_argument("--cache-dir", type=Path, default=atm.DEFAULT_CACHE_DIR, help="大气分钟缓存目录")
    parser.add_argument("--skip-atmosphere", action="store_true", help="只生成总览报告")
    parser.add_argument("--refresh-atm", action="store_true", help="忽略大气缓存重新解析")
    parser.add_argument("--strict", action="store_true", help="有任一数据异常即返回非零状态")
    return run(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
