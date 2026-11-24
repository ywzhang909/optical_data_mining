"""命令行入口：``python -m ui.analysis.thermo``。

示例::

    python -m ui.analysis.thermo --data-root data/daily_data --days 20261014
    python -m ui.analysis.thermo --out data/reports --run-name week42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from .config import ThermoReportConfig
from .pipeline import build
from .spatial import even_heights_from_text, heights_from_text


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（独立函数便于测试）。"""
    parser = argparse.ArgumentParser(
        prog="python -m ui.analysis.thermo",
        description="温度脉动仪多日数据分析：质量检查、统计、湍流参数与报告生成",
    )
    parser.add_argument("--data-root", type=Path, help="原始数据根目录（含 YYYYMMDD 子目录），默认 data/daily_data")
    parser.add_argument("--out", type=Path, help="报告输出根目录，默认 data/reports")
    parser.add_argument(
        "--days",
        nargs="+",
        default=(),
        metavar="YYYYMMDD",
        help="仅分析这些日期；省略表示全部日期",
    )
    parser.add_argument(
        "--max-files-per-day",
        type=int,
        default=None,
        metavar="N",
        help="每个日期每个通道最多读取 N 个文件（快速试跑用）",
    )
    parser.add_argument("--config", type=Path, help="TOML 配置文件路径，命令行参数优先")
    parser.add_argument("--run-name", help="报告目录后缀，默认使用时间戳")
    parser.add_argument("--no-cache", action="store_true", help="忽略解析缓存，强制重新解析原始文件")
    parser.add_argument(
        "--no-aux", action="store_true", help="不读取风速、湿度、能见度与 Cn² 等环境要素（仅做温度分析）"
    )
    parser.add_argument("--slice-rule", help="时间片长度（pandas 频率串，如 10min / 1h），默认 10min")
    parser.add_argument("--no-slices", action="store_true", help="不导出时间片中间数据")
    parser.add_argument("--no-despike", action="store_true", help="关闭抖动异常剔除")
    parser.add_argument("--smooth-minutes", type=int, help="产品平滑窗口（分钟），1 表示不平滑")
    parser.add_argument(
        "--probe-heights",
        default="",
        metavar="tp-1=1.5,tp-2=3.0",
        help="探针安装高度（米）；给出后启用 notebook 的位置插值平均，省略则按跨探针等权聚合",
    )
    parser.add_argument(
        "--probe-heights-even",
        default="",
        metavar="BASE,STEP",
        help='无实测高度时按编号生成等间距假设，例如 "5,2" 表示 tp-1 在 5 m、每层间隔 2 m（报告会标注为假设）',
    )
    parser.add_argument("--no-spatial", action="store_true", help="不计算跨探针空间平均")
    parser.add_argument(
        "--spatial-method",
        choices=("median", "mean"),
        help="未配置高度时的跨探针聚合方式，默认 median（中位数）",
    )
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def config_from_args(args: argparse.Namespace) -> ThermoReportConfig:
    """合并配置文件与命令行参数，返回最终配置。"""
    cfg = ThermoReportConfig.from_toml(args.config) if args.config else ThermoReportConfig()
    paths = cfg.paths
    if args.data_root:
        paths = paths.model_copy(update={"data_root": args.data_root})
    if args.out:
        paths = paths.model_copy(update={"output_dir": args.out})
    updates: dict[str, object] = {"paths": paths}

    export = cfg.export
    if getattr(args, "slice_rule", None):
        export = export.model_copy(update={"slice_rule": args.slice_rule})
    if getattr(args, "no_slices", False):
        export = export.model_copy(update={"enabled": False})
    updates["export"] = export

    filtering = cfg.filtering
    if getattr(args, "no_despike", False):
        filtering = filtering.model_copy(update={"despike_enabled": False})
    if getattr(args, "smooth_minutes", None) is not None:
        filtering = filtering.model_copy(update={"smooth_window_min": max(1, int(args.smooth_minutes))})
    updates["filtering"] = filtering

    probes = cfg.probes
    if getattr(args, "probe_heights", ""):
        probes = probes.model_copy(update={"heights_m": heights_from_text(args.probe_heights)})
    if getattr(args, "probe_heights_even", ""):
        base_m, step_m = even_heights_from_text(args.probe_heights_even)
        if probes.heights_m:
            logger.warning("同时给出 --probe-heights 与 --probe-heights-even，实测高度优先，忽略等间距假设")
        else:
            probes = probes.model_copy(
                update={
                    "heights_even_base_m": base_m,
                    "heights_even_step_m": step_m,
                    "heights_source": f"等间距假设（tp-1 = {base_m:g} m、间隔 {step_m:g} m）",
                }
            )
    updates["probes"] = probes

    spatial_cfg = cfg.spatial
    if getattr(args, "no_spatial", False):
        spatial_cfg = spatial_cfg.model_copy(update={"enabled": False})
    if getattr(args, "spatial_method", None):
        spatial_cfg = spatial_cfg.model_copy(update={"method": args.spatial_method})
    updates["spatial"] = spatial_cfg
    return cfg.model_copy(update=updates)


def main(argv: list[str] | None = None) -> int:
    """CLI 主流程，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    cfg = config_from_args(args)
    logger.info("数据目录 {} → 报告目录 {}", cfg.paths.data_root, cfg.paths.output_dir)
    try:
        report_path = build(
            cfg,
            days=tuple(args.days),
            max_files_per_day=args.max_files_per_day,
            use_cache=not args.no_cache,
            with_auxiliary=not args.no_aux,
            run_name=args.run_name,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("分析失败：{}", exc)
        return 2
    logger.info("报告已生成：{}", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - 直接执行时的入口
    raise SystemExit(main())
