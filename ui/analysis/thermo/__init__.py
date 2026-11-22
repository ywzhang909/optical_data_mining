"""温度脉动仪数据分析与报告生成。

流水线分五步，每一步可单独调用：

1. `discovery` —— 扫描日期目录与小时文件，给出数据清单与截断线索；
2. `ingest` —— 长表 CSV 流式解析为探针宽表，按文件指纹缓存分钟/分块/谱/质量聚合；
3. `channels` —— 加载湿度、能见度、设备 Cn² 等环境要素（同一长表结构，按物理范围剔除异常码）；
4. `aggregate` —— 跨文件聚合：逐日统计、日变化、探针一致性、C_T²/Cn² 反演与设备交叉验证；
5. `report`、`slices` —— 输出自包含 HTML 报告与随时间片的中间数据（CSV/Parquet）。

典型用法::

    from ui.analysis.thermo import ThermoReportConfig, analyze, build

    cfg = ThermoReportConfig()
    result = analyze(cfg, days=("20261014",))      # 只算不落盘
    report = build(cfg, run_name="demo")           # 算完并生成报告

命令行用法见 `python -m ui.analysis.thermo --help`；批量自动化见 `scripts/thermo_auto.ps1`。
"""

from __future__ import annotations

from .aggregate import (
    anomaly_correlation,
    daily_mean_series,
    diurnal_profile,
    ensemble_spread,
    probe_day_summary,
    probe_overall_summary,
    summarize_cn2_agreement,
    turbulence_summary,
)
from .channels import ChannelResult, load_channels
from .config import (
    AnalysisConfig,
    ChannelConfig,
    ExportConfig,
    FilterConfig,
    PathsConfig,
    ProbeConfig,
    QualityConfig,
    ReportConfig,
    ThermoReportConfig,
)
from .discovery import Inventory, SourceFile, discover
from .filters import jitter_mask, moving_average, smooth_minute_frame
from .metrics import (
    cn2_from_ct2,
    ct2_from_structure,
    structure_function,
    structure_noise_floor,
    structure_slope,
)
from .pipeline import AnalysisResult, analyze, build
from .report import write_report
from .slices import build_slice_frame, write_slices

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ChannelConfig",
    "ChannelResult",
    "ExportConfig",
    "FilterConfig",
    "Inventory",
    "PathsConfig",
    "ProbeConfig",
    "QualityConfig",
    "ReportConfig",
    "SourceFile",
    "ThermoReportConfig",
    "__version__",
    "analyze",
    "anomaly_correlation",
    "build",
    "build_slice_frame",
    "cn2_from_ct2",
    "ct2_from_structure",
    "daily_mean_series",
    "diurnal_profile",
    "discover",
    "ensemble_spread",
    "jitter_mask",
    "load_channels",
    "moving_average",
    "probe_day_summary",
    "probe_overall_summary",
    "smooth_minute_frame",
    "structure_function",
    "structure_noise_floor",
    "structure_slope",
    "summarize_cn2_agreement",
    "turbulence_summary",
    "write_report",
    "write_slices",
]
