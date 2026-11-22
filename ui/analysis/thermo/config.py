"""温度脉动仪数据报告 — 配置模型。

所有可调参数集中在此，支持从 TOML 文件加载，便于在不同站点/批次间复用。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    """数据与产物路径配置。"""

    model_config = ConfigDict(frozen=True)

    data_root: Path = Field(default=Path("data/daily_data"), description="按日期分目录的原始数据根目录")
    temperature_subdir: str = Field(default="Temperature", description="温度原始数据子目录")
    wind_subdir: str = Field(default="Wind", description="风速辅助数据子目录（用于 C_T^2 反演）")
    cn2_subdir: str = Field(default="Cn2", description="设备输出 Cn2 子目录（用于交叉验证）")
    cache_dir: Path = Field(default=Path("data/reports/.cache"), description="解析结果缓存目录")
    output_dir: Path = Field(default=Path("data/reports"), description="报告输出根目录")


class ProbeConfig(BaseModel):
    """探针（TP-x）解析与筛选配置。"""

    model_config = ConfigDict(frozen=True)

    probe_prefix: str = Field(default="tp-", description="探针 ID 前缀，区分大小写不敏感匹配")
    include: tuple[str, ...] = Field(default=(), description="仅分析这些探针（空 = 全部）")
    exclude: tuple[str, ...] = Field(default=(), description="排除这些探针")


class QualityConfig(BaseModel):
    """数据质量判定阈值。"""

    model_config = ConfigDict(frozen=True)

    valid_min_c: float = Field(default=-80.0, description="物理下限（°C）")
    valid_max_c: float = Field(default=80.0, description="物理上限（°C）")
    sentinel_values: tuple[float, ...] = Field(
        default=(1220.8, 122.08, 999.9, -999.9),
        description="采集软件写入的异常码，用于质量统计归类（超出物理范围的值同样判为无效）",
    )
    spike_window: int = Field(default=11, description="滑动中值窗口（采样点），用于尖峰检测")
    spike_threshold_c: float = Field(default=0.5, description="偏离滑动中值的判定阈值（°C）")
    stuck_min_seconds: float = Field(
        default=600.0,
        description="同一读数持续超过该时长判为卡值（秒）。0.1 °C 量化下夜间数分钟不变属正常，故阈值取 10 min",
    )
    cross_probe_max_dev_c: float = Field(
        default=5.0,
        description=(
            "与同一时刻跨探针中值相差超过该值的采样判为离群（掉线/错误读数）并从统计中剔除，0 表示关闭。"
            "同址多探针真实差异在 1 °C 量级，5 °C 以上的偏离只可能是传感器故障"
        ),
    )
    quantization_step_c: float = Field(default=0.1, description="量化步长（°C），用于估计量化噪声")
    min_probe_valid_ratio: float = Field(default=0.5, description="低于该有效率的探针不参与湍流统计")


class AnalysisConfig(BaseModel):
    """统计分析参数。"""

    model_config = ConfigDict(frozen=True)

    minute_rule: str = Field(default="1min", description="分钟级聚合窗口")
    block_rule: str = Field(default="10min", description="湍流统计分块窗口")
    min_block_coverage: float = Field(default=0.8, description="分块内有效采样率下限")
    welch_nperseg: int = Field(default=1024, description="Welch 功率谱分段长度")
    spectrum_band_hz: tuple[float, float] = Field(
        default=(0.01, 2.5), description="谱斜率拟合频段；auto_band 开启时作为搜索范围"
    )
    spectrum_auto_band: bool = Field(
        default=True, description="自动收窄拟合频段到量化噪声以上（噪声主导时谱斜率会被拉平）"
    )
    structure_lags_s: tuple[float, ...] = Field(
        default=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), description="结构函数滞后时间（秒）"
    )
    structure_fit_lags_s: tuple[float, float] = Field(default=(2.0, 16.0), description="结构函数幂律拟合滞后区间")
    site_pressure_hpa: float = Field(default=1013.25, description="站点气压（hPa），用于 Cn2 反演，需按实际站点修改")
    detrend_window_s: float = Field(
        default=900.0, description="相关分析前的去趋势滑动窗口（秒）；需显著短于分析时段，否则会削掉全部信号"
    )
    min_span_hours_for_diurnal: float = Field(default=6.0, description="不足该时长不给出日变化节律结论")
    max_workers: int = Field(default=4, description="并行解析进程数")


class ReportConfig(BaseModel):
    """报告呈现配置。"""

    model_config = ConfigDict(frozen=True)

    title: str = Field(default="温度脉动仪数据报告", description="报告标题")
    author: str = Field(default="", description="报告生成者/单位")
    figure_dpi: int = Field(default=130, description="图表分辨率")
    max_series_probes: int = Field(default=8, description="时间序列图最多绘制的探针数量")
    language: str = Field(default="zh", description="报告语言：zh / en")


class ChannelConfig(BaseModel):
    """一路环境要素通道（除温度外）的解析与有效性配置。

    所有通道按同一长表结构读取（``ID, 数值, Timestamp``），因此新增通道通常只需
    在此给出目录名、物理范围与显示名，无需写解析代码。
    """

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="内部标识，同时用于产物文件名")
    subdir: str = Field(description="日期目录下的通道子目录名")
    label: str = Field(description="报告中的中文名称")
    unit: str = Field(default="", description="物理单位")
    valid_min: float = Field(default=float("-inf"), description="物理下限（含）；越界值判为无效")
    valid_max: float = Field(default=float("inf"), description="物理上限（含）；异常码必然越界")
    enabled: bool = True
    is_reference: bool = Field(default=False, description="是否作为独立参考量（如设备 Cn²）参与交叉验证")


def default_channels() -> tuple[ChannelConfig, ...]:
    """本站点已接入的环境要素通道默认值。"""
    return (
        ChannelConfig(key="humidity", subdir="Humidity", label="相对湿度", unit="%", valid_min=0.0, valid_max=100.0),
        ChannelConfig(key="visibility", subdir="Visibility", label="能见度", unit="km", valid_min=0.0, valid_max=100.0),
        ChannelConfig(
            key="cn2",
            subdir="Cn2",
            label="设备 Cn²",
            unit="m^(-2/3)",
            valid_min=1e-22,
            valid_max=1e-10,
            is_reference=True,
        ),
    )


class FilterConfig(BaseModel):
    """异常抖动的剔除与产品平滑。

    噪声治理分两层，互不替代：

    * **抖动剔除**（样本级）：偏离局部中值超过阈值的采样判为抖动异常并从所有统计中剔除，
      与跨探针离群一起构成"数据清洗"。这类值（掉线、瞬时跳变）不是温度脉动，留着会
      直接抬高方差和结构函数。
    * **产品平滑**（分钟级）：导出的时间片数据与趋势图另给一列滑动均值，用于看趋势。
      物理量（谱、结构函数、C_T²）**不使用**平滑序列——低通会把惯性子区抹掉。
    """

    model_config = ConfigDict(frozen=True)

    despike_enabled: bool = Field(default=True, description="是否剔除抖动异常采样")
    despike_window_s: float = Field(default=5.0, description="抖动判据的局部中值窗口（秒）")
    despike_threshold_c: float = Field(default=1.0, description="偏离局部中值超过该值判为抖动（°C）")
    smooth_window_min: int = Field(default=5, description="产品平滑的滑动均值窗口（分钟），1 表示不平滑")


class ExportConfig(BaseModel):
    """随时间片的中间数据导出。"""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="是否导出时间片中间数据")
    slice_rule: str = Field(default="10min", description="时间片长度（pandas 频率串）")
    per_slice_files: bool = Field(default=True, description="每个时间片写一份 CSV（便于增量消费）")
    combined_parquet: bool = Field(default=True, description="同时写一份合并的 Parquet（便于批量分析）")


class ThermoReportConfig(BaseModel):
    """完整配置：由各子配置组合而成。"""

    model_config = ConfigDict(frozen=True)

    paths: PathsConfig = Field(default_factory=PathsConfig)
    probes: ProbeConfig = Field(default_factory=ProbeConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    channels: tuple[ChannelConfig, ...] = Field(
        default_factory=default_channels, description="温度以外的环境要素通道（湿度、能见度、Cn² 等）"
    )
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ThermoReportConfig:
        """从 TOML 文件加载配置（未给出的字段使用默认值）。"""
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        sections: dict[str, Any] = {}
        for name in ("paths", "probes", "quality", "analysis", "filtering", "export", "channels", "report"):
            if name in raw:
                sections[name] = raw[name]
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"未知配置段: {sorted(unknown)}")
        return cls(**sections)

    def to_manifest(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典，写入报告附录。"""
        return self.model_dump(mode="json")
