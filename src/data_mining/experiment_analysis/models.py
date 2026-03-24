from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class ProcessorConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    processor_name: str = Field(default="Processor", description="处理器名称")
    processor_version: str = Field(default="1.0.0", description="处理器版本")
    enabled: bool = Field(default=True, description="是否启用")
    parallel: bool = Field(default=False, description="是否并行处理")
    batch_size: Optional[int] = Field(default=None, description="批处理大小")

class ExperimentType(str, Enum):
    """实验类型枚举"""
    HELIUM_NEBULA = "2"
    OTHER = "1"

class LaserRadiationMode(str, Enum):
    """激光辐射模式"""
    SINGLE_MODE = "单模"
    MULTI_MODE = "多模"
    UNKNOWN = "未知"

class SubBeamType(str, Enum):
    """子束类型"""
    SPECTRAL_CONTROL_BEAM_31 = "31路"
    COMMON_SPECTRUM_33 = "33路"
    COMMON_SPECTRUM_64 = "64路"
    COMMON_SPECTRUM_18 = "18路"
    EMPTY = "空"

class TargetMaterial(str, Enum):
    """靶材类型"""
    STAINLESS_STEEL = "不锈钢"
    ALUMINUM_ALLOY = "铝合金"
    ALUMINUM = "铝"

class TargetShape(str, Enum):
    """靶材形状"""
    CIRCULAR = "圆形"
    SQUARE = "方形"
    UNKNOWN = "未知"

class Experiment(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    experiment_id: int = Field(..., description="实验ID")
    experiment_num: str = Field(..., description="实验编号")
    experiment_group: int = Field(..., description="实验组别")
    experiment_type_id: int = Field(..., description="实验类型ID")
    
    experiment_polarization_state: Optional[str] = Field(None, description="偏振态")
    experiment_radiation_mode: Optional[str] = Field(None, description="辐射模式")
    experiment_radiation_duration: Optional[float] = Field(None, description="辐射时长")
    experiment_radiation_power: Optional[float] = Field(None, description="辐射功率")
    experiment_laser_sub_beam: Optional[str] = Field(None, description="激光子束")
    experiment_extra: Optional[Dict[str, Any]] = Field(None, description="额外参数")
    
    横向风: Optional[float] = Field(None, description="横向风速")
    纵向风: Optional[float] = Field(None, description="纵向风速")
    湍流: Optional[float] = Field(None, description="湍流强度")
    通风: Optional[str] = Field(None, description="通风条件")
    
    厚度: Optional[float] = Field(None, description="靶材厚度")
    靶材: Optional[str] = Field(None, description="靶材类型")
    靶材厚度: Optional[float] = Field(None, description="靶材厚度")
    靶材形状: Optional[str] = Field(None, description="靶材形状")
    材质: Optional[str] = Field(None, description="材质")
    
    距离: Optional[float] = Field(None, description="距离(km)")
    子束: Optional[str] = Field(None, description="子束标识")
    
    experiment_laser_start_time: Optional[datetime] = Field(None, description="激光开始时间")
    date: Optional[datetime] = Field(None, description="实验日期")

class DataQualityReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    total_count: int = Field(..., description="总记录数")
    complete_records: int = Field(..., description="完整记录数")
    incomplete_records: int = Field(..., description="不完整记录数")
    missing_rate: float = Field(..., description="缺失率")
    
    param_missing_stats: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="参数缺失统计"
    )
    
    laser_params_missing: int = Field(..., description="激光参数缺失数")
    laser_params_complete: int = Field(..., description="激光参数完整数")
    
    target_params_missing: int = Field(..., description="靶材参数缺失数")
    target_params_complete: int = Field(..., description="靶材参数完整数")
    
    env_params_missing: int = Field(..., description="环境参数缺失数")
    env_params_complete: int = Field(..., description="环境参数完整数")

class DataCleaningConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    
    processor_name: str = Field(default="DataCleaningProcessor", description="处理器名称")
    
    target_material_map: Dict[str, str] = Field(
        default_factory=lambda: {
            '不锈钢': '不锈钢',
            '铝合金': '铝合金',
            '铝': '铝合金'
        },
        description="靶材类型映射"
    )
    
    sub_beam_map: Dict[str, str] = Field(
        default_factory=lambda: {
            '31路子束': '31路',
            '31路': '31路',
            '31子束': '31路',
            '33路子束': '33路',
            '全子束': '64路',
            '33子束': '33路',
            '高吸收18路': '18路',
        },
        description="子束类型映射"
    )
    
    target_priority: List[str] = Field(
        default_factory=lambda: ['材质', '靶材'],
        description="靶材信息优先级"
    )
    
    drop_na_params: List[str] = Field(
        default_factory=lambda: [],
        description="需要删除NaN的参数列表"
    )
    
    laser_param_groups: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            'laser_params': [
                'experiment_polarization_state',
                'experiment_radiation_mode',
                'experiment_radiation_duration',
                'experiment_radiation_power',
                '_sub_beam'
            ],
            'target_params': [
                '靶材',
                '距离',
                '靶材厚度',
                '材质'
            ],
            'env_params': [
                '横向风',
                '纵向风',
                '湍流',
                '通风'
            ]
        },
        description="参数分组配置"
    )

class DataCleaningResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    original_count: int = Field(..., description="原始记录数")
    cleaned_count: int = Field(..., description="清洗后记录数")
    removed_count: int = Field(..., description="移除记录数")
    removal_rate: float = Field(..., description="移除率")
    
    missing_stats_before: Dict[str, int] = Field(
        default_factory=dict,
        description="清洗前缺失统计"
    )
    missing_stats_after: Dict[str, int] = Field(
        default_factory=dict,
        description="清洗后缺失统计"
    )
    
    cleaning_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="清洗详情"
    )

class AggregationConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    
    processor_name: str = Field(default="DataAggregationProcessor", description="处理器名称")
    
    group_by_params: List[str] = Field(
        default_factory=list,
        description="分组参数列表"
    )
    agg_funcs: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            'count': ['count'],
            'unique_values': ['set'],
            'distinct_count': ['nunique']
        },
        description="聚合函数配置"
    )
    
    output_format: Literal['dict', 'dataframe'] = Field(
        default='dataframe',
        description="输出格式"
    )

class AggregationResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    groups: Dict[str, Any] = Field(
        default_factory=dict,
        description="分组聚合结果"
    )
    group_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="每组记录数"
    )
    total_groups: int = Field(..., description="分组总数")
    
    detailed_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="详细统计信息"
    )

class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    db_user: str = Field(..., description="数据库用户")
    db_password: str = Field(..., description="数据库密码")
    db_host: str = Field(..., description="数据库主机")
    db_port: int = Field(..., description="数据库端口")
    db_name: str = Field(..., description="数据库名称")
    
    connection_url: str = Field(..., description="数据库连接URL")

class ExperimentFilter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    experiment_type_id: Optional[int] = Field(None, description="实验类型ID")
    date_range: Optional[tuple[datetime, datetime]] = Field(None, description="日期范围")
    target_material: Optional[str] = Field(None, description="靶材类型")
    distance: Optional[float] = Field(None, description="距离(km)")
    min_power: Optional[float] = Field(None, description="最小功率")
    max_power: Optional[float] = Field(None, description="最大功率")
    exclude_params: Optional[List[str]] = Field(None, description="排除参数列表")

class DataAnalysisPipeline(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    stages: List[str] = Field(
        default_factory=lambda: ['load', 'quality', 'clean', 'aggregate'],
        description="分析阶段列表"
    )
    config: DataCleaningConfig = Field(
        default_factory=DataCleaningConfig,
        description="数据清洗配置"
    )
    filter_config: ExperimentFilter = Field(
        default_factory=ExperimentFilter,
        description="筛选配置"
    )
    output_format: str = Field(default='dataframe', description="输出格式")
