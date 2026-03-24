from typing import Optional, List
import pandas as pd
from .base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from .models import ExperimentFilter

class ExperimentFilterConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    experiment_type_id: Optional[int] = Field(None, description="实验类型ID")
    date_range: Optional[tuple] = Field(None, description="日期范围")
    target_material: Optional[str] = Field(None, description="靶材类型")
    distance: Optional[float] = Field(None, description="距离(km)")
    min_power: Optional[float] = Field(None, description="最小功率")
    max_power: Optional[float] = Field(None, description="最大功率")
    exclude_params: Optional[List[str]] = Field(None, description="排除参数列表")

class ExperimentFilter(BaseProcessor[pd.DataFrame]):
    config_class = ExperimentFilterConfig
    
    def process(self, config: ExperimentFilterConfig, **kwargs) -> ProcessingResult:
        try:
            df = kwargs.get('data')
            if df is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入数据",
                    error="Data is required"
                )
            
            original_count = len(df)
            
            # 应用过滤条件
            if config.experiment_type_id is not None:
                df = df[df['experiment_type_id'] == config.experiment_type_id]
            
            if config.date_range is not None:
                start_date, end_date = config.date_range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if config.target_material is not None:
                df = df[df['靶材'] == config.target_material]
            
            if config.distance is not None:
                df = df[df['距离'] == config.distance]
            
            if config.min_power is not None:
                df = df[df['experiment_radiation_power'] >= config.min_power]
            
            if config.max_power is not None:
                df = df[df['experiment_radiation_power'] <= config.max_power]
            
            if config.exclude_params:
                for param in config.exclude_params:
                    if param in df.columns:
                        df = df.drop(columns=[param])
            
            filtered_count = len(df)
            removed_count = original_count - filtered_count
            
            result = {
                'original_count': original_count,
                'filtered_count': filtered_count,
                'removed_count': removed_count,
                'filter_rate': removed_count / original_count if original_count > 0 else 0
            }
            
            return ProcessingResult(
                success=True,
                message=f"数据过滤完成，移除 {removed_count} 条记录",
                data=result
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"数据过滤失败: {str(e)}",
                error=str(e)
            )
