from typing import Dict, Any
from pydantic import Field, ConfigDict
import pandas as pd
from .base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from .models import DataQualityReport, ExperimentFilter

class QualityAnalyzerConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    
    processor_name: str = Field(default="QualityAnalyzer", description="处理器名称")
    
    laser_params: list = Field(default_factory=lambda: [
        'experiment_polarization_state',
        'experiment_radiation_mode',
        'experiment_radiation_duration',
        'experiment_radiation_power',
        '_sub_beam'
    ])
    target_params: list = Field(default_factory=lambda: [
        '靶材',
        '距离',
        '靶材厚度',
        '材质'
    ])
    env_params: list = Field(default_factory=lambda: [
        '横向风',
        '纵向风',
        '湍流',
        '通风'
    ])

class QualityAnalyzer(BaseProcessor[DataQualityReport]):
    config_class = QualityAnalyzerConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, str]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "Data is required"
        if hasattr(input_data, '__dataframe__'):
            return True, ""
        if not isinstance(input_data, pd.DataFrame):
            return False, "Data must be a DataFrame"
        return True, ""
    
    def process(self, config: QualityAnalyzerConfig, **kwargs) -> ProcessingResult:
        try:
            df = kwargs.get('data')
            if df is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入数据",
                    error="Data is required"
                )
            
            total_count = len(df)
            complete_records = df.dropna(how='any', subset=config.laser_params + config.target_params + config.env_params).shape[0]
            incomplete_records = total_count - complete_records
            missing_rate = incomplete_records / total_count if total_count > 0 else 0
            
            param_missing_stats = {
                'laser': self._calculate_param_missing(df, config.laser_params),
                'target': self._calculate_param_missing(df, config.target_params),
                'env': self._calculate_param_missing(df, config.env_params)
            }
            
            laser_params_missing = sum(param_missing_stats['laser'].values())
            laser_params_complete = sum(1 for p in config.laser_params if p in df and df[p].notna().any())
            
            target_params_missing = sum(param_missing_stats['target'].values())
            target_params_complete = sum(1 for p in config.target_params if p in df and df[p].notna().any())
            
            env_params_missing = sum(param_missing_stats['env'].values())
            env_params_complete = sum(1 for p in config.env_params if p in df and df[p].notna().any())
            
            report = DataQualityReport(
                total_count=total_count,
                complete_records=complete_records,
                incomplete_records=incomplete_records,
                missing_rate=missing_rate,
                param_missing_stats=param_missing_stats,
                laser_params_missing=laser_params_missing,
                laser_params_complete=laser_params_complete,
                target_params_missing=target_params_missing,
                target_params_complete=target_params_complete,
                env_params_missing=env_params_missing,
                env_params_complete=env_params_complete
            )
            
            return ProcessingResult(
                success=True,
                message="数据质量分析完成",
                data=report
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"质量分析失败: {str(e)}",
                error=str(e)
            )
    
    def _calculate_param_missing(self, df: pd.DataFrame, params: list) -> Dict[str, int]:
        result = {}
        for param in params:
            if param in df.columns:
                result[param] = df[param].isna().sum()
            else:
                result[param] = len(df)
        return result
