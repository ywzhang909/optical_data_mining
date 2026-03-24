from typing import Dict, Any, List
import pandas as pd
from .base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from .models import AggregationConfig, AggregationResult

class DataAggregationProcessor(BaseProcessor[AggregationResult]):
    config_class = AggregationConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, str]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "Data is required"
        if hasattr(input_data, '__dataframe__'):
            return True, ""
        if not isinstance(input_data, pd.DataFrame):
            return False, "Data must be a DataFrame"
        return True, ""
    
    def process(self, config: AggregationConfig, **kwargs) -> ProcessingResult:
        try:
            df = kwargs.get('data')
            if df is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入数据",
                    error="Data is required"
                )
            
            if not config.group_by_params:
                return ProcessingResult(
                    success=False,
                    message="未指定分组参数",
                    error="group_by_params is required"
                )
            
            # 分组聚合
            group_counts = {}
            detailed_stats = {}
            
            for group_key in config.group_by_params:
                if group_key not in df.columns:
                    continue
                
                grouped = df.groupby(group_key)
                group_counts[group_key] = grouped.size().to_dict()
                
                # 计算统计信息
                stats = {}
                for col in df.columns:
                    if col in config.group_by_params:
                        continue
                    
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats[col] = {
                            'mean': grouped[col].mean().to_dict(),
                            'std': grouped[col].std().to_dict(),
                            'min': grouped[col].min().to_dict(),
                            'max': grouped[col].max().to_dict()
                        }
                
                detailed_stats[group_key] = stats
            
            # 生成聚合结果
            groups = {}
            for group_key in config.group_by_params:
                if group_key not in df.columns:
                    continue
                
                grouped = df.groupby(group_key)
                groups[group_key] = grouped.groups if config.output_format == 'dict' else grouped
            
            result = AggregationResult(
                groups=groups,
                group_counts=group_counts,
                total_groups=len(config.group_by_params),
                detailed_stats=detailed_stats
            )
            
            return ProcessingResult(
                success=True,
                message=f"数据聚合完成，分组数: {len(config.group_by_params)}",
                data=result
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"数据聚合失败: {str(e)}",
                error=str(e)
            )
