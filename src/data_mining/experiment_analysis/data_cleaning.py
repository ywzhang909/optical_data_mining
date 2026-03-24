from typing import Dict, Any, List
import pandas as pd
from .base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from .models import DataCleaningConfig, DataCleaningResult

class DataCleaningProcessor(BaseProcessor[DataCleaningResult]):
    config_class = DataCleaningConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, str]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "Data is required"
        if hasattr(input_data, '__dataframe__'):
            return True, ""
        if not isinstance(input_data, pd.DataFrame):
            return False, "Data must be a DataFrame"
        return True, ""
    
    def process(self, config: DataCleaningConfig, **kwargs) -> ProcessingResult:
        try:
            df = kwargs.get('data')
            if df is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入数据",
                    error="Data is required"
                )
            
            original_count = len(df)
            missing_stats_before = self._calculate_missing_stats(df, config)
            
            # 清洗靶材信息
            df = self._clean_material_info(df, config)
            
            # 清洗子束信息
            df = self._clean_sub_beam_info(df, config)
            
            # 删除包含缺失值的记录
            df = self._drop_missing_records(df, config)
            
            cleaned_count = len(df)
            removed_count = original_count - cleaned_count
            removal_rate = removed_count / original_count if original_count > 0 else 0
            
            missing_stats_after = self._calculate_missing_stats(df, config)
            
            details = []
            for param in config.laser_param_groups.get('laser_params', []):
                details.append({
                    'parameter': param,
                    'removed': original_count - cleaned_count
                })
            
            result = DataCleaningResult(
                original_count=original_count,
                cleaned_count=cleaned_count,
                removed_count=removed_count,
                removal_rate=removal_rate,
                missing_stats_before=missing_stats_before,
                missing_stats_after=missing_stats_after,
                cleaning_details=details
            )
            
            return ProcessingResult(
                success=True,
                message=f"数据清洗完成，移除 {removed_count} 条记录",
                data=result
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"数据清洗失败: {str(e)}",
                error=str(e)
            )
    
    def _calculate_missing_stats(self, df: pd.DataFrame, config: DataCleaningConfig) -> Dict[str, int]:
        stats = {}
        for group_name, params in config.laser_param_groups.items():
            total = 0
            for param in params:
                if param in df.columns:
                    missing = df[param].isna().sum()
                    stats[f"{group_name}_{param}"] = missing
                    total += missing
            stats[f"{group_name}_total"] = total
        return stats
    
    def _clean_material_info(self, df: pd.DataFrame, config: DataCleaningConfig) -> pd.DataFrame:
        df = df.copy()
        
        # 靶材映射
        df['靶材'] = df['靶材'].map(config.target_material_map)
        
        # 优先使用材质信息
        for priority_param in config.target_priority:
            if priority_param in df.columns:
                mask = (df['靶材'].isna() | (df['靶材'] == '纯铝')) & df[priority_param].notna()
                df.loc[mask, '靶材'] = df.loc[mask, priority_param]
        
        return df
    
    def _clean_sub_beam_info(self, df: pd.DataFrame, config: DataCleaningConfig) -> pd.DataFrame:
        df = df.copy()
        
        # 子束映射
        df['_sub_beam'] = df['子束'].map(config.sub_beam_map)
        
        # 如果子束为空，统计 experiment_laser_sub_beam 中的 '1' 数量
        mask = df['子束'].isna() | (df['子束'].astype(str).str.strip() == '')
        df.loc[mask, '_sub_beam'] = df.loc[mask, 'experiment_laser_sub_beam'].apply(
            lambda x: f"{x.count('1')}路" if pd.notna(x) and isinstance(x, str) else pd.NaT
        )
        
        return df
    
    def _drop_missing_records(self, df: pd.DataFrame, config: DataCleaningConfig) -> pd.DataFrame:
        if not config.drop_na_params:
            return df
        
        return df.dropna(how='any', subset=config.drop_na_params)
