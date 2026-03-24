import pytest
import pandas as pd
import numpy as np
from typing import Any

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """创建示例DataFrame"""
    return pd.DataFrame({
        'experiment_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'experiment_type_id': [2, 2, 2, 2, 2, 2, 2, 2],
        'experiment_polarization_state': ['线偏振', '线偏振', '圆偏振', '线偏振', '圆偏振', '线偏振', '圆偏振', '线偏振'],
        'experiment_radiation_mode': ['单模', '单模', '多模', '单模', '多模', '单模', '多模', '单模'],
        'experiment_radiation_power': [10.5, 11.2, 9.8, 12.0, 10.8, 11.5, 9.5, 11.8],
        '靶材': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金', '不锈钢', '铝合金', '不锈钢'],
        '距离': [1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0],
        'experiment_radiation_duration': [1.0, 1.5, 1.0, 2.0, 1.5, 1.0, 1.5, 2.0]
    })

def test_data_aggregation_processor_initialization():
    """测试数据聚合处理器初始化"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig()
    processor = DataAggregationProcessor(config)
    assert processor.config.processor_name == "DataAggregationProcessor"

def test_data_aggregation_process_success(sample_dataframe: pd.DataFrame):
    """测试数据聚合成功"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    from data_mining.experiment_analysis.models import AggregationResult
    config = AggregationConfig(group_by_params=['靶材', '距离'])
    processor = DataAggregationProcessor(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == True
    assert result.data is not None
    assert hasattr(result.data, 'total_groups')
    assert result.data.total_groups == 2
    assert hasattr(result.data, 'group_counts')
    assert '靶材' in result.data.group_counts
    assert '距离' in result.data.group_counts

def test_data_aggregation_with_numeric_stats(sample_dataframe: pd.DataFrame):
    """测试数据聚合的数值统计"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig(group_by_params=['靶材'])
    processor = DataAggregationProcessor(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == True
    assert result.data is not None
    assert hasattr(result.data, 'detailed_stats')
    assert '靶材' in result.data.detailed_stats
    stats = result.data.detailed_stats['靶材']
    assert 'experiment_radiation_power' in stats
    assert 'mean' in stats['experiment_radiation_power']

def test_data_aggregation_no_group_params(sample_dataframe: pd.DataFrame):
    """测试未指定分组参数的情况"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig(group_by_params=[])
    processor = DataAggregationProcessor(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == False
    assert result.error is not None

def test_data_aggregation_no_input_data():
    """测试无输入数据的情况"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig(group_by_params=['靶材'])
    processor = DataAggregationProcessor(config)
    
    result = processor.execute({})
    
    assert result.success == False
    assert result.error is not None
    assert 'Data is required' in result.error

def test_data_aggregation_config_defaults():
    """测试数据聚合配置默认值"""
    from data_mining.experiment_analysis.data_aggregation import AggregationConfig
    config = AggregationConfig()
    assert config.agg_funcs is not None
    assert config.output_format == 'dataframe'

def test_data_aggregation_execution_time():
    """测试数据聚合执行时间"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig(group_by_params=['靶材'])
    processor = DataAggregationProcessor(config)
    
    import time
    start = time.time()
    result = processor.execute({'data': pd.DataFrame()})
    elapsed = time.time() - start
    
    assert result.execution_time >= 0
    assert result.execution_time <= elapsed + 0.1

def test_data_aggregation_group_counts(sample_dataframe: pd.DataFrame):
    """测试分组计数"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig(group_by_params=['靶材'])
    processor = DataAggregationProcessor(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == True
    assert result.data is not None
    assert hasattr(result.data, 'group_counts')
    assert isinstance(result.data.group_counts, dict)
    assert '不锈钢' in result.data.group_counts
    assert '铝合金' in result.data.group_counts

def test_data_aggregation_statistics(sample_dataframe: pd.DataFrame):
    """测试统计信息计算"""
    from data_mining.experiment_analysis.data_aggregation import DataAggregationProcessor, AggregationConfig
    config = AggregationConfig(group_by_params=['靶材'])
    processor = DataAggregationProcessor(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == True
    assert result.data is not None
    assert hasattr(result.data, 'detailed_stats')
    assert result.data.detailed_stats is not None
    for group_stats in result.data.detailed_stats.values():
        for param_stats in group_stats.values():
            assert 'mean' in param_stats
            assert 'std' in param_stats
            assert 'min' in param_stats
            assert 'max' in param_stats
