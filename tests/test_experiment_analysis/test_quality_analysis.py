import pytest
import pandas as pd
import numpy as np
from data_mining.experiment_analysis.quality_analysis import QualityAnalyzer, QualityAnalyzerConfig
from data_mining.experiment_analysis.models import DataQualityReport

@pytest.fixture
def sample_dataframe():
    """创建示例DataFrame"""
    return pd.DataFrame({
        'experiment_id': [1, 2, 3, 4, 5],
        'experiment_type_id': [2, 2, 2, 2, 2],
        'experiment_polarization_state': ['线偏振', '线偏振', None, '圆偏振', None],
        'experiment_radiation_mode': ['单模', '单模', '多模', '单模', None],
        'experiment_radiation_duration': [1.0, 1.5, 1.0, 2.0, None],
        'experiment_radiation_power': [10.5, 11.2, 9.8, 12.0, None],
        '_sub_beam': ['31路', '31路', None, '31路', None],
        '靶材': ['不锈钢', '不锈钢', '铝合金', '不锈钢', None],
        '距离': [1.0, 1.0, 1.5, 1.0, None],
        '靶材厚度': [2.0, 2.0, 2.0, 2.0, None],
        '材质': ['不锈钢', '不锈钢', '铝合金', '不锈钢', None],
        '横向风': [2.5, 2.8, 2.3, 2.6, 2.4],
        '纵向风': [1.2, 1.5, 1.1, 1.3, 1.4],
        '湍流': [0.3, 0.4, 0.2, 0.35, 0.25],
        '通风': ['强通风', '弱通风', '强通风', '弱通风', '强通风']
    })

def test_quality_analyzer_initialization():
    """测试质量分析器初始化"""
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    assert processor.config.processor_name == "QualityAnalyzer"

def test_quality_analyzer_process_success(sample_dataframe):
    """测试质量分析成功"""
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == True
    assert isinstance(result.data, DataQualityReport)
    assert result.data.total_count == 5
    assert result.data.complete_records == 3
    assert result.data.incomplete_records == 2
    assert result.data.missing_rate > 0

def test_quality_analyzer_missing_stats(sample_dataframe):
    """测试缺失统计计算"""
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert 'laser' in result.data.param_missing_stats
    assert 'target' in result.data.param_missing_stats
    assert 'env' in result.data.param_missing_stats
    assert result.data.laser_params_missing > 0
    assert result.data.target_params_missing > 0

def test_quality_analyzer_all_complete(sample_dataframe):
    """测试所有数据完整的情况"""
    complete_df = sample_dataframe.dropna()
    
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    
    result = processor.execute({'data': complete_df})
    
    assert result.data.missing_rate == 0
    assert result.data.complete_records == 5
    assert result.data.incomplete_records == 0

def test_quality_analyzer_all_incomplete():
    """测试所有数据缺失的情况"""
    incomplete_df = pd.DataFrame({
        'experiment_id': [1, 2, 3],
        'experiment_type_id': [2, 2, 2],
        'experiment_polarization_state': [None, None, None],
        'experiment_radiation_mode': [None, None, None],
        'experiment_radiation_duration': [None, None, None],
        'experiment_radiation_power': [None, None, None],
        '_sub_beam': [None, None, None],
        '靶材': [None, None, None],
        '距离': [None, None, None],
        '靶材厚度': [None, None, None],
        '材质': [None, None, None],
        '横向风': [None, None, None],
        '纵向风': [None, None, None],
        '湍流': [None, None, None],
        '通风': [None, None, None]
    })
    
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    
    result = processor.execute({'data': incomplete_df})
    
    assert result.data.missing_rate == 1.0
    assert result.data.complete_records == 0

def test_quality_analyzer_no_input_data():
    """测试无输入数据的情况"""
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    
    result = processor.execute({})
    
    assert result.success == False
    assert 'Data is required' in result.error

def test_quality_analyzer_config_defaults():
    """测试质量分析器配置默认值"""
    config = QualityAnalyzerConfig()
    assert len(config.laser_params) > 0
    assert len(config.target_params) > 0
    assert len(config.env_params) > 0

def test_quality_analyzer_execution_time():
    """测试质量分析执行时间"""
    config = QualityAnalyzerConfig()
    processor = QualityAnalyzer(config)
    
    import time
    time.sleep(0.1)
    result = processor.execute({'data': pd.DataFrame()})
    
    assert result.execution_time >= 0.1
