import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from data_mining.experiment_analysis.data_loader import DataLoader, DataLoaderConfig
from data_mining.experiment_analysis.models import DatabaseConfig

@pytest.fixture
def mock_dataframe():
    """创建模拟DataFrame"""
    data = {
        'experiment_id': [1, 2, 3, 4, 5],
        'experiment_num': ['20250101', '20250102', '20250103', '20250104', '20250105'],
        'experiment_group': [1, 1, 2, 2, 1],
        'experiment_type_id': [2, 2, 2, 2, 2],
        'experiment_polarization_state': ['线偏振', '线偏振', '圆偏振', '线偏振', '圆偏振'],
        'experiment_radiation_mode': ['单模', '单模', '多模', '单模', '多模'],
        'experiment_radiation_duration': [1.0, 1.5, 1.0, 2.0, 1.5],
        'experiment_radiation_power': [10.5, 11.2, 9.8, 12.0, 10.8],
        'experiment_laser_sub_beam': ['1111', '1111', '1010', '1111', '1010'],
        'experiment_extra': [
            {'key1': 'value1', 'key2': 'value2'},
            None,
            {'key1': 'value1'},
            None,
            {'key2': 'value2'}
        ],
        '横向风': [2.5, 2.8, 2.3, 2.6, 2.4],
        '纵向风': [1.2, 1.5, 1.1, 1.3, 1.4],
        '湍流': [0.3, 0.4, 0.2, 0.35, 0.25],
        '通风': ['强通风', '弱通风', '强通风', '弱通风', '强通风'],
        '厚度': [2.0, 2.0, 2.0, 2.0, 2.0],
        '靶材': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金'],
        '距离': [1.0, 1.0, 1.5, 1.0, 1.5],
        '靶材厚度': [2.0, 2.0, 2.0, 2.0, 2.0],
        '靶材形状': ['圆形', '圆形', '方形', '圆形', '方形'],
        '材质': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金'],
        '子束': ['31路', '31路', '33路', '31路', '33路']
    }
    return pd.DataFrame(data)

def test_data_loader_initialization():
    """测试数据加载器初始化"""
    config = DataLoaderConfig(table_name='experiment')
    processor = DataLoader(config)
    assert processor.config.processor_name == "DataLoader"
    assert processor.config.enabled == True

def test_data_loader_process_success():
    """测试数据加载成功"""
    config = DataLoaderConfig(table_name='experiment')
    processor = DataLoader(config)
    
    # Mock pandas read_sql_table
    with patch('data_mining.experiment_analysis.data_loader.pd.read_sql_table') as mock_read:
        with patch('data_mining.experiment_analysis.data_loader.pd.to_datetime') as mock_to_datetime:
            with patch('data_mining.experiment_analysis.data_loader.pd.concat') as mock_concat:
                mock_read.return_value = pd.DataFrame()
                mock_to_datetime.return_value = pd.DataFrame()
                mock_concat.return_value = pd.DataFrame()
                
                result = processor.execute({'config': config})
                
                assert result.success == True
                assert 'data' in result.data

def test_data_loader_process_error():
    """测试数据加载失败"""
    config = DataLoaderConfig(table_name='experiment')
    processor = DataLoader(config)
    
    with patch('data_mining.experiment_analysis.data_loader.pd.read_sql_table') as mock_read:
        mock_read.side_effect = Exception("Database connection failed")
        
        result = processor.execute({'config': config})
        
        assert result.success == False
        assert 'error' in result.message

def test_extract_extra_fields():
    """测试提取额外字段"""
    processor = DataLoader(DataLoaderConfig(table_name='test'))
    
    df = pd.DataFrame({
        'experiment_extra': [
            {'key1': 'value1', 'key2': 'value2'},
            None,
            {'key1': 'value1'}
        ]
    })
    
    result_df = processor._extract_extra_fields(df)
    
    assert 'key1' in result_df.columns
    assert 'key2' in result_df.columns

def test_extract_extra_fields_with_missing():
    """测试提取额外字段时处理缺失值"""
    processor = DataLoader(DataLoaderConfig(table_name='test'))
    
    df = pd.DataFrame({
        'experiment_extra': [None, None, None]
    })
    
    result_df = processor._extract_extra_fields(df)
    
    assert 'key1' not in result_df.columns
    assert 'key2' not in result_df.columns

def test_data_loader_with_empty_dataframe():
    """测试空DataFrame处理"""
    config = DataLoaderConfig(table_name='experiment')
    processor = DataLoader(config)
    
    with patch('data_mining.experiment_analysis.data_loader.pd.read_sql_table') as mock_read:
        mock_read.return_value = pd.DataFrame()
        
        result = processor.execute({'config': config})
        
        assert result.success == True
        assert len(result.data) == 0

def test_data_loader_config_validation():
    """测试数据加载器配置验证"""
    config = DataLoaderConfig(table_name='experiment')
    assert config.table_name == 'experiment'
    assert config.use_cache == True

def test_data_loader_execution_time():
    """测试数据加载执行时间"""
    config = DataLoaderConfig(table_name='experiment')
    processor = DataLoader(config)
    
    import time
    time.sleep(0.1)
    result = processor.execute({'config': config})
    
    assert result.execution_time >= 0.1
