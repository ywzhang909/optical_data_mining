import pytest
import pandas as pd
from typing import Any

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """创建示例DataFrame"""
    return pd.DataFrame({
        'experiment_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'experiment_type_id': [2, 2, 2, 2, 2, 2, 2, 2],
        'experiment_polarization_state': ['线偏振', '线偏振', None, '圆偏振', '线偏振', None, '线偏振', '圆偏振'],
        'experiment_radiation_mode': ['单模', '单模', '多模', '单模', None, '单模', '多模', None],
        'experiment_radiation_duration': [1.0, 1.5, 1.0, 2.0, None, 1.0, 1.5, None],
        'experiment_radiation_power': [10.5, 11.2, 9.8, 12.0, None, 10.8, 11.5, None],
        '_sub_beam': ['31路', '31路', None, '31路', None, '33路', None, '33路'],
        '子束': ['31路', '31路', '33路', '31路', '33路', '31路', '33路', '31路'],
        'experiment_laser_sub_beam': ['1111', '1111', '1010', '1111', '1010', '1111', '1010', '1111'],
        '靶材': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金', '不锈钢', '铝合金', '不锈钢'],
        '材质': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金', '不锈钢', '铝合金', '不锈钢'],
        '距离': [1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0],
        '靶材厚度': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        '横向风': [2.5, 2.8, 2.3, 2.6, 2.4, 2.7, 2.5, 2.6],
        '纵向风': [1.2, 1.5, 1.1, 1.3, 1.4, 1.6, 1.2, 1.4]
    })

def test_data_cleaning_processor_initialization():
    """测试数据清洗处理器初始化"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    assert processor.config.processor_name == "DataCleaningProcessor"

def test_data_cleaning_process_success(sample_dataframe: pd.DataFrame):
    """测试数据清洗成功"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    from data_mining.experiment_analysis.models import DataCleaningResult
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    result = processor.execute({'data': sample_dataframe})
    
    assert result.success == True
    assert result.data is not None
    assert hasattr(result.data, 'original_count')
    assert result.data.original_count == 8

def test_clean_material_info(sample_dataframe: pd.DataFrame):
    """测试材料信息清洗"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    result_df = processor._clean_material_info(sample_dataframe, config)
    
    assert '靶材' in result_df.columns

def test_clean_sub_beam_info(sample_dataframe: pd.DataFrame):
    """测试子束信息清洗"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    result_df = processor._clean_sub_beam_info(sample_dataframe, config)
    
    assert '_sub_beam' in result_df.columns

def test_clean_sub_beam_with_empty_sub_beam():
    """测试空子束信息处理"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    df = pd.DataFrame({
        '子束': [None, ''],
        'experiment_laser_sub_beam': ['1111', '1010']
    })
    
    result_df = processor._clean_sub_beam_info(df, config)
    
    assert '_sub_beam' in result_df.columns

def test_drop_missing_records(sample_dataframe: pd.DataFrame):
    """测试删除缺失记录"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    config.drop_na_params = ['experiment_polarization_state', 'experiment_radiation_mode']
    result_df = processor._drop_missing_records(sample_dataframe, config)
    
    assert result_df.dropna(how='any', subset=config.drop_na_params).shape[0] == result_df.shape[0]

def test_data_cleaning_no_input_data():
    """测试无输入数据的情况"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    result = processor.execute({})
    
    assert result.success == False
    assert result.error is not None
    assert 'Data is required' in result.error

def test_data_cleaning_config_defaults():
    """测试数据清洗配置默认值"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningConfig
    config = DataCleaningConfig()
    assert len(config.target_material_map) > 0
    assert len(config.sub_beam_map) > 0
    assert len(config.target_priority) > 0
    assert len(config.laser_param_groups) > 0

def test_data_cleaning_execution_time():
    """测试数据清洗执行时间"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    import time
    start = time.time()
    result = processor.execute({'data': pd.DataFrame()})
    elapsed = time.time() - start
    
    assert result.execution_time >= 0
    assert result.execution_time <= elapsed + 0.1

def test_cleaning_details():
    """测试清洗详情"""
    from data_mining.experiment_analysis.data_cleaning import DataCleaningProcessor, DataCleaningConfig
    config = DataCleaningConfig()
    processor = DataCleaningProcessor(config)
    
    result = processor.execute({'data': pd.DataFrame()})
    
    assert result.data is not None
    assert hasattr(result.data, 'cleaning_details')
    assert isinstance(result.data.cleaning_details, list)
