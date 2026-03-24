import pytest
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Tuple
from data_mining.experiment_analysis.base_processor import (
    BaseProcessor, ProcessingResult, ProcessorConfig,
    DataFrameProcessor, ExperimentDataProcessor
)

class TestProcessorConfig(ProcessorConfig):
    test_value: str = "test"

class MockProcessor(BaseProcessor[dict]):
    config_class = TestProcessorConfig
    
    def validate_input(self, data: dict = None, **kwargs) -> tuple[bool, Optional[str]]:
        if data is None and 'data' not in kwargs:
            return False, "Data cannot be None"
        return True, None
    
    def process(self, data: dict, **kwargs) -> ProcessingResult:
        return ProcessingResult(
            success=True,
            message="Mock processor executed",
            data=data
        )

class DataFrameProcessorImpl(BaseProcessor[dict]):
    config_class = ProcessorConfig
    
    def validate_input(self, data: dict = None, **kwargs) -> tuple[bool, Optional[str]]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "Data cannot be None"
        return True, None
    
    def process(self, data: dict, **kwargs) -> ProcessingResult:
        return ProcessingResult(
            success=True,
            message="DataFrame processed",
            data=data
        )

class ExperimentDataProcessorImpl(BaseProcessor[BaseModel]):
    config_class = ProcessorConfig
    
    def validate_input(self, data: BaseModel = None, **kwargs) -> tuple[bool, Optional[str]]:
        if data is None:
            return False, "Experiment data cannot be None"
        return True, None
    
    def process(self, data: BaseModel, **kwargs) -> ProcessingResult:
        return ProcessingResult(
            success=True,
            message="Experiment processed",
            data=data
        )

def test_base_processor_initialization():
    """测试处理器初始化"""
    processor = MockProcessor()
    assert processor.config.processor_name == "MockProcessor"
    assert processor.config.enabled == True
    assert processor.config.parallel == False

def test_processor_execute():
    """测试处理器执行"""
    processor = MockProcessor()
    result = processor.execute({'test': 'data'})
    assert result.success == True
    assert result.message == "Mock processor executed"
    assert result.data == {'test': 'data'}

def test_processor_validation():
    """测试输入验证"""
    processor = MockProcessor()
    
    is_valid, error = processor.validate_input({'test': 'data'})
    assert is_valid == True
    assert error is None
    
    is_valid, error = processor.validate_input(None)
    assert is_valid == False
    assert error is not None

def test_execution_time_tracking():
    """测试执行时间跟踪"""
    processor = MockProcessor()
    import time
    start = time.time()
    result = processor.execute({'test': 'data'})
    elapsed = time.time() - start
    assert result.execution_time >= 0
    assert result.execution_time <= elapsed + 0.1

def test_average_execution_time():
    """测试平均执行时间计算"""
    processor = MockProcessor()
    
    for _ in range(3):
        processor.execute({'test': 'data'})
    
    stats = processor.get_execution_stats()
    assert stats['execution_count'] == 3

def test_execution_stats():
    """测试执行统计信息"""
    processor = MockProcessor()
    
    for _ in range(3):
        processor.execute({'test': 'data'})
    
    stats = processor.get_execution_stats()
    assert stats['execution_count'] == 3
    assert 'average_time' in stats
    assert 'min_time' in stats
    assert 'max_time' in stats

def test_dataframe_processor_validation():
    """测试DataFrame处理器验证"""
    processor = DataFrameProcessorImpl()
    
    is_valid, error = processor.validate_input({'data': {'key': 'value'}})
    assert is_valid == True
    assert error is None
    
    is_valid, error = processor.validate_input(None)
    assert is_valid == False

def test_experiment_data_processor_validation():
    """测试Experiment数据处理器验证"""
    processor = ExperimentDataProcessorImpl()
    
    is_valid, error = processor.validate_input(BaseModel())
    assert is_valid == True
    
    is_valid, error = processor.validate_input(None)
    assert is_valid == False

def test_processing_result_validation():
    """测试处理结果验证"""
    result = ProcessingResult(
        success=True,
        message="Test result",
        data={'test': 'data'},
        execution_time=1.0
    )
    assert result.success == True
    assert result.message == "Test result"
    assert result.execution_time == 1.0

def test_processor_with_error():
    """测试处理器错误处理"""
    processor = MockProcessor()
    
    result = processor.execute(None)
    assert result.success == False
    assert result.error is not None

def test_processor_config_defaults():
    """测试处理器配置默认值"""
    processor = MockProcessor()
    assert processor.config.processor_version == "1.0.0"
    assert processor.config.batch_size is None
