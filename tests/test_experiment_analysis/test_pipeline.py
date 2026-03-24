import pytest
from unittest.mock import Mock, MagicMock, patch
from data_mining.experiment_analysis.pipeline import (
    PipelineManager, PipelineExecutor, PipelineLogger,
    PipelineStatus, PipelineStage, PipelineStageConfig,
    PipelineStepResult, PipelineExecutionConfig,
    PipelineManagerConfig
)
from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult, ProcessorConfig

class MockProcessorConfig(ProcessorConfig):
    pass

class MockProcessor(BaseProcessor):
    config_class = MockProcessorConfig
    
    def validate_input(self, data):
        return True, None
    
    def process(self, data, **kwargs):
        return ProcessingResult(
            success=True,
            message="Mock processor executed",
            data=data,
            execution_time=0.0
        )

def test_pipeline_manager_initialization():
    """测试管道管理器初始化"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        pipeline_version="1.0.0"
    )
    manager = PipelineManager(config)
    assert manager.config.pipeline_name == "test_pipeline"
    assert manager.current_status == PipelineStatus.PENDING

def test_pipeline_manager_register_processor():
    """测试管道管理器注册处理器"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        stages=[]
    )
    manager = PipelineManager(config)
    
    processor = MockProcessor()
    manager.register_processor('test_stage', processor)
    
    assert 'test_stage' in manager._processors

def test_pipeline_manager_validate_pipeline():
    """测试管道验证"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.LOAD,
                processor_type=MockProcessor,
                enabled=True
            )
        ]
    )
    manager = PipelineManager(config)
    manager.register_processor('LOAD', MockProcessor())
    
    is_valid, error = manager.validate_pipeline()
    assert is_valid == True

def test_pipeline_stage_config():
    """测试管道阶段配置"""
    config = PipelineStageConfig(
        stage_name=PipelineStage.LOAD,
        processor_type=MockProcessor,
        enabled=True,
        dependencies=[]
    )
    assert config.stage_name == PipelineStage.LOAD
    assert config.enabled == True

def test_pipeline_execution_config():
    """测试管道执行配置"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        pipeline_version="1.0.0",
        parallel_execution=True
    )
    assert config.parallel_execution == True

def test_pipeline_logger():
    """测试管道日志记录器"""
    logger = PipelineLogger(enable_logging=True)
    
    logger.log('INFO', 'Test message')
    logger.log('ERROR', 'Test error')
    
    assert len(logger.logs) == 2

def test_pipeline_executor():
    """测试管道执行器"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.LOAD,
                processor_type=MockProcessor,
                enabled=True
            )
        ]
    )
    manager = PipelineManager(config)
    manager.register_processor('LOAD', MockProcessor())
    
    executor = PipelineExecutor(manager)
    
    with patch.object(manager, 'execute_pipeline') as mock_execute:
        mock_execute.return_value = {'data': 'test'}
        result = executor.execute({'data': 'test'})
        
        assert 'data' in result

def test_pipeline_step_result():
    """测试管道步骤结果"""
    result = PipelineStepResult(
        stage=PipelineStage.LOAD,
        success=True,
        execution_time=1.0
    )
    assert result.stage == PipelineStage.LOAD
    assert result.success == True
    assert result.execution_time == 1.0

def test_pipeline_status():
    """测试管道状态枚举"""
    assert PipelineStatus.PENDING.value == "pending"
    assert PipelineStatus.RUNNING.value == "running"
    assert PipelineStatus.COMPLETED.value == "completed"
    assert PipelineStatus.FAILED.value == "failed"

def test_pipeline_stage():
    """测试管道阶段枚举"""
    assert PipelineStage.LOAD.value == "load"
    assert PipelineStage.FILTER.value == "filter"
    assert PipelineStage.QUALITY.value == "quality"

def test_pipeline_manager_get_pipeline_status():
    """测试获取管道状态"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        stages=[]
    )
    manager = PipelineManager(config)
    
    status = manager.get_pipeline_status()
    assert status['status'] == PipelineStatus.PENDING
    assert status['completed_stages'] == []

def test_pipeline_manager_with_dependencies():
    """测试管道依赖关系"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.LOAD,
                processor_type=MockProcessor,
                enabled=True,
                dependencies=[]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.QUALITY,
                processor_type=MockProcessor,
                enabled=True,
                dependencies=[PipelineStage.LOAD]
            )
        ]
    )
    manager = PipelineManager(config)
    manager.register_processor('LOAD', MockProcessor())
    manager.register_processor('QUALITY', MockProcessor())
    
    is_valid, error = manager.validate_pipeline()
    assert is_valid == True

def test_pipeline_manager_empty_stages():
    """测试空管道配置"""
    config = PipelineExecutionConfig(
        pipeline_name="test_pipeline",
        stages=[]
    )
    manager = PipelineManager(config)
    
    is_valid, error = manager.validate_pipeline()
    assert is_valid == True
