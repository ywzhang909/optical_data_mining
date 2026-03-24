from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List, TypeVar, Generic, Type
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
from data_mining.experiment_analysis.models import (
    Experiment,
    DataCleaningResult,
    DataQualityReport,
    AggregationResult,
    ProcessorConfig,
)

T = TypeVar('T', bound=BaseModel)

class ProcessingResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    data: Optional[Any] = Field(None, description="处理数据")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据信息"
    )
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(default=0.0, description="执行时间(秒)")

class BaseProcessor(ABC, Generic[T]):
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or self._default_config()
        self._execution_times: List[float] = []
    
    @classmethod
    def _default_config(cls) -> ProcessorConfig:
        config_cls = getattr(cls, 'config_class', ProcessorConfig)
        kwargs = {}
        
        if hasattr(config_cls, 'model_fields') and 'processor_name' in config_cls.model_fields:
            kwargs['processor_name'] = cls.__name__
        if hasattr(config_cls, 'model_fields') and 'processor_version' in config_cls.model_fields:
            kwargs['processor_version'] = "1.0.0"
        if hasattr(config_cls, 'model_fields') and 'enabled' in config_cls.model_fields:
            kwargs['enabled'] = True
        if hasattr(config_cls, 'model_fields') and 'parallel' in config_cls.model_fields:
            kwargs['parallel'] = False
        
        return config_cls(**kwargs) if kwargs else config_cls()
    
    @property
    @abstractmethod
    def config_class(self) -> Type[ProcessorConfig]:
        return ProcessorConfig
    
    @abstractmethod
    def process(self, data: T, **kwargs) -> ProcessingResult:
        pass
    
    @abstractmethod
    def validate_input(self, data: T) -> Tuple[bool, Optional[str]]:
        pass
    
    def execute(self, data: T, **kwargs) -> ProcessingResult:
        import time
        start_time = time.time()
        
        try:
            is_valid, error_msg = self.validate_input(data, **kwargs)
            if not is_valid:
                execution_time = time.time() - start_time
                return ProcessingResult(
                    success=False,
                    message=f"输入验证失败: {error_msg}",
                    error=error_msg,
                    execution_time=execution_time
                )
            
            result = self.process(data, **kwargs)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.metadata['execution_times'] = self._execution_times
            result.metadata['total_execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            execution_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                message=error_msg,
                error=str(e),
                execution_time=execution_time
            )
    
    def record_execution_time(self, time: float):
        self._execution_times.append(time)
    
    def get_average_execution_time(self) -> float:
        if not self._execution_times:
            return 0.0
        return sum(self._execution_times) / len(self._execution_times)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            'execution_count': len(self._execution_times),
            'average_time': self.get_average_execution_time(),
            'min_time': min(self._execution_times) if self._execution_times else 0.0,
            'max_time': max(self._execution_times) if self._execution_times else 0.0
        }

class DataFrameProcessor(BaseProcessor[Dict[str, Any]]):
    config_class = ProcessorConfig
    
    def validate_input(self, data: Dict[str, Any] = None, **kwargs) -> Tuple[bool, Optional[str]]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "输入数据不能为空"
        if not isinstance(input_data, dict):
            return False, "输入数据必须是字典类型"
        if 'data' not in input_data:
            return False, "输入数据缺少 'data' 键"
        return True, None
    
    def process(self, data: Dict[str, Any], **kwargs) -> ProcessingResult:
        input_data = kwargs.get('data', data)
        return ProcessingResult(
            success=True,
            message="DataFrame processed",
            data=input_data.get('data') if isinstance(input_data, dict) else input_data
        )

class ExperimentDataProcessor(BaseProcessor[Experiment]):
    def validate_input(self, data: Experiment) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, Experiment):
            return False, "输入数据必须是 Experiment 对象"
        if not data.experiment_id:
            return False, "实验数据缺少 experiment_id"
        return True, None

class DataQualityProcessor(BaseProcessor[DataQualityReport]):
    pass

class DataCleaningProcessor(BaseProcessor[DataCleaningResult]):
    pass

class DataAggregationProcessor(BaseProcessor[AggregationResult]):
    pass
