from typing import List, Dict, Any, Optional, Type, Tuple
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import time
import asyncio
from .base_processor import BaseProcessor, ProcessorConfig

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineStage(str, Enum):
    LOAD = "load"
    FILTER = "filter"
    CLEAN = "clean"
    DENOISE = "denoise"
    SEGMENT = "segment"
    FEATURE_EXTRACT = "feature_extract"
    QUALITY = "quality"
    AGGREGATE = "aggregate"
    EXPORT = "export"

class PipelineStageConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    
    stage_name: PipelineStage = Field(..., description="阶段名称")
    processor_type: Type = Field(..., description="处理器类型")
    enabled: bool = Field(default=True, description="是否启用")
    dependencies: List[str] = Field(
        default_factory=list,
        description="依赖的阶段"
    )
    timeout: Optional[float] = Field(None, description="超时时间(秒)")
    retry_count: int = Field(default=0, description="重试次数")

class PipelineStepResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    stage: PipelineStage = Field(..., description="阶段")
    success: bool = Field(..., description="是否成功")
    result: Optional[Any] = Field(None, description="处理结果")
    execution_time: float = Field(..., description="执行时间")
    error: Optional[str] = Field(None, description="错误信息")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="指标信息"
    )

class PipelineExecutionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    pipeline_name: str = Field(default="pipeline", description="管道名称")
    pipeline_version: str = Field(default="1.0.0", description="管道版本")
    stages: List[PipelineStageConfig] = Field(default_factory=list, description="阶段配置")
    parallel_execution: bool = Field(default=False, description="并行执行")
    checkpoint_interval: int = Field(default=1, description="检查点间隔")
    enable_logging: bool = Field(default=True, description="启用日志")
    output_dir: str = Field(default="output", description="输出目录")

class PipelineManager:
    def __init__(self, config: Optional[PipelineExecutionConfig] = None):
        self.config = config or self._default_config()
        self.stages: Dict[str, PipelineStepResult] = {}
        self.current_status: PipelineStatus = PipelineStatus.PENDING
        self._processors: Dict[str, BaseProcessor] = {}
        self._execution_log: List[Dict[str, Any]] = []
    
    @classmethod
    def _default_config(cls) -> PipelineExecutionConfig:
        return PipelineExecutionConfig(
            pipeline_name="experiment_analysis_pipeline",
            pipeline_version="1.0.0",
            parallel_execution=False,
            checkpoint_interval=1
        )
    
    def register_processor(self, stage_name: str, processor: BaseProcessor):
        self._processors[stage_name] = processor
    
    def build_pipeline(self, config: PipelineExecutionConfig):
        self.config = config
        self._initialize_stages()
    
    def _initialize_stages(self):
        for stage_config in self.config.stages:
            if stage_config.enabled:
                self._processors[stage_config.stage_name.value] = stage_config.processor_type()
    
    def validate_pipeline(self) -> Tuple[bool, Optional[str]]:
        stage_names = [s.stage_name.value for s in self.config.stages]
        for stage in self.config.stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    return False, f"依赖阶段 {dep} 不存在"
        
        for stage in self.config.stages:
            if stage.enabled and stage.stage_name.value not in self._processors:
                return False, f"处理器 {stage.stage_name.value} 未注册"
        
        return True, None
    
    async def execute_stage(
        self,
        stage_config: PipelineStageConfig,
        context: Dict[str, Any]
    ) -> PipelineStepResult:
        start_time = time.time()
        stage_name = stage_config.stage_name.value
        
        try:
            processor = self._processors.get(stage_name)
            if not processor:
                raise ValueError(f"处理器 {stage_name} 未注册")
            
            for dep in stage_config.dependencies:
                if dep not in self.stages or not self.stages[dep].success:
                    raise ValueError(f"依赖阶段 {dep} 未成功执行")
            
            result = processor.execute(context.get('data'))
            
            execution_time = time.time() - start_time
            step_result = PipelineStepResult(
                stage=stage_config.stage_name,
                success=result.success,
                result=result.data,
                execution_time=execution_time,
                error=result.error,
                metrics=result.metadata
            )
            
            if result.success:
                context[stage_name] = result.data
                context['results'] = context.get('results', [])
                context['results'].append({
                    'stage': stage_name,
                    'result': result.data,
                    'metadata': result.metadata
                })
            
            return step_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return PipelineStepResult(
                stage=stage_config.stage_name,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def execute_pipeline(self, initial_data: Any) -> Dict[str, Any]:
        self.current_status = PipelineStatus.RUNNING
        
        try:
            is_valid, error_msg = self.validate_pipeline()
            if not is_valid:
                self.current_status = PipelineStatus.FAILED
                return {'status': self.current_status, 'error': error_msg}
            
            context = {
                'data': initial_data,
                'results': []
            }
            
            stage_configs = self.config.stages
            completed_stages = set()
            
            for stage_config in stage_configs:
                if not stage_config.enabled:
                    continue
                
                for dep in stage_config.dependencies:
                    if dep not in completed_stages:
                        raise ValueError(f"依赖阶段 {dep} 未完成")
                
                step_result = await self.execute_stage(stage_config, context)
                self.stages[stage_config.stage_name.value] = step_result
                
                if not step_result.success:
                    self.current_status = PipelineStatus.FAILED
                    return {
                        'status': self.current_status,
                        'error': step_result.error,
                        'completed_stages': list(self.stages.keys()),
                        'execution_log': self._execution_log
                    }
                
                completed_stages.add(stage_config.stage_name.value)
            
            self.current_status = PipelineStatus.COMPLETED
            return {
                'status': self.current_status,
                'data': context,
                'completed_stages': list(completed_stages),
                'execution_log': self._execution_log
            }
            
        except Exception as e:
            self.current_status = PipelineStatus.FAILED
            return {
                'status': self.current_status,
                'error': str(e),
                'completed_stages': list(self.stages.keys()),
                'execution_log': self._execution_log
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        return {
            'status': self.current_status,
            'completed_stages': list(self.stages.keys()),
            'failed_stages': [
                name for name, result in self.stages.items()
                if not result.success
            ],
            'execution_log': self._execution_log
        }

class PipelineExecutor:
    def __init__(self, manager: PipelineManager):
        self.manager = manager
    
    def execute(self, initial_data: Any) -> Dict[str, Any]:
        return asyncio.run(self.manager.execute_pipeline(initial_data))

class PipelineLogger:
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.logs: List[Dict[str, Any]] = []
    
    def log(self, level: str, message: str, **kwargs):
        log_entry = {
            'level': level,
            'message': message,
            'timestamp': time.time(),
            **kwargs
        }
        self.logs.append(log_entry)
        
        if self.enable_logging:
            print(f"[{level.upper()}] {message}")
    
    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs
    
    def export_logs(self, output_path: str):
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

# Backwards-compat alias for tests that import PipelineManagerConfig
class PipelineManagerConfig(PipelineExecutionConfig):
    pass
