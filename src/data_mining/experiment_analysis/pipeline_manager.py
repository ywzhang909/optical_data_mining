from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from .base_processor import BaseProcessor, ProcessingResult
from .pipeline import PipelineManager, PipelineExecutionConfig, PipelineStageConfig, PipelineStage, PipelineExecutor

class PipelineManagerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    manager_name: str = Field(..., description="管理器名称")
    version: str = Field(default="1.0.0", description="版本号")
    default_output_dir: str = Field(default="output/experiment_analysis", description="默认输出目录")
    auto_register_processors: bool = Field(default=True, description="自动注册处理器")

class ExperimentPipelineManager:
    def __init__(self, config: Optional[PipelineManagerConfig] = None):
        self.config = config or self._default_config()
        self.pipeline_managers: Dict[str, PipelineManager] = {}
        self._processor_registry: Dict[str, Type[BaseProcessor]] = {}
    
    @classmethod
    def _default_config(cls) -> PipelineManagerConfig:
        return PipelineManagerConfig(
            manager_name="experiment_analysis_manager",
            version="1.0.0",
            default_output_dir="output/experiment_analysis",
            auto_register_processors=True
        )
    
    def register_processor(self, stage_name: str, processor_class: Type[BaseProcessor]):
        self._processor_registry[stage_name] = processor_class
    
    def build_pipeline(
        self,
        pipeline_name: str,
        stages_config: List[PipelineStageConfig]
    ) -> PipelineManager:
        if pipeline_name not in self.pipeline_managers:
            config = PipelineExecutionConfig(
                pipeline_name=pipeline_name,
                pipeline_version="1.0.0",
                stages=stages_config
            )
            manager = PipelineManager(config)
            self.pipeline_managers[pipeline_name] = manager
        
        return self.pipeline_managers[pipeline_name]
    
    def create_analysis_pipeline(
        self,
        pipeline_name: str = "standard_analysis"
    ) -> PipelineManager:
        self._register_standard_processors()
        
        stages_config = [
            PipelineStageConfig(
                stage_name=PipelineStage.LOAD,
                processor_type=self._processor_registry['DataLoader'],
                enabled=True,
                dependencies=[]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.FILTER,
                processor_type=self._processor_registry['ExperimentFilter'],
                enabled=True,
                dependencies=[PipelineStage.LOAD]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.QUALITY,
                processor_type=self._processor_registry['QualityAnalyzer'],
                enabled=True,
                dependencies=[PipelineStage.FILTER]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.CLEAN,
                processor_type=self._processor_registry['DataCleaner'],
                enabled=True,
                dependencies=[PipelineStage.QUALITY]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.AGGREGATE,
                processor_type=self._processor_registry['DataAggregator'],
                enabled=True,
                dependencies=[PipelineStage.CLEAN]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.EXPORT,
                processor_type=self._processor_registry['DataExporter'],
                enabled=True,
                dependencies=[PipelineStage.AGGREGATE]
            )
        ]
        
        return self.build_pipeline(pipeline_name, stages_config)
    
    def _register_standard_processors(self):
        from .spots_input_stage import SpotsInputStage
        from ..input_sources.input_stage_spots_adapter import SpotsInputStageAdapter  # type: ignore
        from ..input_sources.input_stage_kafka import KafkaInputStage  # type: ignore
        from ..input_sources.input_stage_rabbitmq import RabbitMQInputStage  # type: ignore
        from ..input_sources.input_stage_filesystem import FileSystemInputStage  # type: ignore
        from ..input_sources.input_stage_zip import ZipInputStage  # type: ignore
        from .filter import ExperimentFilter
        from .quality_analysis import QualityAnalyzer
        from .data_cleaning import DataCleaner
        from .data_aggregation import DataAggregator
        from .exporter import DataExporter
        # Use SpotsInputStageAdapter (plugin-based) for the loading step
        self.register_processor('DataLoader', SpotsInputStageAdapter)
        # Additional input stage plugins for extensibility
        self.register_processor('KafkaInputLoader', KafkaInputStage)
        self.register_processor('RabbitMQInputLoader', RabbitMQInputStage)
        self.register_processor('FilesystemInputLoader', FileSystemInputStage)
        self.register_processor('ZipInputLoader', ZipInputStage)
        self.register_processor('ExperimentFilter', ExperimentFilter)
        self.register_processor('QualityAnalyzer', QualityAnalyzer)
        self.register_processor('DataCleaner', DataCleaner)
        self.register_processor('DataAggregator', DataAggregator)
        self.register_processor('DataExporter', DataExporter)
    
    def compile_pipeline(
        self,
        pipeline_name: str,
        stages_config: List[PipelineStageConfig]
    ) -> PipelineManager:
        manager = self.build_pipeline(pipeline_name, stages_config)
        
        is_valid, error_msg = manager.validate_pipeline()
        if not is_valid:
            raise ValueError(f"Pipeline 编译失败: {error_msg}")
        
        return manager
    
    def execute_pipeline(
        self,
        pipeline_name: str,
        initial_data: Any
    ) -> Dict[str, Any]:
        manager = self.pipeline_managers.get(pipeline_name)
        if not manager:
            raise ValueError(f"Pipeline {pipeline_name} 未找到")
        
        executor = PipelineExecutor(manager)
        return executor.execute(initial_data)
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        manager = self.pipeline_managers.get(pipeline_name)
        if not manager:
            raise ValueError(f"Pipeline {pipeline_name} 未找到")
        
        return manager.get_pipeline_status()
    
    def list_pipelines(self) -> List[str]:
        return list(self.pipeline_managers.keys())
