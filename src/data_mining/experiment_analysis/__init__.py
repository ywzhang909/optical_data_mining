from .models import (
    Experiment,
    DataQualityReport,
    DataCleaningConfig,
    DataCleaningResult,
    AggregationConfig,
    AggregationResult,
    DatabaseConfig,
    ExperimentFilter,
    DataAnalysisPipeline
)

from .base_processor import (
    BaseProcessor,
    ProcessingResult,
    ProcessorConfig,
    DataFrameProcessor,
    ExperimentDataProcessor,
    DataQualityProcessor,
    DataCleaningProcessor,
    DataAggregationProcessor
)

from .pipeline import (
    PipelineManager,
    PipelineExecutor,
    PipelineLogger,
    PipelineStatus,
    PipelineStage,
    PipelineStageConfig,
    PipelineStepResult,
    PipelineExecutionConfig
)

from .pipeline_manager import (
    ExperimentPipelineManager,
    PipelineManagerConfig
)

# Re-export input sources for convenient access
from ..input_sources import KafkaInput, RabbitMQInput, FileSystemInput, ZipInput, KafkaInputStage, RabbitMQInputStage, FileSystemInputStage, ZipInputStage  # type: ignore

# Import denoise processor
try:
    from .denoise_processor import (
        ImageDenoiseProcessor,
        SpotImageDenoiseProcessor,
        DenoiseProcessorConfig,
        ImageDenoiseInput,
        ImageDenoiseOutput,
        create_denoise_processor,
        quick_denoise
    )
    _DENOISE_AVAILABLE = True
except ImportError:
    _DENOISE_AVAILABLE = False

# Import segment processor
try:
    from .segment_processor import (
        ImageSegmentProcessor,
        SpotImageSegmentProcessor,
        SegmentProcessorConfig,
        SegmentInput,
        SegmentOutput,
        create_segment_processor,
        quick_segment
    )
    _SEGMENT_AVAILABLE = True
except ImportError:
    _SEGMENT_AVAILABLE = False

# Import feature extract processor
try:
    from .feature_extract_processor import (
        ImageFeatureExtractProcessor,
        SpotImageFeatureExtractProcessor,
        FeatureExtractConfig,
        FeatureExtractInput,
        ExtractedFeatures,
        create_feature_processor,
        quick_extract_features
    )
    _FEATURE_AVAILABLE = True
except ImportError:
    _FEATURE_AVAILABLE = False

__all__ = [
    'Experiment',
    'DataQualityReport',
    'DataCleaningConfig',
    'DataCleaningResult',
    'AggregationConfig',
    'AggregationResult',
    'DatabaseConfig',
    'ExperimentFilter',
    'DataAnalysisPipeline',
    'BaseProcessor',
    'ProcessingResult',
    'ProcessorConfig',
    'DataFrameProcessor',
    'ExperimentDataProcessor',
    'DataQualityProcessor',
    'DataCleaningProcessor',
    'DataAggregationProcessor',
    'PipelineManager',
    'PipelineExecutor',
    'PipelineLogger',
    'PipelineStatus',
    'PipelineStage',
    'PipelineStageConfig',
    'PipelineStepResult',
    'PipelineExecutionConfig',
    'ExperimentPipelineManager',
    'PipelineManagerConfig'
    , 'KafkaInput', 'RabbitMQInput', 'FileSystemInput', 'ZipInput', 'KafkaInputStage', 'RabbitMQInputStage', 'FileSystemInputStage', 'ZipInputStage'
]

# Add denoise exports if available
if _DENOISE_AVAILABLE:
    __all__.extend([
        'ImageDenoiseProcessor',
        'SpotImageDenoiseProcessor',
        'DenoiseProcessorConfig',
        'ImageDenoiseInput',
        'ImageDenoiseOutput',
        'create_denoise_processor',
        'quick_denoise'
    ])

# Add segment exports if available
if _SEGMENT_AVAILABLE:
    __all__.extend([
        'ImageSegmentProcessor',
        'SpotImageSegmentProcessor',
        'SegmentProcessorConfig',
        'SegmentInput',
        'SegmentOutput',
        'create_segment_processor',
        'quick_segment'
    ])

# Add feature extract exports if available
if _FEATURE_AVAILABLE:
    __all__.extend([
        'ImageFeatureExtractProcessor',
        'SpotImageFeatureExtractProcessor',
        'FeatureExtractConfig',
        'FeatureExtractInput',
        'ExtractedFeatures',
        'create_feature_processor',
        'quick_extract_features'
    ])
