Examples: Plugin-based Input Sources

This document demonstrates how to use the plugin-based input system and how to run end-to-end pipelines.

Prerequisites
- Python >= 3.9
- The project already contains the plugin framework under src/data_mining/input_sources
- For real Kafka/RabbitMQ tests, install the client libraries (kafka-python, pika) if you plan to exercise those plugins

1) Using the Spots input (default) with environment configuration
- SPOTS_INPUT_DIR: directory containing spots TIFFs
- SPOTS_OUTPUT_DIR: directory to write outputs
- SPOTS_OUTPUT_FORMAT: parquet/json/csv/xlsx
- This relies on SpotsInputStage and SpotsInputStagePlugin registered in the registry

2) Registering a Kafka input at runtime
```python
from data_mining.input_sources.input_stage_registry import InputStageRegistry
from data_mining.input_sources.input_stage_kafka import KafkaInputStage

InputStageRegistry.register('Kafka', KafkaInputStage)
loader = InputStageRegistry.load('Kafka')
df = loader.load()
```

3) Registering a RabbitMQ input at runtime
```python
from data_mining.input_sources.input_stage_registry import InputStageRegistry
from data_mining.input_sources.input_stage_rabbitmq import RabbitMQInputStage

InputStageRegistry.register('RabbitMQ', RabbitMQInputStage)
loader = InputStageRegistry.load('RabbitMQ')
df = loader.load()
```

4) Using a Filesystem input stage
```python
from data_mining.input_sources.input_stage_registry import InputStageRegistry
from data_mining.input_sources.input_stage_filesystem import FileSystemInputStage

InputStageRegistry.register('FS', FileSystemInputStage)
loader = InputStageRegistry.load('FS')
df = loader.load()
```

5) End-to-end with ExperimentPipelineManager
- Create a standard pipeline, which now uses SpotsInputStage as the input stage by default, then quality, cleaning, and aggregation/export stages.
- Configure environment variables as needed to drive the input sources.

Note: All examples assume you have the required dependencies installed or you mock the external services in tests.
