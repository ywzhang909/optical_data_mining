from typing import Optional
import os
import pandas as pd
from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from pydantic import Field

class KafkaInputConfig(ProcessorConfig):
    bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    topic: str = Field(default="spots", description="Kafka topic")

class KafkaInputStage(BaseProcessor):
    config_class = KafkaInputConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, Optional[str]]:
        return True, None
    
    def process(self, data=None, **kwargs) -> ProcessingResult:
        try:
            from data_mining.input_sources.kafka_input import KafkaInput  # type: ignore
        except Exception as e:
            return ProcessingResult(False, f"KafkaInput not available: {e}", None, {}, str(e))

        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(',')
        topic = os.environ.get("KAFKA_TOPIC", "spots")
        ki = KafkaInput(bootstrap_servers, topic)
        try:
            df = ki.read()
        except Exception as e:
            return ProcessingResult(False, f"Kafka read failed: {e}", None, {}, str(e))

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        return ProcessingResult(True, "Kafka input loaded", df, {}, None)
