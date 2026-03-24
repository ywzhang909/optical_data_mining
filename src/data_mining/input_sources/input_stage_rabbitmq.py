from typing import Optional
import os
import pandas as pd
from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from pydantic import Field

class RabbitMQInputConfig(ProcessorConfig):
    host: str = Field(default="localhost", description="RabbitMQ host")
    queue: str = Field(default="spots", description="RabbitMQ queue")

class RabbitMQInputStage(BaseProcessor):
    config_class = RabbitMQInputConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, Optional[str]]:
        return True, None
    
    def process(self, data=None, **kwargs) -> ProcessingResult:
        try:
            from data_mining.input_sources.rabbitmq_input import RabbitMQInput  # type: ignore
        except Exception as e:
            return ProcessingResult(False, f"RabbitMQInput not available: {e}", None, {}, str(e))

        host = os.environ.get("RABBIT_HOST", "localhost")
        queue = os.environ.get("RABBIT_QUEUE", "spots")
        ri = RabbitMQInput(host, queue)
        try:
            df = ri.read()
        except Exception as e:
            return ProcessingResult(False, f"RabbitMQ read failed: {e}", None, {}, str(e))

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        return ProcessingResult(True, "RabbitMQ input loaded", df, {}, None)
