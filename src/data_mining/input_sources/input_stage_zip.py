from typing import Optional
import os
import pandas as pd
from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from pydantic import Field

class ZipInputConfig(ProcessorConfig):
    zip_path: str = Field(default="", description="Path to ZIP file")

class ZipInputStage(BaseProcessor):
    config_class = ZipInputConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, Optional[str]]:
        return True, None
    
    def process(self, data=None, **kwargs) -> ProcessingResult:
        zip_path = os.environ.get('ZIP_INPUT', None)
        if not zip_path:
            return ProcessingResult(False, "ZIP_INPUT not configured", None, {}, 'config')
        try:
            from data_mining.input_sources.zip_input import ZipInput  # type: ignore
        except Exception as e:
            return ProcessingResult(False, f"ZipInput not available: {e}", None, {}, str(e))
        zi = ZipInput(zip_path)
        df = zi.read()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        return ProcessingResult(True, "ZIP input loaded", df, {}, None)
