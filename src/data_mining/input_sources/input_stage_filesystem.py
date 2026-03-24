from typing import Optional
import os
import pandas as pd
from pathlib import Path
from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from pydantic import Field

class FileSystemInputConfig(ProcessorConfig):
    input_dir: str = Field(default="./spots_input", description="Input directory")

class FileSystemInputStage(BaseProcessor):
    config_class = FileSystemInputConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, Optional[str]]:
        return True, None
    
    def process(self, data=None, **kwargs) -> ProcessingResult:
        input_dir = os.environ.get('FS_INPUT_DIR', './spots_input')
        try:
            from data_mining.input_sources.filesystem_input import FileSystemInput  # type: ignore
        except Exception as e:
            return ProcessingResult(False, f"FilesystemInput not available: {e}", None, {}, str(e))
        fsi = FileSystemInput(input_dir)
        df = fsi.read()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        return ProcessingResult(True, "Filesystem input loaded", df, {}, None)
