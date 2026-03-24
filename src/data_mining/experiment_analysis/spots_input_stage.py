from typing import Optional
import os
import pandas as pd
import numpy as np
from pathlib import Path
from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from pydantic import Field

class SpotsInputConfig(ProcessorConfig):
    input_dir: str = Field(default="./spots_input", description="Input directory for spots data")
    output_dir: str = Field(default="./spots_input", description="Output directory")
    output_format: str = Field(default="parquet", description="Output format")

try:
    from spots_data_pipeline import SpotsDataPipeline  # type: ignore
except Exception:
    SpotsDataPipeline = None  # type: ignore


class SpotsInputStage(BaseProcessor):
    """Spots Data Input Stage: loads data from spots input, using SpotsDataPipeline when available, otherwise fallback to local TIFFs."""
    
    config_class = SpotsInputConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, Optional[str]]:
        return True, None
    
    def process(self, data=None, **kwargs) -> ProcessingResult:
        # Env-driven configuration for input/output
        input_dir = os.environ.get('SPOTS_INPUT_DIR', './spots_input')
        output_dir = os.environ.get('SPOTS_OUTPUT_DIR', input_dir)
        output_format = os.environ.get('SPOTS_OUTPUT_FORMAT', 'parquet')

        df = None
        try:
            if SpotsDataPipeline is not None:
                pipeline = SpotsDataPipeline(input_dir, output_dir, output_format=output_format)
                df = pipeline.run()
        except Exception:
            df = None

        if df is None:
            # Fallback: read TIFFs directly from input_dir
            try:
                from data_mining.image.common import read_tiff_to_numpy  # type: ignore
                input_path = Path(input_dir)
                paths = sorted([p for p in input_path.glob('*') if p.is_file() and p.suffix.lower() in {'.tiff', '.tif', '.tiff'}])
                records = []
                for p in paths:
                    img = read_tiff_to_numpy(str(p))
                    if img is None:
                        continue
                    total_power = float(np.sum(img))
                    cy, cx = 0.0, 0.0
                    try:
                        from scipy.ndimage import center_of_mass
                        cy, cx = center_of_mass(img)
                    except Exception:
                        cy, cx = 0.0, 0.0
                    records.append({
                        'path': str(p),
                        'total_power': total_power,
                        'centroid_x': float(cx),
                        'centroid_y': float(cy)
                    })
                df = pd.DataFrame(records)
            except Exception as e:
                return ProcessingResult(success=False, message="SpotsInputStage failed to read inputs", error=str(e))

        if df is None:
            df = pd.DataFrame()

        return ProcessingResult(success=True, data=df, message="Spots input loaded")
