import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
from data_mining.input_sources.input_stage_base import InputStage
try:
    from spots_data_pipeline import SpotsDataPipeline  # type: ignore
except Exception:
    SpotsDataPipeline = None  # type: ignore

class SpotsInputStageAdapter(InputStage):
    def load(self) -> pd.DataFrame:
        input_dir = os.environ.get('SPOTS_INPUT_DIR', './spots_input')
        output_dir = os.environ.get('SPOTS_OUTPUT_DIR', input_dir)
        output_format = os.environ.get('SPOTS_OUTPUT_FORMAT', 'parquet')
        df = None
        if SpotsDataPipeline is not None:
            try:
                pipeline = SpotsDataPipeline(input_dir, output_dir, output_format=output_format)
                df = pipeline.run()
            except Exception:
                df = None
        if df is None:
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
            except Exception:
                df = pd.DataFrame()
        if df is None:
            df = pd.DataFrame()
        return df
