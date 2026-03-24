import os
import tempfile
from pathlib import Path
import numpy as np

import pandas as pd
import pytest

from data_mining.experiment_analysis.spots_input_stage import SpotsInputStage
from data_mining.image.common import SpotImage


def _make_test_tiff(path: Path, v=1.0):
    from PIL import Image
    arr = (np.ones((10, 10)) * 255 * v).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def test_spots_input_stage_pipeline_path(monkeypatch):
    # Simulate SpotsDataPipeline available and returning a simple DataFrame
    import data_mining.experiment_analysis.spots_input_stage as sis
    class DummyPipeline:
        def __init__(self, inp, out, output_format='parquet'):
            pass
        def run(self):
            df = pd.DataFrame([{'path': 'a.tiff', 'total_power': 1.0}])
            return df
    monkeypatch.setattr(sis, 'SpotsDataPipeline', DummyPipeline)
    stage = SpotsInputStage()
    result = stage.process()
    assert result.success
    assert isinstance(result.data, pd.DataFrame)


def test_spots_input_stage_fallback_reads_tiff(tmp_path, monkeypatch):
    # Ensure SpotsDataPipeline is disabled to trigger fallback
    import data_mining.experiment_analysis.spots_input_stage as sis
    monkeypatch.setattr(sis, 'SpotsDataPipeline', None)

    input_dir = tmp_path / 'spots_in'
    input_dir.mkdir()
    _make_test_tiff(input_dir / 'img1.tiff')
    os.environ['SPOTS_INPUT_DIR'] = str(input_dir)
    # Run the stage
    stage = SpotsInputStage()
    result = stage.process()
    assert result.success
    df = result.data
    assert isinstance(df, pd.DataFrame)
    assert 'path' in df.columns or 'total_power' in df.columns
