import pandas as pd
import pytest
import numpy as np

from data_mining.input_sources.input_stage_registry import InputStageRegistry
from data_mining.input_sources.input_stage_spots import SpotsInputStagePlugin


def test_load_spots_input_plugin(monkeypatch):
    # Ensure the registry can load SpotsInput plugin and call load
    plugin = InputStageRegistry.load('SpotsInput')
    if plugin is None:
        pytest.skip("SpotsInput plugin not registered in registry in this environment")
    # If environment not set, this may just return empty DataFrame
    df = plugin.load()
    assert isinstance(df, pd.DataFrame)
