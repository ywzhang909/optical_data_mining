import pandas as pd
from data_mining.input_sources.input_stage_base import InputStage
from data_mining.input_sources.input_stage_registry import InputStageRegistry


class DummyInput(InputStage):
    def load(self) -> pd.DataFrame:
        return pd.DataFrame([{'val': 1}])


def test_input_stage_registry_basic():
    InputStageRegistry.register('Dummy', DummyInput)
    instance = InputStageRegistry.load('Dummy')
    assert instance is not None
    df = instance.load()
    assert isinstance(df, pd.DataFrame)
