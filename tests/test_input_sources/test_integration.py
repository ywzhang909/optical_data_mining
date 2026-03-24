import pandas as pd
import pytest
from data_mining.input_sources.input_stage_registry import InputStageRegistry
from data_mining.input_sources.input_stage_spots_adapter import SpotsInputStageAdapter
from data_mining.input_sources.input_stage_kafka import KafkaInputStage
from data_mining.input_sources.input_stage_rabbitmq import RabbitMQInputStage


class DummySpotsPipeline:
    def __init__(self, *args, **kwargs):
        pass
    def run(self):
        import pandas as pd
        return pd.DataFrame([{'path': 'spots1.tiff', 'total_power': 1.0}])


def test_registry_and_spots_adapter_load(monkeypatch, tmp_path):
    # Patch SpotsDataPipeline to our dummy to avoid real I/O
    import data_mining.input_sources.input_stage_spots_adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, 'SpotsDataPipeline', DummySpotsPipeline, raising=False)

    # Register and load the Spots adapter
    InputStageRegistry.register('SpotsAdapter', SpotsInputStageAdapter)
    loader = InputStageRegistry.load('SpotsAdapter')
    assert loader is not None
    df = loader.load()
    assert isinstance(df, pd.DataFrame)

def test_kafka_and_rabbitmq_plugins_can_register_and_load(monkeypatch):
    # Prepare mocks for Kafka/RabbitMQ input stages to ensure registry load works
    class MockKafka:
        def __init__(self, *a, **k):
            pass
        def load(self):
            import pandas as pd
            return pd.DataFrame([{'k': 1}])

    class MockRabbit:
        def __init__(self, *a, **k):
            pass
        def load(self):
            import pandas as pd
            return pd.DataFrame([{'r': 2}])

    import data_mining.input_sources.input_stage_kafka as ks
    import data_mining.input_sources.input_stage_rabbitmq as rs
    monkeypatch.setattr(ks, 'KafkaInputStage', MockKafka, raising=False)
    monkeypatch.setattr(rs, 'RabbitMQInputStage', MockRabbit, raising=False)

    InputStageRegistry.register('KafkaTest', MockKafka)
    InputStageRegistry.register('RabbitTest', MockRabbit)
    k = InputStageRegistry.load('KafkaTest')
    r = InputStageRegistry.load('RabbitTest')
    assert k is not None
    assert r is not None
    assert hasattr(k, 'load')
    assert hasattr(r, 'load')
