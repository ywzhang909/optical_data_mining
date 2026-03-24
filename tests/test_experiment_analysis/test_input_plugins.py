import pandas as pd
import numpy as np
import pytest

from data_mining.image.common import SpotImage

def test_kafka_input_stage():
    from data_mining.input_sources.input_stage_kafka import KafkaInputStage
    from data_mining.experiment_analysis.base_processor import ProcessorConfig
    stage = KafkaInputStage()
    assert stage is not None

def test_rabbitmq_input_stage():
    from data_mining.input_sources.input_stage_rabbitmq import RabbitMQInputStage
    stage = RabbitMQInputStage()
    assert stage is not None

def test_filesystem_input_stage(tmp_path):
    from data_mining.input_sources.input_stage_filesystem import FileSystemInputStage
    stage = FileSystemInputStage()
    assert stage is not None

def test_zip_input_stage():
    from data_mining.input_sources.input_stage_zip import ZipInputStage
    stage = ZipInputStage()
    assert stage is not None
