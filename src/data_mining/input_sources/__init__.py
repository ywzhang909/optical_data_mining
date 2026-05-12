"""
data_mining.input_sources — 消息队列输入源

从消息队列（Kafka、RabbitMQ）获取JSON格式的文件信息，
也支持从文件系统和ZIP归档读取数据。
"""

from .base_input import InputSource
from .filesystem_input import FileSystemInput
from .kafka_input import KafkaInput
from .rabbitmq_input import RabbitMQInput
from .zip_input import ZipInput

__all__ = [
    "InputSource",
    "KafkaInput",
    "RabbitMQInput",
    "FileSystemInput",
    "ZipInput",
]
