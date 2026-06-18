import json
from typing import Any

import pandas as pd

from .base_input import InputSource


class RabbitMQInput(InputSource[pd.DataFrame]):
    def __init__(self, host: str, queue: str, max_messages: int = 100, config: object | None = None):
        super().__init__(config)
        self.host = host
        self.queue = queue
        self.max_messages = max_messages

    def read(self) -> pd.DataFrame:
        try:
            import pika  # type: ignore
        except Exception as e:
            raise ImportError(f"pika library is required for RabbitMQInput: {e}") from e

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        channel = connection.channel()
        method_frame, header_frame, body = channel.basic_get(queue=self.queue, auto_ack=True)

        records: list[dict[str, Any]] = []
        count = 0
        while method_frame is not None and count < self.max_messages:
            records.append(json.loads(body))
            count += 1
            method_frame, header_frame, body = channel.basic_get(queue=self.queue, auto_ack=True)

        connection.close()
        df = pd.DataFrame(records)
        return df

    def validate(self, data: pd.DataFrame) -> bool:
        return isinstance(data, pd.DataFrame)
