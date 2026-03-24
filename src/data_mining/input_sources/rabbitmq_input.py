from typing import List, Dict, Any, Optional
import json
import pandas as pd
from .base_input import InputSource

class RabbitMQInput(InputSource[pd.DataFrame]):
    def __init__(self, host: str, queue: str, max_messages: int = 100, config: Optional[object] = None):
        super().__init__(config)
        self.host = host
        self.queue = queue
        self.max_messages = max_messages

    def read(self) -> pd.DataFrame:
        try:
            import pika  # type: ignore
        except Exception as e:
            raise ImportError("pika library is required for RabbitMQInput: {}".format(e))

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        channel = connection.channel()
        method_frame, header_frame, body = channel.basic_get(queue=self.queue, auto_ack=True)

        records: List[Dict[str, Any]] = []
        count = 0
        while method_frame is not None and count < self.max_messages:
            try:
                records.append(json.loads(body))
            except Exception:
                pass
            count += 1
            method_frame, header_frame, body = channel.basic_get(queue=self.queue, auto_ack=True)

        connection.close()
        df = pd.DataFrame(records)
        return df

    def validate(self, data: pd.DataFrame) -> bool:
        return isinstance(data, pd.DataFrame)
