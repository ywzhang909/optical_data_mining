from typing import List, Dict, Any, Optional
import json
import pandas as pd
from .base_input import InputSource

class KafkaInput(InputSource[pd.DataFrame]):
    def __init__(self, bootstrap_servers: List[str], topic: str, group_id: Optional[str] = None, max_messages: int = 100, value_parser: Optional[Any] = None, config: Optional[object] = None):
        super().__init__(config)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.max_messages = max_messages
        self.value_parser = value_parser or (lambda v: json.loads(v.decode('utf-8')) if isinstance(v, (bytes, bytearray)) else json.loads(v))

    def read(self) -> pd.DataFrame:
        try:
            from kafka import KafkaConsumer  # type: ignore
        except Exception as e:
            raise ImportError("kafka-python library is required for KafkaInput: {}".format(e))

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )

        records: List[Dict[str, Any]] = []
        for idx, msg in enumerate(consumer):
            if idx >= self.max_messages:
                break
            try:
                records.append(self.value_parser(msg.value))
            except Exception:
                # fallback to simple JSON decode
                try:
                    records.append(json.loads(msg.value.decode('utf-8')))
                except Exception:
                    # skip unreadable records
                    continue

        df = pd.DataFrame(records)
        return df

    def validate(self, data: pd.DataFrame) -> bool:
        return isinstance(data, pd.DataFrame)
