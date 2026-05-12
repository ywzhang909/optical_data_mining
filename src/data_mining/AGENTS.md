# data_mining — Message Queue Input Sources

**Part of:** optical_data_mining root project

## OVERVIEW
Message queue file JSON consumers. Provides `InputSource` base class with Kafka, RabbitMQ, filesystem, and ZIP implementations. Each consumer reads data and returns a `pd.DataFrame`.

## STRUCTURE
```
input_sources/
├── base_input.py           # Generic[T] ABC: InputSource with read() + validate()
├── kafka_input.py          # Kafka consumer → DataFrame of JSON messages
├── rabbitmq_input.py       # RabbitMQ consumer → DataFrame of JSON messages
├── filesystem_input.py     # Directory reader for CSV/JSON files → DataFrame
└── zip_input.py            # ZIP archive extractor → DataFrame
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Read from Kafka | kafka_input.py | KafkaInput(bootstrap_servers, topic, max_messages) |
| Read from RabbitMQ | rabbitmq_input.py | RabbitMQInput(host, queue, max_messages) |
| Read local files | filesystem_input.py | FileSystemInput(directory), reads CSV/JSON |
| Read ZIP archives | zip_input.py | ZipInput(zip_path), extracts to temp dir |
| Create new input source | base_input.py | Inherit InputSource[T], implement read() and validate() |

## CONVENTIONS
- **All input sources return `pd.DataFrame`** — each row is one parsed record
- **Optional deps** — `kafka-python` and `pika` are try/except guarded with helpful error messages
- **JSON parsing** — Kafka uses configurable `value_parser` (default: `json.loads`); RabbitMQ uses `json.loads` directly

## NOTES
- This is the ONLY module in the pip-installable `data_mining` package
- Analysis/visualization code lives in `ui/analysis/` (not part of this package)
- The `KafkaInput` reads up to `max_messages` (default 100) from earliest offset
- The `RabbitMQInput` uses `basic_get` (polling, not streaming consume)
