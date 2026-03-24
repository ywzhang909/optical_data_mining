import json
from typing import Any

class JsonOutput:
    def write(self, data: Any, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
