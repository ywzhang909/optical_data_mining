import json
from pathlib import Path

class SwanLabOutput:
    def __init__(self, base_dir: str = "swanlab_output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, data, filename: str = "output.json") -> Path:
        path = self.base_dir / filename
        if isinstance(data, (dict, list)):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            # Fallback to string representation
            path.write_text(str(data), encoding='utf-8')
        return path
