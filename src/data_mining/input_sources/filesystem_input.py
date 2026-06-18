from pathlib import Path

import pandas as pd

from .base_input import InputSource


class FileSystemInput(InputSource[pd.DataFrame]):
    def __init__(self, directory: str, config: object | None = None):
        super().__init__(config)
        self.directory = Path(directory)

    def read(self) -> pd.DataFrame:
        if not self.directory.exists() or not self.directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        frames: list[pd.DataFrame] = []
        for p in sorted(self.directory.glob("*")):
            if p.is_file():
                if p.suffix.lower() in [".csv"]:
                    frames.append(pd.read_csv(p))
                elif p.suffix.lower() in [".json"]:
                    frames.append(pd.read_json(p))
                elif p.suffix.lower() in [".jsonl", ".ndjson"]:
                    frames.append(pd.read_json(p, lines=True))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True, sort=False)
        return df

    def validate(self, data: pd.DataFrame) -> bool:
        return isinstance(data, pd.DataFrame)
