import os
import tempfile
import zipfile
from contextlib import suppress
from pathlib import Path

import pandas as pd

from .base_input import InputSource


class ZipInput(InputSource[pd.DataFrame]):
    def __init__(self, zip_path: str, required_files: list[str] | None = None, config: object | None = None):
        super().__init__(config)
        self.zip_path = Path(zip_path)
        self.required_files = required_files or []

    def read(self) -> pd.DataFrame:
        if not self.zip_path.exists() or not self.zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        if self.zip_path.suffix.lower() != ".zip":
            raise ValueError("Input file is not a zip archive")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                zf.extractall(tmpdir)
            # Validate required content if provided
            if self.required_files:
                missing = [f for f in self.required_files if not os.path.exists(os.path.join(tmpdir, f))]
                if missing:
                    raise ValueError(f"Zip missing required files: {missing}")
            # Try to load any CSV/JSON files found
            dfs: list[pd.DataFrame] = []
            root = Path(tmpdir)
            for f in root.rglob("*"):
                if f.is_file():
                    if f.suffix.lower() == ".csv":
                        with suppress(Exception):
                            dfs.append(pd.read_csv(f))
                    elif f.suffix.lower() in [".json", ".jsonl", ".ndjson"]:
                        with suppress(Exception):
                            if f.suffix.lower() == ".jsonl" or f.suffix.lower() == ".ndjson":
                                dfs.append(pd.read_json(f, lines=True))
                            else:
                                dfs.append(pd.read_json(f))
            if not dfs:
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True, sort=False)
            return df

    def validate(self, data: pd.DataFrame) -> bool:
        return isinstance(data, pd.DataFrame)
