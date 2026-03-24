from typing import List, Optional
import zipfile
import tempfile
import os
import pandas as pd
from pathlib import Path
from .base_input import InputSource

class ZipInput(InputSource[pd.DataFrame]):
    def __init__(self, zip_path: str, required_files: Optional[List[str]] = None, config: Optional[object] = None):
        super().__init__(config)
        self.zip_path = Path(zip_path)
        self.required_files = required_files or []

    def read(self) -> pd.DataFrame:
        if not self.zip_path.exists() or not self.zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        if self.zip_path.suffix.lower() != '.zip':
            raise ValueError("Input file is not a zip archive")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                zf.extractall(tmpdir)
            # Validate required content if provided
            if self.required_files:
                missing = [f for f in self.required_files if not os.path.exists(os.path.join(tmpdir, f))]
                if missing:
                    raise ValueError(f"Zip missing required files: {missing}")
            # Try to load any CSV/JSON files found
            dfs: List[pd.DataFrame] = []
            root = Path(tmpdir)
            for f in root.rglob('*'):
                if f.is_file():
                    if f.suffix.lower() == '.csv':
                        try:
                            dfs.append(pd.read_csv(f))
                        except Exception:
                            pass
                    elif f.suffix.lower() in ['.json', '.jsonl', '.ndjson']:
                        try:
                            if f.suffix.lower() == '.jsonl' or f.suffix.lower() == '.ndjson':
                                dfs.append(pd.read_json(f, lines=True))
                            else:
                                dfs.append(pd.read_json(f))
                        except Exception:
                            pass
            if not dfs:
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True, sort=False)
            return df

    def validate(self, data: pd.DataFrame) -> bool:
        return isinstance(data, pd.DataFrame)
