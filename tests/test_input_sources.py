"""Tests for input sources (src/data_mining/input_sources/)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/ to path for data_mining package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_mining.input_sources import FileSystemInput, InputSource, ZipInput


# =============================================================================
# Test: base_input.py — InputSource abstract class
# =============================================================================


class TestInputSource:
    def test_cannot_instantiate_abstract(self):
        """InputSource is abstract — should not be directly instantiable."""
        with pytest.raises(TypeError):
            InputSource()  # type: ignore[abstract]

    def test_accepts_config(self):
        class ConcreteSource(InputSource[dict]):
            def read(self) -> dict:
                return {"key": "value"}

            def validate(self, data: dict) -> bool:
                return isinstance(data, dict)

        source = ConcreteSource(config={"db": "test"})
        assert source.config == {"db": "test"}
        assert source.read() == {"key": "value"}
        assert source.validate({"a": 1})


# =============================================================================
# Test: filesystem_input.py — FileSystemInput
# =============================================================================


class TestFileSystemInput:
    def test_missing_directory_raises(self):
        source = FileSystemInput("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            source.read()

    def test_empty_directory_returns_empty_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = FileSystemInput(tmpdir)
            df = source.read()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_reads_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            df_expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            df_expected.to_csv(dirpath / "data.csv", index=False)

            source = FileSystemInput(str(dirpath))
            df_result = source.read()
            assert len(df_result) == 2
            assert list(df_result.columns) == ["a", "b"]

    def test_reads_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            records = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            with open(dirpath / "data.json", "w") as f:
                json.dump(records, f)

            source = FileSystemInput(str(dirpath))
            df = source.read()
            assert len(df) == 2

    def test_reads_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            pd.DataFrame({"a": [1]}).to_csv(dirpath / "f1.csv", index=False)
            pd.DataFrame({"a": [2]}).to_csv(dirpath / "f2.csv", index=False)

            source = FileSystemInput(str(dirpath))
            df = source.read()
            assert len(df) == 2

    def test_validate_returns_bool(self):
        source = FileSystemInput("/tmp")
        assert source.validate(pd.DataFrame()) is True
        assert source.validate("not a dataframe") is False


# =============================================================================
# Test: zip_input.py — ZipInput
# =============================================================================


class TestZipInput:
    def test_missing_zip_raises(self):
        source = ZipInput("/nonexistent.zip")
        with pytest.raises(FileNotFoundError):
            source.read()

    def test_non_zip_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            source = ZipInput(f.name)
            with pytest.raises(ValueError, match="zip archive"):
                source.read()

    def test_reads_csv_from_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test.zip"
            csv_path = Path(tmpdir) / "data.csv"
            pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)

            import zipfile

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(csv_path, arcname="data.csv")

            source = ZipInput(str(zip_path))
            df = source.read()
            assert len(df) == 2
            assert list(df.columns) == ["a"]

    def test_empty_zip_returns_empty_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "empty.zip"
            import zipfile

            with zipfile.ZipFile(zip_path, "w"):
                pass

            source = ZipInput(str(zip_path))
            df = source.read()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_validate_returns_bool(self):
        source = ZipInput("/tmp/test.zip")
        assert source.validate(pd.DataFrame()) is True
        assert source.validate(None) is False
