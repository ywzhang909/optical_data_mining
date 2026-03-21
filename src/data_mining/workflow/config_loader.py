"""Workflow config loading utilities (JSON/YAML)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_schema import WorkflowConfig


class ConfigFormatError(ValueError):
    """Raised when config format cannot be loaded."""


def load_workflow_config(config: dict[str, Any] | str | Path) -> WorkflowConfig:
    if isinstance(config, dict):
        return WorkflowConfig.model_validate(config)

    path = Path(config)
    suffix = path.suffix.lower()
    raw: dict[str, Any]

    if suffix in {".json"}:
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ConfigFormatError(
                "YAML config requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ConfigFormatError(f"Unsupported config format: {suffix}")

    return WorkflowConfig.model_validate(raw)
