"""Optional Prefect integration for workflow orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .configurable_pipeline import run_pipeline


class PrefectNotInstalledError(ImportError):
    """Raised when Prefect integration is used without installing Prefect."""


def build_prefect_flow(config: dict[str, Any] | str | Path, flow_name: str = "optical-workflow"):
    """Build a Prefect flow wrapping :func:`run_pipeline`.

    Returns a Prefect flow function when Prefect is installed.
    """
    try:
        from prefect import flow  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise PrefectNotInstalledError(
            "Prefect is not installed. Install with `pip install prefect`."
        ) from exc

    @flow(name=flow_name)
    def _prefect_flow():
        return run_pipeline(config)

    return _prefect_flow
