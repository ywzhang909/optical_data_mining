"""A lightweight, config-driven workflow pipeline.

This module provides an internal orchestrator so users can choose data sources
and preprocessing / feature extraction steps without editing Python code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from data_mining.image.common import read_tiff_to_numpy
from data_mining.image.beam_metrics import d4sigma, radius, uniformity
from data_mining.image.shape_features import (
    calculate_xy_diameters,
    ellipse_fit,
    find_spot_border,
)
from .config_loader import load_workflow_config

Record = dict[str, Any]
SourceFunc = Callable[[dict[str, Any]], list[Record]]
StepFunc = Callable[..., dict[str, Any]]


class PipelineConfigError(ValueError):
    """Raised when pipeline configuration is invalid."""


def _source_tiff_dir(config: dict[str, Any]) -> list[Record]:
    directory = Path(config["path"])  # required
    pattern = config.get("pattern", "*.TIFF")

    if not directory.exists():
        raise PipelineConfigError(f"Data source directory does not exist: {directory}")

    records: list[Record] = []
    for path in sorted(directory.glob(pattern)):
        image = read_tiff_to_numpy(path)
        if image is None:
            continue
        records.append({"path": str(path), "image": image.astype(np.float64)})

    if not records:
        raise PipelineConfigError(f"No images loaded from {directory} with pattern {pattern}.")
    return records


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = read_tiff_to_numpy(path)
        if img is None:
            raise PipelineConfigError(f"Cannot read image: {path}")
        return img.astype(np.float64)
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def _source_image_file(config: dict[str, Any]) -> list[Record]:
    path = Path(config["path"])
    if not path.exists():
        raise PipelineConfigError(f"Image file not found: {path}")
    return [{"path": str(path), "image": _read_image(path)}]


def _source_image_folder(config: dict[str, Any]) -> list[Record]:
    directory = Path(config["path"])
    pattern = config.get("pattern", "*")
    if not directory.exists():
        raise PipelineConfigError(f"Image folder not found: {directory}")

    records: list[Record] = []
    for path in sorted(directory.glob(pattern)):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        records.append({"path": str(path), "image": _read_image(path)})
    if not records:
        raise PipelineConfigError(f"No image files found in {directory} with pattern {pattern}")
    return records


def _source_video_file(config: dict[str, Any]) -> list[Record]:
    path = Path(config["path"])
    if not path.exists():
        raise PipelineConfigError(f"Video file not found: {path}")

    frame_step = int(config.get("frame_step", 1))
    max_frames = config.get("max_frames")
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover
        raise PipelineConfigError(
            "video_file source requires imageio. Install with `pip install imageio`."
        ) from exc

    records: list[Record] = []
    kept = 0
    for idx, frame in enumerate(iio.imiter(path)):
        if idx % frame_step != 0:
            continue
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        records.append({"path": f"{path}#frame={idx}", "image": gray.astype(np.float64)})
        kept += 1
        if max_frames is not None and kept >= int(max_frames):
            break

    if not records:
        raise PipelineConfigError(f"No frames extracted from video: {path}")
    return records


def _read_folder_path_from_rabbitmq(config: dict[str, Any]) -> str:
    try:
        import pika
    except ImportError as exc:  # pragma: no cover
        raise PipelineConfigError("rabbitmq_folder_path source requires pika.") from exc

    host = config.get("host", "localhost")
    port = int(config.get("port", 5672))
    queue = config.get("queue")
    if not queue:
        raise PipelineConfigError("rabbitmq_folder_path source requires 'queue'.")

    credentials = None
    if config.get("username") and config.get("password"):
        credentials = pika.PlainCredentials(config["username"], config["password"])
    params = pika.ConnectionParameters(host=host, port=port, credentials=credentials)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    method_frame, _, body = channel.basic_get(queue=queue, auto_ack=True)
    connection.close()

    if method_frame is None or body is None:
        raise PipelineConfigError(f"No message received from RabbitMQ queue: {queue}")
    return body.decode("utf-8").strip()


def _source_rabbitmq_folder_path(config: dict[str, Any]) -> list[Record]:
    folder_path = _read_folder_path_from_rabbitmq(config)
    folder_cfg = {
        "path": folder_path,
        "pattern": config.get("pattern", "*"),
    }
    return _source_image_folder(folder_cfg)


def _source_numpy_files(config: dict[str, Any]) -> list[Record]:
    files = config.get("files", [])
    if not files:
        raise PipelineConfigError("numpy_files source requires a non-empty 'files' list.")

    records: list[Record] = []
    for file in files:
        path = Path(file)
        image = np.load(path)
        if image.ndim != 2:
            raise PipelineConfigError(f"Only 2D arrays are supported, got {image.ndim}D: {path}")
        records.append({"path": str(path), "image": image.astype(np.float64)})
    return records


SOURCE_REGISTRY: dict[str, SourceFunc] = {
    "image_file": _source_image_file,
    "image_folder": _source_image_folder,
    "video_file": _source_video_file,
    "rabbitmq_folder_path": _source_rabbitmq_folder_path,
    "tiff_dir": _source_tiff_dir,
    "numpy_files": _source_numpy_files,
}

FEATURE_REGISTRY: dict[str, StepFunc] = {
    "d4sigma": d4sigma,
    "uniformity": uniformity,
    "radius": radius,
    "ellipse_fit": ellipse_fit,
    "find_spot_border": find_spot_border,
    "calculate_xy_diameters": calculate_xy_diameters,
}


def _resolve_ref(ref: str, context: Record) -> Any:
    """Resolve '$key.nested.path' references from current record context."""
    if not ref.startswith("$"):
        return ref

    parts = ref[1:].split(".")
    value: Any = context
    for part in parts:
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, (list, tuple)):
            value = value[int(part)]
        else:
            raise PipelineConfigError(f"Cannot resolve reference: {ref}")
    return value


def _resolve_params(params: dict[str, Any], context: Record) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("$"):
            resolved[key] = _resolve_ref(value, context)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_ref(v, context) if isinstance(v, str) and v.startswith("$") else v
                for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def _run_step(record: Record, step: dict[str, Any]) -> None:
    step_name = step["name"]
    fn = FEATURE_REGISTRY.get(step_name)
    if fn is None:
        raise PipelineConfigError(f"Unknown step: {step_name}")

    input_key = step.get("input", "image")
    output_key = step.get("output", step_name)
    params = _resolve_params(step.get("params", {}), record)

    if input_key not in record:
        raise PipelineConfigError(f"Step '{step_name}' input key not found: {input_key}")

    output = fn(record[input_key], **params)
    record[output_key] = output


def run_pipeline(config: dict[str, Any] | str | Path) -> list[Record]:
    """Run a configured workflow.

    Config schema (JSON):
    {
      "source": {"type": "tiff_dir", "path": "...", "pattern": "*.TIFF"},
      "steps": [
        {"name": "d4sigma", "input": "image", "output": "moments"},
        {"name": "uniformity", "params": {"center": ["$moments.center_x", "$moments.center_y"]}}
      ]
    }
    """
    cfg = load_workflow_config(config)
    source_config = cfg.source.model_dump(exclude_none=True)
    source_type = source_config.get("type")
    source_fn = SOURCE_REGISTRY.get(source_type)
    if source_fn is None:
        raise PipelineConfigError(f"Unknown source type: {source_type}")

    records = source_fn(source_config)
    for record in records:
        for step in cfg.steps:
            _run_step(record, step.model_dump(exclude_none=True))
    return records
