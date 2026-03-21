import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_mining.workflow import run_pipeline
from data_mining.workflow import configurable_pipeline as cp


def _create_tiff(path: Path, offset_x: int = 0, offset_y: int = 0) -> None:
    size = 64
    y, x = np.mgrid[0:size, 0:size]
    cx = (size - 1) / 2 + offset_x
    cy = (size - 1) / 2 + offset_y
    arr = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 8.0**2))
    arr = (arr / arr.max() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_run_pipeline_with_tiff_source(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _create_tiff(input_dir / "a.TIFF", offset_x=2)
    _create_tiff(input_dir / "b.TIFF", offset_y=-2)

    config = {
        "source": {"type": "tiff_dir", "path": str(input_dir), "pattern": "*.TIFF"},
        "steps": [
            {"name": "d4sigma", "output": "moments"},
            {
                "name": "uniformity",
                "params": {"center": ["$moments.center_x", "$moments.center_y"]},
            },
            {
                "name": "radius",
                "params": {
                    "center": ["$moments.center_x", "$moments.center_y"],
                    "energy": 0.9,
                },
            },
            {"name": "ellipse_fit"},
        ],
    }

    records = run_pipeline(config)
    assert len(records) == 2
    assert "moments" in records[0]
    assert "uniformity" in records[0]
    assert "radius" in records[0]
    assert "ellipse_fit" in records[0]


def test_run_pipeline_can_load_json_file(tmp_path: Path):
    npy = tmp_path / "beam.npy"
    np.save(npy, np.ones((16, 16), dtype=np.float64))

    cfg_path = tmp_path / "pipeline.json"
    cfg_path.write_text(
        json.dumps(
            {
                "source": {"type": "numpy_files", "files": [str(npy)]},
                "steps": [{"name": "d4sigma"}],
            }
        ),
        encoding="utf-8",
    )

    out = run_pipeline(cfg_path)
    assert out[0]["d4sigma"]["D_x"] > 0


def test_run_pipeline_with_image_file_source(tmp_path: Path):
    img = tmp_path / "single.png"
    _create_tiff(img)

    config = {
        "source": {"type": "image_file", "path": str(img)},
        "steps": [{"name": "d4sigma"}],
    }
    out = run_pipeline(config)
    assert len(out) == 1
    assert out[0]["d4sigma"]["D_y"] > 0


def test_run_pipeline_with_image_folder_source(tmp_path: Path):
    folder = tmp_path / "images"
    folder.mkdir()
    _create_tiff(folder / "a.png")
    _create_tiff(folder / "b.png")

    config = {
        "source": {"type": "image_folder", "path": str(folder), "pattern": "*.png"},
        "steps": [{"name": "d4sigma"}],
    }
    out = run_pipeline(config)
    assert len(out) == 2


def test_run_pipeline_with_video_file_source(tmp_path: Path):
    imageio = __import__("imageio.v2", fromlist=["mimsave"])
    video = tmp_path / "beam.gif"
    frames = []
    for i in range(4):
        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[8 + i : 16 + i, 10:18] = 255
        frames.append(arr)
    imageio.mimsave(video, frames, duration=0.1)

    config = {
        "source": {"type": "video_file", "path": str(video), "frame_step": 2},
        "steps": [{"name": "d4sigma"}],
    }
    out = run_pipeline(config)
    assert len(out) == 2
    assert "d4sigma" in out[0]


def test_run_pipeline_with_rabbitmq_folder_path_source(tmp_path: Path, monkeypatch):
    folder = tmp_path / "rmq-images"
    folder.mkdir()
    _create_tiff(folder / "a.TIFF")

    monkeypatch.setattr(cp, "_read_folder_path_from_rabbitmq", lambda _: str(folder))

    config = {
        "source": {"type": "rabbitmq_folder_path", "queue": "spot_folder_queue", "pattern": "*.TIFF"},
        "steps": [{"name": "d4sigma"}],
    }
    out = run_pipeline(config)
    assert len(out) == 1
