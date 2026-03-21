import json

import pytest

from data_mining.workflow import (
    PrefectNotInstalledError,
    build_prefect_flow,
    load_workflow_config,
)


def test_load_workflow_config_from_dict():
    cfg = load_workflow_config(
        {
            "source": {"type": "numpy_files", "files": ["/tmp/a.npy"]},
            "steps": [{"name": "d4sigma"}],
        }
    )
    assert cfg.source.type == "numpy_files"
    assert cfg.steps[0].name == "d4sigma"


def test_load_workflow_config_from_json_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(
        json.dumps(
            {
                "source": {"type": "numpy_files", "files": ["/tmp/a.npy"]},
                "steps": [{"name": "d4sigma"}],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_workflow_config(p)
    assert cfg.source.type == "numpy_files"


def test_load_workflow_config_from_yaml_file(tmp_path):
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "source": {"type": "numpy_files", "files": ["/tmp/a.npy"]},
                "steps": [{"name": "d4sigma"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = load_workflow_config(p)
    assert cfg.steps[0].input == "image"


def test_build_prefect_flow_without_prefect_raises():
    with pytest.raises(PrefectNotInstalledError):
        build_prefect_flow({"source": {"type": "numpy_files", "files": ["a.npy"]}, "steps": []})
