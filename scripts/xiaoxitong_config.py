"""小系统脚本的公共配置：从项目根目录 ``.env`` 读取内部假设参数。

脱敏原则：激光波长、功率、光斑/发散角等数值只用于内部计算与文本替换，
不写入任何 md/csv 产物；脚本中不再硬编码这些数值。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """读取简单的 ``KEY=VALUE`` 配置；已存在的环境变量优先。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


load_env()


def env_float(name: str) -> float:
    """读取必需的数值配置项，缺失或非法时明确报错。"""
    load_env()
    raw = os.environ.get(name)
    if raw is None:
        msg = f"缺少配置项 {name}，请在 {ENV_FILE} 中补充"
        raise RuntimeError(msg)
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"配置项 {name} 不是有效数值: {raw!r}"
        raise RuntimeError(msg) from exc


def env_text(name: str) -> str:
    """读取必需的字符串配置项。"""
    load_env()
    raw = os.environ.get(name)
    if raw is None:
        msg = f"缺少配置项 {name}，请在 {ENV_FILE} 中补充"
        raise RuntimeError(msg)
    return raw
