"""Typed schema for workflow configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    type: str
    path: str | None = None
    pattern: str | None = None
    files: list[str] | None = None
    frame_step: int | None = None
    max_frames: int | None = None
    host: str | None = None
    port: int | None = None
    queue: str | None = None
    username: str | None = None
    password: str | None = None


class StepConfig(BaseModel):
    name: str
    input: str = "image"
    output: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class WorkflowConfig(BaseModel):
    source: SourceConfig
    steps: list[StepConfig]
