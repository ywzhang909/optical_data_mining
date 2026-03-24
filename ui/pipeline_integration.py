"""Bridge for integrating external pipeline_manager event sources with
the in-process UI state used by the FastAPI backend and Streamlit dashboard.

This module provides a small, opt-in adapter that lets your existing
pipeline_manager emit events (start, update, alert) which will be
translated into internal state updates.

Usage (example):
- Import the bridge and wire your pipeline's event emitter to it:
  from ui.pipeline_integration import PipelineEventBridge
  bridge = PipelineEventBridge()
  # Suppose your pipeline has a subscribe function that accepts a callback
  # which will be called with (event_type, payload)
  bridge.wire(your_pipeline_subscribe_function)
"""
from __future__ import annotations

from typing import Callable, Any
import asyncio

from .server.state import start_pipeline, push_update

def _schedule_coroutine(coro: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # If there is no running loop (e.g., during startup), ignore gracefully
        # Users can ensure to call events after the app has started.
        pass

class PipelineEventBridge:
    def __init__(self) -> None:
        self._wired = False

    def wire(self, subscribe_fn: Callable[[Callable[[str, dict], None]], None]) -> None:
        """Attach a callback to the external event source.

        subscribe_fn should accept a single callback parameter which will be
        invoked with (event_type: str, payload: dict).
        """

        def _on_event(event_type: str, payload: dict) -> None:
            # Dispatch events to the internal state asynchronously when possible
            if event_type == "start":
                _schedule_coroutine(start_pipeline())
            elif event_type == "update":
                # payload may contain: progress_delta, throughput, alert
                progress_delta = int(payload.get("progress_delta", 0))
                throughput = int(payload.get("throughput", 0))
                alert = payload.get("alert")
                _schedule_coroutine(push_update(progress_delta=progress_delta, throughput=throughput, alert=alert))
            elif event_type == "alert":
                # forward as an alert in the update path
                alert = payload.get("message") if isinstance(payload, dict) else str(payload)
                _schedule_coroutine(push_update(alert=alert))
            else:
                # Unknown event; ignore gracefully
                pass

        subscribe_fn(_on_event)
        self._wired = True
