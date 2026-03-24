import asyncio
import time
from typing import Any, Dict

_lock = asyncio.Lock()
_state: Dict[str, Any] = {
    "progress": 0,
    "throughput": 0,
    "alerts": [],
    "start_time": None,
}

async def start_pipeline() -> None:
    async with _lock:
        _state["start_time"] = time.time()
        _state["progress"] = 0
        _state["throughput"] = 0
        _state["alerts"] = []

async def push_update(progress_delta: int = 0, throughput: int = 0, alert: str | None = None) -> None:
    async with _lock:
        if _state["start_time"] is None:
            _state["start_time"] = time.time()
        _state["progress"] = min(100, _state["progress"] + int(progress_delta))
        _state["throughput"] = int(throughput)
        if alert:
            _state["alerts"].append(alert)

async def get_status() -> Dict[str, Any]:
    async with _lock:
        status = {
            "progress": _state["progress"],
            "throughput": _state["throughput"],
            "alerts": list(_state["alerts"]),
            "start_time": _state["start_time"],
        }
        if _state["start_time"] is not None:
            status["elapsed"] = time.time() - _state["start_time"]
        else:
            status["elapsed"] = 0.0
        return status
