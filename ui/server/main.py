"""Backend for pipeline_manager UI

FastAPI app exposing:
- /start to initialize a pipeline run
- /update to push progress/throughput/alerts
- /status to fetch current status (polling-friendly)
- /stream to provide real-time SSE updates
"""
from __future__ import annotations

from fastapi import FastAPI, Request
import asyncio
import json
from sse_starlette import EventSourceResponse

from .state import start_pipeline, push_update, get_status

app = FastAPI(title="Pipeline Manager API")

# Enable CORS for local UI (Streamlit) to talk to the backend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/start")
async def start(pipeline_id: str | None = None):
    await start_pipeline()
    return {"status": "started", "pipeline_id": pipeline_id}

@app.post("/update")
async def update(progress_delta: int = 10, throughput: int = 0, alert: str | None = None):
    await push_update(progress_delta=progress_delta, throughput=throughput, alert=alert)
    return {"status": "updated"}

@app.get("/status")
async def status():
    s = await get_status()
    return s

@app.get("/stream")
async def stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            s = await get_status()
            data = json.dumps(s)
            yield f"data: {data}\n\n"
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())
