# UI Package for Pipeline Manager

- Backend: FastAPI app exposing status and SSE stream for real-time updates.
- Frontend: Streamlit dashboard polling the status API to visualize metrics.

How to run:
- Start backend server:
  - cd ui
  - uvicorn ui.server.main:app --reload --port 8000 --host 0.0.0.0
- Start Streamlit dashboard:
  - streamlit run ui/streamlit_dashboard.py --server.port 8501

Notes:
- The backend currently uses an in-memory PipelineState. Replace with real pipeline
  integration by wiring into the actual pipeline events and calling start_pipeline / push_update.
- The SSE endpoint /stream is available for clients that want to subscribe to real-time updates.
