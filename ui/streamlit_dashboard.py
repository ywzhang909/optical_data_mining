"""Streamlit dashboard for monitoring pipeline_manager"""
import streamlit as st
import requests
from time import sleep

# Endpoints exposed by the FastAPI backend in ui/server/main.py
STATUS_URL = "http://127.0.0.1:8000/status"
START_URL = "http://127.0.0.1:8000/start"
UPDATE_URL = "http://127.0.0.1:8000/update"

def start_pipeline():
    try:
        requests.post(START_URL, json={})
        st.success("Pipeline started")
    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")

def render_status(status: dict):
    if not status:
        st.write("No status available yet.")
        return
    st.progress(status.get("progress", 0) / 100.0)
    st.metric(label="Throughput (items/s)", value=status.get("throughput", 0))
    elapsed = status.get("elapsed", 0.0)
    st.write(f"Elapsed: {elapsed:.1f}s")
    alerts = status.get("alerts", [])
    if alerts:
        st.subheader("Alerts")
        for a in alerts:
            st.write(f"- {a}")
    else:
        st.write("No alerts.")

st.title("Pipeline Manager Dashboard")

col_left, col_right = st.columns([3, 1])
with col_right:
    st.button("Start Pipeline", on_click=start_pipeline)
    st.write("")

with col_left:
    # The main status panel
    try:
        status = requests.get(STATUS_URL, timeout=2).json()
        render_status(status)
    except Exception as e:
        st.error(f"Could not fetch status: {e}")

# Auto-refresh support
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2000, key="status_refresh")
except Exception:
    st.info("Install streamlit-autorefresh for automatic refreshing: pip install streamlit-autorefresh")
