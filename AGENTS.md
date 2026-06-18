# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-13
**Commit:** e31970e
**Branch:** master

## OVERVIEW
Laser beam quality analysis toolkit (数字光学数据分析). Python 3.12+ project with three layers: message queue input sources (`src/`), beam analysis algorithms (`ui/analysis/`), and user interfaces (`ui/`).

## STRUCTURE
```
├── src/data_mining/input_sources/   # Message queue file JSON consumers
│   ├── kafka_input.py               # Kafka consumer → pd.DataFrame
│   ├── rabbitmq_input.py            # RabbitMQ consumer → pd.DataFrame
│   ├── filesystem_input.py          # Local file reader → pd.DataFrame
│   └── zip_input.py                 # ZIP archive reader → pd.DataFrame
├── ui/                              # Streamlit + FastAPI applications
│   ├── analysis/                    # Beam analysis algorithms (moved from src/)
│   │   ├── optical_analysis/        # D4σ, Strehl, M², BPP, diffraction
│   │   └── image/                   # Image I/O, feature extraction
│   ├── ao_analysis_app.py           # AO beam quality analysis dashboard
│   ├── streamlit_dashboard.py       # Pipeline monitor dashboard
│   └── server/                      # FastAPI backend
├── notebooks/                       # Jupyter notebooks for interactive analysis
└── docs/                            # Documentation
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Consume Kafka messages | src/data_mining/input_sources/kafka_input.py | Returns DataFrame of JSON messages |
| Consume RabbitMQ messages | src/data_mining/input_sources/rabbitmq_input.py | Returns DataFrame of JSON messages |
| Read files/directories | src/data_mining/input_sources/filesystem_input.py | CSV/JSON files |
| Extract ZIP data | src/data_mining/input_sources/zip_input.py | Extracts to temp dir, reads data |
| Beam quality analysis UI | ui/ao_analysis_app.py | D4σ, PIB, Strehl, BPP, M² |
| Beam analysis algorithms | ui/analysis/optical_analysis/ | beam_analysis, diffraction, metrics |
| Image processing | ui/analysis/image/common.py | read_tiff_to_numpy, get_profiles |
| Pipeline monitor UI | ui/streamlit_dashboard.py | Polls FastAPI backend |
| FastAPI backend | ui/server/main.py | Status/update/stream endpoints |
| Interactive notebooks | notebooks/ | Jupyter notebooks |

## CONVENTIONS
- **No linting/formatting config** — project has no ruff/black/flake8. Follow existing code style.
- **Pydantic v2** — use `model_config = ConfigDict(from_attributes=True)` for config models
- **Chinese docstrings** — internal documentation in Chinese
- **src layout** — package maps to `src/` directory in pyproject.toml
- **sys.path patching** — UI and notebooks add their parent paths to sys.path at import time
- **Optional dependency guards** — `try/except ImportError` for heavy deps (bm3d, aotools)

## ANTI-PATTERNS (THIS PROJECT)
- **DO NOT** add `sys.path` manipulation to library code — only UI and notebooks need this
- **DO NOT** import from `data_mining.image` or `data_mining.optical_analysis` — use `analysis.image` and `analysis.optical_analysis` from `ui/analysis/`
- **DO NOT** ship hardcoded `level="DEBUG"` logging — use environment-based configuration

## UNIQUE STYLES
- Plugin-style input system: `InputSource` base class with Kafka/RabbitMQ/Filesystem/Zip variants
- Analysis code in `ui/analysis/` as a subpackage, not in `src/` — avoids coupling installable package to UI-specific algorithms
- `try/except ImportError` pattern for optional heavy dependencies (aotools, kafka-python, pika)

## COMMANDS
```bash
# Start AO analysis app
streamlit run ui/ao_analysis_app.py

# Start pipeline dashboard + backend
uvicorn ui.server.main:app --reload --port 8000
streamlit run ui/streamlit_dashboard.py --server.port 8501

# Install package in dev mode (input_sources only — analysis is used via sys.path)
pip install -e .
```

## NOTES
- Package name is `data_mining` but lives at `src/data_mining/` (src layout)
- Analysis code (`ui/analysis/`) is NOT part of the pip-installable package — used via sys.path from UI and notebooks
- Kafka requires `kafka-python`; RabbitMQ requires `pika` — both are optional imports
- Diffraction calculations require `aotools` (optional, `try/except` guarded)
- The `notebooks/` directory uses `sys.path` to access both `ui/` (analysis) and `src/` (input_sources)
