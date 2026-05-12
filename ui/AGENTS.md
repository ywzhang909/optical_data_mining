# ui — Frontend Applications & Analysis

## OVERVIEW
User interface for the laser beam analysis toolkit. Provides Streamlit frontends (AO analysis, pipeline dashboard) and a FastAPI backend. Contains all beam analysis algorithms in `analysis/` subpackage.

## STRUCTURE
```
ui/
├── __init__.py
├── analysis/                     # Beam analysis algorithms
│   ├── __init__.py
│   ├── image/
│   │   ├── __init__.py
│   │   └── common.py             # Image I/O, feature extraction
│   └── optical_analysis/
│       ├── __init__.py
│       ├── beam_analysis.py      # D4σ, PIB, Gaussian fitting, BPP, M²
│       ├── beam_analysis_metrics.py  # BeamAnalysisMetrics class
│       ├── diffraction.py        # Fresnel propagation, Strehl ratio
│       ├── history_manager.py    # Analysis session persistence
│       ├── image_utils.py        # Image read, dark field subtraction
│       └── visualization/        # 2D/3D Plotly visualizations
├── server/
│   ├── main.py                   # FastAPI app entry point
│   └── state.py                  # Pipeline state management
├── ao_analysis_app.py            # AO beam quality analysis dashboard
├── streamlit_dashboard.py        # Pipeline monitor dashboard
└── pipeline_integration.py       # Event bridge for pipeline integration
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Start AO analysis | ao_analysis_app.py | `streamlit run ui/ao_analysis_app.py` |
| Start pipeline dashboard | streamlit_dashboard.py | Needs FastAPI backend running |
| Start backend | server/main.py | `uvicorn ui.server.main:app` |
| Beam quality calculation | analysis/optical_analysis/beam_analysis.py | d4sigma, pib_ratio, bpp, m2 |
| Strehl ratio | analysis/optical_analysis/diffraction.py | Energy-conservation method |
| Image I/O | analysis/image/common.py | read_tiff_to_numpy, get_profiles |
| Analysis metrics class | analysis/optical_analysis/beam_analysis_metrics.py | Orchestrates all calculations |
| Session history | analysis/optical_analysis/history_manager.py | Save/load analysis results |

## CONVENTIONS
- **sys.path patching** — UI files add `parent/ui` to `sys.path` to access `analysis` subpackage; notebooks add it manually
- **Streamlit pages** — standalone apps (not multipage)
- **FastAPI + Streamlit** — backend serves data API; Streamlit handles visualization

## ANTI-PATTERNS
- **DO NOT** add `sys.path` manipulation to library code — only UI and notebooks need this
- **DO NOT** import analysis code from `data_mining.*` — use `analysis.optical_analysis` or `analysis.image`
- **DO NOT** hardcode `level="DEBUG"` in production logger — use environment config

## NOTES
- Run AO analysis standalone: `streamlit run ui/ao_analysis_app.py` (no backend needed)
- Run dashboard + backend: `uvicorn ui.server.main:app --port 8000` then `streamlit run ui/streamlit_dashboard.py --server.port 8501`
- Analysis algorithms (`analysis/`) were moved from `src/data_mining/` during refactoring
