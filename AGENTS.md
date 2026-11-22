# optical_data_mining

Laser beam quality analysis toolkit (数字光学数据分析). Python 3.12 project: message-queue input sources (`src/`) feed beam-analysis algorithms (`ui/analysis/`), exposed via Streamlit/FastAPI apps (`ui/`).

## Commands

Use `uv` (project is uv-configured via `.uv/config.toml`: `link-mode=copy`, Tsinghua index, cache in `.uv/cache`; `uv.lock` is **gitignored**).

```bash
uv sync --dev                    # install deps + dev group
uv run streamlit run ui/ao_analysis_app.py          # AO beam analysis (standalone, no backend)
uv run uvicorn ui.server.main:app --reload --port 8000   # FastAPI backend
uv run streamlit run ui/streamlit_dashboard.py --server.port 8501   # needs backend running
uv run pytest tests/ -v          # full suite
uv run pytest tests/test_beam_analysis.py -v   # single file
uv run pytest -m "not slow"      # skip slow/integration tests
uv run ruff check src/ ui/analysis/ ui/
uv run ruff format src/ ui/analysis/ --check
```

CI (`.github/workflows/ci.yml`) only does `pip install -e .` then import-verification — it does **not** run ruff or pytest.

## Package boundaries

- **`data_mining` (pip-installable)** lives at `src/data_mining/`. Contains only `input_sources/` (`InputSource` ABC in `base_input.py` + Kafka/RabbitMQ/Filesystem/Zip). Every consumer returns `pd.DataFrame`. `kafka-python`/`pika` are optional deps (`[project.optional-dependencies]` kafka/rabbitmq), try/except guarded.
- **`ui/analysis/`** holds the beam algorithms — NOT part of the installable package. `ui/analysis/optical_analysis/` = `beam_analysis.py` (D4σ, PIB, BPP, M²), `diffraction.py` (Strehl/Fresnel, needs aotools), `beam_analysis_metrics.py`, `history_manager.py`, `uniform_analysis.py`, `zernike_analysis.py`, `visualization/`. `ui/analysis/image/common.py` = image I/O + profiles.
- pytest is configured with `pythonpath = ["src", "ui"]` (pyproject `[tool.pytest.ini_options]`), so `analysis.*` is importable in tests without manual sys.path setup.

## Gotchas

- **`src/data_mining/optical_analysis/` is a stale stub.** Its `__init__.py` re-exports `.beam_analysis`/`.diffraction`/`.image_utils`/`.uniformity_analysis`, but those modules were moved to `ui/analysis/optical_analysis/` and **do not exist under `src/`**. Importing `data_mining.optical_analysis` raises `ModuleNotFoundError`. Never import analysis code from `data_mining.*` — use `analysis.optical_analysis` / `analysis.image` from `ui/`.
- **`sys.path` patching is intentional** and lives only in UI apps and notebooks (they add their parent `ui/` to `sys.path`). Do NOT add sys.path manipulation to library code in `src/`.
- **`docs/` is partly stale.** `docs/README.md` and `docs/EXAMPLES_INPUT_PLUGINS.md` reference `input_stage_registry.py`, `experiment_analysis/`, and `SpotsInput` that do not exist in the codebase. Trust `src/` and `ui/` over these docs.
- **aotools** is a declared dependency, but diffraction guards it lazily via `_require_aotools()` in `ui/analysis/optical_analysis/diffraction.py` and raises a helpful install error. `bm3d` is optional (`[project.optional-dependencies].bm3d`).
- `docs/` and several root-level files contain GBK-encoded Chinese filenames; don't assume UTF-8 when listing them.

## Conventions

- **Ruff** is configured in `pyproject.toml` (`line-length = 120`, double-quote format). Rule selects/ignores are intentional: `E501`, `N801/N803/N806` ignored for physics notation (`F, Ex, X, Y, Z, NA`) and Chinese docstrings; per-file ignores relax rules for `notebooks/*` and `tests/*`. Run `ruff check`/`ruff format --check` on `src/ ui/analysis/ ui/`.
- **Pydantic v2** — use `model_config = ConfigDict(from_attributes=True)` for config models.
- **Chinese docstrings** for internal documentation; logging via `loguru`, configured by environment — don't hardcode `level="DEBUG"`.
- Streamlit apps are standalone single-file apps (not multipage).

## Where to look

| Task | Location |
|------|----------|
| Add/modify an input source | `src/data_mining/input_sources/base_input.py` (inherit `InputSource[T]`, implement `read()` + `validate()`) |
| Beam metrics (D4σ/PIB/BPP/M²) | `ui/analysis/optical_analysis/beam_analysis.py` |
| Orchestration class | `ui/analysis/optical_analysis/beam_analysis_metrics.py` |
| Strehl / diffraction | `ui/analysis/optical_analysis/diffraction.py` |
| Image I/O + radial profiles | `ui/analysis/image/common.py` |
| 温度脉动仪数据报告 | `ui/analysis/thermo/`（`python -m ui.analysis.thermo`，用法见 `docs/温度脉动仪数据报告.md`） |
| 温度脉动仪自动化 | `scripts/thermo_auto.ps1`（自动发现日期 + 重试 + 日志，产物含 `slices/` 时间片数据） |
| FastAPI backend | `ui/server/main.py` (+ `state.py` in-memory pipeline state) |
| Tests | `tests/` (markers: `slow`, `kafka`, `rabbitmq`, `aotools`) |

Per-subpackage detail: `src/data_mining/AGENTS.md`, `ui/AGENTS.md`.
