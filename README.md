# 数字光学数据分析 — Laser Beam Quality Analysis Toolkit

Python 3.12+ 工具包，从消息队列获取光斑文件信息，进行光束质量分析（D4σ、Strehl、M²、BPP 等），并通过 Streamlit 可视化结果。

## 项目结构

```
src/data_mining/input_sources/     # 消息队列输入源（Kafka / RabbitMQ / 文件系统 / ZIP）
  ├── kafka_input.py               # Kafka 消费者 → pd.DataFrame
  ├── rabbitmq_input.py            # RabbitMQ 消费者 → pd.DataFrame
  ├── filesystem_input.py          # 本地文件读取
  └── zip_input.py                 # ZIP 归档解压读取

ui/                                # Streamlit + FastAPI 应用
  ├── ao_analysis_app.py           # AO 光束质量分析仪表盘（核心功能）
  ├── streamlit_dashboard.py       # Pipeline 监控仪表盘
  ├── server/                      # FastAPI 后端
  └── analysis/                    # 光束分析算法
      ├── optical_analysis/        # D4σ, PIB, Strehl, M², BPP, 衍射
      └── image/                   # 图像 I/O, 特征提取

notebooks/                         # Jupyter 交互式分析脚本
tests/                             # pytest 测试（83 tests）
```

## 快速开始

```bash
# 安装（uv 或 pip）
uv sync --dev
# 或: pip install -e . && pip install -e .[dev]

# AO 光束质量分析（独立运行，无需后端）
uv run streamlit run ui/ao_analysis_app.py

# Pipeline 监控仪表盘（需后端）
uv run uvicorn ui.server.main:app --reload --port 8000 &
uv run streamlit run ui/streamlit_dashboard.py --server.port 8501

# 运行测试
uv run pytest tests/ -v

# 静态检查
uv run ruff check src/ ui/analysis/
uv run ruff format src/ ui/analysis/ --check
```

## 核心功能

### 消息队列输入（`src/data_mining/input_sources/`）

| 类 | 用途 | 可选依赖 |
|---|---|---|
| `KafkaInput` | 从 Kafka topic 消费 JSON 消息 → DataFrame | `kafka-python` |
| `RabbitMQInput` | 从 RabbitMQ 队列消费 JSON 消息 → DataFrame | `pika` |
| `FileSystemInput` | 读取目录下 CSV/JSON 文件 → DataFrame | — |
| `ZipInput` | 解压 ZIP 归档并读取内部 CSV/JSON → DataFrame | — |

消息格式示例（`data_processing_queue`）：
```python
{
    "data_experiment_id": "20260326_001",
    "data_original_path": "/path/to/beam.TIFF",
    "data_info": {"name": "beam.TIFF", "type": "tiff", "size": 5039098},
    "is_unstructured": False
}
```

### 光束分析（`ui/analysis/optical_analysis/`）

| 函数 | 说明 |
|---|---|
| `d4sigma()` | D4σ 光斑直径（ISO 11146） |
| `pib_ratio()` | 桶中功率比（Power In Bucket） |
| `calculate_bpp()` | 光束参数积 |
| `calculate_m2()` | 光束质量因子 M² |
| `calculate_centroid()` | 质心计算 |
| `shift_to_center_fft()` | FFT 亚像素移位居中 |
| `calculate_strehl_ratio_with_energy_conservation()` | 能量守恒斯特列尔比 |
| `fnr3()` | 菲涅尔衍射积分（向量化） |
| `angular_spectrum_propagation()` | 角谱传播法 |
| `BeamAnalysisMetrics` | 综合分析类（一键计算全部指标） |

### 图像处理（`ui/analysis/image/`）

| 函数 | 说明 |
|---|---|
| `read_tiff_to_numpy()` | TIFF/PNG 读取 → numpy 数组 |
| `get_profiles()` | 光斑横纵截面强度分布 |
| `fourier_shift_to_center()` | 傅里叶亚像素平移 |
| `extract_radial_data()` | 径向数据提取 |
| `convert_to_cv()` | numpy → OpenCV uint8 格式 |

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 代码检查
uv run ruff check src/ ui/analysis/
uv run ruff format src/ ui/analysis/ --check

# 类型检查（确保有 pyright/pylance）
# ruff check 已经覆盖常见类型问题

# 测试
uv run pytest tests/ -v --tb=short
uv run pytest tests/ -m "not slow"
```

## 依赖管理

项目使用 `uv` 管理依赖。核心依赖在 `pyproject.toml` 的 `[project.dependencies]` 中声明：

| 组 | 安装命令 | 包含 |
|---|---|---|
| 核心 | `uv sync` | numpy, scipy, opencv, streamlit, fastapi, ... |
| Kafka | `uv sync --group kafka` | `kafka-python` |
| RabbitMQ | `uv sync --group rabbitmq` | `pika` |
| BM3D | `uv sync --group bm3d` | `bm3d` |
| 开发 | `uv sync --dev` | pytest, ruff, ipykernel |

## 许可证

MIT License
