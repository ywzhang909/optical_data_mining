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
      ├── optical_analysis/        # D4σ, PIB, Strehl, M², BPP, 衍射, FTL
      └── image/                   # 图像 I/O, 特征提取, 径向剖面

notebooks/                         # Jupyter 交互式分析脚本
tests/                             # pytest 测试（92 tests）
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
uv run ruff check src/ ui/analysis/ ui/
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
| `fit_flat_topped_lorentz()` | FTL (Flat-Topped Lorentz) 平顶光模型拟合，计算特征半径 R_FL |
| `fitting_gaussian()` | 一维高斯拟合：f(x) = A·exp(-½((x-μ)/σ)²) + b |
| `calculate_xy_diameters()` | 基于高斯拟合的 X/Y 方向直径 |
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
| `compute_radial_profile()` | 径向强度分布（方位角平均），FTL 拟合共用工具 |
| `cartesian_to_polar()` | 直角坐标 → 极坐标转换 |
| `polar_to_cartesian()` | 极坐标 → 直角坐标转换 |
| `fourier_shift_to_center()` | 傅里叶亚像素平移 |
| `extract_radial_data()` | 径向数据提取 |
| `convert_to_cv()` | numpy → OpenCV uint8 格式 |

### AO 光束质量分析仪表盘（`ui/ao_analysis_app.py`）

侧边栏功能：
- **光瞳类型选择**：平顶光 (Flat-Top) / 高斯光 (Gaussian)
  - 平顶光 → FTL 模型拟合 R_FL + Uniformity 分析
  - 高斯光 → 截面高斯拟合 + 束腰位置/直径
- **显示单位**：μm / mm / nm，动态切换所有长度/直径值的显示格式
- 积分球/光轴相机像素尺寸、去暗场方法、光学参数（波长、焦距、入瞳直径）

分析结果：
- D4σ 直径（X/Y/平均），PIB 占比
- 高斯拟合直径（基于 X/Y 截面）
- 包围圆检测与可视化
- FTL 径向拟合图（平顶光）或截面高斯拟合图（高斯光）
- BPP、发散角、M²、斯特列尔比
- 3D Plotly 表面可视化（光轴 vs 理想 vs 光瞳）
- 结果汇总表 + CSV 导出

## 代码审计与重构

项目已通过以下代码质量改进：

- **重复代码移除**：`plot_3d_visualization()` 中范围归一化代码块被复制两次，已删除冗余副本
- **深层嵌套优化**：将 FTL 径向绘图（`_render_ftl_radial_plot`）和光瞳类型分析（`_render_pupil_type_analysis`）提取为独立函数，`main()` 嵌套深度从 6 级降至 4 级
- **工具函数提取**：`compute_radial_profile()` 从 `fit_flat_topped_lorentz()` 提取到 `analysis.image.common`，供模块内复用
- **变量初始化完善**：`uniformity_pupil` 总是通过函数返回值初始化，消除潜在 `NameError`

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 代码检查
uv run ruff check src/ ui/analysis/ ui/
uv run ruff format src/ ui/analysis/ --check

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
