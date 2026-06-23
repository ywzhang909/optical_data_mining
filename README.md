# AO光束质量分析 — Streamlit Cloud

上传光轴(Axis)和光瞳(Pupil)图像，自动计算激光光束质量关键指标：D4σ、PIB、Strehl、BPP、M²、Zernike 波前像差等。

---

## 目录

- [在线访问](#在线访问)
- [本地使用](#本地使用)
- [功能说明](#功能说明)
- [项目结构](#项目结构)
- [二次开发](#二次开发)
- [技术栈](#技术栈)

---

## 在线访问

部署到 [Streamlit Cloud](https://streamlit.io/cloud)：

👉 **在线使用**: [https://opticaldatamining-i2ug6sunezqnmndq6zvn7r.streamlit.app/](https://opticaldatamining-i2ug6sunezqnmndq6zvn7r.streamlit.app/)

自行部署：

1. 将此仓库 fork 或 push 到你的 GitHub
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 **New app** → 选择该仓库 + 分支 `streamlit-cloud`
4. 入口文件：`streamlit_app.py`
5. 部署完成后即可通过生成的 URL 在线访问

### 依赖安装

Streamlit Cloud 会自动读取 `requirements.txt` 安装依赖，无需手动操作。

---

## 本地使用

### 环境要求

- Python 3.12+
- pip 或 uv

### 安装与运行

```bash
# 克隆仓库
git clone <your-repo-url>
cd <repo-dir>
git checkout streamlit-cloud

# 方式一：pip
pip install -r requirements.txt
streamlit run streamlit_app.py

# 方式二：uv（推荐，更快）
uv pip install -r requirements.txt
uv run streamlit run streamlit_app.py

# 方式三：虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

---

## 功能说明

### 分析流程

1. **上传图片**：在侧边栏分别上传光轴(Axis)和/或光瞳(Pupil)图片
2. **设置参数**：配置相机参数（像素尺寸）、处理参数（去暗场方法）
3. **选择光瞳类型**：平顶光(Flat-Top) / 高斯光(Gaussian)
4. **点击计算**：系统自动计算所有适用指标
5. **查看结果**：可视化图表 + 结果汇总表 + CSV 导出

### 支持的计算指标

| 指标 | 说明 | 需要图片 |
|------|------|----------|
| **D4σ 直径** | ISO 11146 标准二阶矩直径（X/Y/平均） | Axis / Pupil |
| **PIB 占比** | 桶中功率比（Power In Bucket） | Axis |
| **高斯拟合直径** | X/Y 截面高斯拟合半高宽直径 | Axis |
| **FTL 特征半径 R_FL** | 平顶光 Flat-Topped Lorentz 拟合 | Pupil（平顶光模式） |
| **均匀度分析** | RMS / P-V 非均匀度 | Pupil（平顶光模式） |
| **包围圆检测** | minEnclosingCircle 拟合 + 椭圆拟合 | Pupil |
| **BPP** | 光束参数积（Beam Parameter Product） | Axis + Pupil |
| **M²** | 光束质量因子 | Axis + Pupil |
| **斯特列尔比** | 能量守恒法 Strehl Ratio | Axis + Pupil |
| **Zernike 波前** | Zernike 多项式（Noll 归一化，基于 `zernike` 包）波前像差分解，亚像素平移到包围圆中心后拟合 | Pupil |
| **3D 可视化** | Plotly 交互式 3D 光强表面 | Axis / Pupil |

### 参数说明

在侧边栏可以配置：

- **相机参数**：光轴/光瞳相机像素尺寸（μm）
- **去暗场方法**：无 / 中值 / 最小值 / 1/e / 手动输入
- **光瞳类型**：平顶光(FTL拟合) / 高斯光(高斯拟合)
- **均匀度边界**：包围圆 / 椭圆 / FTL特征半径 / 二阶矩半径
- **光学参数**：波长、焦距、入瞳直径
- **显示单位**：μm / mm / nm 动态切换
- **Zernike阶数**：4~10阶可配置（使用 `zernike` 包 + 亚像素对齐预处理）

---

## 项目结构

```
├── streamlit_app.py              # 主入口（Streamlit 应用）
├── analysis/                     # 光束分析算法包
│   ├── __init__.py
│   ├── image/
│   │   ├── __init__.py
│   │   └── common.py             # 图像 I/O、特征提取、径向剖面
│   └── optical_analysis/
│       ├── __init__.py           # 模块入口，统一导出
│       ├── beam_analysis.py      # D4σ, PIB, 高斯拟合, BPP, M²
│       ├── beam_analysis_metrics.py  # 综合分析类
│       ├── diffraction.py        # 菲涅尔衍射, 斯特列尔比, FFT居中
│       ├── history_manager.py    # 分析结果历史记录
│       ├── image_utils.py        # 图像读取、去暗场、椭圆拟合
│       ├── uniform_analysis.py   # 平顶光均匀度分析
│       ├── zernike_analysis.py   # Zernike 波前分解
│       └── visualization/
│           ├── __init__.py
│           ├── beam_visualization.py  # Plotly 3D / beam 可视化
│           └── renderers.py           # matplotlib 渲染工具包（纯绘图函数）
├── requirements.txt              # Python 依赖
├── .streamlit/
│   └── config.toml               # Streamlit Cloud 配置
└── README.md
```

### 关键模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **图像输入** | `image/common.py` | TIFF/PNG/JPG 读取 → numpy 数组，径向/横纵截面提取 |
| **光束分析** | `optical_analysis/beam_analysis.py` | 核心算法：d4sigma, pib_ratio, gaussian, fitting_gaussian, calculate_xy_diameters, calculate_bpp |
| **衍射计算** | `optical_analysis/diffraction.py` | shift_to_center_fft, calculate_strehl_ratio_with_energy_conservation |
| **图像工具** | `optical_analysis/image_utils.py` | read_image_to_numpy, subtract_dark_field, find_spot_border, ellipse_fit |
| **波前分析** | `optical_analysis/zernike_analysis.py` | fit_zernike, zernike_order_label |
| **均匀度** | `optical_analysis/uniform_analysis.py` | calculate_uniformity_metrics, plot_uniformity_analysis |
| **可视化** | `visualization/beam_visualization.py` | Plotly 3D 表面图、交互式 Beam 图 |
| **渲染器** | `visualization/renderers.py` | matplotlib 纯绘图函数（D4σ圆、高斯拟合、FTL极坐标、Zernike柱状图） |

---

## 二次开发

### 添加新的分析算法

1. 在 `analysis/optical_analysis/` 下创建新模块（如 `new_analysis.py`）
2. 实现分析函数，遵循现有约定（输入为 numpy 数组，返回 dict）
3. 在 `analysis/optical_analysis/__init__.py` 中导出
4. 在 `streamlit_app.py` 的 `main()` 函数中调用并渲染结果

示例：

```python
# analysis/optical_analysis/new_analysis.py
import numpy as np

def my_metric(image: np.ndarray) -> dict:
    """自定义分析指标"""
    result = np.mean(image)
    return {"my_metric": float(result)}

# analysis/optical_analysis/__init__.py
from .new_analysis import my_metric
```

### 修改前端界面

- `streamlit_app.py` 中的 `main()` 函数控制所有 UI 布局和交互逻辑
- 使用 Streamlit 原生组件：`st.columns`, `st.metric`, `st.pyplot`, `st.plotly_chart`
- 图表绘制使用 `matplotlib`（静态图）和 `plotly`（交互式 3D）
- **渲染分离**：所有绘图函数放在 `visualization/renderers.py` 中（纯 matplotlib，不含 Streamlit 调用），`streamlit_app.py` 只负责调用并显示
- Plotly 交互式可视化位于 `visualization/beam_visualization.py`
- 侧边栏参数通过 `st.sidebar` 管理

### 部署自定义版本

1. Fork 本仓库
2. 在 `streamlit-cloud` 分支上修改
3. 修改 `requirements.txt` 添加新依赖
4. 推送到 GitHub → Streamlit Cloud 自动重新部署

### 本地调试

```bash
# 热重载模式（代码修改后自动刷新）
streamlit run streamlit_app.py --server.runOnSave true

# 指定端口
streamlit run streamlit_app.py --server.port 8080
```

---

## 技术栈

| 技术 | 用途 |
|------|------|
| [Streamlit](https://streamlit.io/) | Web 应用框架 |
| [NumPy](https://numpy.org/) | 数值计算 |
| [SciPy](https://scipy.org/) | 优化拟合、图像处理 |
| [OpenCV](https://opencv.org/) | 图像边缘检测、椭圆拟合 |
| [Matplotlib](https://matplotlib.org/) | 2D 静态图表 |
| [Plotly](https://plotly.com/python/) | 交互式 3D 可视化 |
| [Pillow](https://python-pillow.org/) | 图像文件读写 |
| [scikit-image](https://scikit-image.org/) | 图像处理辅助 |
| [Loguru](https://loguru.readthedocs.io/) | 日志记录 |

---

## License

MIT
