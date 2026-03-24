# Ryven 激光光束分析可视化指南

## 概述

基于 `notebooks/data.py` 中的图像处理功能，我们创建了一个使用 Ryven 库的节点式可视化工作流系统。该系统允许用户通过图形界面拖拽和连接节点来构建激光光束质量分析流程，无需编写代码即可完成复杂的图像分析任务。

## 系统特性

### 🎯 核心功能
- **节点式工作流**: 通过可视化节点构建分析流程
- **图像预处理**: 自适应背景扣除、降噪处理
- **特征提取**: D4σ、椭圆拟合、高斯拟合等
- **质量分析**: Strehl比、BPP、M²值计算
- **实时可视化**: 动态图表和结果展示
- **数据导出**: 支持多种格式输出

### 🔧 技术特点
- 基于 Ryven 框架的节点编程
- 集成现有 `data.py` 中的所有核心算法
- 支持并行处理和性能优化
- 灵活的参数配置和自定义

## 安装和设置

### 1. 环境要求

```bash
# 基础依赖
pip install ryven numpy pandas matplotlib opencv-python
pip install scipy scikit-learn statsmodels
pip install pyyaml pathlib

# 可选依赖（用于增强功能）
pip install plotly bokeh seaborn
pip install jupyter jupyterlab
```

### 2. 文件结构

```
notebooks/
├── ryven_beam_analysis.py    # 主要的Ryven节点实现
├── ryven_config.yaml         # 配置文件
└── data.py                   # 原始图像处理函数

doc/
└── ryven_beam_analysis_guide.md  # 本文档
```

### 3. 启动Ryven

```python
# 方法1: 直接运行
python notebooks/ryven_beam_analysis.py

# 方法2: 在Python中启动
from notebooks.ryven_beam_analysis import setup_ryven_environment
setup_ryven_environment()

# 方法3: 手动启动Ryven GUI
ryven  # 在终端中运行
```

## 节点详解

### 1. 图像输入节点 (Image Input)

**功能**: 加载图像文件并预处理

**输入参数**:
- `directory`: 图像目录路径
- `file_pattern`: 文件匹配模式 (默认: "*.TIFF")
- `process_images`: 是否处理图像 (默认: True)

**输出**:
- `beam_data`: 包含图像数据的DataFrame

**使用场景**:
- 加载激光光斑图像
- 支持多种图像格式
- 自动时间戳解析

### 2. 图像预处理节点 (Image Preprocessing)

**功能**: 对图像进行背景扣除和降噪处理

**输入参数**:
- `beam_data`: 输入光束数据
- `background_method`: 背景扣除方法 ("median"/"min")
- `kernel_size`: 核大小 (默认: 21)
- `sigma_factor`: 阈值倍数 (默认: 3.0)

**输出**:
- `processed_data`: 预处理后的数据

**算法特点**:
- 自适应背景估计
- MAD噪声估计
- 显著性掩码生成

### 3. 特征提取节点 (Feature Extraction)

**功能**: 提取图像的各种几何和物理特征

**输入参数**:
- `beam_data`: 光束数据
- `extract_d4sigma`: 启用D4σ特征 (默认: True)
- `extract_ellipse`: 启用椭圆特征 (默认: True)
- `extract_gaussian`: 启用高斯特征 (默认: False)

**输出**:
- `features_data`: 包含所有特征的DataFrame

**提取特征**:
- **D4σ特征**: 中心位置、直径、强度
- **椭圆特征**: 椭圆参数、圆度、均匀度
- **高斯特征**: 拟合参数、半径

### 4. 光束质量分析节点 (Beam Quality)

**功能**: 计算光束质量指标

**输入参数**:
- `pupil_data`: 光瞳数据
- `axis_data`: 光轴数据
- `focal_length`: 焦距(mm) (默认: 100.0)
- `wavelength`: 波长(nm) (默认: 1064.0)

**输出**:
- `quality_results`: 质量分析结果

**计算指标**:
- **Strehl比**: 光束质量评估
- **BPP**: 光束参数积
- **M²**: 光束质量因子

### 5. 数据可视化节点 (Data Visualization)

**功能**: 生成各种分析图表

**输入参数**:
- `data`: 分析数据
- `plot_type`: 图表类型 (默认: "time_series")
- `features`: 显示特征 (默认: "all")

**输出**:
- `plot_figure`: matplotlib图表对象

**图表类型**:
- 时间序列图
- 散点图矩阵
- 相关性热图
- 统计分布图

### 6. 数据导出节点 (Data Export)

**功能**: 导出分析结果

**输入参数**:
- `data`: 分析数据
- `export_path`: 导出路径
- `export_format`: 导出格式 ("parquet"/"csv")

**输出**:
- `export_status`: 导出状态信息

## 工作流示例

### 简单分析工作流

```
[图像输入] → [预处理] → [特征提取] → [可视化] → [数据导出]
    ↓           ↓           ↓           ↓           ↓
  TIFF文件    去背景      D4σ参数     时间序列    结果文件
```

### 高级分析工作流

```
[光瞳图像] → [预处理] → [特征提取] → 
    ↓           ↓           ↓
[光轴图像] → [预处理] → [特征提取] → [质量分析] → [可视化] → [导出]
    ↓           ↓           ↓           ↓           ↓
  光斑图像    去背景      几何参数    Strehl比    综合图表   数据文件
```

## 使用步骤

### 步骤1: 准备数据

确保图像文件位于指定目录，命名规范如下：
```
data/
├── 光瞳image/
│   └── 20231201_143025(全子束强光平顶)/
│       ├── image_001.TIFF
│       ├── image_002.TIFF
│       └── ...
└── 光轴image/
    └── 20231201_143025(全子束强光平顶)/
        ├── image_001.TIFF
        ├── image_002.TIFF
        └── ...
```

### 步骤2: 启动Ryven GUI

```bash
# 启动Ryven界面
ryven

# 或在Python中
python -c "from ryven import main; main.main()"
```

### 步骤3: 创建工作流

1. **添加节点**: 从节点库拖拽所需的节点到画布
2. **连接节点**: 通过输出输入端口连接节点
3. **设置参数**: 双击节点设置参数
4. **运行工作流**: 点击运行按钮执行分析

### 步骤4: 查看结果

- **实时监控**: 查看节点执行状态和进度
- **图表展示**: 在可视化节点中查看结果图表
- **数据导出**: 将结果导出为文件

## 配置选项

### 修改配置文件

编辑 `notebooks/ryven_config.yaml` 自定义系统行为：

```yaml
# 修改默认参数
data_processing:
  preprocessing:
    kernel_size: 31
    sigma_factor: 2.5

# 修改可视化设置
visualization:
  plots:
    time_series:
      figure_size: [16, 10]
      dpi: 150
```

### 自定义节点

```python
class CustomNode(Node):
    title = '自定义节点'
    color = '#FF0000'
    
    # 定义输入输出
    init_inputs = [...]
    init_outputs = [...]
    
    def update_event(self, inp=-1):
        # 实现节点逻辑
        pass
```

## 性能优化

### 1. 并行处理

```yaml
performance:
  parallel:
    enabled: true
    max_workers: 8
    chunk_size: 50
```

### 2. 内存管理

```yaml
performance:
  memory:
    max_cache_size: "2GB"
    gc_frequency: 50
```

### 3. 批量处理

- 使用批处理节点处理大量图像
- 启用进度显示监控处理状态
- 合理设置缓存大小避免内存溢出

## 故障排除

### 常见问题

1. **Ryven启动失败**
   ```bash
   # 检查安装
   pip list | grep ryven
   
   # 重新安装
   pip uninstall ryven
   pip install ryven
   ```

2. **图像加载错误**
   - 检查文件路径是否正确
   - 确认图像格式支持
   - 验证文件权限

3. **内存不足**
   - 减少批处理大小
   - 启用垃圾回收
   - 降低图像分辨率

4. **节点连接失败**
   - 检查输入输出类型匹配
   - 确认端口连接正确
   - 验证数据格式

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看节点状态
node.get_state()
```

## 扩展功能

### 1. 自定义算法

在 `ryven_beam_analysis.py` 中添加新节点：

```python
class CustomAlgorithmNode(Node):
    def update_event(self, inp=-1):
        # 实现自定义算法
        result = your_algorithm(self.input(0))
        self.set_output_val(0, result)
```

### 2. 集成其他库

```python
# 集成Plotly
import plotly.graph_objects as go

class InteractivePlotNode(Node):
    def create_interactive_plot(self, data):
        fig = go.Figure()
        # 创建交互式图表
        return fig
```

### 3. 批处理脚本

```python
# 自动运行工作流
def run_batch_analysis(data_dirs, output_dir):
    for data_dir in data_dirs:
        # 加载数据
        # 运行分析
        # 保存结果
        pass
```

## 最佳实践

### 1. 工作流设计
- 保持工作流简洁明了
- 合理组织节点布局
- 使用有意义的节点名称

### 2. 参数调优
- 根据实际数据调整预处理参数
- 优化特征提取设置
- 选择合适的可视化类型

### 3. 结果验证
- 对比不同参数设置的结果
- 验证关键指标的合理性
- 检查异常值和错误数据

### 4. 文档记录
- 记录工作流配置
- 保存重要参数设置
- 维护分析结果文档

## 技术支持

### 资源链接
- [Ryven官方文档](https://ryven.org/)
- [激光光束质量分析指南](../光束质量分析.md)
- [图像处理算法说明](../接口文档.md)

### 联系方式
- 技术问题: 查看代码注释和错误信息
- 功能建议: 提交Issue或Pull Request
- 使用咨询: 参考示例和最佳实践

---

*本指南基于 `notebooks/data.py` 的图像处理功能编写，旨在提供一个用户友好的可视化分析工具。*