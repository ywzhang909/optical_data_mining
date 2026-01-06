# 数字光学数据分析

一个全面的激光光束质量分析工具包，支持功率数据分析、光束特征提取、光学质量评估和时间序列分析。

## 🚀 主要功能

### 1. 功率数据分析
- **双Sigmoid脉冲拟合**：适用于激光脉冲、光开关响应等带上升沿和下降沿的功率曲线
- **上升/下降时间计算**：10%-90%幅度时间的精确测量
- **时间序列拟合**：支持多种拟合模型（高斯、双误差函数等）

### 2. 图像处理与分析
- **TIFF图像处理**：高效的TIFF格式图像读取和预处理
- **去噪算法**：中值滤波、最小值滤波等多种去噪方法
- **背景扣除**：自动暗场校正
- **质心计算**：基于强度加权的高精度质心定位

### 3. 光束特征提取
- **D4σ直径计算**：基于一阶矩和二阶矩的光斑尺寸测量
- **高斯拟合**：光束横向分布的高斯拟合分析
- **椭圆拟合**：光束形变和椭圆度分析
- **环围能量分析**：指定能量百分比对应的光斑半径计算

### 4. 光学质量评估
- **Strehl比计算**：基于出瞳与焦平面CCD图像的斯特列尔比计算
- **BPP和M²计算**：光束参数积和光束质量因子的精确测量
- **均匀度分析**：四象限均匀度和RMS均匀度评估
- **平顶拟合**：适用于平顶光束的erf平滑边界拟合

### 5. 时间序列分析
- **平稳性检验**：ADF检验判断时间序列平稳性
- **频谱分析**：Lomb-Scargle周期图分析非均匀采样信号
- **趋势分解**：时间序列的趋势、季节性和残差分解
- **自相关分析**：ACF和PACF分析

### 6. 图像配准与位移检测
- **相位相关法**：亚像素精度的图像位移计算
- **光流分析**：光斑运动轨迹追踪

## 📁 项目结构

```
├── src/data_mining/
│   ├── fitting/
│   │   └── fit_funcs.py          # 拟合函数（DoubleErfPulse, DoubleSigmoid等）
│   ├── image/
│   │   ├── common.py             # 通用图像处理函数
│   │   ├── process.py            # 图像处理流程
│   │   └── zernike.py            # Zernike多项式拟合
│   └── services/
│       └── spots_service.py      # 光斑分析服务
├── notebooks/
│   ├── data.ipynb                # 主要数据分析notebook
│   ├── zernike_fit_near_spot.ipynb # Zernike拟合示例
│   └── atmosphere_analysis.ipynb # 大气分析
├── utils/
│   ├── file_utils.py             # 文件处理工具
│   ├── light_spots_utils.py      # 光斑分析工具
│   └── render_utils.py           # 渲染工具
└── tests/                        # 测试文件
```

## 🛠 安装与依赖

### 系统要求
- Python 3.8+
- Windows/Linux/macOS

### 主要依赖
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
opencv-python>=4.5.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
Pillow>=8.3.0
swifter>=1.1.0
```

### 安装步骤
1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 确保所有依赖项正确安装

## 📖 使用指南

### 快速开始

1. **数据准备**
   - 将TIFF格式的光束图像放在指定目录
   - 准备功率计数据文件（支持TXT格式）
   - 确保时间戳格式正确

2. **运行分析**
   ```python
   # 在Jupyter Notebook中
   import sys
   sys.path.append('../src')
   
   from data_mining.image.common import get_profiles
   from data_mining.fitting.fit_funcs import DoubleErfPulse
   
   # 执行数据分析
   ```

3. **主要分析流程**
   - 功率数据拟合：`DoubleErfPulse` 或 `DoubleSigmoid`
   - 图像特征提取：`d4sigma_feature_extract()`
   - 光学质量计算：`strehl_with_centering()`
   - BPP计算：`calculate_bpp_from_pupil_and_focal()`

### 核心API

#### 功率拟合
```python
from data_mining.fitting.fit_funcs import DoubleErfPulse

# 创建拟合对象
fit = DoubleErfPulse(power_data)
# 执行拟合
params = fit.fit()
# 获取上升/下降时间
rise_time = fit.rise_time
fall_time = fit.fall_time
```

#### 光束特征提取
```python
from data_mining.image.common import get_profiles, d4sigma

# 提取光束截面
profiles = get_profiles(image_array, center=(cx, cy))
# 计算D4σ直径
d4sigma_features = d4sigma(image_array)
```

#### Strehl比计算
```python
from data_mining.image.common import strehl_with_centering

result = strehl_with_centering(pupil_image, focal_image)
strehl_ratio = result['strehl']
```

## 🔬 分析对象

### 1. 光轴Image（焦平面）
- **质心分析**：光轴抖动频谱、光强比较
- **高斯拟合**：光束半径测量
- **椭圆拟合**：形心、半径、角度分析
- **BPP计算**：X、Y方向光束质量
- **Strehl比**：近场光斑与理想光斑比较

### 2. 光瞳Image（出瞳平面）
- **均匀度分析**：RMS均匀度、变异系数
- **平顶拟合**：erf边界平顶函数拟合
- **环围能量**：指定能量百分比半径计算

### 3. 功率计数据
- **稳定功率分析**：最大功率、激光器功率比较
- **脉冲特性**：半功率时间、上升/下降沿分析

### 4. 波前数据
- **波前RMS**：包含/不包含倾斜离焦的分析
- **Zernike拟合**：多项式波前重建
- **频谱分析**：波前误差频域特性

## 📊 结果展示

### 可视化功能
- **实时绘图**：matplotlib集成的动态图表
- **统计分析**：多维度统计特征计算
- **趋势分析**：时间序列趋势和周期性展示
- **频谱分析**：功率谱密度和主频率识别

### 导出格式
- **Parquet**：高效的数据存储格式
- **CSV**：通用数据交换格式
- **图表**：PNG/PDF格式的图表导出

## 🧪 测试

运行测试套件：
```bash
python -m pytest tests/ -v
```

## 📚 文档

详细文档请参考：
- [`doc/光束质量分析.md`](doc/光束质量分析.md) - 光束质量分析原理
- [`doc/接口文档.md`](doc/接口文档.md) - API接口说明
- [`notebooks/`](notebooks/) - 示例notebook

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 常见问题

### Q: 导入错误怎么办？
A: 确保正确设置Python路径：
```python
sys.path.append('../src')
```

### Q: TIFF文件读取失败？
A: 检查文件格式和权限，确保使用正确的文件路径。

### Q: 拟合效果不好？
A: 调整初始参数或检查数据质量，确保信号具有足够的信噪比。

## 📞 联系信息

如有问题，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

---

**注意**：本工具包专为激光光束质量分析设计，使用前请确保理解相关光学原理和测量方法。