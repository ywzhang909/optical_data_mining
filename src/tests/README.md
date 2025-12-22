# 测试说明

## 目录结构

```
tests/
├── __init__.py
├── conftest.py
├── test_process.py
└── README.md
```

## 运行测试

### 安装依赖

确保已安装所有必需的依赖项：

```bash
pip install pytest numpy scipy scikit-image pandas
```

### 运行所有测试

在项目根目录下运行以下命令：

```bash
cd src
python -m pytest tests/ -v
```

或者直接在项目根目录运行：

```bash
python -m pytest src/tests/ -v
```

### 运行特定测试文件

```bash
python -m pytest src/tests/test_process.py -v
```

### 运行特定测试函数

```bash
python -m pytest src/tests/test_process.py::test_find_centroid -v
```

## 测试内容

`test_process.py` 文件包含了对 `src/data_mining/image/process.py` 中所有函数的单元测试：

1. `find_centroid` - 测试质心计算功能
2. `calculate_beam_width` - 测试光束宽度计算功能
3. `fit_hyperbola` - 测试双曲线拟合功能
4. `calculate_m2_factor` - 测试M²因子计算功能
5. `calculate_strehl_ratio` - 测试斯特列尔比计算功能
6. `encircled_energy` - 测试包围能量计算功能
7. `calculate_ellipticity` - 测试椭圆度计算功能
8. `calculate_symmetry` - 测试对称性计算功能
9. `calculate_uniformity` - 测试均匀性计算功能

## 测试数据

测试使用合成数据来验证各函数的正确性：

1. 使用高斯分布样式的图像数据测试图像处理函数
2. 使用线性/二次数据测试拟合函数
3. 使用典型物理参数测试物理计算函数

## 测试覆盖

测试覆盖了以下方面：

1. 函数的基本功能
2. 边界条件处理
3. 返回值的正确性
4. 异常输入的处理