# Swifter 使用指南

## 快速开始

### 1. 安装 swifter

```bash
pip install swifter
```

### 2. 基本用法

```python
import pandas as pd
import swifter

# 原始代码
df['result'] = df['column'].apply(function)

# 使用 swifter 优化
df['result'] = df['column'].swifter.apply(function)
```

## 配置选项

### 基本配置

```python
import swifter

# 设置默认配置
swifter.set_defaults(
    npartitions=4,           # 分区数量（建议设置为CPU核心数）
    dask_threshold=100,      # 数据量阈值，超过此数量使用dask
    disable_cache=False,     # 启用缓存
    progress_bar=True        # 显示进度条
)
```

### 高级配置

```python
# 针对特定操作的配置
df['result'] = df['column'].swifter(
    npartitions=8,           # 自定义分区数
    dask_threshold=50,       # 自定义阈值
    progress_bar=False,      # 不显示进度条
    allow_dask_on_strings=True  # 允许对字符串使用dask
).apply(function)
```

## 最佳实践

### 1. 何时使用 swifter

**推荐使用的情况：**
- 数据量 > 100 行
- apply 函数计算复杂度较高
- 函数执行时间 > 1ms
- CPU 密集型操作

**不推荐使用的情况：**
- 数据量 < 50 行
- 函数执行时间很短（< 1ms）
- I/O 密集型操作
- 函数内部已有并行处理

### 2. 性能优化技巧

```python
# 1. 合理设置分区数
swifter.set_defaults(npartitions=4)  # 通常设置为CPU核心数

# 2. 启用缓存
swifter.set_defaults(disable_cache=False)

# 3. 调整阈值
swifter.set_defaults(dask_threshold=100)  # 根据数据量调整

# 4. 显示进度条
swifter.set_defaults(progress_bar=True)
```

### 3. 内存管理

```python
# 对于大数据集，注意内存使用
import gc

# 处理完一批数据后清理内存
df['result'] = df['column'].swifter.apply(function)
gc.collect()  # 手动垃圾回收
```

## 常见问题

### 1. 类型错误

**问题：** 某些复杂的numpy数组操作可能出现类型错误

**解决方案：**
```python
# 确保返回值类型一致
def safe_function(x):
    result = some_complex_operation(x)
    return float(result)  # 明确指定返回类型
```

### 2. 内存不足

**问题：** 处理大数据集时内存不足

**解决方案：**
```python
# 减少分区数
swifter.set_defaults(npartitions=2)

# 分批处理
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    batch['result'] = batch['column'].swifter.apply(function)
    # 处理批次结果...
```

### 3. 调试困难

**问题：** 并行执行时调试信息不清晰

**解决方案：**
```python
# 临时禁用 swifter 进行调试
# df['result'] = df['column'].swifter.apply(function)
df['result'] = df['column'].apply(function)  # 串行执行便于调试
```

## 性能监控

### 1. 使用装饰器监控

```python
import time
from functools import wraps

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

@time_function
def your_function(x):
    # 你的处理逻辑
    return result
```

### 2. 比较性能

```python
import time

# 测试普通 apply
start = time.time()
result1 = df['column'].apply(function)
normal_time = time.time() - start

# 测试 swifter apply
start = time.time()
result2 = df['column'].swifter.apply(function)
swifter_time = time.time() - start

print(f"普通 apply: {normal_time:.4f}s")
print(f"swifter: {swifter_time:.4f}s")
print(f"加速比: {normal_time/swifter_time:.2f}x")
```

## 实际应用示例

### 图像处理优化

```python
import pandas as pd
import numpy as np
import swifter
from PIL import Image

def process_image(file_path):
    """处理图像文件"""
    image = Image.open(file_path)
    image_array = np.asarray(image, dtype=np.float32)
    
    # 复杂的图像处理逻辑
    result = np.sum(image_array) / np.median(image_array)
    return result

# 优化前
df['result'] = df['file_path'].apply(process_image)

# 优化后
df['result'] = df['file_path'].swifter.apply(process_image)
```

### 数值计算优化

```python
def complex_calculation(data):
    """复杂的数值计算"""
    # 模拟复杂计算
    result = np.sum(data**2) / np.std(data)
    return result

# 优化前
df['result'] = df['data_column'].apply(complex_calculation)

# 优化后
df['result'] = df['data_column'].swifter.apply(complex_calculation)
```

## 总结

Swifter 是一个简单而强大的工具，可以显著提升 pandas apply 操作的性能。通过合理的配置和使用，可以在不改变代码逻辑的情况下获得 2-8 倍的性能提升。

**关键要点：**
1. 正确配置 swifter 参数
2. 了解何时使用 swifter
3. 注意内存管理和类型兼容性
4. 使用性能监控工具评估效果