# 实验数据分析 Pipeline

基于抽象基类和 Pipeline 编排系统的实验数据分析框架。

## 架构设计

### 核心组件

1. **数据模型 (Pydantic BaseModel)**
   - `Experiment`: 实验数据模型
   - `DataQualityReport`: 数据质量报告
   - `DataCleaningConfig`: 数据清洗配置
   - `DataCleaningResult`: 数据清洗结果
   - `AggregationConfig`: 聚合配置
   - `AggregationResult`: 聚合结果
   - `DatabaseConfig`: 数据库配置
   - `ExperimentFilter`: 实验筛选配置
   - `DataAnalysisPipeline`: 分析流程配置

2. **抽象基类**
   - `BaseProcessor`: 处理器抽象基类
   - `ProcessingResult`: 处理结果模型
   - `ProcessorConfig`: 处理器配置模型
   - `DataFrameProcessor`: DataFrame 处理器基类
   - `ExperimentDataProcessor`: Experiment 数据处理器基类

3. **Pipeline 编排系统**
   - `PipelineManager`: Pipeline 管理器
   - `PipelineExecutor`: Pipeline 执行器
   - `PipelineLogger`: Pipeline 日志记录器
   - `PipelineStage`: 管道阶段枚举
   - `PipelineStatus`: 管道状态枚举

4. **Pipeline 管理器**
   - `ExperimentPipelineManager`: 实验分析 Pipeline 管理器
   - `PipelineManagerConfig`: 管理器配置

### 处理器模块

1. **数据加载器**
   - `DataLoader`: 从数据库加载数据

2. **过滤器**
   - `ExperimentFilter`: 实验数据过滤

3. **质量分析器**
   - `QualityAnalyzer`: 数据质量分析

4. **数据清洗器**
   - `DataCleaningProcessor`: 数据清洗

5. **数据聚合器**
   - `DataAggregator`: 数据聚合

6. **导出器**
   - `DataExporter`: 数据导出

## 使用示例

### 基本使用

```python
from data_mining.experiment_analysis import (
    ExperimentPipelineManager,
    ExperimentFilter,
    DataCleaningConfig
)

# 创建管理器
manager = ExperimentPipelineManager()

# 创建标准分析 Pipeline
pipeline = manager.create_analysis_pipeline("standard_analysis")

# 准备输入数据
input_data = {
    'config': {
        'db_config': {
            'db_user': 'root',
            'db_password': '123456',
            'db_host': 'localhost',
            'db_port': 3306,
            'db_name': 'data_mining'
        },
        'table_name': 'experiment'
    },
    'filter_config': ExperimentFilter(
        experiment_type_id=2
    ),
    'cleaning_config': DataCleaningConfig()
}

# 执行 Pipeline
result = manager.execute_pipeline("standard_analysis", input_data)

# 处理结果
if result['status'] == 'completed':
    cleaned_data = result.get('cleaned_data')
    quality_report = result.get('quality_report')
    aggregation_result = result.get('aggregation_result')
    print(f"分析完成，清洗后数据: {len(cleaned_data)} 条")
```

### 自定义 Pipeline

```python
from data_mining.experiment_analysis import (
    DataLoader,
    QualityAnalyzer,
    DataAggregator
)
from data_mining.experiment_analysis.pipeline import (
    PipelineExecutionConfig,
    PipelineStageConfig,
    PipelineStage
)

# 创建管理器
manager = ExperimentPipelineManager()

# 注册处理器
manager.register_processor('DataLoader', DataLoader)
manager.register_processor('QualityAnalyzer', QualityAnalyzer)
manager.register_processor('DataAggregator', DataAggregator)

# 创建自定义 Pipeline
stages_config = [
    PipelineStageConfig(
        stage_name=PipelineStage.LOAD,
        processor_type=DataLoader,
        enabled=True
    ),
    PipelineStageConfig(
        stage_name=PipelineStage.QUALITY,
        processor_type=QualityAnalyzer,
        enabled=True
    ),
    PipelineStageConfig(
        stage_name=PipelineStage.AGGREGATE,
        processor_type=DataAggregator,
        enabled=True
    )
]

custom_pipeline = manager.compile_pipeline("custom_pipeline", stages_config)

# 执行
input_data = {'config': {'table_name': 'experiment'}}
result = manager.execute_pipeline("custom_pipeline", input_data)
```

### 单独使用处理器

```python
from data_mining.experiment_analysis import (
    ExperimentFilter,
    QualityAnalyzer
)

# 创建过滤器
filter_processor = ExperimentFilter()
config = ExperimentFilter(
    experiment_type_id=2,
    target_material='不锈钢'
)
result = filter_processor.execute({'data': df, 'config': config})

# 创建质量分析器
analyzer = QualityAnalyzer()
result = analyzer.execute({'data': df})
quality_report = result.data
```

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_experiment_analysis/

# 运行特定测试
pytest tests/test_experiment_analysis/test_base_processor.py::test_base_processor_initialization

# 显示详细输出
pytest -v tests/test_experiment_analysis/

# 显示打印输出
pytest -s tests/test_experiment_analysis/
```

## Pipeline 阶段

1. **LOAD**: 数据加载
2. **FILTER**: 数据过滤
3. **QUALITY**: 质量分析
4. **CLEAN**: 数据清洗
5. **AGGREGATE**: 数据聚合
6. **EXPORT**: 数据导出

## 配置说明

### 数据库配置

```python
db_config = DatabaseConfig(
    db_user="root",
    db_password="password",
    db_host="localhost",
    db_port=3306,
    db_name="data_mining"
)
```

### 筛选配置

```python
filter_config = ExperimentFilter(
    experiment_type_id=2,
    date_range=(start_date, end_date),
    target_material="不锈钢",
    distance=1.0,
    min_power=10.0,
    max_power=15.0,
    exclude_params=['experiment_extra']
)
```

### 清洗配置

```python
cleaning_config = DataCleaningConfig(
    target_material_map={
        '不锈钢': '不锈钢',
        '铝合金': '铝合金',
        '铝': '铝合金'
    },
    sub_beam_map={
        '31路子束': '31路',
        '31路': '31路',
        '33路子束': '33路',
        '全子束': '64路'
    },
    target_priority=['材质', '靶材']
)
```

## 输出格式

### 数据质量报告

```python
{
    'total_count': 100,
    'complete_records': 85,
    'incomplete_records': 15,
    'missing_rate': 0.15,
    'param_missing_stats': {
        'laser': {...},
        'target': {...},
        'env': {...}
    },
    'laser_params_missing': 10,
    'laser_params_complete': 75,
    'target_params_missing': 5,
    'target_params_complete': 80,
    'env_params_missing': 0,
    'env_params_complete': 100
}
```

### 数据清洗结果

```python
{
    'original_count': 100,
    'cleaned_count': 85,
    'removed_count': 15,
    'removal_rate': 0.15,
    'missing_stats_before': {...},
    'missing_stats_after': {...},
    'cleaning_details': [...]
}
```

### 聚合结果

```python
{
    'groups': {...},
    'group_counts': {...},
    'total_groups': 2,
    'detailed_stats': {...}
}
```

## 错误处理

所有处理器都返回 `ProcessingResult` 对象，包含：

- `success`: 是否成功
- `message`: 结果消息
- `data`: 处理数据
- `error`: 错误信息
- `execution_time`: 执行时间

```python
result = processor.execute(data)
if result.success:
    # 处理成功
    data = result.data
else:
    # 处理失败
    print(f"错误: {result.error}")
```

## 性能优化

1. **并行执行**: 支持异步并行执行 Pipeline 阶段
2. **缓存机制**: 支持数据缓存
3. **批处理**: 支持批处理大数据集
4. **执行时间跟踪**: 自动记录和处理执行时间

## 扩展开发

### 创建自定义处理器

```python
from data_mining.experiment_analysis.base_processor import (
    BaseProcessor,
    ProcessingResult,
    ProcessorConfig
)

class CustomProcessor(BaseProcessor[pd.DataFrame]):
    config_class = ProcessorConfig
    
    def validate_input(self, data: pd.DataFrame) -> tuple[bool, Optional[str]]:
        if not isinstance(data, pd.DataFrame):
            return False, "输入必须是 DataFrame"
        return True, None
    
    def process(self, data: pd.DataFrame, **kwargs) -> ProcessingResult:
        # 实现处理逻辑
        return ProcessingResult(
            success=True,
            message="处理成功",
            data=result_data
        )
```

## 文件结构

```
src/data_mining/experiment_analysis/
├── __init__.py              # 模块初始化
├── models.py                # Pydantic 数据模型
├── base_processor.py        # 抽象基类
├── pipeline.py              # Pipeline 编排系统
├── pipeline_manager.py      # Pipeline 管理器
├── data_loader.py           # 数据加载器
├── filter.py                # 过滤器
├── quality_analysis.py      # 质量分析器
├── data_cleaning.py         # 数据清洗器
├── data_aggregation.py      # 数据聚合器
├── exporter.py              # 数据导出器
└── example_usage.py         # 使用示例

tests/test_experiment_analysis/
├── __init__.py
├── test_base_processor.py
├── test_data_loader.py
├── test_quality_analysis.py
├── test_data_cleaning.py
├── test_data_aggregation.py
└── test_pipeline.py
```

## 依赖项

- pydantic==2.4.2
- pandas>=2.3.0
- numpy>=2.3.0
- sqlalchemy>=2.0.45
- pytest==7.4.3

## 许可证

MIT License
