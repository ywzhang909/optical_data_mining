# Ryven激光光束分析工作流

本项目将notebooks/data.ipynb中的计算函数转换为Ryven节点，形成一个完整的激光光束质量分析可视化工作流。

## 概述

Ryven是一个基于节点的可视化编程环境，本项目定义了多个用于激光光束分析的算子节点，可以将复杂的分析流程以图形化方式连接起来。

## 算子节点列表

### 1. PowerDataAnalysisNode (功率数据分析)
- **功能**: 对激光功率时间序列进行分析和拟合
- **输入**: 
  - time_data: 时间数据
  - power_data: 功率数据
  - fit_method: 拟合方法（默认'sigmoid'）
- **输出**:
  - fitted_data: 拟合数据
  - parameters: 拟合参数
  - rise_time: 上升时间
  - fall_time: 下降时间

### 2. ImagePreprocessingNode (图像预处理)
- **功能**: 对激光光斑图像进行预处理，包括背景扣除
- **输入**:
  - image: 图像数据
  - method: 预处理方法（默认'background_subtraction'）
  - kernel_size: 核大小（默认21）
  - sigma_factor: 阈值倍数（默认3.0）
- **输出**:
  - processed_image: 预处理图像
  - background: 背景图像
  - mask: 有效区域掩码

### 3. D4SigmaFeatureNode (D4σ特征提取)
- **功能**: 计算激光光束的D4σ直径等特征参数
- **输入**:
  - image: 图像数据
  - pixel_size_um: 像素尺寸（默认1.0μm）
- **输出**:
  - d4sigma_x: X方向D4σ直径
  - d4sigma_y: Y方向D4σ直径
  - center_x: X中心坐标
  - center_y: Y中心坐标
  - total_power: 总功率

### 4. EllipseFitNode (椭圆拟合)
- **功能**: 对光斑进行椭圆拟合，提取形状参数
- **输入**:
  - image: 图像数据
  - threshold: 阈值（默认0.5）
- **输出**:
  - ellipse_params: 椭圆参数
  - major_axis: 长轴
  - minor_axis: 短轴
  - eccentricity: 离心率

### 5. StrehlRatioNode (Strehl比计算)
- **功能**: 计算激光光束的Strehl比
- **输入**:
  - pupil_image: 光瞳图像
  - focus_image: 焦平面图像
  - pixel_size_pupil_um: 光瞳像素尺寸（默认5.5μm）
  - pixel_size_focus_um: 焦面像素尺寸（默认3.45μm）
  - focal_length_mm: 焦距（默认100.0mm）
  - wavelength_um: 波长（默认1.064μm）
- **输出**:
  - strehl_ratio: Strehl比
  - ideal_matched: 匹配理想光斑

### 6. BPPM2Node (BPP和M²计算)
- **功能**: 计算激光光束的BPP和M²因子
- **输入**:
  - pupil_diameter_mm: 光瞳直径
  - focal_diameter_mm: 焦面直径
  - focal_length_mm: 焦距（默认100.0mm）
  - wavelength_nm: 波长（默认1064.0nm）
- **输出**:
  - bpp_mm_mrad: BPP (mm·mrad)
  - m2: M²因子
  - theta_mrad: 发散角
  - all_results: 所有结果

### 7. TimeSeriesAnalysisNode (时间序列分析)
- **功能**: 对时间序列数据进行统计分析
- **输入**:
  - data: 时间序列数据
  - analysis_type: 分析类型（默认'acf_pacf'）
- **输出**:
  - result: 分析结果
  - plot: 分析图表

### 8. WavefrontGradientNode (波前梯度分析)
- **功能**: 基于TIE方程计算波前梯度
- **输入**:
  - focus_image: 焦点图像
  - defocus_image: 离焦图像
  - delta_z: 离焦距离（默认3.0）
  - wavelength: 波长（默认1064e-9）
- **输出**:
  - gradient_map: 梯度图
  - wavefront_reconstruction: 波前重建

### 9. DataMergeNode (数据合并)
- **功能**: 合并多个数据集
- **输入**:
  - data1: 数据集1
  - data2: 数据集2
  - merge_on: 合并键（默认'time'）
  - merge_type: 合并类型（默认'left'）
- **输出**:
  - merged_data: 合并后数据

## 使用方法

### 1. 环境准备
```bash
pip install ryven
```

### 2. 启动Ryven GUI
```bash
ryven --project=beam_analysis
```

### 3. 在Ryven中使用节点
1. 启动Ryven后，导入`ryven_operators.py`中定义的节点
2. 从节点库中拖拽所需节点到工作区
3. 连接节点的输入输出端口
4. 设置节点参数
5. 运行分析流程

### 4. 典型工作流示例
```
ImageInputNode -> ImagePreprocessingNode -> D4SigmaFeatureNode
ImageInputNode -> ImagePreprocessingNode -> EllipseFitNode
D4SigmaFeatureNode, EllipseFitNode -> BPPM2Node
ImageInputNode -> StrehlRatioNode
PowerDataAnalysisNode -> TimeSeriesAnalysisNode
```

## 工作流示例

`ryven_workflow_example.py`文件包含一个完整的激光光束分析工作流示例，展示了如何连接各个节点以形成完整的分析流程。

## 依赖项

- numpy
- pandas
- scipy
- scikit-image
- opencv-python
- matplotlib
- statsmodels
- ryven

## 应用场景

1. **激光光束质量评估**: 计算M²因子、BPP、Strehl比等关键指标
2. **光束稳定性分析**: 通过时间序列分析评估光束参数的稳定性
3. **光斑形状分析**: 椭圆拟合、D4σ直径计算等
4. **功率特性分析**: 功率时间序列拟合与特征提取
5. **波前分析**: 波前梯度计算和分析

## 注意事项

1. 确保输入数据格式正确
2. 根据实际物理参数设置节点参数（如像素尺寸、波长等）
3. 大数据集处理时注意内存使用
4. 某些计算可能需要较长的处理时间

## 扩展性

这些节点设计为模块化，可以轻松扩展以支持更多的激光光束分析功能。新的分析算法可以被封装为新的节点并集成到工作流中。