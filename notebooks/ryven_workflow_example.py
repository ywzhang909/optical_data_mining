"""
Ryven工作流示例
展示如何将计算参考ryven的算子组合成完整的激光光束分析流程
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ryven.NENV import *
from ryven.main import run_gui

# 导入我们定义的算子
from ryven_operators import DataAnalysisNodes

def create_beam_analysis_workflow():
    """
    创建激光光束分析的完整工作流
    这个工作流包括以下步骤：
    1. 加载图像数据
    2. 图像预处理（背景扣除）
    3. 特征提取（D4σ, 椭圆拟合等）
    4. 功率数据分析
    5. 光束质量分析（Strehl比, BPP, M²）
    6. 结果可视化
    """
    
    # 注册所有节点
    DataAnalysisNodes.register_nodes()
    
    print("激光光束分析工作流创建完成！")
    print("\n工作流包含以下节点类型：")
    
    for node_class in DataAnalysisNodes.get_nodes():
        print(f"- {node_class.title}")
    
    print("\n工作流连接示例：")
    print("1. ImageInputNode -> ImagePreprocessingNode")
    print("2. ImagePreprocessingNode -> D4SigmaFeatureNode")
    print("3. ImagePreprocessingNode -> EllipseFitNode") 
    print("4. PowerDataAnalysisNode -> TimeSeriesAnalysisNode")
    print("5. D4SigmaFeatureNode, EllipseFitNode -> BPPM2Node")
    print("6. ImagePreprocessingNode -> StrehlRatioNode")
    print("7. 各分析结果 -> DataVisualizationNode")
    
    return True

def run_beam_analysis_demo():
    """
    运行激光光束分析演示
    """
    print("启动激光光束分析Ryven工作流演示...")
    
    try:
        # 创建工作流
        success = create_beam_analysis_workflow()
        
        if success:
            print("\n要启动Ryven GUI，请运行以下命令：")
            print("ryven --project=beam_analysis")
            print("\n或者在Python中使用：")
            print("from ryven.main import run_gui")
            print("run_gui()")
        
        return success
        
    except Exception as e:
        print(f"运行演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_workflow_json():
    """
    创建一个示例工作流JSON配置
    """
    workflow_json = {
        "title": "激光光束质量分析工作流",
        "nodes": [
            {
                "type": "ImageInputNode",
                "id": 1,
                "title": "图像输入",
                "inputs": {
                    "directory": "/path/to/image/data",
                    "file_pattern": "*.TIFF",
                    "process_images": True
                },
                "outputs": ["beam_data"]
            },
            {
                "type": "ImagePreprocessingNode", 
                "id": 2,
                "title": "图像预处理",
                "inputs": {
                    "image": {"source_node": 1, "output": "beam_data"},
                    "method": "background_subtraction",
                    "kernel_size": 21,
                    "sigma_factor": 3.0
                },
                "outputs": ["processed_image", "background", "mask"]
            },
            {
                "type": "D4SigmaFeatureNode",
                "id": 3,
                "title": "D4σ特征提取",
                "inputs": {
                    "image": {"source_node": 2, "output": "processed_image"},
                    "pixel_size_um": 5.5
                },
                "outputs": ["d4sigma_x", "d4sigma_y", "center_x", "center_y", "total_power"]
            },
            {
                "type": "EllipseFitNode",
                "id": 4,
                "title": "椭圆拟合",
                "inputs": {
                    "image": {"source_node": 2, "output": "processed_image"},
                    "threshold": 0.5
                },
                "outputs": ["ellipse_params", "major_axis", "minor_axis", "eccentricity"]
            },
            {
                "type": "PowerDataAnalysisNode",
                "id": 5,
                "title": "功率数据分析",
                "inputs": {
                    "time_data": [0, 1, 2, 3, 4, 5],  # 示例数据
                    "power_data": [0.1, 0.5, 1.2, 0.8, 0.2, 0.1],
                    "fit_method": "sigmoid"
                },
                "outputs": ["fitted_data", "parameters", "rise_time", "fall_time"]
            },
            {
                "type": "BPPM2Node",
                "id": 6,
                "title": "BPP和M²计算",
                "inputs": {
                    "pupil_diameter_mm": {"source_node": 3, "output": "d4sigma_x"},
                    "focal_diameter_mm": {"source_node": 3, "output": "d4sigma_y"},
                    "focal_length_mm": 100.0,
                    "wavelength_nm": 1064.0
                },
                "outputs": ["bpp_mm_mrad", "m2", "theta_mrad", "all_results"]
            },
            {
                "type": "TimeSeriesAnalysisNode",
                "id": 7,
                "title": "时间序列分析",
                "inputs": {
                    "data": {"source_node": 5, "output": "fitted_data"},
                    "analysis_type": "acf_pacf"
                },
                "outputs": ["result", "plot"]
            }
        ],
        "connections": [
            {"from_node": 1, "from_output": "beam_data", "to_node": 2, "to_input": "image"},
            {"from_node": 2, "from_output": "processed_image", "to_node": 3, "to_input": "image"},
            {"from_node": 2, "from_output": "processed_image", "to_node": 4, "to_input": "image"},
            {"from_node": 3, "from_output": "d4sigma_x", "to_node": 6, "to_input": "pupil_diameter_mm"},
            {"from_node": 3, "from_output": "d4sigma_y", "to_node": 6, "to_input": "focal_diameter_mm"},
            {"from_node": 5, "from_output": "fitted_data", "to_node": 7, "to_input": "data"}
        ]
    }
    
    return workflow_json

if __name__ == "__main__":
    print("激光光束分析Ryven工作流")
    print("="*50)
    
    # 运行演示
    run_beam_analysis_demo()
    
    print("\n" + "="*50)
    print("示例工作流JSON结构已定义")
    print("可以使用此结构在Ryven中创建预配置的工作流")