# %%
"""
基于 Ryven 的激光光束质量分析可视化工作流

这个模块使用 Ryven 库创建一个节点式的可视化界面，
用于激光光束图像的预处理、特征提取和分析。
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入现有的图像处理函数
from notebooks.data import (
    read_tiff_to_numpy,
    adaptive_background_subtraction,
    process_beam_data,
    d4sigma_feature_extract,
    shape_feature_extract,
    strehl_with_centering,
    calculate_bpp_from_pupil_and_focal
)

# Ryven 相关导入
try:
    from ryven.NENV import *
    from ryven.core.reaction import Reaction
    from ryven.gui.node_datas import *
    from ryven.gui.icons import *
    from ryven.Style import *
    RYVEN_AVAILABLE = True
except ImportError:
    print("警告: Ryven 库未安装。请使用 'pip install ryven' 安装")
    RYVEN_AVAILABLE = False


# 配置matplotlib支持中文
try:
    import matplotlib.pyplot as plt
    import warnings
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
except ImportError:
    pass

# %%
class BeamAnalysisNodes:
    """激光光束分析节点集合"""
    
    if RYVEN_AVAILABLE:
        
        # === 图像输入节点 ===
        class ImageInputNode(Node):
            """图像输入节点"""
            title = '图像输入'
            color = '#FF6B6B'
            
            init_inputs = [
                NodeInputBP('directory', '数据目录'),
                NodeInputBP('file_pattern', '文件模式', default='*.TIFF'),
                NodeInputBP('process_images', '是否处理图像', dtype='bool', default=True)
            ]
            
            init_outputs = [
                NodeOutputBP('beam_data', '光束数据')
            ]
            
            def __init__(self, params):
                super().__init__(params)
                self.current_data = None
            
            def update_event(self, inp=-1):
                if inp == 0:  # directory changed
                    self.load_images()
            
            def load_images(self):
                """加载图像数据"""
                directory = self.input(0)
                file_pattern = self.input(1)
                process_images = self.input(2)
                
                # 检查是否是测试模式
                test_mode = getattr(self, 'test_mode', False)
                
                if directory and (test_mode or os.path.exists(directory)):
                    try:
                        # 查找图像文件
                        img_paths = list(Path(directory).glob(file_pattern))
                        if img_paths:
                            # 创建DataFrame
                            beam_df = pd.DataFrame([{'path': path} for path in img_paths])
                            
                            if process_images:
                                # 读取图像数组
                                beam_df['img_array'] = beam_df['path'].apply(read_tiff_to_numpy)
                                # 处理时间列
                                beam_df = process_beam_data(beam_df)
                            
                            self.current_data = beam_df
                            self.set_output_val(0, beam_df)
                        else:
                            self.current_data = None
                            self.set_output_val(0, None)
                    except Exception as e:
                        print(f"加载图像时出错: {e}")
                        self.current_data = None
                        self.set_output_val(0, None)
        
        # === 图像预处理节点 ===
        class ImagePreprocessingNode(Node):
            """图像预处理节点"""
            title = '图像预处理'
            color = '#4ECDC4'
            
            init_inputs = [
                NodeInputBP('beam_data', '光束数据'),
                NodeInputBP('background_method', '背景扣除方法', default='median'),
                NodeInputBP('kernel_size', '核大小', dtype='int', default=21),
                NodeInputBP('sigma_factor', '阈值倍数', dtype='float', default=3.0)
            ]
            
            init_outputs = [
                NodeOutputBP('processed_data', '预处理数据')
            ]
            
            def update_event(self, inp=-1):
                beam_data = self.input(0)
                if beam_data is not None and not beam_data.empty:
                    # 应用自适应背景扣除
                    processed_data = beam_data.copy()
                    
                    # 为每个图像应用背景扣除
                    denoise_results = []
                    for _, row in processed_data.iterrows():
                        denoised, background, mask = adaptive_background_subtraction(
                            row['denoise_img_array'],
                            kernel_size=int(self.input(2)),
                            sigma_factor=float(self.input(3))
                        )
                        denoise_results.append({
                            'denoised': denoised,
                            'background': background,
                            'mask': mask
                        })
                    
                    # 添加结果到DataFrame
                    processed_data['adaptive_denoised'] = [r['denoised'] for r in denoise_results]
                    processed_data['adaptive_background'] = [r['background'] for r in denoise_results]
                    processed_data['adaptive_mask'] = [r['mask'] for r in denoise_results]
                    
                    self.set_output_val(0, processed_data)
                else:
                    self.set_output_val(0, None)
        
        # === 特征提取节点 ===
        class FeatureExtractionNode(Node):
            """特征提取节点"""
            title = '特征提取'
            color = '#45B7D1'
            
            init_inputs = [
                NodeInputBP('beam_data', '光束数据'),
                NodeInputBP('extract_d4sigma', 'D4σ特征', dtype='bool', default=True),
                NodeInputBP('extract_ellipse', '椭圆特征', dtype='bool', default=True),
                NodeInputBP('extract_gaussian', '高斯特征', dtype='bool', default=False)
            ]
            
            init_outputs = [
                NodeOutputBP('features_data', '特征数据')
            ]
            
            def update_event(self, inp=-1):
                beam_data = self.input(0)
                if beam_data is not None and not beam_data.empty:
                    processed_data = beam_data.copy()
                    
                    # D4σ特征提取
                    if self.input(2):
                        processed_data = d4sigma_feature_extract(processed_data)
                    
                    # 椭圆特征提取
                    if self.input(3):
                        processed_data = shape_feature_extract(processed_data)
                    
                    # 高斯特征（可扩展）
                    if self.input(4):
                        # 这里可以添加高斯拟合相关功能
                        pass
                    
                    self.set_output_val(0, processed_data)
                else:
                    self.set_output_val(0, None)
        
        # === 光束质量分析节点 ===
        class BeamQualityNode(Node):
            """光束质量分析节点"""
            title = '光束质量分析'
            color = '#F7DC6F'
            
            init_inputs = [
                NodeInputBP('pupil_data', '光瞳数据'),
                NodeInputBP('axis_data', '光轴数据'),
                NodeInputBP('focal_length', '焦距(mm)', dtype='float', default=100.0),
                NodeInputBP('wavelength', '波长(nm)', dtype='float', default=1064.0)
            ]
            
            init_outputs = [
                NodeOutputBP('quality_results', '质量分析结果')
            ]
            
            def update_event(self, inp=-1):
                pupil_data = self.input(0)
                axis_data = self.input(1)
                
                if pupil_data is not None and axis_data is not None:
                    try:
                        # 合并光瞳和光轴数据
                        merged_data = pd.merge_asof(
                            pupil_data.sort_values('time'),
                            axis_data.sort_values('time'),
                            on='time',
                            direction='nearest',
                            suffixes=('_pupil', '_axis')
                        )
                        
                        # 计算Strehl比
                        strehl_results = []
                        for _, row in merged_data.iterrows():
                            if 'denoise_img_array_pupil' in row and 'denoise_img_array_axis' in row:
                                result = strehl_with_centering(
                                    row['denoise_img_array_pupil'],
                                    row['denoise_img_array_axis']
                                )
                                strehl_results.append(result)
                        
                        if strehl_results:
                            merged_data['strehl'] = [r['strehl'] for r in strehl_results]
                        
                        # 计算BPP和M²
                        bpp_results = []
                        for _, row in merged_data.iterrows():
                            if 'avg_sigma2_pupil' in row and 'avg_sigma2_axis' in row:
                                result = calculate_bpp_from_pupil_and_focal(
                                    pupil_diameter_mm=row['avg_sigma2_pupil'],
                                    focal_diameter_mm=row['avg_sigma2_axis'],
                                    focal_length_mm=float(self.input(2)),
                                    wavelength_nm=float(self.input(3))
                                )
                                bpp_results.append(result)
                        
                        if bpp_results:
                            for key in bpp_results[0].keys():
                                merged_data[f'bpp_{key}'] = [r[key] for r in bpp_results]
                        
                        self.set_output_val(0, merged_data)
                    except Exception as e:
                        print(f"质量分析出错: {e}")
                        self.set_output_val(0, None)
                else:
                    self.set_output_val(0, None)
        
        # === 数据可视化节点 ===
        class DataVisualizationNode(Node):
            """数据可视化节点"""
            title = '数据可视化'
            color = '#BB8FCE'
            
            init_inputs = [
                NodeInputBP('data', '分析数据'),
                NodeInputBP('plot_type', '图表类型', default='time_series'),
                NodeInputBP('features', '显示特征', default='all')
            ]
            
            init_outputs = [
                NodeOutputBP('plot_figure', '图表对象')
            ]
            
            def update_event(self, inp=-1):
                data = self.input(0)
                plot_type = self.input(1)
                
                if data is not None and not data.empty:
                    try:
                        import matplotlib.pyplot as plt
                        
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        fig.suptitle('激光光束质量分析结果')
                        
                        if plot_type == 'time_series':
                            # 时间序列图
                            numeric_cols = data.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                time_data = data['time'] if 'time' in data else range(len(data))
                                
                                axes[0, 0].plot(time_data, data[numeric_cols[0]], 'b-')
                                axes[0, 0].set_title('特征时间序列')
                                axes[0, 0].set_xlabel('时间')
                                axes[0, 0].set_ylabel(numeric_cols[0])
                                
                                # Strehl比
                                if 'strehl' in data:
                                    axes[0, 1].plot(time_data, data['strehl'], 'r-')
                                    axes[0, 1].set_title('Strehl比')
                                    axes[0, 1].set_xlabel('时间')
                                    axes[0, 1].set_ylabel('Strehl比')
                                
                                # M²值
                                if 'bpp_M2' in data:
                                    axes[1, 0].plot(time_data, data['bpp_M2'], 'g-')
                                    axes[1, 0].set_title('M²值')
                                    axes[1, 0].set_xlabel('时间')
                                    axes[1, 0].set_ylabel('M²')
                                
                                # 直方图
                                if len(numeric_cols) > 1:
                                    axes[1, 1].hist(data[numeric_cols[1]].dropna(), bins=20, alpha=0.7)
                                    axes[1, 1].set_title(f'{numeric_cols[1]}分布')
                                    axes[1, 1].set_xlabel(numeric_cols[1])
                                    axes[1, 1].set_ylabel('频次')
                        
                        plt.tight_layout()
                        self.set_output_val(0, fig)
                    except Exception as e:
                        print(f"可视化出错: {e}")
                        self.set_output_val(0, None)
                else:
                    self.set_output_val(0, None)
        
        # === 数据导出节点 ===
        class DataExportNode(Node):
            """数据导出节点"""
            title = '数据导出'
            color = '#85C1E9'
            
            init_inputs = [
                NodeInputBP('data', '分析数据'),
                NodeInputBP('export_path', '导出路径'),
                NodeInputBP('export_format', '导出格式', default='parquet')
            ]
            
            init_outputs = [
                NodeOutputBP('export_status', '导出状态')
            ]
            
            def update_event(self, inp=-1):
                data = self.input(0)
                export_path = self.input(1)
                export_format = self.input(2)
                
                if data is not None and export_path:
                    try:
                        export_path = Path(export_path)
                        export_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if export_format == 'parquet':
                            data.to_parquet(export_path, compression='zstd')
                        elif export_format == 'csv':
                            # 只导出数值列
                            numeric_data = data.select_dtypes(include=[np.number])
                            numeric_data.to_csv(export_path)
                        else:
                            raise ValueError(f"不支持的格式: {export_format}")
                        
                        self.set_output_val(0, f"数据已导出到: {export_path}")
                    except Exception as e:
                        self.set_output_val(0, f"导出失败: {e}")
                else:
                    self.set_output_val(0, "数据或路径为空")
        
        # === 节点注册函数 ===
        @classmethod
        def get_nodes(cls):
            """获取所有节点类"""
            return [
                cls.ImageInputNode,
                cls.ImagePreprocessingNode,
                cls.FeatureExtractionNode,
                cls.BeamQualityNode,
                cls.DataVisualizationNode,
                cls.DataExportNode
            ]
        
        @classmethod
        def register_nodes(cls):
            """注册所有节点到Ryven"""
            if RYVEN_AVAILABLE:
                for node_class in cls.get_nodes():
                    register_node(node_class)
                print(f"已注册 {len(cls.get_nodes())} 个激光光束分析节点")
            else:
                print("Ryven 不可用，无法注册节点")

# %%
class BeamAnalysisWorkflow:
    """激光光束分析工作流管理器"""
    
    def __init__(self):
        self.nodes = {}
        self.connections = []
        
    def create_simple_workflow(self, data_dir: str, output_dir: str = None):
        """创建一个简单的分析工作流"""
        if not RYVEN_AVAILABLE:
            print("Ryven 不可用，无法创建可视化工作流")
            return None
            
        try:
            # 创建节点实例
            workflow = {
                'image_input': BeamAnalysisNodes.ImageInputNode,
                'preprocessing': BeamAnalysisNodes.ImagePreprocessingNode,
                'feature_extraction': BeamAnalysisNodes.FeatureExtractionNode,
                'quality_analysis': BeamAnalysisNodes.BeamQualityNode,
                'visualization': BeamAnalysisNodes.DataVisualizationNode,
                'data_export': BeamAnalysisNodes.DataExportNode
            }
            
            # 设置参数
            params = {
                'image_input': {
                    'directory': data_dir,
                    'file_pattern': '*.TIFF',
                    'process_images': True
                },
                'data_export': {
                    'export_path': output_dir or 'analysis_results.parquet',
                    'export_format': 'parquet'
                }
            }
            
            return workflow, params
            
        except Exception as e:
            print(f"创建工作流失败: {e}")
            return None, None
    
    def create_advanced_workflow(self):
        """创建高级分析工作流（包含更多节点）"""
        if not RYVEN_AVAILABLE:
            print("Ryven 不可用")
            return None
            
        # 这里可以添加更多复杂的节点连接
        # 例如：多路输入、批处理、并行分析等
        pass

# %%
def setup_ryven_environment():
    """设置Ryven环境"""
    if RYVEN_AVAILABLE:
        try:
            # 注册节点
            BeamAnalysisWorkflow.register_nodes()
            
            # 打印可用节点信息
            print("\n=== 激光光束分析Ryven节点 ===")
            print("可用节点:")
            for node_class in BeamAnalysisWorkflow.get_nodes():
                print(f"  - {node_class.title}")
            
            print("\n使用方法:")
            print("1. 启动Ryven GUI")
            print("2. 从节点库中选择上述节点")
            print("3. 连接节点创建分析工作流")
            print("4. 设置参数并运行分析")
            
            return True
        except Exception as e:
            print(f"设置Ryven环境失败: {e}")
            return False
    else:
        print("Ryven库未安装，请先安装: pip install ryven")
        return False

# %%
def demo_analysis():
    """演示分析功能（不使用Ryven GUI）"""
    print("=== 激光光束分析演示 ===")
    
    # 示例数据目录（需要根据实际情况修改）
    data_dir = project_root / 'data' / 'digitaloptical4Floor'
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        print("请修改data_dir变量指向实际的图像数据目录")
        return
    
    try:
        # 创建工作流
        workflow_manager = BeamAnalysisWorkflow()
        workflow, params = workflow_manager.create_simple_workflow(str(data_dir))
        
        if workflow and params:
            print("工作流创建成功!")
            print("注意: 这是一个代码演示，实际的Ryven GUI需要手动操作")
            
    except Exception as e:
        print(f"演示分析失败: {e}")

# %%
if __name__ == "__main__":
    # 设置Ryven环境
    setup_success = setup_ryven_environment()
    
    if setup_success:
        print("\n" + "="*50)
        print("Ryven环境设置完成!")
        print("="*50)
    
    # 运行演示
    demo_analysis()
