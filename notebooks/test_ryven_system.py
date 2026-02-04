# %%
"""
Ryven 激光光束分析系统测试和示例

这个脚本提供了对Ryven节点的单元测试和使用示例，
帮助验证系统功能并演示如何使用。
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 测试数据生成函数
def generate_test_images(num_images=5, image_size=(256, 256)):
    """生成测试用的模拟激光光斑图像"""
    test_images = []
    
    for i in range(num_images):
        # 创建模拟的激光光斑（高斯分布）
        x = np.linspace(-3, 3, image_size[1])
        y = np.linspace(-3, 3, image_size[0])
        X, Y = np.meshgrid(x, y)
        
        # 添加一些随机性
        center_x = 2 * np.random.randn() + image_size[1] // 2
        center_y = 2 * np.random.randn() + image_size[0] // 2
        sigma = 0.5 + 0.3 * np.random.randn()
        intensity = 1000 + 500 * np.random.randn()
        background = 50 + 20 * np.random.randn()
        
        # 生成高斯光斑
        gaussian = intensity * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        
        # 添加噪声
        noise = np.random.normal(0, 10, image_size)
        
        # 组合图像
        image = gaussian + background + noise
        image = np.maximum(image, 0)  # 确保非负
        
        test_images.append(image.astype(np.uint16))
    
    return test_images

def create_test_dataframe(test_images):
    """创建测试用的DataFrame"""
    data = []
    for i, img in enumerate(test_images):
        data.append({
            'index': i,
            'img_array': img,
            'time': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i),
            'path': Path(f'test_image_{i:03d}.TIFF')
        })
    
    df = pd.DataFrame(data)
    
    # 应用现有的处理函数
    try:
        from notebooks.data import process_beam_data, d4sigma_feature_extract, shape_feature_extract
        df = process_beam_data(df)
        df = d4sigma_feature_extract(df)
        df = shape_feature_extract(df)
        print("✓ 成功应用现有的图像处理函数")
    except Exception as e:
        print(f"⚠ 处理函数应用失败: {e}")
    
    return df

# 节点功能测试
def test_image_input_functionality():
    """测试图像输入功能"""
    print("\n=== 测试图像输入功能 ===")
    
    # 生成测试数据
    test_images = generate_test_images(3)
    test_df = create_test_dataframe(test_images)
    
    print(f"✓ 生成了 {len(test_images)} 个测试图像")
    print(f"✓ DataFrame形状: {test_df.shape}")
    print(f"✓ 包含列: {list(test_df.columns)}")
    
    # 验证图像数据
    if 'denoise_img_array' in test_df.columns:
        sample_img = test_df['denoise_img_array'].iloc[0]
        print(f"✓ 样本图像形状: {sample_img.shape}")
        print(f"✓ 样本图像范围: [{sample_img.min()}, {sample_img.max()}]")
    
    return test_df

def test_preprocessing_functionality(test_df):
    """测试预处理功能"""
    print("\n=== 测试预处理功能 ===")
    
    try:
        from notebooks.data import adaptive_background_subtraction
        
        # 测试自适应背景扣除
        sample_img = test_df['denoise_img_array'].iloc[0]
        denoised, background, mask = adaptive_background_subtraction(
            sample_img, kernel_size=21, sigma_factor=3.0
        )
        
        print(f"✓ 预处理成功")
        print(f"  - 原始图像范围: [{sample_img.min()}, {sample_img.max()}]")
        print(f"  - 去噪图像范围: [{denoised.min()}, {denoised.max()}]")
        print(f"  - 有效区域像素数: {mask.sum()}")
        
        return True
    except Exception as e:
        print(f"✗ 预处理测试失败: {e}")
        return False

def test_feature_extraction_functionality(test_df):
    """测试特征提取功能"""
    print("\n=== 测试特征提取功能 ===")
    
    try:
        # 检查D4σ特征
        d4sigma_columns = [col for col in test_df.columns if 'center' in col or 'D_' in col]
        print(f"✓ D4σ特征列: {d4sigma_columns}")
        
        if d4sigma_columns:
            sample_features = test_df[d4sigma_columns].iloc[0]
            print(f"  - 中心位置: ({sample_features.get('center_x', 'N/A'):.2f}, {sample_features.get('center_y', 'N/A'):.2f})")
            print(f"  - D4σ直径: ({sample_features.get('D_x', 'N/A'):.2f}, {sample_features.get('D_y', 'N/A'):.2f})")
        
        # 检查椭圆特征
        ellipse_columns = [col for col in test_df.columns if 'ellipse' in col or 'uniformity' in col]
        print(f"✓ 椭圆特征列: {ellipse_columns}")
        
        if ellipse_columns:
            sample_ellipse = test_df[ellipse_columns].iloc[0]
            print(f"  - 椭圆度: {sample_ellipse.get('ellipticity', 'N/A'):.3f}")
            print(f"  - 均匀度: {sample_ellipse.get('uniformity', 'N/A'):.3f}")
        
        return True
    except Exception as e:
        print(f"✗ 特征提取测试失败: {e}")
        return False

def test_quality_analysis_functionality(test_df):
    """测试质量分析功能"""
    print("\n=== 测试质量分析功能 ===")
    
    try:
        from notebooks.data import strehl_with_centering, calculate_bpp_from_pupil_and_focal
        
        # 测试Strehl比计算
        if len(test_df) >= 2:
            img1 = test_df['denoise_img_array'].iloc[0]
            img2 = test_df['denoise_img_array'].iloc[1]
            
            strehl_result = strehl_with_centering(img1, img2)
            print(f"✓ Strehl比计算成功: {strehl_result['strehl']:.4f}")
        
        # 测试BPP计算
        if 'avg_sigma2' in test_df.columns:
            pupil_diameter = test_df['avg_sigma2'].iloc[0]
            focal_diameter = pupil_diameter * 0.1  # 模拟焦斑直径
            
            bpp_result = calculate_bpp_from_pupil_and_focal(
                pupil_diameter_mm=pupil_diameter,
                focal_diameter_mm=focal_diameter,
                focal_length_mm=100.0,
                wavelength_nm=1064.0
            )
            
            print(f"✓ BPP计算成功:")
            print(f"  - BPP: {bpp_result['BPP_mm_mrad']:.4f} mm·mrad")
            print(f"  - M²: {bpp_result['M2']:.4f}")
            print(f"  - 发散角: {bpp_result['theta_mrad']:.4f} mrad")
        
        return True
    except Exception as e:
        print(f"✗ 质量分析测试失败: {e}")
        return False

def test_visualization_functionality(test_df):
    """测试可视化功能"""
    print("\n=== 测试可视化功能 ===")
    
    try:
        # 创建测试图表
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('激光光束分析测试结果', fontsize=16)
        
        # 图像示例
        if 'denoise_img_array' in test_df.columns:
            sample_img = test_df['denoise_img_array'].iloc[0]
            axes[0, 0].imshow(sample_img, cmap='hot')
            axes[0, 0].set_title('示例光斑图像')
            axes[0, 0].axis('off')
        
        # 特征时间序列
        numeric_cols = test_df.select_dtypes(include=[np.number]).columns[:3]
        if len(numeric_cols) > 0:
            time_index = range(len(test_df))
            for i, col in enumerate(numeric_cols[:3]):
                if i < 2:
                    axes[0, 1].plot(time_index, test_df[col], label=col, marker='o')
            axes[0, 1].set_title('特征时间序列')
            axes[0, 1].set_xlabel('图像序号')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 直方图
        if len(numeric_cols) > 0:
            axes[1, 0].hist(test_df[numeric_cols[0]].dropna(), bins=10, alpha=0.7, color='skyblue')
            axes[1, 0].set_title(f'{numeric_cols[0]}分布')
            axes[1, 0].set_xlabel(numeric_cols[0])
            axes[1, 0].set_ylabel('频次')
        
        # 散点图
        if len(numeric_cols) >= 2:
            axes[1, 1].scatter(test_df[numeric_cols[0]], test_df[numeric_cols[1]], alpha=0.6)
            axes[1, 1].set_xlabel(numeric_cols[0])
            axes[1, 1].set_ylabel(numeric_cols[1])
            axes[1, 1].set_title('特征相关性')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_analysis_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ 可视化测试成功")
        print("  - 图表已保存为: test_analysis_results.png")
        
        return True
    except Exception as e:
        print(f"✗ 可视化测试失败: {e}")
        return False

def test_ryven_nodes():
    """测试Ryven节点定义"""
    print("\n=== 测试Ryven节点定义 ===")
    
    try:
        from notebooks.ryven_beam_analysis import BeamAnalysisNodes
        
        # 检查节点类是否存在
        nodes = BeamAnalysisNodes.get_nodes()
        print(f"✓ 成功获取 {len(nodes)} 个节点类:")
        
        for node_class in nodes:
            print(f"  - {node_class.title} (颜色: {node_class.color})")
        
        # 尝试创建节点实例
        if nodes:
            test_node = nodes[0]({})  # 空的参数字典
            print(f"✓ 成功创建节点实例: {test_node.title}")
        
        return True
    except Exception as e:
        print(f"✗ Ryven节点测试失败: {e}")
        return False

# 综合测试函数
def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("Ryven 激光光束分析系统 - 综合测试")
    print("=" * 60)
    
    test_results = []
    
    # 1. 测试图像输入功能
    test_df = test_image_input_functionality()
    test_results.append(("图像输入", test_df is not None))
    
    # 2. 测试预处理功能
    preprocessing_ok = test_preprocessing_functionality(test_df)
    test_results.append(("图像预处理", preprocessing_ok))
    
    # 3. 测试特征提取功能
    features_ok = test_feature_extraction_functionality(test_df)
    test_results.append(("特征提取", features_ok))
    
    # 4. 测试质量分析功能
    quality_ok = test_quality_analysis_functionality(test_df)
    test_results.append(("质量分析", quality_ok))
    
    # 5. 测试可视化功能
    viz_ok = test_visualization_functionality(test_df)
    test_results.append(("数据可视化", viz_ok))
    
    # 6. 测试Ryven节点
    nodes_ok = test_ryven_nodes()
    test_results.append(("Ryven节点", nodes_ok))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:12} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(test_results)} 项测试通过")
    
    if passed == len(test_results):
        print("🎉 所有测试通过！系统准备就绪。")
    else:
        print("⚠ 部分测试失败，请检查相关功能。")
    
    return test_results

# 使用示例
def create_usage_example():
    """创建使用示例"""
    print("\n" + "=" * 60)
    print("Ryven 使用示例")
    print("=" * 60)
    
    print("""
1. 启动Ryven环境:
   from notebooks.ryven_beam_analysis import setup_ryven_environment
   setup_ryven_environment()

2. 创建工作流:
   from notebooks.ryven_beam_analysis import BeamAnalysisWorkflow
   workflow_manager = BeamAnalysisWorkflow()
   workflow, params = workflow_manager.create_simple_workflow(
       data_dir="/path/to/your/images",
       output_dir="/path/to/output"
   )

3. 手动运行分析:
   # 加载图像
   from notebooks.data import process_beam_data
   beam_data = process_beam_data(your_dataframe)
   
   # 特征提取
   from notebooks.data import d4sigma_feature_extract
   features = d4sigma_feature_extract(beam_data)
   
   # 可视化
   import matplotlib.pyplot as plt
   plt.plot(features['time'], features['center_x'])
   plt.show()

4. 配置Ryven:
   # 编辑 ryven_config.yaml 自定义设置
   # 重启Ryven以应用新配置
""")

# 主函数
if __name__ == "__main__":
    # 运行综合测试
    test_results = run_comprehensive_test()
    
    # 显示使用示例
    create_usage_example()
    
    print(f"\n测试完成时间: {pd.Timestamp.now()}")
    print("如需详细使用说明，请参考 doc/ryven_beam_analysis_guide.md")