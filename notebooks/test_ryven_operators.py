"""
测试Ryven算子是否可以正确导入
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_operator_imports():
    """测试算子导入"""
    print("测试Ryven算子导入...")
    
    try:
        from ryven_operators import DataAnalysisNodes
        print("✓ 成功导入DataAnalysisNodes")
        
        # 获取所有节点类
        nodes = DataAnalysisNodes.get_nodes()
        print(f"✓ 获取到 {len(nodes)} 个节点类:")
        
        for node_class in nodes:
            print(f"  - {node_class.title}")
        
        print("\n✓ 所有算子导入测试通过!")
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_functions():
    """测试单个函数功能"""
    print("\n测试单个函数功能...")
    
    try:
        import numpy as np
        # 测试D4σ特征提取功能
        from ryven_operators import DataAnalysisNodes
        
        # 创建一个简单的测试图像
        test_img = np.zeros((100, 100))
        # 添加一个高斯光斑
        y, x = np.ogrid[:100, :100]
        center_y, center_x = 50, 50
        sigma = 10
        test_img = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # 测试D4σ特征提取
        node_instance = DataAnalysisNodes.D4SigmaFeatureNode(params={})
        features = node_instance.calculate_d4sigma_features(test_img, pixel_size_um=1.0)
        
        print(f"✓ D4σ特征提取测试成功:")
        print(f"  - d4sigma_x: {features['d4sigma_x']:.2f} μm")
        print(f"  - d4sigma_y: {features['d4sigma_y']:.2f} μm")
        print(f"  - center_x: {features['center_x']:.2f}")
        print(f"  - center_y: {features['center_y']:.2f}")
        print(f"  - total_power: {features['total_power']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 函数功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Ryven算子测试")
    print("="*50)
    
    success1 = test_operator_imports()
    success2 = test_individual_functions()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败!")
        
    print("\n下一步:")
    print("1. 安装ryven: pip install ryven")
    print("2. 运行GUI: ryven --project=beam_analysis")
    print("3. 在GUI中使用这些算子创建工作流")