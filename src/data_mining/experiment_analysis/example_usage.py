#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用示例：实验数据分析 Pipeline
演示如何使用 ExperimentPipelineManager 进行数据分析
"""

from data_mining.experiment_analysis import (
    ExperimentPipelineManager,
    PipelineManagerConfig,
    ExperimentFilter,
    DataCleaningConfig
)

def example_1_standard_pipeline():
    """示例1：使用标准分析 Pipeline"""
    print("=" * 60)
    print("示例1：标准分析 Pipeline")
    print("=" * 60)
    
    # 1. 创建管理器
    manager = ExperimentPipelineManager()
    
    # 2. 创建标准分析 Pipeline
    pipeline = manager.create_analysis_pipeline("standard_analysis")
    
    # 3. 准备输入数据
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
    
    # 4. 执行 Pipeline
    try:
        result = manager.execute_pipeline("standard_analysis", input_data)
        
        # 5. 处理结果
        if result['status'] == 'completed':
            print("✓ Pipeline 执行成功")
            print(f"清洗后数据: {len(result.get('cleaned_data', []))} 条")
            print(f"质量报告: {result.get('quality_report', {})}")
        else:
            print("✗ Pipeline 执行失败")
            
    except Exception as e:
        print(f"✗ 错误: {str(e)}")

def example_2_custom_pipeline():
    """示例2：自定义 Pipeline"""
    print("\n" + "=" * 60)
    print("示例2：自定义 Pipeline")
    print("=" * 60)
    
    from data_mining.experiment_analysis import (
        DataLoader,
        QualityAnalyzer,
        DataAggregator
    )
    
    # 创建管理器
    manager = ExperimentPipelineManager()
    
    # 注册自定义处理器
    manager.register_processor('DataLoader', DataLoader)
    manager.register_processor('QualityAnalyzer', QualityAnalyzer)
    manager.register_processor('DataAggregator', DataAggregator)
    
    # 创建自定义 Pipeline
    from data_mining.experiment_analysis.pipeline import (
        PipelineExecutionConfig,
        PipelineStageConfig,
        PipelineStage
    )
    
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
    
    # 执行自定义 Pipeline
    input_data = {
        'config': {
            'table_name': 'experiment'
        }
    }
    
    try:
        result = manager.execute_pipeline("custom_pipeline", input_data)
        print("✓ 自定义 Pipeline 执行成功")
    except Exception as e:
        print(f"✗ 错误: {str(e)}")

def example_3_filter_and_clean():
    """示例3：单独使用过滤器"""
    print("\n" + "=" * 60)
    print("示例3：单独使用过滤器")
    print("=" * 60)
    
    from data_mining.experiment_analysis import ExperimentFilter
    
    # 创建过滤器
    filter_processor = ExperimentFilter()
    
    # 准备测试数据
    test_data = pd.DataFrame({
        'experiment_id': [1, 2, 3, 4, 5],
        'experiment_type_id': [2, 2, 2, 2, 2],
        'experiment_polarization_state': ['线偏振', '线偏振', None, '圆偏振', None],
        'experiment_radiation_mode': ['单模', '单模', '多模', '单模', None],
        'experiment_radiation_power': [10.5, 11.2, 9.8, 12.0, None],
        '靶材': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金'],
        '距离': [1.0, 1.0, 1.5, 1.0, 1.5]
    })
    
    # 应用过滤器
    config = ExperimentFilter(
        experiment_type_id=2,
        target_material='不锈钢'
    )
    
    result = filter_processor.execute({'data': test_data, 'config': config})
    
    if result.success:
        print(f"✓ 过滤成功: {result.data['filtered_count']} 条数据")
        print(f"移除: {result.data['removed_count']} 条数据")
    else:
        print(f"✗ 错误: {result.error}")

def example_4_quality_analysis():
    """示例4：质量分析"""
    print("\n" + "=" * 60)
    print("示例4：质量分析")
    print("=" * 60)
    
    from data_mining.experiment_analysis import QualityAnalyzer, DataQualityReport
    
    # 创建质量分析器
    analyzer = QualityAnalyzer()
    
    # 准备测试数据
    test_data = pd.DataFrame({
        'experiment_id': [1, 2, 3, 4, 5],
        'experiment_type_id': [2, 2, 2, 2, 2],
        'experiment_polarization_state': ['线偏振', '线偏振', None, '圆偏振', None],
        'experiment_radiation_mode': ['单模', '单模', '多模', '单模', None],
        'experiment_radiation_power': [10.5, 11.2, 9.8, 12.0, None],
        '靶材': ['不锈钢', '不锈钢', '铝合金', '不锈钢', '铝合金']
    })
    
    # 执行质量分析
    config = QualityAnalyzer()
    result = analyzer.execute({'data': test_data})
    
    if result.success:
        quality_report = result.data
        print(f"✓ 质量分析完成")
        print(f"总记录数: {quality_report.total_count}")
        print(f"完整记录: {quality_report.complete_records}")
        print(f"缺失率: {quality_report.missing_rate:.2%}")
        print(f"激光参数缺失: {quality_report.laser_params_missing}")
        print(f"靶材参数缺失: {quality_report.target_params_missing}")
    else:
        print(f"✗ 错误: {result.error}")

if __name__ == "__main__":
    import pandas as pd
    
    # 运行示例
    example_1_standard_pipeline()
    example_2_custom_pipeline()
    example_3_filter_and_clean()
    example_4_quality_analysis()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成")
    print("=" * 60)
