#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
降噪Pipeline集成示例

展示如何将降噪步骤集成到数据处理Pipeline中
"""

import asyncio
import numpy as np
from pathlib import Path

# 导入Pipeline相关类
from data_mining.experiment_analysis.pipeline import (
    PipelineManager,
    PipelineExecutor,
    PipelineStage,
    PipelineStageConfig,
    PipelineExecutionConfig
)

# 导入降噪处理器
from data_mining.experiment_analysis.denoise_processor import (
    ImageDenoiseProcessor,
    DenoiseProcessorConfig
)


def create_sample_images(count: int = 3, size: int = 128) -> list:
    """创建示例图像（带噪声的高斯光斑）"""
    images = []
    for i in range(count):
        # 创建干净的高斯光斑
        x = np.arange(size) - (size // 2)
        y = np.arange(size) - (size // 2)
        xx, yy = np.meshgrid(x, y)
        sigma = 10.0 + i * 2  # 不同的光斑大小
        clean = 255 * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # 添加噪声
        noise = np.random.normal(0, 25, clean.shape)
        noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
        
        images.append(noisy)
    
    return images


def example_1_basic_denoise_pipeline():
    """示例1：基本降噪Pipeline"""
    print("=" * 60)
    print("示例1：基本降噪Pipeline")
    print("=" * 60)
    
    # 创建示例图像
    images = create_sample_images(count=3)
    print(f"创建了 {len(images)} 张示例图像")
    
    # 配置Pipeline
    config = PipelineExecutionConfig(
        pipeline_name="denoise_pipeline",
        pipeline_version="1.0.0",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.DENOISE,
                processor_type=ImageDenoiseProcessor,
                enabled=True,
                dependencies=[]
            )
        ]
    )
    
    # 创建Pipeline管理器
    manager = PipelineManager(config)
    
    # 初始化处理器
    manager._initialize_stages()
    
    # 配置降噪处理器（使用高斯滤波）
    processor = manager._processors[PipelineStage.DENOISE.value]
    processor.config = DenoiseProcessorConfig(
        method="gaussian",
        gaussian_kernel_size=5,
        gaussian_sigma=1.5,
        evaluate_quality=True
    )
    
    # 执行Pipeline
    executor = PipelineExecutor(manager)
    result = executor.execute({
        'images': images,
        'original_images': images.copy()
    })
    
    # 显示结果
    if result['status'] == 'completed':
        print("✓ Pipeline执行成功！")
        denoise_result = result['data']['results'][0]['result']
        print(f"  处理方法: {denoise_result.method_used}")
        print(f"  处理时间: {denoise_result.processing_time:.3f}秒")
        print(f"  降噪图像数量: {len(denoise_result.denoised_images)}")
        
        if denoise_result.evaluation_metrics:
            print("  质量评估指标:")
            for i, metrics in enumerate(denoise_result.evaluation_metrics):
                print(f"    图像 {i+1}: SNR提升 = {metrics.get('snr_improvement', 'N/A'):.2f} dB")
    else:
        print(f"✗ Pipeline执行失败: {result.get('error', '未知错误')}")
    
    print()


def example_2_bm3d_denoise_pipeline():
    """示例2：使用BM3D的降噪Pipeline"""
    print("=" * 60)
    print("示例2：BM3D降噪Pipeline")
    print("=" * 60)
    
    images = create_sample_images(count=2)
    print(f"创建了 {len(images)} 张示例图像")
    
    # 配置Pipeline
    config = PipelineExecutionConfig(
        pipeline_name="bm3d_denoise_pipeline",
        pipeline_version="1.0.0",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.DENOISE,
                processor_type=ImageDenoiseProcessor,
                enabled=True,
                dependencies=[]
            )
        ]
    )
    
    manager = PipelineManager(config)
    manager._initialize_stages()
    
    # 配置BM3D降噪
    try:
        import bm3d
        processor = manager._processors[PipelineStage.DENOISE.value]
        processor.config = DenoiseProcessorConfig(
            method="bm3d",
            bm3d_sigma_psd=25.0,
            evaluate_quality=True
        )
        
        executor = PipelineExecutor(manager)
        result = executor.execute({
            'images': images,
            'original_images': images.copy()
        })
        
        if result['status'] == 'completed':
            print("✓ BM3D Pipeline执行成功！")
            denoise_result = result['data']['results'][0]['result']
            print(f"  处理方法: {denoise_result.method_used}")
            print(f"  处理时间: {denoise_result.processing_time:.3f}秒")
        else:
            print(f"✗ 执行失败: {result.get('error', '未知错误')}")
    except ImportError:
        print("⚠ BM3D库未安装，跳过此示例")
        print("  安装命令: pip install bm3d")
    
    print()


def example_3_multi_stage_pipeline():
    """示例3：多阶段Pipeline（过滤+降噪）"""
    print("=" * 60)
    print("示例3：多阶段Pipeline（过滤+降噪）")
    print("=" * 60)
    
    images = create_sample_images(count=3)
    print(f"创建了 {len(images)} 张示例图像")
    
    # 创建自定义的过滤处理器
    from data_mining.experiment_analysis.base_processor import (
        BaseProcessor, ProcessingResult, ProcessorConfig
    )
    
    class ImageFilterProcessor(BaseProcessor):
        """示例图像过滤处理器"""
        config_class = ProcessorConfig
        
        def validate_input(self, data, **kwargs):
            if data is None or 'images' not in data:
                return False, "缺少图像数据"
            return True, None
        
        def process(self, data, **kwargs):
            images = data['images']
            # 简单的过滤：只保留大于平均亮度的像素
            filtered = []
            for img in images:
                mean_val = np.mean(img)
                img_filtered = img.copy()
                img_filtered[img_filtered < mean_val * 0.5] = 0
                filtered.append(img_filtered)
            
            return ProcessingResult(
                success=True,
                message="图像过滤完成",
                data={'images': filtered}
            )
    
    # 配置Pipeline
    config = PipelineExecutionConfig(
        pipeline_name="multi_stage_pipeline",
        pipeline_version="1.0.0",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.FILTER,
                processor_type=ImageFilterProcessor,
                enabled=True,
                dependencies=[]
            ),
            PipelineStageConfig(
                stage_name=PipelineStage.DENOISE,
                processor_type=ImageDenoiseProcessor,
                enabled=True,
                dependencies=[PipelineStage.FILTER.value]
            )
        ]
    )
    
    manager = PipelineManager(config)
    manager._initialize_stages()
    
    # 配置降噪处理器
    manager._processors[PipelineStage.DENOISE.value].config = DenoiseProcessorConfig(
        method="nlmeans",
        nlmeans_h=10
    )
    
    executor = PipelineExecutor(manager)
    result = executor.execute({
        'images': images,
        'original_images': images.copy()
    })
    
    if result['status'] == 'completed':
        print("✓ 多阶段Pipeline执行成功！")
        print(f"  完成阶段: {result['completed_stages']}")
        print(f"  处理结果数量: {len(result['data']['results'])}")
    else:
        print(f"✗ 执行失败: {result.get('error', '未知错误')}")
    
    print()


def example_4_auto_noise_estimation():
    """示例4：自动噪声估计"""
    print("=" * 60)
    print("示例4：自动噪声估计")
    print("=" * 60)
    
    images = create_sample_images(count=2)
    print(f"创建了 {len(images)} 张示例图像（带噪声）")
    
    config = PipelineExecutionConfig(
        pipeline_name="auto_estimate_pipeline",
        pipeline_version="1.0.0",
        stages=[
            PipelineStageConfig(
                stage_name=PipelineStage.DENOISE,
                processor_type=ImageDenoiseProcessor,
                enabled=True,
                dependencies=[]
            )
        ]
    )
    
    manager = PipelineManager(config)
    manager._initialize_stages()
    
    # 启用自动噪声估计
    processor = manager._processors[PipelineStage.DENOISE.value]
    processor.config = DenoiseProcessorConfig(
        method="gaussian",
        auto_estimate_noise=True,
        noise_estimate_method="mad",
        evaluate_quality=True
    )
    
    executor = PipelineExecutor(manager)
    result = executor.execute({
        'images': images,
        'original_images': images.copy()
    })
    
    if result['status'] == 'completed':
        print("✓ 自动噪声估计Pipeline执行成功！")
        denoise_result = result['data']['results'][0]['result']
        print(f"  估计的噪声水平: {denoise_result.noise_estimates}")
        print(f"  实际使用的参数: {denoise_result.parameters}")
    else:
        print(f"✗ 执行失败: {result.get('error', '未知错误')}")
    
    print()


def example_5_quick_denoise():
    """示例5：快速降噪函数"""
    print("=" * 60)
    print("示例5：快速降噪函数")
    print("=" * 60)
    
    from data_mining.experiment_analysis.denoise_processor import quick_denoise
    
    images = create_sample_images(count=3)
    print(f"创建了 {len(images)} 张示例图像")
    
    # 使用快速降噪
    print("执行快速中值滤波...")
    denoised = quick_denoise(images, method="median", kernel_size=5)
    
    print(f"✓ 快速降噪完成！")
    print(f"  输入图像数量: {len(images)}")
    print(f"  输出图像数量: {len(denoised)}")
    print(f"  图像尺寸: {denoised[0].shape}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("降噪Pipeline集成示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_1_basic_denoise_pipeline()
    example_2_bm3d_denoise_pipeline()
    example_3_multi_stage_pipeline()
    example_4_auto_noise_estimation()
    example_5_quick_denoise()
    
    print("=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)
