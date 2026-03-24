#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像降噪Pipeline处理器

提供Pipeline集成的图像降噪功能
"""

from typing import Optional, Dict, Any, Tuple, List, Union
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult
from data_mining.experiment_analysis.models import ProcessorConfig
from data_mining.image.denoising import (
    denoise,
    batch_denoise,
    estimate_noise_sigma,
    evaluate_denoising,
    DenoiseMethod,
    DenoiseResult
)
from data_mining.image.common import SpotImage


class DenoiseProcessorConfig(ProcessorConfig):
    """降噪处理器配置"""
    model_config = ConfigDict(extra='forbid')
    
    # 降噪方法
    method: str = Field(default="gaussian", description="降噪方法")
    
    # 高斯滤波参数
    gaussian_kernel_size: int = Field(default=5, description="高斯核大小")
    gaussian_sigma: float = Field(default=1.5, description="高斯核标准差")
    
    # 中值滤波参数
    median_kernel_size: int = Field(default=5, description="中值滤波窗口大小")
    
    # 非局部均值参数
    nlmeans_h: float = Field(default=10, description="NLMeans滤波强度")
    nlmeans_template_size: int = Field(default=7, description="NLMeans模板窗口大小")
    nlmeans_search_size: int = Field(default=21, description="NLMeans搜索窗口大小")
    
    # BM3D参数
    bm3d_sigma_psd: float = Field(default=25.0, description="BM3D噪声标准差")
    bm3d_stage: str = Field(default="all", description="BM3D处理阶段")
    
    # 双边滤波参数
    bilateral_d: int = Field(default=9, description="双边滤波邻域直径")
    bilateral_sigma_color: float = Field(default=75, description="双边滤波颜色空间sigma")
    bilateral_sigma_space: float = Field(default=75, description="双边滤波坐标空间sigma")
    
    # 小波去噪参数
    wavelet_type: str = Field(default="db1", description="小波类型")
    wavelet_level: int = Field(default=2, description="小波分解层数")
    wavelet_mode: str = Field(default="soft", description="小波阈值模式")
    
    # 噪声估计
    auto_estimate_noise: bool = Field(default=False, description="自动估计噪声水平")
    noise_estimate_method: str = Field(default="mad", description="噪声估计方法")
    
    # 批量处理
    batch_size: Optional[int] = Field(default=None, description="批量处理大小")
    
    # 评估
    evaluate_quality: bool = Field(default=False, description="评估降噪质量")
    
    # 输出
    save_denoised_images: bool = Field(default=False, description="保存降噪后的图像")
    output_dir: Optional[str] = Field(default=None, description="输出目录")


class ImageDenoiseInput(BaseModel):
    """图像降噪输入数据模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    images: List[np.ndarray] = Field(default_factory=list, description="图像列表")
    image_paths: Optional[List[str]] = Field(default=None, description="图像路径列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ImageDenoiseOutput(BaseModel):
    """图像降噪输出数据模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    denoised_images: List[np.ndarray] = Field(default_factory=list, description="降噪后的图像")
    noise_estimates: List[float] = Field(default_factory=list, description="噪声估计值")
    evaluation_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="评估指标")
    processing_time: float = Field(default=0.0, description="处理时间")
    method_used: str = Field(default="", description="使用的降噪方法")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="处理参数")


class ImageDenoiseProcessor(BaseProcessor[ImageDenoiseInput]):
    """图像降噪处理器
    
    集成到Pipeline中的图像降噪处理器，支持多种降噪方法
    
    示例:
        >>> config = DenoiseProcessorConfig(method="bm3d", bm3d_sigma_psd=25)
        >>> processor = ImageDenoiseProcessor(config)
        >>> 
        >>> input_data = ImageDenoiseInput(images=[image1, image2])
        >>> result = processor.execute(input_data)
    """
    
    config_class = DenoiseProcessorConfig
    
    def __init__(self, config: Optional[DenoiseProcessorConfig] = None):
        super().__init__(config)
        self.config: DenoiseProcessorConfig = config or self._default_config()
    
    def validate_input(self, data: Union[ImageDenoiseInput, Dict, List[np.ndarray]], **kwargs) -> Tuple[bool, Optional[str]]:
        """验证输入数据"""
        try:
            if data is None:
                return False, "输入数据不能为空"
            
            # 处理不同类型的输入
            if isinstance(data, dict):
                if 'images' not in data:
                    return False, "输入字典缺少 'images' 键"
                images = data['images']
            elif isinstance(data, list):
                images = data
            elif isinstance(data, ImageDenoiseInput):
                images = data.images
            else:
                return False, f"不支持的输入类型: {type(data)}"
            
            if not images:
                return False, "图像列表为空"
            
            if not isinstance(images, list):
                return False, "images必须是列表"
            
            # 验证每个图像
            for i, img in enumerate(images):
                if not isinstance(img, np.ndarray):
                    return False, f"第 {i} 个图像不是numpy数组"
                if img.ndim not in [2, 3]:
                    return False, f"第 {i} 个图像维度不正确，应为2或3"
            
            return True, None
            
        except Exception as e:
            return False, f"验证输入时出错: {str(e)}"
    
    def process(self, data: Union[ImageDenoiseInput, Dict, List[np.ndarray]], **kwargs) -> ProcessingResult:
        """处理图像降噪"""
        import time
        start_time = time.time()
        
        try:
            # 标准化输入
            if isinstance(data, dict):
                images = data['images']
                original_images = data.get('original_images', images)
            elif isinstance(data, list):
                images = data
                original_images = images
            elif isinstance(data, ImageDenoiseInput):
                images = data.images
                original_images = images
            else:
                return ProcessingResult(
                    success=False,
                    message="输入数据格式错误",
                    error="不支持的输入类型"
                )
            
            # 获取降噪方法
            method = self.config.method.lower()
            
            # 准备降噪参数
            denoise_params = self._get_denoise_params(method)
            
            # 自动估计噪声（如果启用）
            noise_estimates = []
            if self.config.auto_estimate_noise:
                for img in images[:5]:  # 只估计前5张
                    sigma = estimate_noise_sigma(img, method=self.config.noise_estimate_method)
                    noise_estimates.append(sigma)
                
                avg_noise = np.mean(noise_estimates)
                if method == "bm3d":
                    denoise_params['sigma_psd'] = avg_noise
                elif method == "gaussian":
                    denoise_params['sigma'] = avg_noise / 10  # 调整参数
            
            # 执行降噪
            if len(images) == 1:
                # 单张图像
                result = denoise(images[0], method, **denoise_params)
                denoised_images = [result.image]
                noise_estimate = getattr(result, 'noise_estimate', None)
                if noise_estimate:
                    noise_estimates = [noise_estimate]
            else:
                # 批量处理
                results = batch_denoise(images, method, **denoise_params)
                denoised_images = [r.image if r else img for r, img in zip(results, images)]
                noise_estimates = [getattr(r, 'noise_estimate', None) for r in results]
            
            # 评估降噪质量
            evaluation_metrics = []
            if self.config.evaluate_quality:
                for orig, denoised in zip(original_images, denoised_images):
                    metrics = evaluate_denoising(orig, denoised, orig if len(original_images) == len(denoised_images) else None)
                    evaluation_metrics.append(metrics)
            
            # 保存图像（如果启用）
            if self.config.save_denoised_images and self.config.output_dir:
                self._save_images(denoised_images, method)
            
            # 构建输出
            processing_time = time.time() - start_time
            output = ImageDenoiseOutput(
                denoised_images=denoised_images,
                noise_estimates=[float(n) for n in noise_estimates if n is not None],
                evaluation_metrics=evaluation_metrics,
                processing_time=processing_time,
                method_used=method,
                parameters=denoise_params
            )
            
            return ProcessingResult(
                success=True,
                message=f"成功处理 {len(images)} 张图像，使用 {method} 方法",
                data=output,
                metadata={
                    'image_count': len(images),
                    'method': method,
                    'processing_time': processing_time,
                    'noise_estimates': noise_estimates,
                    'has_evaluation': len(evaluation_metrics) > 0
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"降噪处理失败: {str(e)}",
                error=str(e)
            )
    
    def _get_denoise_params(self, method: str) -> Dict[str, Any]:
        """获取降噪参数"""
        params_map = {
            'gaussian': {
                'kernel_size': self.config.gaussian_kernel_size,
                'sigma': self.config.gaussian_sigma
            },
            'median': {
                'kernel_size': self.config.median_kernel_size
            },
            'nlmeans': {
                'h': self.config.nlmeans_h,
                'template_window_size': self.config.nlmeans_template_size,
                'search_window_size': self.config.nlmeans_search_size
            },
            'bm3d': {
                'sigma_psd': self.config.bm3d_sigma_psd,
                'stage_arg': self.config.bm3d_stage
            },
            'bilateral': {
                'd': self.config.bilateral_d,
                'sigma_color': self.config.bilateral_sigma_color,
                'sigma_space': self.config.bilateral_sigma_space
            },
            'wavelet': {
                'wavelet': self.config.wavelet_type,
                'level': self.config.wavelet_level,
                'mode': self.config.wavelet_mode
            }
        }
        
        return params_map.get(method, {})
    
    def _save_images(self, images: List[np.ndarray], method: str):
        """保存降噪后的图像"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img in enumerate(images):
            filename = f"denoised_{method}_{timestamp}_{i:04d}.png"
            filepath = output_dir / filename
            
            import cv2
            if img.ndim == 3:
                cv2.imwrite(str(filepath), img)
            else:
                cv2.imwrite(str(filepath), img)


class SpotImageDenoiseProcessor(ImageDenoiseProcessor):
    """SpotImage专用降噪处理器
    
    专门处理SpotImage对象的降噪处理器
    """
    
    def validate_input(self, data: Union[List[SpotImage], Dict], **kwargs) -> Tuple[bool, Optional[str]]:
        """验证SpotImage输入"""
        if data is None:
            return False, "输入数据不能为空"
        
        if isinstance(data, dict):
            images = data.get('images', [])
        elif isinstance(data, list):
            images = data
        else:
            return False, f"不支持的输入类型: {type(data)}"
        
        if not images:
            return False, "图像列表为空"
        
        for i, img in enumerate(images):
            if not isinstance(img, SpotImage):
                return False, f"第 {i} 个对象不是SpotImage类型"
        
        return True, None
    
    def process(self, data: Union[List[SpotImage], Dict], **kwargs) -> ProcessingResult:
        """处理SpotImage降噪"""
        try:
            if isinstance(data, dict):
                spot_images = data['images']
            else:
                spot_images = data
            
            method = self.config.method
            params = self._get_denoise_params(method)
            
            # 处理每个SpotImage
            denoised_spots = []
            for spot in spot_images:
                # 使用denoise_spot_image函数
                from data_mining.image.denoising import denoise_spot_image
                result = denoise_spot_image(spot, method, **params)
                
                # 创建新的SpotImage对象
                denoised_spot = SpotImage(
                    result.image,
                    f"{spot.meta_info.get('name', 'unknown')}_denoised",
                    spot.meta_info.copy()
                )
                denoised_spot.features = spot.features.copy()
                denoised_spots.append(denoised_spot)
            
            return ProcessingResult(
                success=True,
                message=f"成功处理 {len(spot_images)} 个SpotImage",
                data={
                    'denoised_spots': denoised_spots,
                    'original_count': len(spot_images)
                },
                metadata={
                    'method': method,
                    'count': len(spot_images)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"SpotImage降噪失败: {str(e)}",
                error=str(e)
            )


# 工厂函数
def create_denoise_processor(
    method: str = "gaussian",
    **kwargs
) -> ImageDenoiseProcessor:
    """创建降噪处理器的工厂函数
    
    参数:
        method: 降噪方法
        **kwargs: 其他配置参数
    
    返回:
        ImageDenoiseProcessor: 降噪处理器实例
    
    示例:
        >>> processor = create_denoise_processor("bm3d", bm3d_sigma_psd=25)
        >>> result = processor.execute({'images': [image]})
    """
    config = DenoiseProcessorConfig(method=method, **kwargs)
    return ImageDenoiseProcessor(config)


# 快速降噪函数
def quick_denoise(
    images: Union[np.ndarray, List[np.ndarray]],
    method: str = "gaussian",
    **kwargs
) -> Union[np.ndarray, List[np.ndarray]]:
    """快速降噪函数
    
    无需创建处理器实例的快速降噪函数
    
    参数:
        images: 图像或图像列表
        method: 降噪方法
        **kwargs: 降噪参数
    
    返回:
        降噪后的图像或图像列表
    
    示例:
        >>> denoised = quick_denoise(image, "median", kernel_size=5)
        >>> denoised_list = quick_denoise([img1, img2], "bm3d", sigma_psd=25)
    """
    processor = create_denoise_processor(method, **kwargs)
    
    if isinstance(images, np.ndarray):
        images = [images]
        return_single = True
    else:
        return_single = False
    
    result = processor.execute({'images': images})
    
    if result.success and result.data:
        output = result.data
        if isinstance(output, ImageDenoiseOutput):
            denoised = output.denoised_images
            return denoised[0] if return_single else denoised
    
    return images[0] if return_single else images
