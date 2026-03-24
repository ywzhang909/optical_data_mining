#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像分割Pipeline处理器

提供Pipeline集成的图像分割功能
"""

from typing import Optional, Dict, Any, Tuple, List
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import cv2

from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult
from data_mining.experiment_analysis.models import ProcessorConfig
from data_mining.image.common import SpotImage


class SegmentProcessorConfig(ProcessorConfig):
    """分割处理器配置"""
    model_config = ConfigDict(extra='forbid')
    
    method: str = Field(default="otsu", description="分割方法: otsu, adaptive, watershed, grabcut, threshold")
    threshold_value: Optional[float] = Field(None, description="固定阈值(0-255)")
    block_size: int = Field(default=11, description="自适应分割块大小")
    c_constant: float = Field(default=2.0, description="自适应分割C常数")
    min_region_size: int = Field(default=50, description="最小区域大小")
    morph_iterations: int = Field(default=2, description="形态学迭代次数")
    kernel_size: int = Field(default=5, description="形态学核大小")


class SegmentInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    image: Any = Field(..., description="输入图像")
    mask: Optional[Any] = Field(None, description="可选掩码")


class SegmentOutput(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    segmented_image: Any = Field(..., description="分割后的图像")
    mask: Any = Field(..., description="分割掩码")
    regions: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的区域")
    method_used: str = Field(..., description="使用的分割方法")
    num_regions: int = Field(..., description="区域数量")


class ImageSegmentProcessor(BaseProcessor[SegmentOutput]):
    config_class = SegmentProcessorConfig
    
    def __init__(self, config: Optional[SegmentProcessorConfig] = None):
        super().__init__(config)
        self._execution_times: List[float] = []
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, str]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "图像数据是必需的"
        return True, ""
    
    def process(self, config: SegmentProcessorConfig, **kwargs) -> ProcessingResult:
        try:
            image = kwargs.get('data')
            if image is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入图像",
                    error="Image is required"
                )
            
            if isinstance(image, SpotImage):
                image_data = image.image
            else:
                image_data = image
            
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data.copy()
            
            if config.method == "otsu":
                mask, regions = self._otsu_segment(gray, config)
            elif config.method == "adaptive":
                mask, regions = self._adaptive_segment(gray, config)
            elif config.method == "threshold":
                mask, regions = self._threshold_segment(gray, config)
            elif config.method == "watershed":
                mask, regions = self._watershed_segment(gray, config)
            else:
                mask, regions = self._otsu_segment(gray, config)
            
            segmented = cv2.bitwise_and(image_data, image_data, mask=mask)
            
            result = SegmentOutput(
                segmented_image=segmented,
                mask=mask,
                regions=regions,
                method_used=config.method,
                num_regions=len(regions)
            )
            
            return ProcessingResult(
                success=True,
                message=f"分割完成，检测到 {len(regions)} 个区域",
                data=result.model_dump(),
                metadata={
                    "method": config.method,
                    "num_regions": len(regions),
                    "region_areas": [r.get("area", 0) for r in regions]
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"分割失败: {str(e)}",
                error=str(e)
            )
    
    def _otsu_segment(self, gray: np.ndarray, config: SegmentProcessorConfig) -> Tuple[np.ndarray, List[Dict]]:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = self._post_process_mask(mask, config)
        return mask, self._extract_regions(mask)
    
    def _adaptive_segment(self, gray: np.ndarray, config: SegmentProcessorConfig) -> Tuple[np.ndarray, List[Dict]]:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, config.block_size, config.c_constant
        )
        mask = self._post_process_mask(mask, config)
        return mask, self._extract_regions(mask)
    
    def _threshold_segment(self, gray: np.ndarray, config: SegmentProcessorConfig) -> Tuple[np.ndarray, List[Dict]]:
        if config.threshold_value is None:
            threshold = 127
        else:
            threshold = config.threshold_value
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = self._post_process_mask(mask, config)
        return mask, self._extract_regions(mask)
    
    def _watershed_segment(self, gray: np.ndarray, config: SegmentProcessorConfig) -> Tuple[np.ndarray, List[Dict]]:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((config.kernel_size, config.kernel_size), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=config.morph_iterations)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for i in range(2, markers.max() + 1):
            mask[markers == i] = 255
        
        mask = self._post_process_mask(mask, config)
        return mask, self._extract_regions(mask)
    
    def _post_process_mask(self, mask: np.ndarray, config: SegmentProcessorConfig) -> np.ndarray:
        kernel = np.ones((config.kernel_size, config.kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_iterations)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= config.min_region_size:
                filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def _extract_regions(self, mask: np.ndarray) -> List[Dict[str, Any]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 0:
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                regions.append({
                    "id": idx,
                    "area": float(area),
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "centroid": (int(cx), int(cy)),
                    "perimeter": float(perimeter),
                    "circularity": float(circularity)
                })
        
        return regions


class SpotImageSegmentProcessor(ImageSegmentProcessor):
    """SpotImage专用的分割处理器"""
    
    def process(self, config: SegmentProcessorConfig, **kwargs) -> ProcessingResult:
        try:
            spot_image = kwargs.get('data')
            if spot_image is None or not isinstance(spot_image, SpotImage):
                return ProcessingResult(
                    success=False,
                    message="需要一个SpotImage对象",
                    error="SpotImage required"
                )
            
            return super().process(config, data=spot_image.image)
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"SpotImage分割失败: {str(e)}",
                error=str(e)
            )


def create_segment_processor(
    method: str = "otsu",
    **kwargs
) -> ImageSegmentProcessor:
    """创建分割处理器的工厂函数"""
    config = SegmentProcessorConfig(method=method, **kwargs)
    return ImageSegmentProcessor(config)


def quick_segment(
    image: np.ndarray,
    method: str = "otsu",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """快速分割函数"""
    config = SegmentProcessorConfig(method=method, **kwargs)
    processor = ImageSegmentProcessor(config)
    result = processor.execute(image)
    
    if result.success:
        data = result.data
        return data["segmented_image"], data["mask"]
    else:
        raise ValueError(result.error)
