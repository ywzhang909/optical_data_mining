#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像特征提取Pipeline处理器

提供Pipeline集成的图像特征提取功能
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops

from data_mining.experiment_analysis.base_processor import BaseProcessor, ProcessingResult
from data_mining.experiment_analysis.models import ProcessorConfig
from data_mining.image.common import SpotImage


class FeatureExtractConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    
    extract_intensity: bool = Field(default=True, description="提取强度特征")
    extract_shape: bool = Field(default=True, description="提取形状特征")
    extract_texture: bool = Field(default=True, description="提取纹理特征")
    extract_histogram: bool = Field(default=False, description="提取直方图特征")
    extract_morphological: bool = Field(default=False, description="提取形态学特征")
    
    histogram_bins: int = Field(default=256, description="直方图 bins")
    texture_distances: List[int] = Field(default_factory=lambda: [1, 2, 3], description="纹理距离")
    texture_angles: List[int] = Field(default_factory=lambda: [0, np.pi/4, np.pi/2, 3*np.pi/4], description="纹理角度")


class FeatureExtractInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    image: Any = Field(..., description="输入图像")
    mask: Optional[Any] = Field(None, description="分割掩码")


class ExtractedFeatures(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    image_id: Optional[str] = Field(None, description="图像ID")
    intensity_features: Dict[str, float] = Field(default_factory=dict, description="强度特征")
    shape_features: Dict[str, float] = Field(default_factory=dict, description="形状特征")
    texture_features: Dict[str, float] = Field(default_factory=dict, description="纹理特征")
    histogram_features: Dict[str, float] = Field(default_factory=dict, description="直方图特征")
    morphological_features: Dict[str, float] = Field(default_factory=dict, description="形态学特征")
    all_features: Dict[str, float] = Field(default_factory=dict, description="所有特征汇总")


class ImageFeatureExtractProcessor(BaseProcessor[ExtractedFeatures]):
    config_class = FeatureExtractConfig
    
    def __init__(self, config: Optional[FeatureExtractConfig] = None):
        super().__init__(config)
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, str]:
        input_data = kwargs.get('data', data)
        if input_data is None:
            return False, "图像数据是必需的"
        return True, ""
    
    def process(self, config: FeatureExtractConfig, **kwargs) -> ProcessingResult:
        try:
            image = kwargs.get('data')
            if image is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入图像",
                    error="Image is required"
                )
            
            if isinstance(image, dict):
                image_data = image.get('image', image.get('segmented_image'))
                mask = image.get('mask')
            else:
                image_data = image
                mask = None
            
            if image_data is None:
                return ProcessingResult(
                    success=False,
                    message="图像数据无效",
                    error="Invalid image data"
                )
            
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data.astype(np.uint8)
            
            features = ExtractedFeatures()
            
            if config.extract_intensity:
                features.intensity_features = self._extract_intensity_features(gray, mask)
            
            if config.extract_shape:
                features.shape_features = self._extract_shape_features(gray, mask)
            
            if config.extract_texture:
                features.texture_features = self._extract_texture_features(gray, mask)
            
            if config.extract_histogram:
                features.histogram_features = self._extract_histogram_features(gray, mask, config.histogram_bins)
            
            if config.extract_morphological:
                features.morphological_features = self._extract_morphological_features(gray, mask)
            
            features.all_features = {
                **features.intensity_features,
                **features.shape_features,
                **features.texture_features,
                **features.histogram_features,
                **features.morphological_features
            }
            
            return ProcessingResult(
                success=True,
                message=f"特征提取完成，共 {len(features.all_features)} 个特征",
                data=features.model_dump(),
                metadata={
                    "num_features": len(features.all_features),
                    "feature_groups": {
                        "intensity": len(features.intensity_features),
                        "shape": len(features.shape_features),
                        "texture": len(features.texture_features),
                        "histogram": len(features.histogram_features),
                        "morphological": len(features.morphological_features)
                    }
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"特征提取失败: {str(e)}",
                error=str(e)
            )
    
    def _extract_intensity_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        if mask is not None:
            roi = gray[mask > 0]
        else:
            roi = gray.flatten()
        
        if len(roi) == 0:
            return {}
        
        return {
            "intensity_mean": float(np.mean(roi)),
            "intensity_std": float(np.std(roi)),
            "intensity_min": float(np.min(roi)),
            "intensity_max": float(np.max(roi)),
            "intensity_median": float(np.median(roi)),
            "intensity_sum": float(np.sum(roi)),
            "intensity_range": float(np.max(roi) - np.min(roi)),
            "intensity_percentile_25": float(np.percentile(roi, 25)),
            "intensity_percentile_75": float(np.percentile(roi, 75)),
            "intensity_iqr": float(np.percentile(roi, 75) - np.percentile(roi, 25)),
            "intensity_skewness": float(self._skewness(roi)),
            "intensity_kurtosis": float(self._kurtosis(roi))
        }
    
    def _extract_shape_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        if mask is None:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area <= 0 or perimeter <= 0:
            return {}
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w / h) if h > 0 else 0
        extent = float(area / (w * h)) if (w * h) > 0 else 0
        
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0
        
        equi_diameter = np.sqrt(4 * area / np.pi)
        
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (ma, ma_ratio), angle = ellipse
            eccentricity = float(np.sqrt(1 - (min(ma, ma * ma_ratio) / max(ma, ma * ma_ratio))**2))
        except Exception:
            eccentricity = 0
        
        return {
            "shape_area": float(area),
            "shape_perimeter": float(perimeter),
            "shape_aspect_ratio": aspect_ratio,
            "shape_extent": extent,
            "shape_solidity": solidity,
            "shape_equi_diameter": float(equi_diameter),
            "shape_eccentricity": eccentricity,
            "shape_compactness": float(perimeter ** 2 / (4 * np.pi * area)) if area > 0 else 0,
            "shape_bounding_rect_area": float(w * h)
        }
    
    def _extract_texture_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        if mask is not None:
            roi = gray[mask > 0]
            if len(roi) == 0:
                return {}
            roi_min, roi_max = roi.min(), roi.max()
            if roi_max > roi_min:
                gray_norm = ((gray - roi_min) / (roi_max - roi_min) * 255).astype(np.uint8)
            else:
                gray_norm = gray
        else:
            gray_norm = gray
        
        gray_norm = (gray_norm / 16).astype(np.uint8)
        
        try:
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray_norm, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
        except Exception:
            return {}
        
        features = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            try:
                values = graycoprops(glcm, prop)
                features[f'texture_{prop}_mean'] = float(np.mean(values))
                features[f'texture_{prop}_std'] = float(np.std(values))
            except Exception:
                pass
        
        return features
    
    def _extract_histogram_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None, bins: int = 256) -> Dict[str, float]:
        if mask is not None:
            roi = gray[mask > 0]
        else:
            roi = gray.flatten()
        
        if len(roi) == 0:
            return {}
        
        hist, _ = np.histogram(roi, bins=bins, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        
        cumsum = np.cumsum(hist)
        
        features = {
            "histogram_entropy": float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))),
            "histogram_uniformity": float(np.sum(hist ** 2)),
            "histogram_max_bin": float(np.argmax(hist)),
            "histogram_max_value": float(np.max(hist)),
        }
        
        for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
            features[f'histogram_percentile_{int(threshold*100)}'] = float(np.searchsorted(cumsum, threshold))
        
        return features
    
    def _extract_morphological_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        if mask is None:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        mask_bool = mask > 0
        
        try:
            skeleton = morphology.skeletonize(mask_bool)
            skeleton_length = int(np.sum(skeleton))
        except Exception:
            skeleton_length = 0
        
        try:
            dist_transform = ndimage.distance_transform_edt(mask_bool)
            max_dist = float(np.max(dist_transform))
            mean_dist = float(np.mean(dist_transform[mask_bool])) if mask_bool.any() else 0.0
        except Exception:
            max_dist = 0.0
            mean_dist = 0.0
        
        try:
            labeled = measure.label(mask_bool)
            num_regions = int(labeled.max())
        except Exception:
            num_regions = 0
        
        return {
            "morph_skeleton_length": float(skeleton_length),
            "morph_max_distance": float(max_dist),
            "morph_mean_distance": float(mean_dist),
            "morph_num_regions": float(num_regions)
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _kurtosis(self, data: np.ndarray) -> float:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return float(np.mean(((data - mean) / std) ** 4)) - 3


class SpotImageFeatureExtractProcessor(ImageFeatureExtractProcessor):
    """SpotImage专用的特征提取处理器"""
    
    def process(self, config: FeatureExtractConfig, **kwargs) -> ProcessingResult:
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
                message=f"SpotImage特征提取失败: {str(e)}",
                error=str(e)
            )


def create_feature_processor(
    extract_intensity: bool = True,
    extract_shape: bool = True,
    extract_texture: bool = True,
    **kwargs
) -> ImageFeatureExtractProcessor:
    """创建特征提取处理器的工厂函数"""
    config = FeatureExtractConfig(
        extract_intensity=extract_intensity,
        extract_shape=extract_shape,
        extract_texture=extract_texture,
        **kwargs
    )
    return ImageFeatureExtractProcessor(config)


def quick_extract_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, float]:
    """快速特征提取函数"""
    config = FeatureExtractConfig(**kwargs)
    processor = ImageFeatureExtractProcessor(config)
    result = processor.execute({'image': image, 'mask': mask})
    
    if result.success:
        return result.data["all_features"]
    else:
        raise ValueError(result.error)
