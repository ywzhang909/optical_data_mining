from typing import Optional, Any, Callable
from singleton_decorator import singleton

import os
from pathlib import Path
from datetime import datetime

from PIL import Image
import numpy as np
import pandas as pd

img_process_func = Callable[
    ["SpotImage", Optional[dict[str, Any]]],
    tuple["SpotImage", Optional[dict[str, Any]]]
    ]


@singleton
class FunctionRegistry:
    _registry: dict[str, img_process_func] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[img_process_func], img_process_func]:
        def decorator(func: img_process_func) -> img_process_func:
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[img_process_func]:
        return cls._registry.get(name)

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())
    


def read_tiff_to_numpy(file_path):
    """
    Read a TIFF image file and convert it to a NumPy array.

    Args:
        file_path (str): The path to the TIFF image file.

    Returns:
        np.ndarray: A NumPy array representing the TIFF image.
    """
    try:
        # Open the TIFF image using PIL
        image = Image.open(file_path)
        if image.mode != 'L':
            image = image.convert('L')
        # Convert the image to a NumPy array
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error reading the TIFF file: {e}")
        return None
    


class SpotImage:
    
    def __init__(self, image_array, name: str, meta_info: Optional[dict[str, Any]] = None):
        self.image_array = image_array
        if image_array.ndim != 2:
            raise ValueError("SpotImage must be a 2D array")
        
        self.meta_info = meta_info or {}
        self.meta_info['name'] = name
        self.meta_info['_analysis_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta_info['_process_list'] = []
        
        self.w, self.h = image_array.shape[0], image_array.shape[1]
        self.xv, self.yv = np.arange(self.w), np.arange(self.h)
        self.xx, self.yy = np.meshgrid(self.xv, self.yv)
        
        self.features = {}
        
    @classmethod
    def from_tiff(cls, file_path):
        image_array = read_tiff_to_numpy(file_path)
        meta_info = {
            'source_file': str(file_path),
            'creation_time': datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
            }
        if image_array is None:
            raise ValueError(f"Failed to read TIFF file: {file_path}")
        return cls(image_array, name=Path(file_path).stem, meta_info=meta_info)
    
    def add_process(self, process: img_process_func, parameters: dict[str, Any]):
        self.meta_info['_process_list'].append({
            'process_name': process.__name__,
            'parameters': parameters,
            })
        self, res = process(self, parameters)
        if res is not None:
            self.features.update(res)
            
    def get_features(self):
        res = self.features.copy()
        res.update(self.meta_info)
        return res
            
    @property
    def image(self):
        return self.image_array
    
    
class SpotImageCollection:
    def __init__(self, spot_images: list[SpotImage]):
        self.spot_images = spot_images
        
    def get_feature_dataframe(self):
        df = pd.DataFrame([img.features for img in self.spot_images])
        return df