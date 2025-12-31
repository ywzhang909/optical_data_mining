from typing import Optional, Any, Callable
from singleton_decorator import singleton

import os
from pathlib import Path
from datetime import datetime

from PIL import Image
import numpy as np
import pandas as pd
import math
from scipy.ndimage import fourier_shift

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
    
def process_time_columns(df: pd.DataFrame):
    """
    处理DataFrame中的时间相关列
    
    Args:
        df (pd.DataFrame): 要处理的DataFrame
    
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
    df = df.copy()
    # 添加time列
    df['time'] = df['path'].apply(lambda x: x.stem)
    # 提取括号内信息
    df['info'] = df['time'].str.extract(r'[（\(]([^）\)]*)[）\)]')[0]
    # 转换时间格式
    df['time'] = pd.to_datetime(
        df['time'].str.split('(').str[0].str.replace('：', ':', regex=False),
        format='%Y%m%d %H:%M:%S.%f'
    )
    return df

def cartesian_to_polar(x, y, c_x=0, c_y=0):
    """
    Convert Cartesian coordinates to polar coordinates with a custom origin.

    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        c_x (float): x-coordinate of the custom origin.
        c_y (float): y-coordinate of the custom origin.

    Returns:
        tuple: A tuple containing the radial distances (r) and angles (theta).
    """
    dx = x - c_x
    dy = y - c_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return r, theta

def polar_to_cartesian(r, theta, c_x, c_y):
    """
    Convert polar coordinates to Cartesian coordinates with a custom origin.

    Args:
        r (np.ndarray): Radial distances from the custom origin.
        theta (np.ndarray): Angles in radians.
        c_x (float): x-coordinate of the custom origin.
        c_y (float): y-coordinate of the custom origin.

    Returns:
        tuple: A tuple containing the x-coordinates (x) and y-coordinates (y).
    """
    x = c_x + r * np.cos(theta)
    y = c_y + r * np.sin(theta)
    return x, y

def extract_radial_data(image, centroid_x, centroid_y, angle):
    """
    Extract data along a radial line from the centroid at a given angle.

    Args:
        image (np.ndarray): The input image array.
        centroid_x (int): The x-coordinate of the centroid.
        centroid_y (int): The y-coordinate of the centroid.
        angle (float): The angle in degrees.

    Returns:
        np.ndarray: The extracted data along the radial line.
    """
    height, width = image.shape
    angle_rad = np.deg2rad(angle)
    max_length = int(max(
        math.sqrt(centroid_x**2 + centroid_y**2),
        math.sqrt((width - centroid_x)**2 + centroid_y**2),
        math.sqrt(centroid_x**2 + (height - centroid_y)**2),
        math.sqrt((width - centroid_x)**2 + (height - centroid_y)**2)
    ))
    distances = np.arange(-max_length, max_length + 1)
    x_coords = np.round(centroid_x + distances * np.cos(angle_rad)).astype(int)
    y_coords = np.round(centroid_y + distances * np.sin(angle_rad)).astype(int)
    valid_mask = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    return image[y_coords, x_coords]

def fourier_shift_to_center(image, cx, cy):
    """
    使用傅里叶移位定理实现亚像素平移。
    
    参数:
        image: 2D array
        shift: (dy, dx) —— 平移量（可为小数）
    
    返回:
        shifted_image: 平移后的图像
    """
    h, w = image.shape
    target_x, target_y = w // 2, h // 2
    dx, dy = cx - target_x, cy - target_y
    return fourier_shift(image, (dy, dx))
    

def get_profiles(img, center, line_width=5):
    h,w = img.shape
    center_x, center_y = center

    start_x = int(max(0, center_x - line_width // 2))
    end_x = int(min(w, center_x + line_width // 2 + 1))
    vertical = np.mean(img[:, start_x:end_x], axis=1)  # 沿X方向平均

    start_y = int(max(0, center_y - line_width // 2))
    end_y = int(min(h, center_y + line_width // 2 + 1))
    horizontal = np.mean(img[start_y:end_y, :], axis=0) # 沿Y方向平均
    return {
        'vertical': vertical, 'horizontal': horizontal
    }

def normalize_data(data):
    """
    Normalize the data to the range [0, 1].
    Args:
        data (np.ndarray): The input data array.
    Returns:
        np.ndarray: The normalized data array.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

class Image2D:

    def __init__(self, image_array):
        self.image_array = image_array.astype(np.float32) \
            if isinstance(image_array, np.ndarray) else np.array(image_array, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError("SpotImage must be a 2D array")
        
        self.w, self.h = image_array.shape[0], image_array.shape[1]
        self.xv, self.yv = np.arange(self.w), np.arange(self.h)
        self.xx, self.yy = np.meshgrid(self.xv, self.yv)

    @property
    def normalized_image(self):
        return normalize_data(self.image_array)
    
    def uint8_image(self):
        return (self.normalized_image * 255).astype(np.uint8)


class SpotImage(Image2D):
    
    def __init__(self, image_array, name: str, meta_info: Optional[dict[str, Any]] = None):
        super(SpotImage, self).__init__(image_array)
        self.meta_info = meta_info or {}
        self.meta_info['name'] = name
        self.meta_info['_analysis_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta_info['_process_list'] = []
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