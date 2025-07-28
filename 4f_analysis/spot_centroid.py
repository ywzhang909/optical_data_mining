from PIL import Image
import math
from glob import glob
from tqdm import tqdm
from loguru import logger

from datetime import datetime
import re

import numpy as np

from scipy.optimize import curve_fit

from pathlib import Path
current_dir = Path(__file__).resolve().parent

logger.info(f"Current directory: {current_dir}")

# file operations
def read_tiff_to_numpy(file_path) -> np.ndarray:
    """
    Read a TIFF image file and convert it to a NumPy array.

    Args:
        file_path (str): The path to the TIFF image file.

    Returns:
        np.ndarray: A NumPy array representing the TIFF image.
    """
    # Open the TIFF image using PIL
    image = Image.open(file_path)
    if image.mode != 'L':
        image = image.convert('L')
    # Convert the image to a NumPy array
    image_array = np.array(image)
    return image_array


def read_all_tiff_in_dir(dir_path) -> list[np.ndarray]:
    file_list = glob(f'{dir_path}/*.TIFF')
    corpus = []
    for file_path in file_list:
        try:
            corpus.append(read_tiff_to_numpy(file_path))
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
    logger.info(f"Read {len(corpus)} TIFF images")
    return corpus
    
# visual algorithms
def find_centroid(img):
    """
    Calculate the centroid coordinates of an object in a binary image.

    Args:
        image (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the (y, x) coordinates of the centroid.
    """
    # Calculate centroid
    total = np.sum(img)
    if total == 0:
        raise ValueError("Empty image - all pixel values are zero")
    # Get image dimensions
    height, width = img.shape
    # Create coordinate grids
    x, y = np.indices((width, height))
    # Calculate weighted coordinates
    x_center = int(np.sum(x * img.T) / total)
    y_center = int(np.sum(y * img.T) / total)
    
    return x_center, y_center
    
def find_lightest_centroid(img):
    """
    Calculate the centroid coordinates of the lightest object in a binary image.

    Args:
        image (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the (y, x) coordinates of the centroid.
    """
    # Calculate centroid
    total = np.sum(img)
    if total == 0:
        raise ValueError("Empty image - all pixel values are zero")
    _img = img.copy()
    max_intensity = np.max(_img)
    _img[img!=max_intensity] = 0

    return find_centroid(_img)

def gaussian(x, b, mu, sigma):
    """
    Define the Gaussian function.

    Args:
        x (np.ndarray): Input x values.
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Output values of the Gaussian function.
    """
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b

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

def normalize(data:np.ndarray, keep_min=False):
    """
    Normalize the input data to the range [0, 1].

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Normalized data array.
    """
    if keep_min:
        return (data) / np.max(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def fit_one_image_with_particular_angle(image:np.ndarray, angle:float):
    """
    Fit a Gaussian curve to the radial data extracted from the image at a given angle.

    Args:
        image (np.ndarray): Input image array.
        angle (float): Angle in degrees.

    Returns:
        dict: Fitting parameters (b, mu, sigma) and diameter.
    """
    assert np.max(image) == 1, "normalize the image first"
    c_x, c_y = find_lightest_centroid(image)
    y_data = extract_radial_data(image, c_x, c_y, angle)
    x_data = np.arange(len(y_data))
    initial_guess = [0, np.argmax(y_data), 10]
    # Perform the curve fitting using the least squares method
    params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    logger.info("fit end with cov: {}".format(covariance))
    return params, covariance

@logger.catch
def calculate_one_image(image_path:str) -> dict:
    image = read_tiff_to_numpy(image_path)
    assert np.max(image) > 50, "Image is too dark"
    _image = normalize(image)
    params_x, _ = fit_one_image_with_particular_angle(_image, 0)
    params_y, _ = fit_one_image_with_particular_angle(_image, 90)
    
    def pack_result(param):
        param = dict(zip(['b', 'mu', 'wrist_radiu'], param))
        param['wrist_radiu'] = 2 * math.sqrt(2) * param['wrist_radiu']
        return param
    
    params_x = pack_result(params_x)
    params_y = pack_result(params_y)
    
    logger.info(f"fit result: {params_x}, {params_y}")
    
    return {"centroid_x": params_x['mu'],
            "centroid_y": params_y['mu'],
            "average noise": np.mean([params_x['b'], params_y['b']]),
            "wrist radius(x)": params_x['wrist_radiu'],
            "wrist radius(y)": params_y['wrist_radiu']}

def parse_timestamp(timestamp_str: str) -> datetime:
    """
    从指定格式的字符串中解析时间戳
    
    Args:
        timestamp_str: 时间字符串（格式示例："20250611 15：52：12.35(60%（31束）)"）
        
    Returns:
        datetime: 解析后的时间对象
    """
    # 正则表达式匹配：提取日期（8位数字）、时间（时:分:秒.毫秒）
    # 注意原字符串中使用全角冒号"："，需要匹配
    pattern = r"(\d{8}) (\d{2})：(\d{2})：(\d{2})\.(\d{2})"
    match = re.search(pattern, timestamp_str)
    if not match:
        raise ValueError(f"无法解析时间字符串: {timestamp_str}")
    
    # 提取各时间分量（年、月、日、时、分、秒、毫秒）
    year = int(match.group(1)[:4])
    month = int(match.group(1)[4:6])
    day = int(match.group(1)[6:8])
    hour = int(match.group(2))
    minute = int(match.group(3))
    second = int(match.group(4))
    millisecond = int(match.group(5))  # 毫秒（两位）
    
    # 构造datetime对象（注意：datetime的微秒参数需要6位，这里将毫秒转为微秒）
    return datetime(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        microsecond=millisecond * 10000  # 2位毫秒 → 6位微秒（如35ms → 35000μs）
    )

def merge_dict_list(dict_list: list[dict]) -> dict:
    """
    将相同键的字典列表转换为值为列表的字典
    
    Args:
        dict_list: 包含多个字典的列表（要求所有字典有相同键）
        
    Returns:
        合并后的字典，每个键对应的值是原列表中所有字典该键值的列表
    """
    merged = {}
    # 遍历每个字典
    for d in dict_list:
        # 遍历字典中的每个键值对
        for key, value in d.items():
            # 初始化键或追加值
            if key not in merged:
                merged[key] = []
            merged[key].append(value)
    return merged

def main():
    file_list = glob(f'{current_dir}/*.TIFF')
    data = {Path(_image).stem: calculate_one_image(_image) for _image in tqdm(file_list)}
    return data

if __name__ == '__main__':
    import os
    import pandas as pd
    
    data = main()
    data = {k:v for k,v in data.items() if v}
    merged_result = merge_dict_list(list(data.values()))
    df = pd.DataFrame(merged_result, index=data.keys())
    df['time'] = df.index.map(lambda x: parse_timestamp(x))
    print(df)
    current_dir_name = Path(__file__).parent.name
    df.to_pickle(os.path.join(current_dir, f'{current_dir_name}.pkl'))
    
    df.to_csv(os.path.join(current_dir, f'{current_dir_name}.csv'))
    logger.info(f"save to {os.path.join(current_dir, 'centroid.csv')}")
