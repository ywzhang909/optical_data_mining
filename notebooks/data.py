# %%
import pandas as pd
from PIL import Image
import numpy as np
from scipy.ndimage import center_of_mass
import cv2
import math

from pathlib import Path
from functools import partial

# %%
root_dir = Path('D:/Projects/TIFO/data-mining/data')

axis_beam_dir = root_dir / Path('20250611/1F/光轴image/20250611 15：38：48(20%（31束）)')
pupil_beam_dir = root_dir / Path('20250611/1F/光轴image/20250611 15：38：48(20%（31束）)')

axis_beam_img_path = axis_beam_dir.glob('*.TIFF')
pupil_beam_img_path = pupil_beam_dir.glob('*.TIFF')

axis_beam = pd.DataFrame([{'path': path} for path in axis_beam_img_path])
pupil_beam = pd.DataFrame([{'path': path} for path in pupil_beam_img_path])
# %%
def read_tiff_to_numpy(file_path):
    """
    Read a TIFF image file and convert it to a NumPy array.

    Args:
        file_path (str): The path to the TIFF image file.

    Returns:
        np.ndarray: A NumPy array representing the TIFF image.
    """
    image = Image.open(file_path)
    if image.mode != 'L':
        image = image.convert('L')
    image_array = np.asarray(image, dtype=np.float32)
    return image_array
    
axis_beam['img_array'] = axis_beam['path'].apply(read_tiff_to_numpy)
pupil_beam['img_array'] = pupil_beam['path'].apply(read_tiff_to_numpy)

# %%
axis_beam['time'] = axis_beam['path'].apply(lambda x: x.stem)
# 提取括号内信息
axis_beam['info'] = axis_beam['time'].str.extract(r'[（\(]([^）\)]*)[）\)]')[0]
axis_beam['time'] = pd.to_datetime(
    axis_beam['time'].str.split('(').str[0].str.replace('：', ':', regex=False),
    format='%Y%m%d %H:%M:%S.%f'
)
# %%
# 去暗场
axis_beam['black'] = axis_beam['img_array'].apply(lambda x: np.median(x))
# pupil_beam['black'] = pupil_beam['img_array'].apply(lambda x: x.min())

axis_beam['denoise_img_array'] = axis_beam.apply(lambda x: np.where(x['img_array'] > x['black'], x['img_array'] - x['black'], 0), axis=1)
# pupil_beam['denoise_img_array'] = pupil_beam.apply(lambda x: np.where(x['img_array'] > x['black'], x['img_array'] - x['black'], 0), axis=1)

axis_beam['intensity'] = axis_beam['denoise_img_array'].apply(np.sum)
# pupil_beam['intensity'] = pupil_beam['denoise_img_array'].apply(np.sum)
# %%
# 筛选
valid_axis_beam = axis_beam[axis_beam['intensity'] > 0].copy()
# valid_pupil_beam = pupil_beam[pupil_beam['intensity'] > 0]

# %%
# 提取一阶矩、二阶矩
valid_axis_beam[['center_x', 'center_y']] = pd.DataFrame(
    valid_axis_beam['denoise_img_array'].apply(lambda x: center_of_mass(x)[::-1]).tolist(), index=valid_axis_beam.index)

def d4sigma(img : np.ndarray):
    total = img.sum()
    cy, cx = center_of_mass(img)
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 二阶中心矩（光强加权）
    mu_xx = np.sum((x - cx)**2 * img) / total  # σ_x²
    mu_yy = np.sum((y - cy)**2 * img) / total  # σ_y²
    Dx = 4 * np.sqrt(max(mu_xx, 0))
    Dy = 4 * np.sqrt(max(mu_yy, 0))
    
    return Dx, Dy

valid_axis_beam[['D_x', 'D_y']] = pd.DataFrame(
    valid_axis_beam['denoise_img_array'].apply(d4sigma).tolist(), index=valid_axis_beam.index)

# %%
# 椭圆拟合
def find_spot_border(image):
    """
    处理光斑图片，计算噪声阈值，去除噪声并拟合包含光斑的圆形。

    参数:
    image (numpy.ndarray): 输入的光斑图片，应为单通道灰度图像。

    返回:
    numpy.ndarray: 去除噪声后的图像。
    tuple: 拟合圆形的圆心坐标 (x, y) 和半径。
    """
    # 步骤 1: 高斯降噪
    denoised_image = cv2.GaussianBlur(image, (3, 3), 0)
    # 步骤 2: canny边缘检测
    noise_threshold = np.max(denoised_image) * (1/math.e)
    denoised_image = np.where(denoised_image > noise_threshold, denoised_image, 0)
    # 步骤 3: 去除噪声
    denoised_image = cv2.fastNlMeansDenoising(denoised_image, None, 10, 7, 21)
    # denoised_image = cv2.Canny(image, 1, 1)
    # 步骤 4: 二值化
    near_binary_img = cv2.threshold(denoised_image, noise_threshold, 255, cv2.THRESH_BINARY)[1]
    # 步骤 5: 拟合一个圆形正好包含光斑
    contours, _ = cv2.findContours(near_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
    else:
        center = (0, 0)
        radius = 0

    return denoised_image, (center, radius)

def ellipse_fit(image):
    # 将float32图像转换为uint8类型以适配OpenCV函数
    # 首先归一化到0-255范围
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max > image_min:
        # 归一化到0-255范围
        normalized_image = (image - image_min) / (image_max - image_min) * 255
        uint8_image = normalized_image.astype(np.uint8)
    else:
        # 如果图像是常量，创建全零图像
        uint8_image = np.zeros_like(image, dtype=np.uint8)
    
    # 计算噪声阈值（使用30%作为阈值）
    noise_threshhold = np.max(uint8_image) * 0.3
    
    # 二值化处理
    binary_image = cv2.threshold(uint8_image, noise_threshhold, 255, cv2.THRESH_BINARY)[1]
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        (ellipse_center_x, ellipse_center_y),(short_axis, long_axis),angle = cv2.fitEllipse(largest_contour)
        return (ellipse_center_x, ellipse_center_y),(short_axis, long_axis),angle
    else:
        # 如果没有找到轮廓，返回默认值
        h, w = image.shape
        return (w//2, h//2), (0, 0), 0
    
valid_axis_beam[['ellipse_center', 'axis', 'angle']] = pd.DataFrame(
    valid_axis_beam['denoise_img_array'].apply(ellipse_fit).tolist(), index=valid_axis_beam.index)
valid_axis_beam[['short_axis', 'long_axis']] = pd.DataFrame(valid_axis_beam['axis'].tolist(), index=valid_axis_beam.index)
valid_axis_beam[['ellipse_center_x', 'ellipse_center_y']] = pd.DataFrame(valid_axis_beam['ellipse_center'].tolist(), index=valid_axis_beam.index)
valid_axis_beam['axis_ratio']  = valid_axis_beam['long_axis'] / valid_axis_beam['short_axis']
# %%
# 光瞳光轴对齐

# %%
# strehl
def shift_to_center_fft(image, target_center=None):
    """
    使用傅里叶移位将光斑移到图像中心（无插值，保全信息）
    
    参数:
        image: 2D array
        target_center: (cx, cy) 目标中心，默认为图像几何中心
    
    返回:
        shifted_image: 光斑已居中的图像
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    
    if target_center is None:
        target_cx, target_cy = w // 2, h // 2
    else:
        target_cx, target_cy = target_center

    # 当前质心
    cy, cx = center_of_mass(image)
    
    # 需要平移的量（从当前到目标）
    dx = target_cx - cx
    dy = target_cy - cy

    # 傅里叶移位：在频域乘以相位因子
    # 创建频率网格
    u = np.fft.fftfreq(w).reshape(1, -1)
    v = np.fft.fftfreq(h).reshape(-1, 1)

    # 相位因子：exp(-2πi (u*dx + v*dy))
    phase = np.exp(-2j * np.pi * (u * dx + v * dy))

    # 应用移位
    F = np.fft.fft2(image)
    F_shifted = F * phase
    shifted = np.real(np.fft.ifft2(F_shifted))

    # 保留非负强度（数值误差可能导致微小负值）
    shifted = np.clip(shifted, 0, None)
    return shifted

def strehl_with_centering(pupil_img, focal_img):
    """
    自动处理非中心光斑的 Strehl 比计算
    """
    # 1. 将入瞳光斑移到中心
    pupil_centered = shift_to_center_fft(pupil_img)
    
    # 2. 构建复振幅（假设相位=0）
    amplitude = np.sqrt(np.clip(pupil_centered, 0, None))
    amplitude = amplitude / np.sqrt(amplitude.sum()**2 + 1e-12)  # 归一化功率
    
    # 3. 计算理想 PSF（已在中心）
    U_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(amplitude)))
    ideal_psf = np.abs(U_focal) ** 2
    
    # 4. 将实测焦斑也移到中心（便于取峰值）
    focal_centered = shift_to_center_fft(focal_img)
    
    # 5. 能量归一化（使总功率一致）
    total_actual = focal_centered.sum()
    if total_actual > 0:
        ideal_psf = ideal_psf / ideal_psf.sum() * total_actual
    
    # 6. 计算 Strehl
    strehl = focal_centered.max() / ideal_psf.max()
    return strehl, ideal_psf, pupil_centered, focal_centered

# %%
import pygwalker

def filter_pygwalker_supported_columns(df):
    """
    筛选出 pygwalker 支持的数据类型的列
    
    Args:
        df (pd.DataFrame): 输入的数据框
        
    Returns:
        pd.DataFrame: 只包含 pygwalker 支持列的数据框
    """
    supported_dtypes = ['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 
                       'int8', 'uint8', 'uint16', 'uint32', 'datetime64[ns]', 
                       'category', 'string', 'object']
    
    supported_columns = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        
        # 检查是否是基本支持的数据类型
        if dtype in supported_dtypes:
            # 对于 object 类型，需要进一步检查是否为复杂对象（如 numpy 数组）
            if dtype == 'object':
                # 检查是否为简单的可序列化对象（如字符串、Path等）
                sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample_value is not None:
                    # 检查是否为 numpy 数组或其他复杂对象
                    if isinstance(sample_value, (np.ndarray, tuple, list, dict)):
                        continue  # 跳过复杂对象
                supported_columns.append(col)
            else:
                supported_columns.append(col)
    
    return df[supported_columns]

def analyze_with_pygwalker(df):
    """
    筛选出 pygwalker 支持的列并调用 pygwalker 进行数据探索
    
    Args:
        df (pd.DataFrame): 输入的数据框
    """
    print("原始数据框形状:", df.shape)
    print("原始数据框列:", list(df.columns))
    
    # 筛选支持的列
    supported_df = filter_pygwalker_supported_columns(df)
    
    print("筛选后数据框形状:", supported_df.shape)
    print("筛选后数据框列:", list(supported_df.columns))
    print("筛选掉的列:", [col for col in df.columns if col not in supported_df.columns])
    
    # 调用 pygwalker
    pygwalker.walk(supported_df)

# 调用函数进行分析
analyze_with_pygwalker(valid_axis_beam)
# %%
