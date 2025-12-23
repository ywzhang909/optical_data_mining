# %%
import pandas as pd
from PIL import Image
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
import cv2
import math

from pathlib import Path
from functools import partial
import swanlab

# %%
root_dir = Path('D:/Projects/TIFO/data-mining/data')

axis_beam_dir = root_dir / Path('20250611/1F/光轴image/20250611 15：38：48(20%（31束）)')
pupil_beam_dir = root_dir / Path('20250611/1F/光瞳image/20250611 15：38：48(20%（31束）)')

axis_beam_img_path = axis_beam_dir.glob('*.TIFF')
pupil_beam_img_path = pupil_beam_dir.glob('*.TIFF')

axis_beam = pd.DataFrame([{'path': path} for path in axis_beam_img_path])
pupil_beam = pd.DataFrame([{'path': path} for path in pupil_beam_img_path])

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
def process_time_columns(df):
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

def process_image_data(df, denoise_method='median'):
    """
    处理图像数据：去暗场和计算强度
    
    Args:
        df (pd.DataFrame): 包含img_array列的DataFrame
        denoise_method (str): 去暗场方法，'median' 或 'min'
    
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
    df = df.copy()
    
    # 去暗场
    if denoise_method == 'median':
        df['black'] = df['img_array'].apply(lambda x: np.median(x))
    elif denoise_method == 'min':
        df['black'] = df['img_array'].apply(lambda x: np.min(x))
    
    # 去噪后的图像
    df['denoise_img_array'] = df.apply(lambda x: np.where(x['img_array'] > x['black'], x['img_array'] - x['black'], 0), axis=1)
    
    # 计算总强度
    df['intensity'] = df['denoise_img_array'].apply(np.sum)
    
    return df

def process_beam_data(beam_df, denoise_method='median'):
    """
    统一处理光束数据的预处理
    
    Args:
        beam_df (pd.DataFrame): 原始光束数据
        denoise_method (str): 去暗场方法
    
    Returns:
        pd.DataFrame: 处理后的数据
    """
    # 处理时间列
    beam_df = process_time_columns(beam_df)
    # 处理图像数据
    beam_df = process_image_data(beam_df, denoise_method)
    return beam_df

# 同时处理两个DataFrame
axis_beam = process_beam_data(axis_beam, 'median')
pupil_beam = process_beam_data(pupil_beam, 'median')

# 筛选有效数据
valid_axis_beam = axis_beam[axis_beam['intensity'] > 0].copy()
valid_pupil_beam = pupil_beam[pupil_beam['intensity'] > 0].copy()

# %%
# 提取一阶矩、二阶矩
def d4sigma(img : np.ndarray, pixel_size_um=1.0):
    """
    计算图像的 D4σ 直径（一阶矩和二阶矩）
    
    Args:
        img (np.ndarray): 输入图像（2D数组）
        pixel_size_um (float): 像素尺寸（微米）
    
    Returns:
        tuple: (中心x, 中心y, D4σ_x, D4σ_y)
    """
    total = img.sum()
    cy, cx = center_of_mass(img)
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 二阶中心矩（光强加权）
    mu_xx = np.sum((x - cx)**2 * img) / total  # σ_x²
    mu_yy = np.sum((y - cy)**2 * img) / total  # σ_y²
    Dx = 4 * np.sqrt(max(mu_xx, 0)) * pixel_size_um
    Dy = 4 * np.sqrt(max(mu_yy, 0)) * pixel_size_um
    
    return cx, cy, Dx, Dy

def d4sigma_feature_extract(df):
    df = df.copy()
    df[['center_x', 'center_y', 'D_x', 'D_y']] = pd.DataFrame(
        df['denoise_img_array'].apply(d4sigma).tolist(), index=df.index)
    df['avg_sigma2'] = np.sqrt(df['D_x'] * df['D_y'])
    return df

valid_axis_beam = d4sigma_feature_extract(valid_axis_beam)
valid_pupil_beam = d4sigma_feature_extract(valid_pupil_beam)
# %%
# 高斯拟合
def cartesian_to_polar(x, y, c_x, c_y):
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

def gaussian(x, mu, sigma, A, b):
    """
    Define the Gaussian function.

    Args:
        x (np.ndarray): Input x values.
        A (float): Amplitude of the Gaussian.
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Output values of the Gaussian function.
    """
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b

def fitting_gaussian(data):
    """
    Fit a Gaussian function to a given data series.
    Args:
        data (np.ndarray): The data series to fit a Gaussian function.
    Returns:
        tuple: A tuple containing the fitted parameters (A, b, mu, sigma) and the fitted curve.
    """
    x_data = np.arange(len(data))
    initial_guess = [np.argmax(data), 10, np.max(data), 0]
    (mu, sigma, A, b), covariance = curve_fit(gaussian, x_data, data, p0=initial_guess)

    return (mu, sigma, A, b), covariance

def calculate_diameter(sigma):
    diameter = 2 * sigma
    return diameter

def calculate_xy_diameters(image):
    """
    Calculate the diameters at y = 1/e + b in x and y directions.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.

    Returns:
        tuple: A tuple containing the x-direction diameter and y-direction diameter.
    """
    # 获取图像中心作为默认质心
    height, width = image.shape
    center_x = width // 2
    center_y = height // 2
    
    coords = center_of_mass(image)
    center_y = int(coords[0])  # type: ignore
    center_x = int(coords[1])  # type: ignore
    # 确保在边界内
    center_x = max(0, min(center_x, width - 1))
    center_y = max(0, min(center_y, height - 1))

    # Extract data for x and y directions
    y_data = image[:, center_x]
    x_data = image[center_y, :]

    # Calculate diameters
    (mu, sigma, A, b), conv = fitting_gaussian(x_data)
    x_diameter = calculate_diameter(sigma)
    (mu, sigma, A, b), conv = fitting_gaussian(y_data)
    y_diameter = calculate_diameter(sigma)

    return x_diameter, y_diameter

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

def calculate_diameter_at_angle(image, centroid_x, centroid_y, angle):
    """
    Calculate the diameter at y = 1/e + b at a given angle.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.
        angle (float): The angle in degrees.

    Returns:
        float: The calculated diameter, or None if fitting fails or A <= 0.
    """
    radial_data = extract_radial_data(image, centroid_x, centroid_y, angle)
    (mu, sigma, A, b), conv = fitting_gaussian(radial_data)
    return calculate_diameter(sigma)

valid_axis_beam[['gaussian_dia_x', 'gaussian_dia_y']] = pd.DataFrame(
    valid_axis_beam['denoise_img_array'].apply(calculate_xy_diameters).tolist(), index=valid_axis_beam.index)
# %%
# 椭圆拟合
def convert_to_cv(img):
    # 归一化到0-255范围
    image_min = np.min(img)
    image_max = np.max(img)
    if image_max > image_min:
        # 归一化到0-255范围
        normalized_image = (img - image_min) / (image_max - image_min) * 255
        uint8_image = normalized_image.astype(np.uint8)
    else:
        # 如果图像是常量，创建全零图像
        uint8_image = np.zeros_like(img, dtype=np.uint8)
    return uint8_image

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
    else:
        x, y = 0, 0
        radius = 0

    return x, y, radius

def ellipse_fit(uint8_image):
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
        h, w = uint8_image.shape
        return (w//2, h//2), (0, 0), 0

def shape_feature_extract(df):
    df = df.copy()
    uint8_images = df['denoise_img_array'].apply(convert_to_cv)
    df[['ellipse_center', 'axis', 'angle']] = pd.DataFrame(
        uint8_images.apply(ellipse_fit).tolist(), index=df.index)
    df[['short_axis', 'long_axis']] = pd.DataFrame(df['axis'].tolist(), index=df.index)
    df[['ellipse_center_x', 'ellipse_center_y']] = pd.DataFrame(df['ellipse_center'].tolist(), index=df.index)
    df['axis_ratio']  = df['long_axis'] / df['short_axis']
    df[['border_x', 'border_y', 'border_radius']] = pd.DataFrame(
        uint8_images.apply(find_spot_border).tolist(), index=df.index)
    return df
    
uint8_images = valid_axis_beam['denoise_img_array'].apply(convert_to_cv)
    
valid_axis_beam = shape_feature_extract(valid_axis_beam)
# TODO valid_pupil_beam = shape_feature_extract(valid_pupil_beam)

# %%
# TODO zernike 

# %%
# 光瞳光轴对齐
valid_axis_beam.sort_values('time', inplace=True)
valid_pupil_beam.sort_values('time', inplace=True)

# 合并数据，按时间nearest join
if len(valid_axis_beam) < len(valid_pupil_beam):
    merged_beam = pd.merge_asof(
        valid_axis_beam,
        valid_pupil_beam,
        on='time',
        direction='nearest',
        suffixes=('_axis', '_pupil')
    )
else:
    merged_beam = pd.merge_asof(
        valid_pupil_beam,
        valid_axis_beam,
        on='time',
        direction='nearest',
        suffixes=('_pupil', '_axis')
    )
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
    ideal_psf = np.abs(U_focal) ** 2  # type: ignore
    
    # 4. 将实测焦斑也移到中心（便于取峰值）
    focal_centered = shift_to_center_fft(focal_img)
    
    # 5. 能量归一化（使总功率一致）
    total_actual = focal_centered.sum()
    if total_actual > 0:
        ideal_psf = ideal_psf / ideal_psf.sum() * total_actual
    
    # 6. 计算 Strehl
    strehl = focal_centered.max() / ideal_psf.max()
    # return strehl, ideal_psf, pupil_centered, focal_centered
    return {
        'strehl': strehl,
        'ideal_psf': ideal_psf,
        # 'pupil_centered': pupil_centered,
        # 'focal_centered': focal_centered
    }

strehl_results = merged_beam.apply(
    lambda row: strehl_with_centering(row['denoise_img_array_pupil'], row['denoise_img_array_axis']),
    axis=1,
    result_type='expand'
)
merged_beam = pd.concat([merged_beam, strehl_results], axis=1)
# %%
# 计算BPP和M²
def calculate_bpp_from_pupil_and_focal(
    pupil_diameter_mm,
    focal_diameter_mm,
    focal_length_mm,
    wavelength_nm=1064.0
):
    """
    使用出瞳和焦斑直径（单位：mm）计算 BPP 和 M²
    
    Parameters:
        pupil_diameter_mm: 出瞳 D4σ 直径（毫米）
        focal_diameter_mm: 焦平面 D4σ 直径（毫米）
        focal_length_mm: 透镜焦距（毫米）
        wavelength_nm: 波长（纳米）
    
    Returns:
        dict with results in mm / mm·mrad / mrad
    """
    # 转为半径（mm）
    w_pupil = pupil_diameter_mm / 2.0      # mm
    w_focal = focal_diameter_mm / 2.0      # mm
    f = focal_length_mm                    # mm
    
    # 发散角 θ ≈ w_focal / f （单位：弧度）
    theta_rad = w_focal / f
    theta_mrad = theta_rad * 1000.0        # 转为毫弧度（mrad）
    
    # BPP = w_pupil * θ （单位：mm·rad → 转为 mm·mrad）
    bpp_mm_mrad = w_pupil * theta_mrad
    
    # 衍射极限 BPP (mm·mrad) = λ(μm) / π
    wavelength_um = wavelength_nm * 1e-3   # nm → μm
    bpp_diffraction_mm_mrad = wavelength_um / np.pi
    
    M2 = bpp_mm_mrad / bpp_diffraction_mm_mrad
    
    return {
        "BPP_mm_mrad": bpp_mm_mrad,
        "M2": M2,
        "theta_mrad": theta_mrad,
        "pupil_radius_mm": w_pupil,
    }

    
# 计算BPP和M²
def calculate_bpp_m2_for_row(row):
    """
    为merged_beam的每一行计算BPP和M²
    """
    waist_diameter = row['avg_sigma2_pupil']
    farfield_diameter = row['avg_sigma2_axis']
    
    result = calculate_bpp_from_pupil_and_focal(
        pupil_diameter_mm=waist_diameter,
        focal_diameter_mm=farfield_diameter,
        focal_length_mm= 1.0e3,
        wavelength_nm = 1064.0e-3,
    )
    return result

# 为每一行计算BPP和M²
bpp_m2_results = merged_beam.apply(calculate_bpp_m2_for_row, axis=1, result_type='expand')
merged_beam = pd.concat([merged_beam, bpp_m2_results], axis=1)
# %%
import pygwalker

def filter_pygwalker_supported_columns(df):
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
    # 筛选支持的列
    supported_df = filter_pygwalker_supported_columns(df)
    # 调用 pygwalker
    pygwalker.walk(supported_df)

# 1. 按time列排序merged_beam
merged_beam = merged_beam.sort_values('time')
# 调用函数进行分析
analyze_with_pygwalker(merged_beam)
# %%
# swanlab 初始化已在文件开头完成
swanlab.init(
    # 设置项目名
    project="beam_analysis_test",
    # TODO experiment_name="beam_analysis_test",
    # 设置超参数
    config={
        "focal_data_path": str(axis_beam_dir),
        "pupil_data_path": str(pupil_beam_dir),
        "process": '->'.join(['denoise', 'center', 'strehl', 'sort_by_time', 'bpp_m2_calculation', 'dict_convert', 'swanlab_log']),
    }
)

for step, (idx, row) in enumerate(merged_beam.iterrows()):
    # 转换为dict，过滤掉不能序列化的列
    row_dict = {}
    for col in merged_beam.columns:
        value = row[col]
        # 处理不同类型的numpy数组
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                # 二维数组转换为图像
                row_dict[col] = swanlab.Image(value)
            elif value.ndim == 1:
                # 一维数组转换为图表数据
                chart = swanlab.echarts.Line()
                chart.add_xaxis(merged_beam['time'].tolist())
                chart.add_yaxis(col, value.tolist())
                row_dict[col] = chart
            elif value.ndim == 0:
                # 0维数组转换为标量
                row_dict[col] = float(value)
            else:
                # 多维数组跳过
                continue
        elif isinstance(value, (pd.Timestamp)):
            # 转换为Python datetime对象
            row_dict[col] = value.to_pydatetime().timestamp()
        elif isinstance(value, (tuple, list, dict, str, Path)):
            # 跳过其他复杂数据类型
            continue
        else:
            row_dict[col] = value
    
    # 记录到swanlab，添加step参数
    swanlab.log(row_dict, step=step)
swanlab.finish()
# %%
