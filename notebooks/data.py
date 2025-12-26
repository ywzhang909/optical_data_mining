# %%
import os
from typing import Literal

import pandas as pd

from PIL import Image
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import lombscargle
from sklearn.metrics import r2_score, mean_squared_error
import cv2
import math
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
import swifter
import swanlab
import time
from functools import wraps


# 配置swifter参数以优化性能
swifter.set_defaults(
    npartitions=os.cpu_count(),  # 分区数量，根据CPU核心数调整
    dask_threshold=100,  # 数据量阈值，超过此数量使用dask
    disable_cache=False,  # 启用缓存
    progress_bar=True  # 显示进度条
)

def time_function(func):
    """装饰器：记录函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

root_dir = Path('D:/Projects/TIFO/data-mining/data')

# %%
# load power meter data
power_file = root_dir / Path('002/digitaloptical4Floor/功率计数据/20251219 09：29：37(20%平顶31).TXT')
power_data = pd.read_csv(power_file, sep='\t', encoding='gbk', header=None)
power_data.columns = ['time', 'power']
# %%
power_data['time'] = pd.to_datetime(power_data['time'], format='%Y-%m-%d-%H：%M：%S.%f')
power_data['delta_time'] = (power_data['time'] - power_data['time'].iloc[0]).dt.total_seconds().astype(float)

def double_erf_pulse(t, P0, A, t1, sigma1, t2, sigma2):
    """
    对于**带上升沿和下降沿的功率曲线（如激光脉冲、光开关响应等）**，且**上升/下降时间未知**，推荐使用 **“平滑阶跃函数”组合模型**。这类函数既能描述平台区（恒定功率），又能用可调参数刻画上升/下降沿的**形状和时间位置**。

    **双误差函数模型（Double Error Function）**

    $$
    P(t) = P_0 + \frac{A}{2} \left[ 
    \operatorname{erf}\left( \frac{t - t_1}{\sigma_1} \right) 
    - \operatorname{erf}\left( \frac{t - t_2}{\sigma_2} \right) 
    \right]
    $$

    #### 🔍 参数含义：
    | 参数 | 物理意义 |
    |------|----------|
    | $P_0$ | 基线功率（脉冲前/后的背景） |
    | $A$ | 脉冲幅度（平台高度） |
    | $t_1$ | **上升沿中心时间**（50% 上升点附近） |
    | $\sigma_1$ | **上升沿陡峭度**（越小越陡，$\tau_{\text{rise}} \propto \sigma_1$） |
    | $t_2$ | **下降沿中心时间**（50% 下降点附近） |
    | $\sigma_2$ | **下降沿陡峭度** |
    """
    return P0 + (A / 2.0) * (erf((t - t1) / sigma1) - erf((t - t2) / sigma2))

def estimate_initial_params_erf(t, P):
    P_min, P_max = np.min(P), np.max(P)
    P0_est = P_min
    A_est = P_max - P_min
    
    # 找 50% 幅度对应的大概时间
    mid_level = P0_est + 0.5 * A_est
    idx_rise = np.where(P >= mid_level)[0][0]  # 第一个超过50%的点（上升）
    idx_fall = np.where(P >= mid_level)[0][-1] # 最后一个超过50%的点（下降）
    
    t1_est = t[idx_rise]
    t2_est = t[idx_fall]
    
    # 估计 sigma：假设上升/下降跨越 ~5个数据点
    dt = np.mean(np.diff(t))
    sigma1_est = sigma2_est = 2 * dt  # 初始猜测较平缓
    
    return [P0_est, A_est, t1_est, sigma1_est, t2_est, sigma2_est]

def bounds_erf(t, P):
    bounds_low = [
        np.min(P) - 1,   # P0
        0.1,                  # A > 0
        np.min(t),       # t1
        1e-6,                 # sigma1 > 0
        np.min(t),       # t2
        1e-6                  # sigma2 > 0
    ]
    bounds_high = [
        np.max(P),       # P0
        (np.max(P_data) - np.min(P_data)),  # A
        np.max(t_data),       # t1
        (np.max(t) - np.min(t)) / 2,  # sigma1
        np.max(t),       # t2
        (np.max(t) - np.min(t)) / 2   # sigma2
    ]
    return bounds_low, bounds_high

def double_sigmoid(t, P0, A1, k, t1, k1, t2, k2):
    """
    双 Sigmoid 脉冲模型

    $$
    P(t) = P_0 + A \left[ 
    \frac{1}{1 + e^{-k_1 (t - t_1)}} 
    - \frac{1}{1 + e^{-k_2 (t - t_2)}}
    \right]
    $$

    #### 🔍 参数说明：
    | 参数 | 含义 |
    |------|------|
    | $P_0$ | 基线功率（脉冲前/后的背景值） |
    | $A$ | 脉冲平台高度（幅度） |
    | $t_1$ | 上升沿中心时间（Sigmoid 中点，≈50% 上升点） |
    | $k_1$ | 上升沿陡峭度（越大越陡，$k_1 > 0$） |
    | $t_2$ | 下降沿中心时间（Sigmoid 中点，≈50% 下降点） |
    | $k_2$ | 下降沿陡峭度（越大越陡，$k_2 > 0$） |
    """
    rise = 1 / (1 + np.exp(-k1 * (t - t1)))
    fall = 1 / (1 + np.exp(-k2 * (t - t2)))
    return P0 + A1 * (rise - k * fall)

def estimate_initial_params_sigmoid(t, P):
    """
    从数据粗略估计双 Sigmoid 的初始参数
    """
    P_min, P_max = np.min(P), np.max(P)
    P0_est = P_min
    A_est = P_max - P_min
    
    # 找 50% 幅度对应的时间（近似 t1, t2）
    mid_level = P0_est + 0.5 * A_est
    above_mid = np.where(P >= mid_level)[0]
    
    if len(above_mid) == 0:
        raise ValueError("无法估计脉冲位置")
    
    t1_est = t[above_mid[0]]   # 第一个超过50%的点 → 上升中点
    t2_est = t[above_mid[-1]]  # 最后一个超过50%的点 → 下降中点
    
    # 估计 k：假设上升/下降跨越 ~5个时间单位
    dt = np.mean(np.diff(t))
    k1_est = k2_est = 2.0 / dt  # 初始陡峭度（经验值）
    
    return [P0_est, A_est, 1, t1_est, k1_est, t2_est, k2_est]

def bounds_sigmoid(t, P):
    avg_t = np.diff(t).mean()
    bounds_low = [
        np.min(P) - 0.5,      # P0
        0.1,              # A > 0
        1,
        np.min(t),            # t1
        0.1,              # k1 > 0
        np.min(t),            # t2
        0.1               # k2 > 0
    ]
    bounds_high = [
        np.max(P),            # P0
        np.max(P) - np.min(P),  # A
        2,
        np.max(t),            # t1
        10.0 / avg_t,    # k1 上限（非常陡）
        np.max(t),            # t2
        10.0 / avg_t     # k2 上限
    ]
    return bounds_low, bounds_high

# 3. 计算 10%-90% 上升/下降时间（更工程化）
t_data = power_data['delta_time'].to_numpy()
P_data = power_data['power'].to_numpy()
p0 = estimate_initial_params_sigmoid(t_data, P_data)

popt, pcov = curve_fit(
    double_sigmoid, t_data, P_data,
    p0=p0,
    bounds=bounds_sigmoid(t_data, P_data),
    maxfev=5000
)
P0, A1, k, t1, k1, t2, k2 = popt

t_fine = np.linspace(np.min(t_data), np.max(t_data), 10000)
P_fine = double_sigmoid(t_fine, *popt)

def find_time_for_fraction(frac):
    target = P0 + frac * A1
    # 在拟合曲线上插值找时间
    idx = np.argmin(np.abs(P_fine - target))
    return t_fine[idx]

t_rise_10 = find_time_for_fraction(0.1)
t_rise_90 = find_time_for_fraction(0.9)
rise_time = t_rise_90 - t_rise_10

# 注意：下降沿是从高到低，所以要找下降段的90%和10%
# 更准确做法：在 t > (t1+t2)/2 区域找
t_mid = (t1 + t2) / 2
mask_fall = t_fine > t_mid
t_fall_90 = t_fine[mask_fall][np.argmin(np.abs(P_fine[mask_fall] - (P0 + 0.9*A1*k)))]
t_fall_10 = t_fine[mask_fall][np.argmin(np.abs(P_fine[mask_fall] - (P0 + 0.1*A1*k)))]
fall_time = t_fall_10 - t_fall_90  # 应为正数

plt.figure(figsize=(10, 5))
plt.scatter(t_data, P_data, s=15, alpha=0.6, label='Measured Data')
plt.plot(t_fine, P_fine, 'r-', linewidth=2, label='Fit Data')
plt.axhline(P0 + 0.1*A1, color='gray', linestyle='--', alpha=0.5)
plt.axhline(P0 + 0.9*A1, color='gray', linestyle='--', alpha=0.5)
plt.title(
    f'R2 score: {r2_score(P_data, double_sigmoid(t_data, *popt)):.3f}, Rise Time: {rise_time:.3f}, Fall Time: {fall_time:.3f}')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# %%
# load image data
ref_axis_dir = root_dir / Path('002/digitaloptical4Floor/光轴image/20251219 09：28：25(全子束弱光平顶)')

axis_beam_dir = root_dir / Path('002/digitaloptical4Floor/光轴image/20251219 09：29：37(20%平顶31)')
pupil_beam_dir = root_dir / Path('002/digitaloptical4Floor/光瞳image/20251219 09：29：37(20%平顶31)')

axis_beam_img_path = axis_beam_dir.glob('*.TIFF')
pupil_beam_img_path = pupil_beam_dir.glob('*.TIFF')

ref_axis = pd.DataFrame([{'path': path} for path in ref_axis_dir.glob('*.TIFF')])
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
    
ref_axis['img_array'] = ref_axis['path'].apply(read_tiff_to_numpy)
axis_beam['img_array'] = axis_beam['path'].apply(read_tiff_to_numpy)
pupil_beam['img_array'] = pupil_beam['path'].apply(read_tiff_to_numpy)

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
        df['black'] = df['img_array'].swifter.apply(lambda x: np.median(x))
    elif denoise_method == 'min':
        df['black'] = df['img_array'].swifter.apply(lambda x: np.min(x))
    
    # 去噪后的图像
    black_threshold = df['black'].max()
    df['denoise_img_array'] = df.apply(lambda x: np.where(x['img_array'] > black_threshold, x['img_array'] - black_threshold, 0), axis=1)
    
    # 计算总强度
    df['intensity'] = df['denoise_img_array'].swifter.apply(np.sum)
    
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

ref_axis = process_beam_data(ref_axis, 'median')
axis_beam = process_beam_data(axis_beam, 'median')
pupil_beam = process_beam_data(pupil_beam, 'median')

# 筛选有效数据
valid_ref_axis = ref_axis[ref_axis['intensity'] > 10]
valid_axis_beam = axis_beam[axis_beam['intensity'] > 10]
valid_pupil_beam = pupil_beam[pupil_beam['intensity'] > 10]

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
    
    return {
        'center_x': float(cx),
        'center_y': float(cy),
        'D_x': float(Dx),
        'D_y': float(Dy),
        'center_intensity': float(img[int(cy), int(cx)]),
    }

def d4sigma_feature_extract(df: pd.DataFrame):
    d4sigma_features = df.swifter.apply(lambda x: d4sigma(x['denoise_img_array']), axis=1, result_type='expand')
    d4sigma_features['avg_sigma2'] = np.sqrt(d4sigma_features['D_x'] * d4sigma_features['D_y'])
    return pd.concat([df, d4sigma_features], axis=1)

valid_ref_axis = d4sigma_feature_extract(valid_ref_axis)
valid_axis_beam = d4sigma_feature_extract(valid_axis_beam)
valid_pupil_beam = d4sigma_feature_extract(valid_pupil_beam)

ref_axis_center_x = valid_ref_axis['center_x'].mean()
ref_axis_center_y = valid_ref_axis['center_y'].mean()
valid_axis_beam['center_offset_x'] = valid_axis_beam['center_x'] - ref_axis_center_x
valid_axis_beam['center_offset_y'] = valid_axis_beam['center_y'] - ref_axis_center_y
valid_axis_beam['center_offset'] = np.sqrt(
    (valid_axis_beam['center_x'] - ref_axis_center_x) ** 2 +
    (valid_axis_beam['center_y'] - ref_axis_center_y) ** 2
)
# %%
# 均匀度
def blur_img(img: np.ndarray, kernel_size=100):
    """
    对图像进行模糊处理
    
    Args:
        img (np.ndarray): 输入图像（2D数组）
        kernel_size (int): 模糊核大小（奇数）
    
    Returns:
        np.ndarray: 模糊处理后的图像
    """
    return cv2.blur(img, (kernel_size, kernel_size))

def uniformity(img: np.ndarray, center: tuple, clip_level=0.85):
    """
    计算图像的均匀度
    
    Args:
        img (np.ndarray): 输入图像（2D数组）
    
    Returns:
        float: rms, 四象限均匀度
    """
    cx, cy = int(center[0]), int(center[1])
    
    # 四象限均匀度
    q1 = img[:cy, :cx].sum()
    q2 = img[:cy, cx:].sum()
    q3 = img[cy:, cx:].sum()
    q4 = img[cy:, :cx].sum()
    four_quadrant_rms = np.sqrt((q1 + q2 + q3 + q4) / 4)
    return {
        'rms_4_quadrant': four_quadrant_rms
    }

uniformity_features = valid_pupil_beam.swifter.apply(
    lambda x: uniformity(x['denoise_img_array'], (x['center_x'], x['center_y'])), axis=1, result_type='expand'
)
valid_pupil_beam = pd.concat([valid_pupil_beam, uniformity_features], axis=1)

# %%
# 平顶拟合
# 定义理想平顶光强模型（erf 平滑边界）
def top_hat_erf_model(x, A, w, sigma, offset):
    """
    带 erf 软边界的平顶函数
    :param x: 横坐标 (像素或 mm)
    :param A: 平顶高度
    :param w: 平顶全宽 (FWHM of plateau)
    :param sigma: 边缘过渡宽度参数
    :param offset: 背景偏移
    :return: 光强 I(x)
    """
    return (A / 2.0) * (erf((x + w / 2.0) / sigma) - erf((x - w / 2.0) / sigma)) + offset

def fit_top_hat_profile(img: np.ndarray, center: tuple, orient: Literal['vertical', 'horizontal']='vertical', line_width=5, plot=True):
    """
    从图像中提取中心截面，拟合到理想平顶模型，并返回拟合指标
    
    参数:
        img: 输入图像（2D数组）
        center: 中心坐标 (x, y)
        line_width: 沿中心线取多行平均（抗噪），默认5行
        plot: 是否绘图
    
    返回:
        dict: 包含拟合参数和指标
    """
    # 2. 提取center截面
    h, w = img.shape
    center_x, center_y = center
    if orient == 'vertical':
        start_y = int(max(0, center_y - line_width // 2))
        end_y = int(min(h, center_y + line_width // 2 + 1))
        profile = img[start_y:end_y, int(center_x)]  # 沿Y方向平均
    elif orient == 'horizontal':
        start_x = int(max(0, center_x - line_width // 2))
        end_x = int(min(w, center_x + line_width // 2 + 1))
        profile = img[int(center_y), start_x:end_x]  # 沿X方向平均

    x = np.arange(len(profile))

    # 3. 初始参数估计
    A0 = np.max(profile) - np.min(profile)
    offset0 = np.min(profile)
    
    # 估计平顶宽度：找强度 > 80% 最大值的区域
    threshold = 0.8 * (np.max(profile) - offset0) + offset0
    above = np.where(profile > threshold)[0]
    if len(above) > 0:
        w0 = above[-1] - above[0]
    else:
        w0 = len(profile) * 0.6
    
    sigma0 = 2.0  # 初始边缘宽度

    p0 = [A0, w0, sigma0, offset0]

    # 4. 执行非线性最小二乘拟合
    try:
        popt, pcov = curve_fit(
            top_hat_erf_model, x, profile,
            p0=p0,
            bounds=([0, 0, 0.1, 0], [np.inf, len(profile), 20, np.max(profile)])
        )
        A_fit, w_fit, sigma_fit, offset_fit = popt
        fitted_profile = top_hat_erf_model(x, *popt)

        # 5. 计算拟合优度
        rmse = np.sqrt(mean_squared_error(profile, fitted_profile))
        nrmse = rmse / (np.max(profile) - np.min(profile)) if (np.max(profile) - np.min(profile)) > 0 else 0
        
        # 6. 计算w_fit范围内，实际与A_fit的RMS
        w_fit_range = np.arange(0, w_fit, 0.1)
        A_fit_range = top_hat_erf_model(w_fit_range, A_fit, w_fit, sigma_fit, offset_fit)
        A_diff = np.abs(A_fit_range - A_fit)
        A_diff_rms = np.sqrt(np.mean(A_diff**2))

        return {
                "plateau_width_w": w_fit,
                "flat_rms": A_diff_rms,
                "edge_sigma": sigma_fit,
                "NRMSE": nrmse  # 归一化均方根误差
        }
    except (RuntimeError, ValueError):
        return {
                "plateau_width_w": np.nan,
                "flat_rms": np.nan,
                "edge_sigma": np.nan,
                "NRMSE": np.nan
        }

fit_features = valid_pupil_beam.swifter.apply(
    lambda x: fit_top_hat_profile(x['denoise_img_array'], (x['center_x'], x['center_y'])), axis=1, result_type='expand'
)
valid_pupil_beam = pd.concat([valid_pupil_beam, fit_features], axis=1)


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
    try:
        (mu, sigma, A, b), covariance = curve_fit(gaussian, x_data, data, p0=initial_guess)
    except RuntimeError:
        return (np.nan, np.nan, np.nan, np.nan), np.nan

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
    coords = center_of_mass(image)
    center_y = int(coords[0])  # type: ignore
    center_x = int(coords[1])  # type: ignore
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
    valid_axis_beam['denoise_img_array'].swifter.apply(calculate_xy_diameters).tolist(), index=valid_axis_beam.index)
# %%
# 椭圆拟合
def convert_to_cv(float_image) -> np.ndarray:
    """
    将图像转换为OpenCV兼容的uint8格式
    """
    assert isinstance(float_image, np.ndarray), f"{float_image} must be a numpy array, but got {type(float_image)}"
    image_min = np.min(float_image)
    image_max = np.max(float_image)

    normalized_image = (float_image - image_min) / (image_max - image_min) * 255
    uint8_image = normalized_image.astype(np.uint8)
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
    # 验证输入
    if not isinstance(image, np.ndarray):
        raise TypeError(f"find_spot_border expects numpy array, got {type(image)}")
    
    # 步骤 1: 高斯降噪
    denoised_image = cv2.GaussianBlur(image, (3, 3), 0)
    # 步骤 2: canny边缘检测
    noise_threshold = np.max(denoised_image) * (1/math.e)
    denoised_image = np.where(denoised_image > noise_threshold, denoised_image, 0)
    # 确保数据类型正确
    if denoised_image.dtype != np.uint8:
        denoised_image = denoised_image.astype(np.uint8)
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
        x, y = np.nan, np.nan
        radius = np.nan

    return {
        'border_x': x,
        'border_y': y,
        'border_radius': radius
    }

def ellipse_fit(uint8_image):
    # 计算噪声阈值（使用30%作为阈值）
    noise_threshhold = np.max(uint8_image) * 0.3
    
    # 二值化处理
    binary_image = cv2.threshold(uint8_image, noise_threshhold, 255, cv2.THRESH_BINARY)[1]
    
    # 查找轮廓
    try:
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert contours, "No contours found"
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        (ellipse_center_x, ellipse_center_y),(short_axis, long_axis),angle = cv2.fitEllipse(largest_contour)
    except (AssertionError, ValueError):
        return {
            'ellipse_center_x': np.nan,
            'ellipse_center_y': np.nan,
            'short_axis': np.nan,
            'long_axis': np.nan,
            'ellipticity': np.nan,
            'angle': np.nan,
            'uniformity': np.nan
        }
    
    area = cv2.contourArea(largest_contour)
    # 创建掩膜用于后续处理
    mask = np.zeros_like(uint8_image, dtype=np.uint8)
    if area > 100:
        # 将主轮廓内部填充为白色 (255)
        cv2.drawContours(mask, [largest_contour], -1, (255,), thickness=cv2.FILLED)
        # 使用掩膜提取光斑内的所有像素
        mean_val, std_val = cv2.meanStdDev(uint8_image, mask=mask)
        mean_intensity = mean_val[0][0]
        std_intensity = std_val[0][0]
        uniformity = std_intensity / mean_intensity
    else:
        uniformity = np.nan
    
    return {
        'ellipse_center_x': ellipse_center_x,
        'ellipse_center_y': ellipse_center_y,
        'short_axis': short_axis,
        'long_axis': long_axis,
        'ellipticity': long_axis / short_axis,
        'angle': angle,
        'uniformity': uniformity
    }

def shape_feature_extract(df: pd.DataFrame, use_gpu=False):
    """
    提取形状特征，支持GPU加速
    
    参数:
    df: 包含denoise_img_array列的DataFrame
    use_gpu: 是否使用GPU加速
    
    返回:
    包含形状特征的DataFrame
    """
    # 转换图像格式
    uint8_images = df['denoise_img_array'].apply(convert_to_cv)
    
    ellipse_features = pd.DataFrame(uint8_images.swifter.apply(ellipse_fit).tolist(), index=df.index)
    circle_features = pd.DataFrame(uint8_images.swifter.apply(find_spot_border).tolist(), index=df.index)
    
    return pd.concat([df, ellipse_features, circle_features], axis=1)
    
valid_axis_beam = shape_feature_extract(valid_axis_beam, use_gpu=True)
valid_pupil_beam = shape_feature_extract(valid_pupil_beam, use_gpu=True)

# %%
# Zernike多项式拟合
import numpy as np
from scipy.special import factorial
from scipy.optimize import least_squares

def zernike_radial_polynomial(n, m, rho):
    """
    计算Zernike径向多项式
    
    参数:
    n, m: Zernike多项式的阶数
    rho: 归一化径向坐标 (0-1)
    
    返回:
    R_nm: 径向多项式值
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    
    R = np.zeros_like(rho)
    for k in range(0, (n - m) // 2 + 1):
        coefficient = ((-1) ** k * factorial(n - k)) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        R += coefficient * rho ** (n - 2 * k)
    
    return R

def zernike_function(n, m, rho, theta, normalization='Noll'):
    """
    计算Zernike函数
    
    参数:
    n, m: Zernike多项式的阶数
    rho, theta: 极坐标
    normalization: 归一化方法 ('Noll' 或 'orthonormal')
    
    返回:
    Z: Zernike函数值
    """
    R = zernike_radial_polynomial(n, m, rho)
    
    if m == 0:
        Z = R
    elif m > 0:
        Z = np.sqrt(2) * R * np.cos(m * theta)
    else:
        Z = np.sqrt(2) * R * np.sin(-m * theta)
    
    # 归一化
    if normalization == 'Noll':
        if m == 0:
            norm = np.sqrt(n + 1)
        else:
            norm = np.sqrt(2 * (n + 1))
        Z = Z / norm
    
    return Z

def generate_zernike_basis(max_order=6):
    """
    生成Zernike基函数集合
    
    参数:
    max_order: 最大阶数
    
    返回:
    basis_functions: 基函数列表，每个元素是(n, m, 函数)
    """
    basis_functions = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if n >= abs(m):
                def make_zernike(n=n, m=m):
                    def zernike_func(rho, theta):
                        return zernike_function(n, m, rho, theta)
                    return zernike_func
                basis_functions.append((n, m, make_zernike(n, m)))
    
    return basis_functions

def fit_zernike_to_image(image, max_order=6, mask=None):
    """
    将图像拟合并到Zernike多项式
    
    参数:
    image: 输入图像
    max_order: 最大阶数
    mask: 掩码，用于指定有效区域
    
    返回:
    coeffs: Zernike系数
    fitted_image: 拟合后的图像
    """
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 转换为极坐标
    center_y, center_x = h // 2, w // 2
    rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / min(h, w) * 2
    theta = np.arctan2(y - center_y, x - center_x)
    
    # 创建掩码
    if mask is None:
        mask = rho <= 1.0
    
    # 有效点
    valid_points = mask & (rho <= 1.0)
    rho_valid = rho[valid_points]
    theta_valid = theta[valid_points]
    intensity_valid = image[valid_points]
    
    # 生成基函数
    basis_functions = generate_zernike_basis(max_order)
    
    # 构建设计矩阵
    design_matrix = []
    for n, m, func in basis_functions:
        basis_values = func(rho_valid, theta_valid)
        design_matrix.append(basis_values)
    
    design_matrix = np.column_stack(design_matrix)
    
    # 最小二乘拟合
    coeffs = np.linalg.lstsq(design_matrix, intensity_valid, rcond=None)[0]
    
    # 重构图像
    fitted_image = np.zeros_like(image)
    for i, (n, m, func) in enumerate(basis_functions):
        basis_image = func(rho, theta) * coeffs[i]
        fitted_image += basis_image
    
    return coeffs, fitted_image

def calculate_zernike_coefficients(df, max_order=6):
    """
    为DataFrame中的图像计算Zernike系数
    
    参数:
    df: 包含denoise_img_array列的DataFrame
    max_order: 最大阶数
    
    返回:
    zernike_df: 包含Zernike系数的DataFrame
    """
    def extract_zernike_features(img_array):
        try:
            coeffs, fitted = fit_zernike_to_image(img_array, max_order=max_order)
            
            # 计算拟合残差
            residual = np.sqrt(np.mean((img_array - fitted) ** 2))
            
            # 提取主要像差
            zernike_dict = {
                'zernike_residual': residual,
                'zernike_rms': np.sqrt(np.mean(coeffs ** 2))
            }
            
            # 添加前几项系数
            for i, (n, m, _) in enumerate(generate_zernike_basis(max_order)):
                if i < len(coeffs):
                    zernike_dict[f'Z_{n}_{m}'] = coeffs[i]
            
            return zernike_dict
        except:
            return {f'Z_{n}_{m}': np.nan for n, m, _ in generate_zernike_basis(max_order)}
    
    zernike_features = df['denoise_img_array'].swifter.apply(extract_zernike_features)
    zernike_df = pd.DataFrame(zernike_features.tolist(), index=df.index)
    
    return pd.concat([df, zernike_df], axis=1) 


# %%
# 光流分析
def optical_flow(prev_img, next_img):
    """
    计算两帧之间的光流向量
    
    参数:
    prev_img: 前一帧图像（灰度）
    next_img: 当前帧图像（灰度）
    
    返回:
    flow: 光流向量数组，形状为 (h, w, 2)，其中 flow[..., 0] 是水平方向，flow[..., 1] 是垂直方向
    """
    # 确保输入是浮点数
    prev_img = prev_img.astype(np.float32)
    next_img = next_img.astype(np.float32)
    
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    return flow

def analyze_optical_flow(flow, mask=None):
    """
    分析光流特征
    
    参数:
    flow: 光流向量数组
    mask: 分析掩码
    
    返回:
    dict: 光流统计特征
    """
    if flow is None:
        return {
            'flow_magnitude_mean': np.nan,
            'flow_magnitude_std': np.nan,
            'flow_angle_mean': np.nan,
            'flow_angle_std': np.nan,
            'flow_dominant_direction': np.nan,
            'flow_divergence': np.nan,
            'flow_curl': np.nan
        }
    
    # 计算光流幅度和角度
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)
    
    # 应用掩码
    if mask is not None:
        magnitude = magnitude[mask]
        angle = angle[mask]
        fx = fx[mask]
        fy = fy[mask]
    
    # 计算光流特征
    mag_mean = np.mean(magnitude)
    mag_std = np.std(magnitude)
    angle_mean = np.mean(angle)
    angle_std = np.std(angle)
    
    # 主导方向（幅度加权的角度平均）
    weights = magnitude / (np.sum(magnitude) + 1e-8)
    dominant_angle = np.sum(angle * weights)
    
    # 计算光流散度和旋度（简化版本）
    # 对于离散数据，使用有限差分
    if len(fx.shape) >= 2:
        dx_fx = np.gradient(fx, axis=1)
        dy_fy = np.gradient(fy, axis=0)
        divergence = np.mean(dx_fx + dy_fy)
        
        dx_fy = np.gradient(fy, axis=1)
        dy_fx = np.gradient(fx, axis=0)
        curl = np.mean(dx_fy - dy_fx)
    else:
        divergence = np.nan
        curl = np.nan
    
    return {
        'flow_magnitude_mean': mag_mean,
        'flow_magnitude_std': mag_std,
        'flow_angle_mean': angle_mean,
        'flow_angle_std': angle_std,
        'flow_dominant_direction': dominant_angle,
        'flow_divergence': divergence,
        'flow_curl': curl
    }

def calculate_optical_flow_features(df, ref_col='denoise_img_array'):
    """
    计算DataFrame中连续帧之间的光流特征
    
    参数:
    df: 包含图像数据的DataFrame（按时间排序）
    ref_col: 参考图像列名
    
    返回:
    flow_df: 包含光流特征的DataFrame
    """
    flow_features = []
    
    for i in range(len(df) - 1):
        prev_img = df.iloc[i][ref_col]
        next_img = df.iloc[i + 1][ref_col]
        
        # 确保是灰度图
        if len(prev_img.shape) == 3:
            prev_img = cv2.cvtColor(prev_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if len(next_img.shape) == 3:
            next_img = cv2.cvtColor(next_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 计算光流
        flow = optical_flow(prev_img, next_img)
        
        # 分析光流
        flow_stats = analyze_optical_flow(flow)
        flow_features.append(flow_stats)
    
    # 创建光流特征DataFrame
    flow_df = pd.DataFrame(flow_features)
    flow_df.index = df.index[:-1]  # 最后一帧没有光流
    
    return flow_df

def visualize_optical_flow(flow, prev_img, step=10):
    """
    可视化光流
    
    参数:
    flow: 光流数组
    prev_img: 参考图像
    step: 采样步长
    """
    h, w = flow.shape[:2]
    
    # 创建网格
    y, x = np.mgrid[step//2:h:step, step//2:w:step]
    
    # 采样光流
    fx = flow[y, x, 0]
    fy = flow[y, x, 1]
    
    # 绘制
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(prev_img, cmap='gray')
    plt.quiver(x, y, fx, fy, color='red', scale=1, scale_units='xy', angles='xy')
    plt.title('光流向量')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    plt.imshow(magnitude, cmap='hot')
    plt.colorbar(label='光流幅度')
    plt.title('光流幅度')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


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

strehl_results = merged_beam.swifter.apply(
    lambda row: strehl_with_centering(row['denoise_img_array_pupil'], row['denoise_img_array_axis']),
    axis=1,
    result_type='expand'
)
merged_beam = pd.concat([merged_beam, strehl_results], axis=1)
# %%
# 计算BPP和M²
# TODO 考虑缩束比
def calculate_bpp_from_pupil_and_focal(
    pupil_diameter_mm,
    focal_diameter_mm,
    focal_length_mm,
    wavelength_nm=1064.0,
    beam_expansion_ratio=1.0,
    pupil_diameter_input_mm=None
):
    """
    使用出瞳和焦斑直径（单位：mm）计算 BPP 和 M²，考虑缩束比
    
    Parameters:
        pupil_diameter_mm: 出瞳 D4σ 直径（毫米）
        focal_diameter_mm: 焦平面 D4σ 直径（毫米）
        focal_length_mm: 透镜焦距（毫米）
        wavelength_nm: 波长（纳米）
        beam_expansion_ratio: 光束扩束/缩束比 (>1 表示扩束, <1 表示缩束)
        pupil_diameter_input_mm: 输入光束直径（毫米），用于计算有效扩束比
    
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
    
    # 如果提供了输入光束直径，计算实际扩束比
    if pupil_diameter_input_mm is not None:
        actual_expansion_ratio = pupil_diameter_mm / pupil_diameter_input_mm
        effective_expansion_ratio = actual_expansion_ratio
    else:
        effective_expansion_ratio = beam_expansion_ratio
        actual_expansion_ratio = beam_expansion_ratio
    
    # 计算理想扩束后的理论BPP
    bpp_theoretical_input = None
    if pupil_diameter_input_mm is not None:
        w_input = pupil_diameter_input_mm / 2.0
        theta_theoretical = theta_rad  # 假设扩束后发散角不变
        bpp_theoretical_input = w_input * theta_mrad
    
    # 衍射极限 BPP (mm·mrad) = λ(μm) / π
    wavelength_um = wavelength_nm * 1e-3   # nm → μm
    bpp_diffraction_mm_mrad = wavelength_um / np.pi
    
    M2 = bpp_mm_mrad / bpp_diffraction_mm_mrad
    
    # 考虑缩束比的影响
    M2_corrected = M2 / effective_expansion_ratio if effective_expansion_ratio != 0 else np.nan
    
    return {
        "BPP_mm_mrad": bpp_mm_mrad,
        "M2": M2,
        "M2_corrected": M2_corrected,
        "theta_mrad": theta_mrad,
        "pupil_radius_mm": w_pupil,
        "expansion_ratio": actual_expansion_ratio,
        "bpp_theoretical_input": bpp_theoretical_input,
        "bpp_diffraction_limit": bpp_diffraction_mm_mrad,
        "strehl_estimate": np.exp(-M2_corrected**2) if not np.isnan(M2_corrected) else np.nan
    }

    
# 计算BPP和M²
def calculate_bpp_m2_for_row(row):
    """
    为merged_beam的每一行计算BPP和M²
    """
    waist_diameter = row['avg_sigma2_pupil']
    farfield_diameter = row['avg_sigma2_axis']
    
    # 尝试从数据中推断缩束比（如果存在的话）
    expansion_ratio = getattr(row, 'expansion_ratio', 1.0)
    
    result = calculate_bpp_from_pupil_and_focal(
        pupil_diameter_mm=waist_diameter,
        focal_diameter_mm=farfield_diameter,
        focal_length_mm=1.0e3,
        wavelength_nm=1064.0,
        beam_expansion_ratio=expansion_ratio,
    )
    return result

# 为每一行计算BPP和M²
bpp_m2_results = merged_beam.swifter.apply(calculate_bpp_m2_for_row, axis=1, result_type='expand')
merged_beam = pd.concat([merged_beam, bpp_m2_results], axis=1)

# %%
# save result
# drop columns with object dtype
saved_beam = merged_beam.drop(columns=merged_beam.select_dtypes(include=['object']).columns)
saved_beam.to_parquet(root_dir / 'merged_beam_analysis.parquet', compression='zstd')

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
    pygwalker.walk(supported_df, appearance='light')

# 1. 按time列排序merged_beam
merged_beam = merged_beam.sort_values('time')
# 调用函数进行分析
analyze_with_pygwalker(merged_beam)
# %%
# swanlab 初始化已在文件开头完成
swanlab.init(
    # 设置项目名
    project="beam_analysis_test",
    experiment_name=axis_beam_dir.name,
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
# 时间序列分析
valued_features = merged_beam.select_dtypes(include=['float64','float32', 'float16'])
t = (merged_beam['time'] - merged_beam['time'].iloc[0]).dt.total_seconds()
def lomb_scargle_analysis(t, y, f_min=2./max(t), f_max=50.0, df=0.01):
    """
    对非均匀采样信号进行 Lomb-Scargle 分析。
    
    参数:
    t: 时间数组（非均匀采样）
    y: 信号数组（对应 t）
    f_min: 最小频率（Hz）
    f_max: 最大频率（Hz）
    df: 频率步长（Hz）
    
    返回:
    freqs: 频率数组
    power: Lomb-Scargle 功率谱
    """
    # 定义要搜索的频率范围
    freqs = np.arange(f_min, f_max, df)
    # 计算角频率（Lomb-Scargle 要求 omega = 2πf）
    omega = 2 * np.pi * freqs
    # 执行 Lomb-Scargle
    power = lombscargle(t, y, omega)
    
    peak_idx = np.argmax(power)
    estimated_freq = freqs[peak_idx]
    
    return freqs, power, estimated_freq

totol_cols = len(valued_features.columns)*2//2+1
plt.figure(figsize=(16, 4*totol_cols))
for i, col in enumerate(valued_features.columns):
    y = merged_beam[col]
    freqs, power, estimated_freq = lomb_scargle_analysis(t, y, f_max=len(valued_features)/2)
    
    plt.subplot(totol_cols, 4, 2*i+1)
    plt.plot(freqs, power, 'b-')
    plt.axvline(estimated_freq, color='g', linestyle=':', label=f'Est. freq = {estimated_freq:.2f} Hz')
    plt.title(f'{col}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(len(valued_features.columns)*2//2+1, 4, 2*i+2)
    plt.plot(t, y, 'r-')
    plt.title(f'{col}')
    plt.xlabel('Time (s)')
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# 对valued_features的每个列进行统计分析
def calculate_statistical_analysis(df, columns=None):
    """
    对DataFrame的数值列进行全面的统计分析
    
    参数:
    df: 输入DataFrame
    columns: 要分析的列名列表，默认为所有数值列
    
    返回:
    stats_df: 包含统计指标的DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    stats_list = []
    
    for col in columns:
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
            
        # 基本统计量
        stats = {
            'column': col,
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'median': data.median(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,  # 变异系数
        }
        
        # 异常值检测
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        stats['outlier_count'] = len(outliers)
        stats['outlier_ratio'] = len(outliers) / len(data) if len(data) > 0 else 0
        
        # 正态性检验（Shapiro-Wilk）
        try:
            from scipy.stats import shapiro
            if len(data) >= 3:  # Shapiro-Wilk 需要至少3个数据点
                stat, p_value = shapiro(data[:min(len(data), 5000)])  # 限制样本大小
                stats['normality_pvalue'] = p_value
                stats['is_normal'] = p_value > 0.05
            else:
                stats['normality_pvalue'] = np.nan
                stats['is_normal'] = np.nan
        except:
            stats['normality_pvalue'] = np.nan
            stats['is_normal'] = np.nan
        
        # 稳定性指标
        if len(data) > 1:
            # 相对标准偏差
            stats['relative_stability'] = data.std() / (data.mean() + 1e-12)
            
            # 漂移（线性趋势）
            x = np.arange(len(data))
            try:
                slope = np.polyfit(x, data, 1)[0]
                stats['linear_trend'] = slope
                stats['drift_per_sample'] = slope
            except:
                stats['linear_trend'] = np.nan
                stats['drift_per_sample'] = np.nan
        else:
            stats['relative_stability'] = np.nan
            stats['linear_trend'] = np.nan
            stats['drift_per_sample'] = np.nan
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)

def plot_statistical_analysis(stats_df):
    """
    绘制统计分析结果
    
    参数:
    stats_df: 统计分析结果DataFrame
    """
    if len(stats_df) == 0:
        print("没有数据可绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 变异系数分布
    cv_data = stats_df['cv'].dropna()
    axes[0, 0].hist(cv_data, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('变异系数分布')
    axes[0, 0].set_xlabel('变异系数')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].axvline(cv_data.mean(), color='red', linestyle='--', label=f'均值: {cv_data.mean():.3f}')
    axes[0, 0].legend()
    
    # 2. 偏度分布
    skew_data = stats_df['skewness'].dropna()
    axes[0, 1].hist(skew_data, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('偏度分布')
    axes[0, 1].set_xlabel('偏度')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='正态分布')
    axes[0, 1].legend()
    
    # 3. 异常值比例分布
    outlier_data = stats_df['outlier_ratio'].dropna()
    axes[1, 0].hist(outlier_data, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('异常值比例分布')
    axes[1, 0].set_xlabel('异常值比例')
    axes[1, 0].set_ylabel('频数')
    
    # 4. 均值vs标准差
    mean_data = stats_df['mean'].dropna()
    std_data = stats_df['std'].dropna()
    min_len = min(len(mean_data), len(std_data))
    axes[1, 1].scatter(mean_data[:min_len], std_data[:min_len], alpha=0.6)
    axes[1, 1].set_title('均值 vs 标准差')
    axes[1, 1].set_xlabel('均值')
    axes[1, 1].set_ylabel('标准差')
    
    plt.tight_layout()
    plt.show()
    
    return fig


# %%
# 对valued_features的每个列进行趋势、周期分析
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def trend_analysis(t, y, max_degree=3):
    """
    对时间序列进行趋势分析

    参数:
    t: 时间数组
    y: 数据数组
    max_degree: 多项式拟合的最高阶数

    返回:
    dict: 包含趋势分析结果
    """
    t = np.array(t).reshape(-1, 1)
    y = np.array(y)

    results = {}

    # 1. 线性趋势
    lr = LinearRegression()
    lr.fit(t, y)
    linear_slope = lr.coef_[0]
    linear_intercept = lr.intercept_
    linear_r2 = lr.score(t, y)

    results['linear_slope'] = linear_slope
    results['linear_intercept'] = linear_intercept
    results['linear_r2'] = linear_r2

    # 2. 多项式趋势
    best_degree = 1
    best_r2 = linear_r2

    for degree in range(2, max_degree + 1):
        try:
            poly_features = PolynomialFeatures(degree=degree)
            t_poly = poly_features.fit_transform(t)
            lr_poly = LinearRegression()
            lr_poly.fit(t_poly, y)
            r2_poly = lr_poly.score(t_poly, y)

            if r2_poly > best_r2:
                best_r2 = r2_poly
                best_degree = degree
                results['polynomial_coefficients'] = lr_poly.coef_
                results['polynomial_intercept'] = lr_poly.intercept_
                results['polynomial_degree'] = degree
                results['polynomial_r2'] = r2_poly
        except:
            continue

    # 3. 趋势强度分类
    if abs(linear_slope) < 1e-10:
        trend_type = "平稳"
    elif linear_slope > 0:
        trend_type = "上升"
    else:
        trend_type = "下降"

    if best_r2 > 0.8:
        trend_strength = "强"
    elif best_r2 > 0.5:
        trend_strength = "中等"
    elif best_r2 > 0.2:
        trend_strength = "弱"
    else:
        trend_strength = "无明显趋势"

    results['trend_type'] = trend_type
    results['trend_strength'] = trend_strength
    results['trend_explanation'] = f"{trend_strength}{trend_type}趋势 (R²={best_r2:.3f})"

    return results

def periodicity_analysis(t, y, min_period=None, max_period=None):
    """
    对时间序列进行周期性分析

    参数:
    t: 时间数组
    y: 数据数组
    min_period: 最小周期
    max_period: 最大周期

    返回:
    dict: 包含周期分析结果
    """
    if len(t) < 10:
        return {
            'dominant_period': np.nan,
            'period_strength': np.nan,
            'frequency_spectrum': np.array([]),
            'frequencies': np.array([]),
            'has_significant_period': False
        }

    # 确保时间序列均匀采样
    dt = np.median(np.diff(t))

    # FFT分析
    fft_vals = fft(y - np.mean(y))  # 去均值
    freqs = fftfreq(len(y), dt)
    power_spectrum = np.abs(fft_vals) ** 2

    # 只保留正频率
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power_spectrum[:len(power_spectrum)//2]

    # 转换为周期
    periods = 1.0 / (positive_freqs[1:] + 1e-12)  # 避免除零
    powers = positive_power[1:]  # 去掉直流分量

    # 周期范围过滤
    if min_period is None:
        min_period = 2 * dt
    if max_period is None:
        max_period = (t[-1] - t[0]) / 2

    mask = (periods >= min_period) & (periods <= max_period)
    filtered_periods = periods[mask]
    filtered_powers = powers[mask]

    results = {
        'dominant_period': np.nan,
        'period_strength': np.nan,
        'frequency_spectrum': filtered_powers,
        'frequencies': 1.0 / filtered_periods,
        'periods': filtered_periods,
        'has_significant_period': False
    }

    if len(filtered_periods) == 0:
        return results

    # 找到最强周期
    max_power_idx = np.argmax(filtered_powers)
    dominant_period = filtered_periods[max_power_idx]
    period_strength = filtered_powers[max_power_idx] / np.sum(filtered_powers)

    results['dominant_period'] = dominant_period
    results['period_strength'] = period_strength
    results['has_significant_period'] = period_strength > 0.1  # 10%阈值

    return results

def comprehensive_time_series_analysis(df, time_col='time', columns=None):
    """
    对DataFrame进行综合的时间序列分析

    参数:
    df: 输入DataFrame
    time_col: 时间列名
    columns: 要分析的列名列表

    返回:
    analysis_df: 包含分析结果的DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col in columns:
            columns.remove(time_col)

    t = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values

    analysis_results = []

    for col in columns:
        y = df[col].dropna().values

        # 对齐时间序列
        if len(y) != len(t):
            # 简单处理：取最小长度
            min_len = min(len(y), len(t))
            t_aligned = t[:min_len]
            y_aligned = y[:min_len]
        else:
            t_aligned = t
            y_aligned = y

        # 趋势分析
        trend_results = trend_analysis(t_aligned, y_aligned)

        # 周期分析
        period_results = periodicity_analysis(t_aligned, y_aligned)

        # 组合结果
        result = {
            'column': col,
            **trend_results,
            **period_results
        }

        analysis_results.append(result)

    return pd.DataFrame(analysis_results)

def plot_time_series_analysis(analysis_df, df, time_col='time'):
    """
    绘制时间序列分析结果

    参数:
    analysis_df: 分析结果DataFrame
    df: 原始数据DataFrame
    time_col: 时间列名
    """
    if len(analysis_df) == 0:
        print("没有数据可绘制")
        return

    t = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values

    n_cols = min(len(analysis_df), 4)  # 最多显示4个特征
    n_rows = (len(analysis_df) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4*n_cols, 8*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, (_, row) in enumerate(analysis_df.iterrows()):
        col = row['column']
        y = df[col].values

        if len(y) != len(t):
            min_len = min(len(y), len(t))
            t_plot = t[:min_len]
            y_plot = y[:min_len]
        else:
            t_plot = t
            y_plot = y

        row_idx = i // n_cols
        col_idx = i % n_cols

        # 上图：原始数据和趋势
        ax1 = axes[row_idx * 2][col_idx]
        ax1.plot(t_plot, y_plot, 'b-', alpha=0.7, label='原始数据')

        # 绘制趋势线
        if not np.isnan(row['linear_slope']):
            trend_line = row['linear_intercept'] + row['linear_slope'] * t_plot
            ax1.plot(t_plot, trend_line, 'r--', label=f"线性趋势 (斜率={row['linear_slope']:.2e})")

        ax1.set_title(f'{col}\\n趋势: {row.get("trend_explanation", "N/A")}')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel(col)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 下图：频谱分析
        ax2 = axes[row_idx * 2 + 1][col_idx]
        if len(row.get('frequencies', [])) > 0:
            ax2.semilogy(row['frequencies'], row['frequency_spectrum'])

            # 标记主导周期
            if not np.isnan(row['dominant_period']):
                dom_freq = 1.0 / row['dominant_period']
                ax2.axvline(dom_freq, color='red', linestyle='--',
                           label=f"主导周期: {row['dominant_period']:.1f}s")
                ax2.legend()

        ax2.set_title(f'频谱分析')
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('功率谱')
        ax2.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(analysis_df), n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx * 2][col_idx].axis('off')
        if row_idx * 2 + 1 < len(axes):
            axes[row_idx * 2 + 1][col_idx].axis('off')

    plt.tight_layout()
    plt.show()

    return fig


# %%
# 对valued_features的每个列进行自相关分析
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

def autocorrelation_analysis(y, nlags=None, alpha=0.05):
    """
    对时间序列进行自相关分析

    参数:
    y: 时间序列数据
    nlags: 滞后期数，默认min(10*log10(n), n-1)
    alpha: 显著性水平

    返回:
    dict: 包含自相关分析结果
    """
    y = np.array(y)
    y = y - np.mean(y)  # 去均值

    if len(y) < 10:
        return {
            'acf_values': np.array([]),
            'pacf_values': np.array([]),
            'significant_lags_acf': [],
            'significant_lags_pacf': [],
            'has_autocorrelation': False,
            'autocorr_strength': 0.0
        }

    # 设置默认滞后期数
    if nlags is None:
        nlags = min(int(10 * np.log10(len(y))), len(y) - 1)

    nlags = min(nlags, len(y) - 1)

    try:
        # 计算自相关函数
        acf_vals = acf(y, nlags=nlags, alpha=alpha)
        if len(acf_vals.shape) > 1:
            acf_values = acf_vals[:, 0]
            acf_confint = acf_vals[:, 1:]  # 置信区间
        else:
            acf_values = acf_vals
            acf_confint = None

        # 计算偏自相关函数
        pacf_vals = pacf(y, nlags=nlags, alpha=alpha)
        if len(pacf_vals.shape) > 1:
            pacf_values = pacf_vals[:, 0]
            pacf_confint = pacf_vals[:, 1:]
        else:
            pacf_values = pacf_vals
            pacf_confint = None

        # 找到显著的自相关滞后
        significant_lags_acf = []
        significant_lags_pacf = []

        if acf_confint is not None:
            for i in range(1, len(acf_values)):
                if abs(acf_values[i]) > acf_confint[i, 1] - acf_confint[i, 0]:  # 超过置信区间
                    significant_lags_acf.append(i)

        if pacf_confint is not None:
            for i in range(1, len(pacf_values)):
                if abs(pacf_values[i]) > pacf_confint[i, 1] - pacf_confint[i, 0]:
                    significant_lags_pacf.append(i)

        # 判断是否存在显著自相关
        has_autocorrelation = len(significant_lags_acf) > 0 or len(significant_lags_pacf) > 0

        # 计算自相关强度（前几个滞后的平均绝对自相关）
        autocorr_strength = np.mean(np.abs(acf_values[1:min(6, len(acf_values))]))

        return {
            'acf_values': acf_values,
            'pacf_values': pacf_values,
            'significant_lags_acf': significant_lags_acf,
            'significant_lags_pacf': significant_lags_pacf,
            'has_autocorrelation': has_autocorrelation,
            'autocorr_strength': autocorr_strength,
            'nlags': nlags
        }

    except Exception as e:
        print(f"自相关分析失败: {e}")
        return {
            'acf_values': np.array([]),
            'pacf_values': np.array([]),
            'significant_lags_acf': [],
            'significant_lags_pacf': [],
            'has_autocorrelation': False,
            'autocorr_strength': 0.0
        }

def comprehensive_autocorrelation_analysis(df, time_col='time', columns=None):
    """
    对DataFrame进行综合的自相关分析

    参数:
    df: 输入DataFrame
    time_col: 时间列名
    columns: 要分析的列名列表

    返回:
    analysis_df: 包含自相关分析结果的DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col in columns:
            columns.remove(time_col)

    analysis_results = []

    for col in columns:
        y = df[col].dropna().values

        if len(y) < 10:
            continue

        # 自相关分析
        acf_results = autocorrelation_analysis(y)

        # 组合结果
        result = {
            'column': col,
            **acf_results
        }

        analysis_results.append(result)

    return pd.DataFrame(analysis_results)

def plot_autocorrelation_analysis(analysis_df, df, time_col='time'):
    """
    绘制自相关分析结果

    参数:
    analysis_df: 自相关分析结果DataFrame
    df: 原始数据DataFrame
    time_col: 时间列名
    """
    if len(analysis_df) == 0:
        print("没有数据可绘制")
        return

    n_cols = min(len(analysis_df), 4)  # 最多显示4个特征
    n_rows = (len(analysis_df) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4*n_cols, 8*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, (_, row) in enumerate(analysis_df.iterrows()):
        col = row['column']
        y = df[col].dropna().values

        row_idx = i // n_cols
        col_idx = i % n_cols

        # 上图：原始时间序列
        ax1 = axes[row_idx * 2][col_idx]
        ax1.plot(y, 'b-', alpha=0.7)
        ax1.set_title(f'{col}\\n自相关强度: {row.get("autocorr_strength", 0):.3f}')
        ax1.set_xlabel('样本点')
        ax1.set_ylabel(col)
        ax1.grid(True, alpha=0.3)

        # 下图：自相关函数
        ax2 = axes[row_idx * 2 + 1][col_idx]
        acf_values = row.get('acf_values', np.array([]))
        if len(acf_values) > 0:
            lags = np.arange(len(acf_values))
            ax2.bar(lags, acf_values, alpha=0.7, color='blue', width=0.8)

            # 标记显著滞后
            significant_lags = row.get('significant_lags_acf', [])
            if len(significant_lags) > 0:
                ax2.scatter(significant_lags, acf_values[significant_lags],
                           color='red', s=50, zorder=5, label='显著自相关')

            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axhline(y=1.96/np.sqrt(len(y)), color='red', linestyle='--', alpha=0.5, label='95%置信区间')
            ax2.axhline(y=-1.96/np.sqrt(len(y)), color='red', linestyle='--', alpha=0.5)

            if len(significant_lags) > 0:
                ax2.legend()

        ax2.set_title('自相关函数 (ACF)')
        ax2.set_xlabel('滞后期')
        ax2.set_ylabel('自相关系数')
        ax2.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(analysis_df), n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx * 2][col_idx].axis('off')
        if row_idx * 2 + 1 < len(axes):
            axes[row_idx * 2 + 1][col_idx].axis('off')

    plt.tight_layout()
    plt.show()

    return fig

# 对valued_features的每个列进行自相关分析
print("进行自相关分析...")
try:
    if len(merged_beam) > 10:
        acf_analysis = comprehensive_autocorrelation_analysis(
            merged_beam,
            time_col='time',
            columns=valued_features.columns.tolist()[:5]  # 限制分析前5个特征以节省时间
        )
        print(f"自相关分析完成，共分析了 {len(acf_analysis)} 个特征")
        # 显示自相关分析结果
        plot_autocorrelation_analysis(acf_analysis, merged_beam, time_col='time')
    else:
        print("数据点不足，无法进行自相关分析")
except Exception as e:
    print(f"自相关分析失败: {e}")

# %%
# 扩展时间序列分析统计量
from scipy.stats import entropy
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def calculate_hurst_exponent(time_series, max_lag=100):
    """
    计算Hurst指数，用于判断时间序列的长期记忆性

    Hurst指数计算算法：
    1. 将时间序列分割为多个子序列
    2. 对每个子序列计算累积偏差的范围R和标准差S
    3. 计算log(R/S)与log(子序列长度)的关系
    4. Hurst指数H = 线性拟合的斜率

    H < 0.5: 反持久性（均值回归）
    H = 0.5: 随机游走
    H > 0.5: 持久性（长期记忆）

    参数:
    time_series: 时间序列数据
    max_lag: 最大滞后期

    返回:
    hurst_exponent: Hurst指数
    """
    if len(time_series) < 20:
        return np.nan

    # 去趋势
    y = np.array(time_series) - np.mean(time_series)

    # 计算不同尺度的R/S统计量
    lags = range(2, min(max_lag, len(y)//2))
    rs_values = []

    for lag in lags:
        # 分割为lag个子序列
        n_subseq = len(y) // lag
        if n_subseq < 2:
            continue

        rs_sub = []
        for i in range(n_subseq):
            subseq = y[i*lag:(i+1)*lag]
            if len(subseq) < lag:
                continue

            # 计算累积偏差
            cum_dev = np.cumsum(subseq - np.mean(subseq))

            # 计算范围R
            R = np.max(cum_dev) - np.min(cum_dev)

            # 计算标准差S
            S = np.std(subseq)

            if S > 0:
                rs_sub.append(R / S)

        if rs_sub:
            rs_values.append(np.mean(rs_sub))

    if len(rs_values) < 3:
        return np.nan

    # 对数线性回归
    x = np.log(lags[:len(rs_values)])
    y_rs = np.log(rs_values)

    try:
        slope = np.polyfit(x, y_rs, 1)[0]
        hurst_exponent = slope
    except:
        hurst_exponent = np.nan

    return hurst_exponent

def calculate_sample_entropy(time_series, m=2, r=0.2):
    """
    计算样本熵（Sample Entropy），用于衡量时间序列的复杂性和规律性

    样本熵算法：
    1. 将时间序列分割为m维向量
    2. 计算向量之间的相似度（Chebyshev距离）
    3. 计算匹配模板的比例
    4. 样本熵 = -ln(匹配比例)

    样本熵越小，表示序列越有规律；越大，表示越随机

    参数:
    time_series: 时间序列数据
    m: 嵌入维度
    r: 相似性阈值（相对于标准差的比例）

    返回:
    sample_entropy: 样本熵值
    """
    if len(time_series) < m + 1:
        return np.nan

    # 标准化数据
    y = np.array(time_series)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)

    r_threshold = r * np.std(y)

    def count_matches(template, m_val):
        """计算匹配模板的数量"""
        count = 0
        for i in range(len(template) - m_val + 1):
            for j in range(i + 1, len(template) - m_val + 1):
                # 计算Chebyshev距离
                dist = np.max(np.abs(template[i:i+m_val] - template[j:j+m_val]))
                if dist <= r_threshold:
                    count += 1
        return count

    # 计算m维和m+1维的匹配数
    N = len(y)
    B = count_matches(y, m)
    A = count_matches(y, m + 1)

    if B == 0 or A == 0:
        return np.nan

    # 样本熵计算
    sample_entropy = -np.log(A / B)

    return sample_entropy

def calculate_permutation_entropy(time_series, order=3, delay=1):
    """
    计算排列熵（Permutation Entropy），用于衡量时间序列的复杂性

    排列熵算法：
    1. 将时间序列重构为(order, delay)相空间
    2. 对每个向量计算排列模式
    3. 计算各模式的概率分布
    4. 计算Shannon熵

    排列熵越小，表示序列越有序；越大，表示越复杂

    参数:
    time_series: 时间序列数据
    order: 嵌入维度
    delay: 时间延迟

    返回:
    permutation_entropy: 排列熵
    """
    if len(time_series) < order * delay:
        return np.nan

    y = np.array(time_series)

    # 构建嵌入向量
    n_vectors = len(y) - (order - 1) * delay
    vectors = np.zeros((n_vectors, order))

    for i in range(n_vectors):
        for j in range(order):
            vectors[i, j] = y[i + j * delay]

    # 计算排列模式
    permutations = []
    for vec in vectors:
        # 获取排序索引
        sorted_indices = np.argsort(vec)
        # 转换为排列模式（从0开始的连续整数）
        perm = np.empty_like(sorted_indices)
        perm[sorted_indices] = np.arange(order)
        permutations.append(tuple(perm))

    # 计算概率分布
    unique_perms, counts = np.unique(permutations, return_counts=True)
    probabilities = counts / len(permutations)

    # 计算Shannon熵
    permutation_entropy = entropy(probabilities)

    # 归一化到[0,1]
    max_entropy = np.log(order)
    if max_entropy > 0:
        permutation_entropy /= max_entropy

    return permutation_entropy

def adf_stationarity_test(time_series, significance_level=0.05):
    """
    ADF检验（Augmented Dickey-Fuller Test）用于检验时间序列的平稳性

    ADF检验算法：
    1. 建立回归方程：Δy_t = α + βt + γy_{t-1} + δ₁Δy_{t-1} + ... + δₖΔy_{t-k} + ε_t
    2. 原假设H0: γ = 0（存在单位根，非平稳）
    3. 备择假设H1: γ < 0（平稳）
    4. 计算检验统计量并与临界值比较

    参数:
    time_series: 时间序列数据
    significance_level: 显著性水平

    返回:
    dict: 包含检验结果
    """
    if len(time_series) < 10:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'critical_values': {},
            'is_stationary': False,
            'test_result': '数据点不足'
        }

    try:
        result = adfuller(time_series, autolag='AIC')

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < significance_level,
            'test_result': '平稳' if result[1] < significance_level else '非平稳'
        }
    except:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'critical_values': {},
            'is_stationary': False,
            'test_result': '检验失败'
        }

def calculate_seasonal_decomposition(time_series, period=None):
    """
    时间序列季节性分解

    分解算法：
    1. 趋势提取：使用移动平均
    2. 季节性提取：计算季节性因子
    3. 残差计算：原始数据减去趋势和季节性

    参数:
    time_series: 时间序列数据
    period: 季节性周期（如果为None，自动检测）

    返回:
    dict: 包含分解结果
    """
    if len(time_series) < 2 * (period or 7):
        return {
            'trend': np.array([]),
            'seasonal': np.array([]),
            'residual': np.array([]),
            'decomposition_success': False
        }

    try:
        # 如果没有指定周期，尝试自动检测
        if period is None:
            # 简单的方法：假设周期为数据长度的1/4到1/10之间
            period = max(2, min(len(time_series) // 10, len(time_series) // 4))

        # 使用简单的移动平均分解
        y = np.array(time_series)

        # 趋势：使用周期长度的移动平均
        trend = np.convolve(y, np.ones(period)/period, mode='valid')
        # 填充边界
        trend = np.concatenate([
            np.full(period//2, trend[0]),
            trend,
            np.full(len(y) - len(trend) - period//2, trend[-1])
        ])

        # 季节性：计算每个周期位置的平均偏差
        seasonal = np.zeros_like(y)
        for i in range(period):
            mask = np.arange(i, len(y), period)
            if len(mask) > 0:
                seasonal[i::period] = np.mean(y[mask] - trend[mask])

        # 残差
        residual = y - trend - seasonal

        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'period': period,
            'decomposition_success': True
        }
    except:
        return {
            'trend': np.array([]),
            'seasonal': np.array([]),
            'residual': np.array([]),
            'decomposition_success': False
        }

def comprehensive_advanced_time_series_analysis(df, time_col='time', columns=None):
    """
    综合高级时间序列分析

    参数:
    df: 输入DataFrame
    time_col: 时间列名
    columns: 要分析的列名列表

    返回:
    analysis_df: 包含高级分析结果的DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col in columns:
            columns.remove(time_col)

    analysis_results = []

    for col in columns:
        y = df[col].dropna().values

        if len(y) < 20:  # 需要足够的数据点
            continue

        result = {'column': col}

        # Hurst指数
        result['hurst_exponent'] = calculate_hurst_exponent(y)

        # 样本熵
        result['sample_entropy'] = calculate_sample_entropy(y)

        # 排列熵
        result['permutation_entropy'] = calculate_permutation_entropy(y)

        # ADF平稳性检验
        adf_result = adf_stationarity_test(y)
        result.update({f'adf_{k}': v for k, v in adf_result.items()})

        # 季节性分解
        decomp_result = calculate_seasonal_decomposition(y)
        result['seasonal_decomposition_success'] = decomp_result['decomposition_success']
        if decomp_result['decomposition_success']:
            result['seasonal_strength'] = np.std(decomp_result['seasonal']) / (np.std(y) + 1e-12)
            result['trend_strength'] = np.std(decomp_result['trend']) / (np.std(y) + 1e-12)
            result['noise_ratio'] = np.std(decomp_result['residual']) / (np.std(y) + 1e-12)

        analysis_results.append(result)

    return pd.DataFrame(analysis_results)

# 应用高级时间序列分析
print("进行高级时间序列分析...")
try:
    if len(merged_beam) > 20:
        advanced_ts_analysis = comprehensive_advanced_time_series_analysis(
            merged_beam,
            time_col='time',
            columns=valued_features.columns.tolist()[:3]  # 限制分析前3个特征
        )
        print(f"高级时间序列分析完成，共分析了 {len(advanced_ts_analysis)} 个特征")
        print("分析结果预览:")
        print(advanced_ts_analysis.head())
    else:
        print("数据点不足，无法进行高级时间序列分析")
except Exception as e:
    print(f"高级时间序列分析失败: {e}")

# %%
# 时间序列分析统计量总结

"""
# 时间序列分析统计量算法说明

## 1. Hurst指数 (Hurst Exponent)
**算法原理**: 通过计算不同时间尺度的R/S统计量来判断时间序列的长期记忆性

**计算步骤**:
1. 将时间序列分割为多个子序列
2. 对每个子序列计算累积偏差的范围R和标准差S
3. 计算log(R/S)与log(子序列长度)的线性关系
4. Hurst指数H = 线性拟合的斜率

**解释**:
- H < 0.5: 反持久性（均值回归）
- H = 0.5: 随机游走
- H > 0.5: 持久性（长期记忆）

## 2. 样本熵 (Sample Entropy)
**算法原理**: 衡量时间序列的复杂性和规律性

**计算步骤**:
1. 将时间序列分割为m维向量
2. 计算向量之间的Chebyshev距离
3. 计算匹配模板的比例
4. 样本熵 = -ln(匹配比例)

**解释**:
- 值越小：序列越有规律
- 值越大：序列越随机/复杂

## 3. 排列熵 (Permutation Entropy)
**算法原理**: 通过分析时间序列的排列模式来衡量复杂性

**计算步骤**:
1. 将时间序列重构为(order, delay)相空间
2. 对每个向量计算排列模式
3. 计算各模式的概率分布
4. 计算归一化的Shannon熵

**解释**:
- 值越小：序列越有序
- 值越大：序列越复杂

## 4. ADF检验 (Augmented Dickey-Fuller Test)
**算法原理**: 检验时间序列的平稳性

**回归方程**:
Δy_t = α + βt + γy_{t-1} + δ₁Δy_{t-1} + ... + δₖΔy_{t-k} + ε_t

**假设检验**:
- H0: γ = 0（存在单位根，非平稳）
- H1: γ < 0（平稳序列）

## 5. 季节性分解 (Seasonal Decomposition)
**算法原理**: 将时间序列分解为趋势、季节性和残差成分

**计算步骤**:
1. 趋势提取：使用移动平均
2. 季节性提取：计算季节性因子
3. 残差计算：原始数据减去趋势和季节性

**强度指标**:
- 季节性强度 = std(季节性) / std(原始序列)
- 趋势强度 = std(趋势) / std(原始序列)
- 噪声比例 = std(残差) / std(原始序列)

## 6. 自相关分析 (Autocorrelation Analysis)
**算法原理**: 分析时间序列在不同滞后期的相关性

**主要指标**:
- ACF (自相关函数): 直接相关性
- PACF (偏自相关函数): 去除中间变量后的相关性
- 自相关强度: 前几个滞后的平均绝对相关系数

## 7. 趋势分析 (Trend Analysis)
**算法原理**: 检测和量化时间序列的长期变化趋势

**方法**:
- 线性趋势: 一阶多项式拟合
- 多项式趋势: 高阶多项式拟合
- 趋势强度分类: 基于R²值的强弱判断

## 8. 周期分析 (Periodicity Analysis)
**算法原理**: 通过频谱分析检测周期性成分

**方法**:
- FFT (快速傅里叶变换)
- Lomb-Scargle周期图（适用于非均匀采样）
- 周期强度评估

这些统计量共同提供了时间序列的全面特征描述，包括平稳性、复杂性、记忆性、周期性和趋势性等多个维度。
"""