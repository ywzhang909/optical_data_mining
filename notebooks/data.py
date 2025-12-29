# %%
import os
from typing import Literal

import pandas as pd

from PIL import Image
import numpy as np
from scipy.ndimage import center_of_mass, fourier_shift, median_filter, generic_filter
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.signal import lombscargle
from scipy import fftpack
from scipy.fft import fft2, ifftshift, fftshift, ifft2
from scipy.interpolate import RegularGridInterpolator
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
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

root_dir = Path(f'{__file__}').parent.parent / 'data'
assert root_dir.exists(), f"数据目录不存在: {root_dir}"
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

def adaptive_background_subtraction(
    img,
    kernel_size=21,
    sigma_factor=3.0,
    min_background_percentile=10,
    preserve_energy=True
):
    """
    自适应背景扣除与降噪，适用于激光光斑图像。
    
    参数:
        img: 输入图像 (2D array, float or uint)
        kernel_size: 背景估计窗口大小（奇数，建议 15~51）
        sigma_factor: 判定信号的阈值倍数（通常 2.5~4.0）
        min_background_percentile: 全局最小背景强度百分位（防过扣除）
        preserve_energy: 是否在降噪后保持总能量（用于斯特列尔比计算）
    
    返回:
        denoised_img: 降噪并背景扣除后的图像（≥0）
        background: 估计的背景图像
        mask: 有效信号区域掩码 (bool)
    """
    img = img.astype(np.float64)
    
    # ----------------------------
    # 1. 局部背景估计：滚动中位数
    # ----------------------------
    # 中位数对强信号不敏感，能逼近真实背景
    background = median_filter(img, size=kernel_size, mode='constant', cval=0)

    # 防止背景过高（例如光斑很大时）
    global_bg_floor = np.percentile(img, min_background_percentile)
    background = np.minimum(background, img)  # 背景不能高于原图
    background = np.maximum(background, global_bg_floor)  # 也不能低于全局底噪

    # ----------------------------
    # 2. 局部噪声强度估计：MAD
    # ----------------------------
    # MAD = median(|x - median(x)|) ≈ 0.6745 * σ (对高斯噪声)
    def mad_func(window):
        med = np.median(window)
        return np.median(np.abs(window - med))
    
    mad_map = generic_filter(img, mad_func, size=kernel_size, mode='constant', cval=0)
    sigma_map = mad_map / 0.6745  # 转换为标准差估计

    # 防止 sigma 过小（数值稳定）
    sigma_map = np.maximum(sigma_map, np.percentile(sigma_map, 10))

    # ----------------------------
    # 3. 构建显著性掩码
    # ----------------------------
    signal = img - background
    threshold = sigma_factor * sigma_map
    mask = signal > threshold  # 信噪比 > sigma_factor 的区域保留

    # 可选：形态学闭操作连接断裂信号
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ).astype(bool)

    # ----------------------------
    # 4. 输出降噪图像
    # ----------------------------
    denoised = np.where(mask, img - background, 0.0)
    denoised = np.maximum(denoised, 0.0)

    if preserve_energy and np.sum(denoised) > 0:
        # 可选：将被阈值切除的能量按比例加回（保守做法）
        # 更常见的是直接使用 denoised，因噪声无物理意义
        pass

    return denoised, background, mask


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

def fourier_shift_manual(image, dx, dy):
    """
    使用傅里叶移位定理实现亚像素平移。
    
    参数:
        image: 2D array
        shift: (dy, dx) —— 平移量（可为小数）
    
    返回:
        shifted_image: 平移后的图像
    """
    h, w = image.shape

    # 1. 计算 FFT
    F = fft2(image)

    # 2. 构建频率网格（归一化）
    u = np.fft.fftfreq(w).reshape(1, -1)  # [0, 1/w, ..., -1/w, ...]
    v = np.fft.fftfreq(h).reshape(-1, 1)  # 列向量

    # 3. 构造相位因子
    phase = np.exp(-2j * np.pi * (u * dx + v * dy))

    # 4. 频域相乘
    F_shifted = F * phase

    # 5. 逆变换
    shifted = np.real(ifft2(F_shifted))

    return shifted

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
# TODO zernike : 适用zernike拟合valid_pupil_beam光斑，返回zernike系数


# %%
# %%
# 二维时序处理，计算前后两个光瞳的偏移
def phase_correlation_shift(img1, img2):
    """
    使用相位相关法计算两幅图像之间的亚像素位移。
    
    Args:
        img1: 参考图像 (numpy array)
        img2: 待配准图像 (numpy array)
    
    Returns:
        dx: x方向位移 (像素)
        dy: y方向位移 (像素)
        confidence: 匹配置信度 (相关峰的高度)
    """
    # 确保图像为 float32 类型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 计算互功率谱的相位相关
    # cv2.DFT_COMPLEX_OUTPUT 表示输出是复数
    dft1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # 计算互功率谱: (F1 * F2*) / (|F1 * F2*|)
    # np.conj() 计算复共轭
    product = dft1 * np.conj(dft2)
    # 计算幅度谱并加上极小值防止除零
    magnitude = np.sqrt(product[:,:,0]**2 + product[:,:,1]**2 + 1e-12)
    # 归一化到复平面上的单位圆
    cross_power = product / magnitude[:, :, np.newaxis]
    
    # 逆傅里叶变换得到相关平面
    correlation = cv2.idft(cross_power, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # 找到相关平面中的最大值位置 (即最可能的位移)
    # minMaxLoc 返回最小值、最大值及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation)
    
    # 获取图像尺寸
    h, w = img1.shape
    
    # 亚像素精修 (质心法或抛物线拟合)
    # 这里采用简单的质心法围绕峰值寻找亚像素位置
    radius = 2 # 搜索窗口半径
    x1, y1 = max_loc
    # 边界检查
    x1 = max(radius, min(w-radius-1, x1))
    y1 = max(radius, min(h-radius-1, y1))
    
    # 提取局部区域
    patch = correlation[int(y1-radius):int(y1+radius+1), int(x1-radius):int(x1+radius+1)]
    # 创建坐标网格
    x_coords = np.arange(-radius, radius+1)
    y_coords = np.arange(-radius, radius+1)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 计算质心 (相对于峰值位置的偏移)
    # 加上一个小常数防止除零
    total = np.sum(patch) + 1e-12
    dx_sub = np.sum(X * patch) / total
    dy_sub = np.sum(Y * patch) / total
    
    # 总位移
    dx = x1 + dx_sub - (w // 2) # 转换为相对于图像中心的位移
    dy = y1 + dy_sub - (h // 2)
    
    return dx, dy, max_val

pupil_img_array = valid_pupil_beam['denoise_img_array']
continuis_img = pd.DataFrame({'prev': pupil_img_array.iloc[:-1].values,
                              'next': pupil_img_array.iloc[1:].values}, index=valid_pupil_beam['time'].iloc[1:])
def compute_phase_correlation(row):
    dx, dy, confidence = phase_correlation_shift(row['prev'], row['next'])
    return pd.Series({'shift_dx': dx, 'shift_dy': dy, 'confidence': confidence})
valid_pupil_beam = pd.concat([valid_pupil_beam, continuis_img.swifter.apply(compute_phase_correlation, axis=1)], axis=1)
# TODO 光流valid_pupil_beam光斑


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


def calculate_strehl_ratio_with_energy_conservation(
    pupil_img,
    focus_img,
    pixel_size_pupil_um=5.5,
    pixel_size_focus_um=3.45,
    N=0.2,
    f_mm=100.0,
    wavelength_um=1.064,
    background_subtract=True,
    roi_fraction=0.8
):
    """
    基于能量守恒的斯特列尔比计算。
    
    新增特性:
        - 背景扣除
        - 能量归一化（使理想与实际总能量一致）
        - 可选 ROI 避免边缘噪声影响
    
    返回:
        strehl_ratio (float)
        ideal_matched (np.ndarray): 能量匹配后的理想光斑（与 focus_img 同尺寸）
    """

    # ----------------------------
    # 1. 背景扣除（可选但推荐）
    # ----------------------------
    def subtract_background(img):
        # 使用图像边缘区域估计背景（假设中心是光斑）
        h, w = img.shape
        margin = int(min(h, w) * 0.1)
        bg = np.median(img[margin:-margin, margin:-margin])
        return np.maximum(img.astype(np.float64) - bg, 0.0)

    if background_subtract:
        pupil_img = subtract_background(pupil_img)
        focus_img = subtract_background(focus_img)

    # ----------------------------
    # 2. 物理尺度校准
    # ----------------------------
    dx_pupil_phys_um = pixel_size_pupil_um / N  # 关键：除以缩束比 N
    H, W = pupil_img.shape
    Lx_pupil_um = W * dx_pupil_phys_um
    Ly_pupil_um = H * dx_pupil_phys_um

    # ----------------------------
    # 3. 构建理想复振幅（假设相位为0）
    # ----------------------------
    pupil_field = np.sqrt(np.maximum(pupil_img, 0)) + 0j  # 复数类型

    # ----------------------------
    # 4. 计算理想聚焦光斑（透镜后焦面）
    # ----------------------------
    ideal_focus_field = fftshift(fft2(ifftshift(pupil_field)))
    ideal_focus_intensity = np.abs(ideal_focus_field)**2

    # ----------------------------
    # 5. 计算理想光斑物理网格
    # ----------------------------
    f_um = f_mm * 1000.0
    dx_ideal_um = (wavelength_um * f_um) / Lx_pupil_um
    dy_ideal_um = (wavelength_um * f_um) / Ly_pupil_um
    # 理想光斑总物理尺寸
    Lx_ideal_um = W * dx_ideal_um
    Ly_ideal_um = H * dy_ideal_um

    x_ideal = np.linspace(-Lx_ideal_um/2, Lx_ideal_um/2 - dx_ideal_um, W)
    y_ideal = np.linspace(-Ly_ideal_um/2, Ly_ideal_um/2 - dy_ideal_um, H)
    X_ideal, Y_ideal = np.meshgrid(x_ideal, y_ideal)

    # ----------------------------
    # 6. 实际光斑物理网格
    # ----------------------------
    Hf, Wf = focus_img.shape
    x_actual = (np.arange(Wf) - Wf // 2) * pixel_size_focus_um
    y_actual = (np.arange(Hf) - Hf // 2) * pixel_size_focus_um
    X_actual, Y_actual = np.meshgrid(x_actual, y_actual)

    # ----------------------------
    # 7. 插值：理想 → 实际网格
    # ----------------------------
    interp_func = RegularGridInterpolator(
        (y_ideal, x_ideal),
        ideal_focus_intensity,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    points = np.stack([Y_actual.ravel(), X_actual.ravel()], axis=-1)
    ideal_on_actual = interp_func(points).reshape(Hf, Wf)

    # ----------------------------
    # 8. 【关键】能量守恒校准
    # ----------------------------
    # 可选：使用中心 ROI 计算能量，避免边缘噪声
    if roi_fraction < 1.0:
        h_roi = int(Hf * roi_fraction)
        w_roi = int(Wf * roi_fraction)
        y_start = (Hf - h_roi) // 2
        x_start = (Wf - w_roi) // 2
        
        actual_roi = focus_img[y_start:y_start+h_roi, x_start:x_start+w_roi]
        ideal_roi = ideal_on_actual[y_start:y_start+h_roi, x_start:x_start+w_roi]
    else:
        actual_roi = focus_img
        ideal_roi = ideal_on_actual

    total_energy_actual = np.sum(actual_roi)
    total_energy_ideal = np.sum(ideal_roi)

    if total_energy_ideal == 0:
        raise ValueError("理想光斑总能量为零，请检查输入光瞳图像。")

    # 缩放理想光斑，使其总能量 = 实际总能量
    scaling_factor = total_energy_actual / total_energy_ideal
    ideal_energy_matched = ideal_on_actual * scaling_factor

    # ----------------------------
    # 9. 计算斯特列尔比
    # ----------------------------
    peak_actual = np.max(actual_roi)
    peak_ideal = np.max(ideal_energy_matched[
        y_start:y_start+h_roi, x_start:x_start+w_roi
    ]) if roi_fraction < 1.0 else np.max(ideal_energy_matched)

    strehl = peak_actual / (peak_ideal + 1e-12)

    return strehl, ideal_energy_matched

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
def compute_wavefront_gradient_tie(intensity_in_focus, intensity_defocus, 
                                 delta_z, wavelength):
    """
    基于光强传输方程 (TIE) 计算波前梯度。
    该方法需要两张图：一张在焦点，一张离焦 (defocus)。
    
    Args:
        intensity_in_focus: 焦点处的光强图像 (2D array)
        intensity_defocus: 离焦处的光强图像 (2D array)，通常离焦量为 delta_z
        delta_z: 离焦距离 (传播距离)
        wavelength: 激光波长
    
    Returns:
        gradient_x: 波前在 x 方向的梯度 (偏导数)
        gradient_y: 波前在 y 方向的梯度 (偏导数)
    """
    # 1. 预处理：背景扣除和归一化
    # 假设背景是均匀的，取边缘均值
    def preprocess(img):
        # 简单的背景扣除 (取边缘10%)
        h, w = img.shape
        margin = int(min(h, w) * 0.1)
        bg = np.mean(img[margin:-margin, margin:-margin])
        img_proc = np.maximum(img - bg, 0)
        return img_proc / np.mean(img_proc) # 归一化平均光强
    
    I0 = preprocess(intensity_in_focus)
    I1 = preprocess(intensity_defocus)
    
    # 2. 计算光强沿传播方向的导数 (dI/dz)
    # 使用中心差分近似
    dI_dz = (I1 - I0) / delta_z
    
    # 3. 计算光强的拉普拉斯算子 (用于TIE方程的分母项处理)
    # 使用频域拉普拉斯算子计算更稳定
    def laplacian_fft(img):
        h, w = img.shape
        # 生成频率坐标
        fx = np.fft.fftfreq(w).reshape(1, -1)
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        # 拉普拉斯算子在频域是 -(fx^2 + fy^2)
        k_sq = -(fx**2 + fy**2)
        # 对图像做FFT，乘以算子，再IFFT回来
        img_fft = fftpack.fft2(img)
        lap_fft = img_fft * k_sq
        return np.real(fftpack.ifft2(lap_fft))
    
    # 4. 求解泊松方程 (简化版 TIE)
    # TIE方程: -dI/dz = div(I * grad(phi))
    # 在小相位扰动下，可以近似求解 grad(phi)
    # 这里我们直接计算导致光强变化的“力”场
    # 注意：严格求解需要解泊松方程，这里为了演示梯度趋势，我们计算归一化的 dI/dz
    
    # 波前梯度与光强导数成正比 (在均匀照明假设下)
    # 实际上，我们需要解: nabla^2(phi) = - (1/I) * dI/dz
    # 这里返回的是“源项”
    source_term = - dI_dz / (I0 + 1e-6) # 加上小量防止除零
    
    # 5. (可选) 通过解泊松方程获得平滑的梯度场
    # 这里我们直接返回源项作为梯度的“指示器”
    # 如果你需要真实的连续波前，需要对 source_term 进行泊松反演
    
    return source_term

def wavefront_tie(df: pd.DataFrame, delta_z=3, wavelength=1064e-9):
    """
    对DataFrame中的每一行计算波前梯度图
    """
    df['wavefront_gradient_map'] = df.swifter.apply(
        lambda row: compute_wavefront_gradient_tie(
            row['denoise_img_array_axis'],
            row['denoise_img_array_axis_defocus'],
            delta_z,
            wavelength
        ),
        axis=1
    )
    return df

merged_beam = wavefront_tie(merged_beam, delta_z=3, wavelength=1064e-9)




# %%
# save result
# drop columns with object dtype
# saved_beam = merged_beam.drop(columns=merged_beam.select_dtypes(include=['object']).columns)
# saved_beam.to_parquet(root_dir / 'merged_beam_analysis.parquet', compression='zstd')

merged_beam = pd.read_parquet(root_dir / 'merged_beam_analysis.parquet')
# %%
import pygwalker
from notebooks.data import pupil_img_array

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
# TODO valued_features趋势、周期分析
def calculate_statistical_features(series):
    """
    计算时间序列的基础统计特征
    """
    features = {
        'mean': series.mean(),
        'std': series.std(),
        'var': series.var(),
        'min': series.min(),
        'max': series.max(),
        'skewness': series.skew(),   # 偏度
        'kurtosis': series.kurtosis(), # 峰度
        'q_05': series.quantile(0.05), # 5%分位数
        'q_95': series.quantile(0.95)  # 95%分位数
    }
    return features

def check_stationarity(series, cutoff_p=0.05):
    """
    使用 ADF 检验判断平稳性
    """
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < cutoff_p
    }
    
def decompose_series(series):
    """
    分解时间序列
    """
    # 自动推断频率或手动指定
    # 对于非日期索引，freq 需要指定具体的整数（如 12 代表年周期，7 代表周周期）
    try:
        # x must contain 2 complete cycles.
        result = seasonal_decompose(series, model='additive', period=12)
        # result.plot()
        # plt.suptitle('时间序列分解图', y=1.02)
        # plt.tight_layout()
        # plt.show()
        
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'resid': result.resid
        }
    except Exception as e:
        print("分解失败:", e)
        return {
            'trend': np.nan,
            'seasonal': np.nan,
            'resid': np.nan
        }
        


# %%
# TODO valued_features自相关分析
def acf_pacf_analysis(series, lags=40):
    """
    计算 ACF 和 PACF
    
    ACF - 自相关函数, 含义是序列与其滞后值之间的相关性。
    PACF - 偏自相关函数, 含义是序列与其滞后值之间的相关性，剔除掉中间滞后值的影响。
    通过 ACF 和 PACF 可以帮助识别时间序列模型的阶数
    """
    acf_values = acf(series.dropna(), nlags=lags)
    pacf_values = pacf(series.dropna(), nlags=lags)
    return {
        'acf_values': acf_values,
        'pacf_values': pacf_values
    }
