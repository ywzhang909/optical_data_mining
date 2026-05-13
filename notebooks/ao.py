# %%
import sys
from pathlib import Path

sys.path.append('../ui')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

from analysis.image.common import read_tiff_to_numpy
from analysis.optical_analysis.beam_analysis import (
    gaussian, fitting_gaussian, center_of_mass_numpy, d4sigma, pib_ratio,
    calculate_xy_diameters,
)

FOCAL_CAM_PIXEL = 2.9e-6
AXIS_CAM_PIXEL = 5.5e-6

plt.gray()
# %%
axis_image = Image.open(r"D:\Projects\TIFO\data-mining\data\AO调试数据\20260326\调控区image\光轴image\（AO闭环）20260326 17：09：49(全子束-40W-平顶-5s)\20260326 17：09：49.79.jpg")
axis_array = np.array(axis_image)
h,w = axis_array.shape
yv, xv = np.mgrid[0:h, 0:w].astype(np.float64)

cx, cy = center_of_mass_numpy(axis_array, xv, yv, 3)
print(cx,cy)
plt.imshow(axis_image)
plt.scatter(cx, cy, c='red')
# %%

axis_imgs_dir = Path('../data/AO调试数据/20260326/数控区-光轴相机/20260326 17：02：43AO')
pupil_imgs_dir = Path('../data/AO调试数据/20260326/数控区-光瞳相机/20260326 17：02：43AO')
axis_imgs = pd.DataFrame(list(axis_imgs_dir.glob('*.TIFF')), columns=['path'])
pupil_imgs = pd.DataFrame(list(pupil_imgs_dir.glob('*.TIFF')), columns=['path'])
len(axis_imgs), len(pupil_imgs)

# %%
axis_imgs['img_array'] = axis_imgs['path'].apply(read_tiff_to_numpy)
pupil_imgs['img_array'] = pupil_imgs['path'].apply(read_tiff_to_numpy)

fig, [ax1, ax2] = plt.subplots(1,2)
ax1.imshow(axis_imgs.iloc[0].img_array)
ax2.imshow(pupil_imgs.iloc[0].img_array)
plt.show()
# %%
def process_time_columns(df, c='path'):
    """
    处理DataFrame中的时间相关列
    
    Args:
        df (pd.DataFrame): 要处理的DataFrame
    
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
    df = df.copy()
    # 添加time列
    df['time'] = df[c].apply(lambda x: x.stem)
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
    elif denoise_method == '1_e':
        df['black'] = df['img_array'].apply(lambda x: np.max(x)/np.e)
    
    # 去噪后的图像
    black_threshold = df['black'].max()
    df['denoise_img_array'] = df.apply(lambda x: np.where(x['img_array'] > black_threshold, x['img_array'] - black_threshold, 0), axis=1)
    
    # 计算总强度
    df['intensity'] = df['denoise_img_array'].apply(np.sum)
    
    return df

pupil_beam = process_time_columns(process_image_data(pupil_imgs))
axis_beam = process_time_columns(process_image_data(axis_imgs, denoise_method='1_e'))

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

def d4sigma_feature_extract(df: pd.DataFrame, pixel_size_um=1.0):
    d4sigma_features = df.apply(lambda x: d4sigma(x['denoise_img_array'], pixel_size_um), axis=1, result_type='expand')
    d4sigma_features['avg_sigma2'] = np.sqrt(d4sigma_features['D_x'] * d4sigma_features['D_y'])
    return pd.merge(df, d4sigma_features, left_index=True, right_index=True)

valid_axis_beam = d4sigma_feature_extract(axis_beam, AXIS_CAM_PIXEL)
valid_pupil_beam = d4sigma_feature_extract(pupil_beam, FOCAL_CAM_PIXEL)

# %%
# PIB 占比
def pib_ratio(img, center, r=5.0):
    """
    计算 PIB 占比
    
    Args:
        img (np.ndarray): 输入图像（2D数组）
        center (tuple): 中心坐标 (x, y)
        r (float): 半径（默认5.0）
    
    Returns:
        float: PIB 占比
    """
    cx, cy = center
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    pib_intensity = img[mask].sum()
    total_intensity = img.sum()
    return pib_intensity / total_intensity

pib_ratio = valid_axis_beam.apply(lambda x: pib_ratio(x['img_array'], (x['center_x'], x['center_y'])), axis=1)
valid_axis_beam['pib_ratio'] = pib_ratio

pib_ratio
# %%
# 高斯拟合
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

def calculate_xy_diameters(image, center_x, center_y, pix_size=1.0):
    """
    Calculate the diameters at y = 1/e + b in x and y directions.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.

    Returns:
        tuple: A tuple containing the x-direction diameter and y-direction diameter.
    """
    # Extract data for x and y directions
    y_data = image[:, int(center_x)]
    x_data = image[int(center_y), :]

    # Calculate diameters
    (mu, sigma, A, b), conv = fitting_gaussian(x_data)
    x_diameter = calculate_diameter(sigma)
    (mu, sigma, A, b), conv = fitting_gaussian(y_data)
    y_diameter = calculate_diameter(sigma)

    return {'gaussian_dia_x(um)': x_diameter*pix_size, 'gaussian_dia_y(um)': y_diameter*pix_size}

guassian_dia = valid_axis_beam.apply(
    lambda x: calculate_xy_diameters(x['img_array'], x['center_x'], x['center_y'], pix_size=AXIS_CAM_PIXEL),
    axis=1, result_type='expand'
)

valid_axis_beam = pd.merge(valid_axis_beam, guassian_dia, left_index=True, right_index=True)
guassian_dia

# %%
def shift_to_center_fft(image, cx, cy):
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
    dx = w//2 - cx
    dy = h//2 - cy
    
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
    return np.where(shifted < 1e-3, 0, shifted)

valid_pupil_beam['shifted_img_array'] = valid_pupil_beam.apply(
    lambda x: shift_to_center_fft(x['denoise_img_array'], x['center_x'], x['center_y']), axis=1
)
valid_axis_beam['shifted_img_array'] = valid_axis_beam.apply(
    lambda x: shift_to_center_fft(x['denoise_img_array'], x['center_x'], x['center_y']), axis=1
)

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
    
merged_beam.describe()
# %%
def fnr3(Ex, input_pixel_size, output_pixel_size, zz, lambda_m):
    """
    菲涅尔衍射积分（向量化实现 - 矩阵乘法）
    
    参数:
        Ex (np.ndarray): 输入光场复振幅，尺寸 [Ny1, Nx1] (行对应 y，列对应 x)
        input_pixel_size_um (float): 输入平面像素尺寸 [微米]
        output_pixel_size_um (float): 输出平面像素尺寸 [微米]
        output_shape (tuple): 输出光场的形状 (Ny2, Nx2)
        zz (float): 传播距离 [米]
        lambda_um (float): 波长 [微米]
    
    返回:
        Ex2 (np.ndarray): 输出光场复振幅，尺寸 [Ny2, Nx2]
    """
    dx1 = input_pixel_size
    dy1 = dx1  # 假设正方形像素
    dx2 = output_pixel_size
    dy2 = dx2  # 假设正方形像素

    k0 = 2 * np.pi / lambda_m

    # 获取输入和输出光场的尺寸
    Ny1, Nx1 = Ex.shape
    Ny2, Nx2 = Ex.shape

    # ----- 在输入平面生成坐标网格 (原点在中心) -----
    x1v = (np.arange(Nx1) - Nx1 // 2) * dx1  # [Nx1,]
    y1v = (np.arange(Ny1) - Ny1 // 2) * dy1  # [Ny1,]

    # ----- 在输出平面生成坐标网格 (原点在中心) -----
    x2v = (np.arange(Nx2) - Nx2 // 2) * dx2  # [Nx2,]
    y2v = (np.arange(Ny2) - Ny2 // 2) * dy2  # [Ny2,]

    # ----- 输入平面二次相位因子（广播计算）-----
    phase_in = np.exp(1j * k0 / (2 * zz) * (x1v[np.newaxis, :]**2 + y1v[:, np.newaxis]**2))  # [Ny1, Nx1]
    Ex_hat = Ex * phase_in    # 调制后的输入场

    # ----- 对 y 方向做傅里叶变换（积分）-----
    F_y = np.exp(-1j * 2 * np.pi / (lambda_m * zz) * np.outer(y1v, y2v))  # [Ny1, Ny2]
    temp = (F_y.T @ Ex_hat) * dy1  # [Ny2, Nx1]

    # ----- 对 x 方向做傅里叶变换（积分）-----
    F_x = np.exp(-1j * 2 * np.pi / (lambda_m * zz) * np.outer(x1v, x2v).T)  # [Nx1, Nx2]
    # 注意：这里遵循原始MATLAB代码，不乘以 dx1
    Ex2 = temp @ F_x  # [Ny2, Nx2]

    # ----- 输出平面二次相位因子及常数因子-----
    phase_out = np.exp(1j * k0 * zz + 1j * k0 / (2 * zz) * (x2v[np.newaxis, :]**2 + y2v[:, np.newaxis]**2))  # [Ny2, Nx2]
    Ex2 = Ex2 * phase_out / (1j * lambda_m * zz)
    
    return Ex2
# %%
def calculate_strehl_ratio_with_energy_conservation(
    pupil_img,
    focus_img,
    f_m=3,
    wavelength_m=1064e-9,
    background_subtract=False,
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
    H, W = pupil_img.shape
    crop_size = (W - H) // 2
    
    pupil_img = pupil_img[:,crop_size:-crop_size]

    # ----------------------------
    # 3. 构建理想复振幅（假设相位为0）
    # ----------------------------
    ideal_focus_intensity = np.abs(fnr3(
        pupil_img, AXIS_CAM_PIXEL*32, AXIS_CAM_PIXEL, 3, wavelength_m
    ))
    # ----------------------------
    # 8. 【关键】能量守恒校准
    # ----------------------------
    total_energy_actual = np.sum(focus_img)
    total_energy_ideal = np.sum(ideal_focus_intensity)

    if total_energy_ideal == 0:
        raise ValueError("理想光斑总能量为零，请检查输入光瞳图像。")

    # 缩放理想光斑，使其总能量 = 实际总能量
    scaling_factor = total_energy_actual / total_energy_ideal
    ideal_energy_matched = ideal_focus_intensity * scaling_factor

    # ----------------------------
    # 9. 计算斯特列尔比
    # ----------------------------
    peak_actual = np.max(focus_img)
    peak_ideal = np.max(ideal_energy_matched)

    strehl = peak_actual / (peak_ideal)

    return strehl, ideal_energy_matched

strehl_results_with_scaler = merged_beam.apply(
    lambda row: calculate_strehl_ratio_with_energy_conservation(row['shifted_img_array_pupil'], row['shifted_img_array_axis']),
    axis=1,
    result_type='expand'
)
strehl_results_with_scaler.columns = ['strehl_ratio', 'ideal_matched']
# merged_beam = pd.merge(merged_beam, strehl_results_with_scaler, left_index=True, right_index=True)
strehl_results_with_scaler

# %% [markdown]
# ## 计算BPP和M²

# %%
def calculate_bpp_from_pupil_and_focal(
    pupil_diameter_mm,
    focal_diameter_mm,
    focal_length_mm = 3e3,
    wavelength_nm=1064.0,
    beam_expansion_ratio=1.0/32,
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
    # bpp_theoretical_input = None
    # if pupil_diameter_input_mm is not None:
    #     w_input = pupil_diameter_input_mm / 2.0
    #     bpp_theoretical_input = w_input * theta_mrad
    
    # 衍射极限 BPP (mm·mrad) = λ(μm) / π
    # wavelength_um = wavelength_nm * 1e-3   # nm → μm
    # bpp_diffraction_mm_mrad = wavelength_um / np.pi
    
    # M2 = bpp_mm_mrad / bpp_diffraction_mm_mrad
    
    # 考虑缩束比的影响
    # M2_corrected = M2 / effective_expansion_ratio if effective_expansion_ratio != 0 else np.nan
    
    return {
        "BPP_mm_mrad": bpp_mm_mrad,
    }
    
bpp_results = merged_beam.swifter.apply(
    lambda row: calculate_bpp_from_pupil_and_focal(row['avg_sigma2_pupil'], row['avg_sigma2_axis']),
    axis=1, result_type='expand')

merged_beam = pd.merge(merged_beam, bpp_results, left_index=True, right_index=True)
bpp_results
