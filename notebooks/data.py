# %%
import pandas as pd

import numpy as np
from scipy.ndimage import median_filter, generic_filter
from scipy import fftpack
from scipy.fft import fft2, ifftshift, fftshift
from scipy.interpolate import RegularGridInterpolator
import cv2

# %%
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
# %%
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
# %%
# 计算BPP和M²
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