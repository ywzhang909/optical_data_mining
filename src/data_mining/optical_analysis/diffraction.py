"""
衍射计算模块
=============
提供光束传播和衍射计算的算法

功能:
- 菲涅尔衍射积分 (FNR3)
- FFT居中处理
- 斯特列尔比计算
"""

import numpy as np
from typing import Tuple, Optional


def crop_to_square(img: np.ndarray) -> np.ndarray:
    """
    将图像裁剪成正方形
    
    Args:
        img: 输入图像（2D数组）
    
    Returns:
        裁剪后的正方形图像
    """
    h, w = img.shape
    if h == w:
        return img
    elif h > w:
        crop_border = (h - w) // 2
        return img[crop_border:-crop_border, :]
    else:
        crop_border = (w - h) // 2
        return img[:, crop_border:-crop_border]


def shift_to_center_fft(image: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """
    使用傅里叶移位将光斑移到图像中心，并裁剪成正方形
    
    Args:
        image: 输入图像（2D数组）
        cx: 质心X坐标
        cy: 质心Y坐标
    
    Returns:
        移位并裁剪后的图像
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    dx = w//2 - cx
    dy = h//2 - cy
    
    # 傅里叶移位
    u = np.fft.fftfreq(w).reshape(1, -1)
    v = np.fft.fftfreq(h).reshape(-1, 1)
    phase = np.exp(-2j * np.pi * (u * dx + v * dy))
    
    F = np.fft.fft2(image)
    F_shifted = F * phase
    shifted = np.real(np.fft.ifft2(F_shifted))
    # 保留非负值，去除数值误差导致的小负值
    shifted = np.clip(shifted, 0, None)
    
    # 裁剪成正方形
    return crop_to_square(shifted)


def fnr3(
    Ex: np.ndarray,
    input_pixel_size: float,
    output_pixel_size: float,
    zz: float,
    lambda_m: float,
    focal_length_m: Optional[float] = None
) -> np.ndarray:
    """
    菲涅尔衍射积分（向量化实现）
    
    参数:
        Ex: 输入光场复振幅
        input_pixel_size: 输入平面像素尺寸 (米)
        output_pixel_size: 输出平面像素尺寸 (米)
        zz: 传播距离 (米)
        lambda_m: 波长 (米)
        focal_length_m: 聚焦透镜焦距（可选），如果提供则在输入平面添加聚焦相位
    
    返回:
        输出光场复振幅
    """
    dx1 = input_pixel_size
    dy1 = dx1
    dx2 = output_pixel_size
    dy2 = dx2

    k0 = 2 * np.pi / lambda_m

    Ny1, Nx1 = Ex.shape
    Ny2, Nx2 = Ex.shape

    # 输入平面坐标
    x1v = (np.arange(Nx1) - Nx1 // 2) * dx1
    y1v = (np.arange(Ny1) - Ny1 // 2) * dy1

    # 输出平面坐标
    x2v = (np.arange(Nx2) - Nx2 // 2) * dx2
    y2v = (np.arange(Ny2) - Ny2 // 2) * dy2

    # 添加聚焦相位（透镜相位）
    if focal_length_m is not None and focal_length_m > 0:
        # 聚焦相位: exp(-i * k * r² / (2f))
        r1_sq = x1v[np.newaxis, :]**2 + y1v[:, np.newaxis]**2
        lens_phase = np.exp(-1j * k0 * r1_sq / (2 * focal_length_m))
        Ex = Ex * lens_phase

    # 输入平面二次相位因子
    phase_in = np.exp(1j * k0 / (2 * zz) * (x1v[np.newaxis, :]**2 + y1v[:, np.newaxis]**2))
    Ex_hat = Ex * phase_in

    # y方向傅里叶变换
    F_y = np.exp(-1j * 2 * np.pi / (lambda_m * zz) * np.outer(y1v, y2v))
    temp = (F_y.T @ Ex_hat) * dy1

    # x方向傅里叶变换
    F_x = np.exp(-1j * 2 * np.pi / (lambda_m * zz) * np.outer(x1v, x2v).T)
    Ex2 = temp @ F_x

    # 输出平面二次相位因子
    phase_out = np.exp(1j * k0 * zz + 1j * k0 / (2 * zz) * (x2v[np.newaxis, :]**2 + y2v[:, np.newaxis]**2))
    Ex2 = Ex2 * phase_out / (1j * lambda_m * zz)
    
    return Ex2


def calculate_strehl_ratio_with_energy_conservation(
    pupil_img: np.ndarray,
    focus_img: np.ndarray,
    f_m: float = 3,
    wavelength_m: float = 1064e-9,
    focal_length_m: float = 3.0,
    input_pixel_size: Optional[float] = None,
    output_pixel_size: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    基于能量守恒的斯特列尔比计算
    
    参数:
        pupil_img: 光瞳图像（复振幅的强度）
        focus_img: 焦平面图像
        f_m: 传播距离（米）
        wavelength_m: 波长（米）
        focal_length_m: 聚焦透镜焦距（米）
        input_pixel_size: 输入像素尺寸（米）
        output_pixel_size: 输出像素尺寸（米）
    
    返回:
        tuple: (strehl_ratio, ideal_energy_matched)
            - strehl_ratio: 斯特列尔比
            - ideal_energy_matched: 能量校准后的理想焦斑
    """
    # 裁剪成正方形
    pupil_img = crop_to_square(pupil_img)
    focus_img = crop_to_square(focus_img)

    # 默认像素尺寸
    if input_pixel_size is None:
        input_pixel_size = 2.9e-6
    if output_pixel_size is None:
        output_pixel_size = 5.5e-6

    # 构建理想复振幅（将光瞳强度作为振幅，使用聚焦相位）
    # 注意：光瞳图像被用作复振幅的振幅，相位设为0，然后应用聚焦透镜的相位
    ideal_focus_intensity = np.abs(fnr3(
        pupil_img, 
        input_pixel_size, 
        output_pixel_size, 
        f_m, 
        wavelength_m,
        focal_length_m=focal_length_m  # 传入聚焦焦距
    ))

    # 能量守恒校准
    total_energy_actual = np.sum(focus_img)
    total_energy_ideal = np.sum(ideal_focus_intensity)

    if total_energy_ideal == 0:
        return 0.0, np.zeros_like(focus_img)

    scaling_factor = total_energy_actual / total_energy_ideal
    ideal_energy_matched = ideal_focus_intensity * scaling_factor

    # 斯特列尔比
    peak_actual = np.max(focus_img)
    peak_ideal = np.max(ideal_energy_matched)

    strehl = peak_actual / peak_ideal if peak_ideal > 0 else 0

    return strehl, ideal_energy_matched


def propagate_through_lens(
    field: np.ndarray,
    pixel_size: float,
    wavelength: float,
    focal_length: float
) -> np.ndarray:
    """
    通过透镜传播计算焦平面场
    
    参数:
        field: 输入光场（复振幅）
        pixel_size: 像素尺寸 (米)
        wavelength: 波长 (米)
        focal_length: 透镜焦距 (米)
    
    返回:
        焦平面光场强度
    """
    # 使用FNR3传播到焦平面
    output_field = fnr3(
        field,
        pixel_size,
        pixel_size,  # 输出像素尺寸与输入相同
        focal_length,
        wavelength,
        focal_length_m=focal_length
    )
    
    return np.abs(output_field)


def angular_spectrum_propagation(
    field: np.ndarray,
    pixel_size: float,
    wavelength: float,
    distance: float
) -> np.ndarray:
    """
    角谱传播法
    
    参数:
        field: 输入光场（复振幅）
        pixel_size: 像素尺寸 (米)
        wavelength: 波长 (米)
        distance: 传播距离 (米)
    
    返回:
        传播后的光场
    """
    field = np.asarray(field, dtype=complex)
    h, w = field.shape
    
    # 频率坐标
    fx = np.fft.fftfreq(w, pixel_size).reshape(1, -1)
    fy = np.fft.fftfreq(h, pixel_size).reshape(-1, 1)
    
    # 传递函数
    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    
    # 避免负数平方根
    kz_sq = k**2 - kx**2 - ky**2
    kz = np.where(kz_sq >= 0, np.sqrt(kz_sq), 0)
    
    transfer_function = np.exp(1j * kz * distance)
    
    # 傅里叶变换、乘以传递函数、逆变换
    F = np.fft.fft2(field)
    F_propagated = F * transfer_function
    result = np.fft.ifft2(F_propagated)
    
    return np.abs(result)
