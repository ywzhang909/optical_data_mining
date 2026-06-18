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
from loguru import logger

try:
    from aotools.opticalpropagation import twoStepFresnel
except ImportError as exc:
    twoStepFresnel = None
    _AOTOOLS_IMPORT_ERROR = exc
else:
    _AOTOOLS_IMPORT_ERROR = None


def _require_aotools() -> None:
    if twoStepFresnel is None:
        raise RuntimeError(
            "aotools is required for fnr3, propagate_through_lens, and Strehl propagation. "
            "Install the project dependencies or repair the local aotools/numba installation."
        ) from _AOTOOLS_IMPORT_ERROR


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
    dx = w // 2 - cx
    dy = h // 2 - cy

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


def suggest_output_grid_for_focusing(
    wavelength: float,
    focal_length: float | None = None,
    NA: float | None = None,
    f_number: float | None = None,
    input_aperture_diameter: float | None = None,
    pixels_per_min_feature: int = 3,
    num_pixels: int = 1024,
) -> tuple[float, float]:
    """
    根据 NA、f/# 或焦距+孔径，自动建议聚焦仿真的输出像素尺寸和物理视场半宽。

    参数:
        wavelength (float): 波长 λ (米)
        focal_length (float, optional): 透镜焦距 f (米)
        NA (float, optional): 数值孔径，NA = n * sin(θ)，空气中 n≈1
        f_number (float, optional): f/# = f / D
        input_aperture_diameter (float, optional): 入瞳直径 D (米)
        pixels_per_min_feature (int): 每个最小光斑特征（如艾里斑半高宽）的像素数，建议 2~4
        num_pixels (int): 输出阵列总像素数（单边），用于计算总视场

    返回:
        output_pixel_size (float): 推荐的输出平面像素尺寸 (米)
        half_FOV (float): 推荐的输出平面半视场（即从中心到边缘的物理距离，米）
    """
    # Step 1: 确定 NA
    if NA is not None:
        pass
    elif f_number is not None:
        # 对于小角度，NA ≈ 1 / (2 * f/#)
        NA = 1.0 / (2 * f_number)
    elif focal_length is not None and input_aperture_diameter is not None:
        # NA = sin(arctan(D/(2f))) ≈ D/(2f) 当 D << f
        NA = np.sin(np.arctan(input_aperture_diameter / (2 * focal_length)))
    else:
        raise ValueError("Must provide one of: NA, f_number, or (focal_length + input_aperture_diameter)")

    assert NA
    if NA <= 0 or NA >= 1.0:
        raise ValueError(f"Invalid NA: {NA}. Should be in (0, ~0.95) for air.")

    # Step 2: 计算衍射极限光斑尺寸
    # 艾里斑第一零点半径（Airy disk radius to first zero）:
    airy_radius = 1.22 * wavelength / (2 * NA)  # = 0.61 * λ / NA

    # 半高全宽（FWHM）近似（对 Airy pattern）:
    airy_fwhm = 0.514 * wavelength / NA  # 经验公式

    # 或者对高斯光束（理想聚焦）: w0 = λ * f / (π * w_in) = λ / (π * NA) （当 NA 小时）
    gaussian_waist = wavelength / (np.pi * NA)

    # 我们取最严格的（最小的特征尺寸）作为参考
    min_feature_size = min(airy_fwhm, gaussian_waist)

    # Step 3: 根据奈奎斯特采样确定像素尺寸
    # 至少每 min_feature_size 有 `pixels_per_min_feature` 个像素
    output_pixel_size = min_feature_size / pixels_per_min_feature

    # Step 4: 确定视场（FOV）
    # 建议视场至少覆盖 3~5 倍艾里斑半径，以看到旁瓣
    FOV_radius = 3 * airy_radius  # 半视场（从中心到边缘）

    # 但也要确保 FOV 能被 num_pixels 整除
    # 实际半视场 = (num_pixels // 2) * output_pixel_size
    actual_half_FOV = (num_pixels // 2) * output_pixel_size

    # 如果实际 FOV 太小，扩大像素尺寸以覆盖所需视场
    if actual_half_FOV < FOV_radius:
        # 重新设定 output_pixel_size 以覆盖 FOV_radius
        output_pixel_size = FOV_radius / (num_pixels // 2)
        # 注意：此时采样率可能略低于理想，但保证视场完整

    half_FOV = (num_pixels // 2) * output_pixel_size

    return output_pixel_size, half_FOV


def fnr3(
    Ex: np.ndarray,
    input_pixel_size: float,
    output_pixel_size: float,
    zz: float,
    lambda_m: float,
    focal_length_m: float | None = None,
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
    if zz <= 0:
        raise ValueError("Propagation distance zz must be positive.")
    focal_length_m = focal_length_m if focal_length_m else zz

    dx1 = input_pixel_size
    dy1 = dx1
    dx2 = output_pixel_size
    dy2 = dx2

    k0 = 2 * np.pi / lambda_m
    Ny, Nx = Ex.shape

    # 输入平面坐标（中心对齐）
    x1v = (np.arange(Nx) - Nx // 2) * dx1
    y1v = (np.arange(Ny) - Ny // 2) * dy1

    # 输出平面坐标（同样像素数，但可不同物理尺寸）
    x2v = (np.arange(Nx) - Nx // 2) * dx2
    y2v = (np.arange(Ny) - Ny // 2) * dy2

    # 调试信息
    logger.debug(f"Input FOV: [{x1v[0]:.6e}, {x1v[-1]:.6e}] m")
    logger.debug(f"Output FOV: [{x2v[0]:.6e}, {x2v[-1]:.6e}] m")

    # 添加聚焦相位（薄透镜模型，精确球面波）
    r1_sq = x1v[np.newaxis, :] ** 2 + y1v[:, np.newaxis] ** 2
    phase = (2 * np.pi / lambda_m) * (focal_length_m - np.sqrt(focal_length_m**2 + r1_sq))
    lens_phase = np.exp(1j * phase)
    Ex = Ex * lens_phase
    logger.debug(f"Max lens phase shift: {np.max(np.abs(phase)):.3f} rad")

    # # 输入二次相位因子
    # phase_in = np.exp(1j * k0 / (2 * zz) * (x1v[np.newaxis, :]**2 + y1v[:, np.newaxis]**2))
    # Ex_hat = Ex * phase_in

    # K = 2 * np.pi / (lambda_m * zz)
    # F_y = np.exp(-1j * K * np.outer(y1v, y2v))   # shape (Ny, Ny_out) = (Ny, Ny)
    # F_x = np.exp(-1j * K * np.outer(x1v, x2v))   # shape (Nx, Nx_out) = (Nx, Nx)

    # # 执行分离变量的菲涅尔积分：Ex2 = F_y^T @ Ex_hat @ F_x
    # temp = F_y.T @ Ex_hat          # (Ny, Ny) @ (Ny, Nx) -> (Ny, Nx)
    # Ex2 = temp @ F_x               # (Ny, Nx) @ (Nx, Nx) -> (Ny, Nx)

    # # 输出二次相位因子 + 常数因子
    # phase_out = np.exp(1j * k0 * zz + 1j * k0 / (2 * zz) * (x2v[np.newaxis, :]**2 + y2v[:, np.newaxis]**2))
    # Ex2 = Ex2 * phase_out * (dx1 * dy1) / (1j * lambda_m * zz)

    # logger.debug(f"Input power: {np.sum(np.abs(Ex)**2) * dx1 * dy1:.6e}")
    # logger.debug(f"Output power: {np.sum(np.abs(Ex2)**2) * dx2 * dy2:.6e}")
    _require_aotools()
    assert twoStepFresnel is not None
    Ex2 = twoStepFresnel(Ex, lambda_m, dx1, dx2, zz)

    return Ex2


def calculate_strehl_ratio_with_energy_conservation(
    pupil_img: np.ndarray,
    focus_img: np.ndarray,
    f_m: float = 3,
    wavelength_m: float = 1064e-9,
    focal_length_m: float = 3.0,
    input_pixel_size: float = 2.9e-6,
    output_pixel_size: float = 5.5e-6,
) -> tuple[float, np.ndarray]:
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

    # 构建理想复振幅（将光瞳强度作为振幅，使用聚焦相位）
    N = pupil_img.shape[0]
    x = (np.arange(N) - N // 2) * input_pixel_size
    y = (np.arange(N) - N // 2) * input_pixel_size
    X, Y = np.meshgrid(x, y, indexing="xy")
    k = 2 * np.pi / wavelength_m
    lens_phase = np.exp(-1j * k * (X**2 + Y**2) / (2 * f_m))
    E_in = complex(1) * pupil_img * lens_phase
    _require_aotools()
    assert twoStepFresnel is not None
    E_out = twoStepFresnel(E_in, wavelength_m, input_pixel_size, output_pixel_size, f_m)

    ideal_focus_intensity = np.abs(E_out) ** 2

    # 能量守恒校准
    total_energy_actual = np.sum(focus_img[focus_img > 10])
    total_energy_ideal = np.sum(ideal_focus_intensity[ideal_focus_intensity > 10])

    if total_energy_ideal == 0:
        return 0.0, np.zeros_like(focus_img)

    scaling_factor = total_energy_actual / total_energy_ideal
    ideal_energy_matched = ideal_focus_intensity * scaling_factor

    # 斯特列尔比
    peak_actual = np.max(focus_img)
    peak_ideal = np.max(ideal_energy_matched)

    strehl = peak_actual / peak_ideal if peak_ideal > 0 else 0

    return strehl, ideal_energy_matched


def propagate_through_lens(field: np.ndarray, pixel_size: float, wavelength: float, focal_length: float) -> np.ndarray:
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
        focal_length_m=focal_length,
    )

    return np.abs(output_field)


def angular_spectrum_propagation(
    field: np.ndarray, pixel_size: float, wavelength: float, distance: float
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
