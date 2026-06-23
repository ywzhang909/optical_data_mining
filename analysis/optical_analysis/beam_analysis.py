"""
光束质量分析模块
=================
提供光斑分析和光束质量评估的核心算法

功能:
- D4σ直径计算（一阶矩和二阶矩）
- PIB占比计算
- 高斯拟合直径
- BPP (Beam Parameter Product) 计算
- 斯特列尔比(Strehl Ratio) 计算辅助
"""

from typing import Any

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

from ..image.common import compute_radial_profile


def gaussian(x: np.ndarray, mu: float, sigma: float, A: float, b: float) -> np.ndarray:
    """
    高斯函数

    Args:
        x: 自变量数组
        mu: 中心位置
        sigma: 标准差
        A: 振幅
        b: 基线偏移

    Returns:
        高斯函数值数组
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + b


def fitting_gaussian(data: np.ndarray) -> tuple[tuple[float, float, float, float], Any]:
    """
    拟合高斯函数

    Args:
        data: 输入一维数据数组

    Returns:
        (mu, sigma, A, b): 拟合参数元组
        covariance: 协方差矩阵
    """
    x_data = np.arange(len(data))
    initial_guess = [np.argmax(data), 10, np.max(data), 0]
    try:
        (mu, sigma, A, b), covariance = curve_fit(
            gaussian, x_data, data, p0=initial_guess
        )
        return (mu, sigma, A, b), covariance
    except RuntimeError:
        return (np.nan, np.nan, np.nan, np.nan), np.nan


def center_of_mass_numpy(
    intensity: np.ndarray, xv: np.ndarray, yv: np.ndarray, moment: int = 1
) -> tuple[float, float]:
    """
    计算光强的中心位置

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param moment: 中心位置的阶数
    :return center_x, center_y: 光强的中心位置
    """
    _intensity = intensity.copy().astype(np.float32) ** moment
    total_intensity = np.sum(_intensity)
    c_x = np.sum(xv * _intensity) / total_intensity
    c_y = np.sum(yv * _intensity) / total_intensity
    return (float(c_x), float(c_y))


def d4sigma(
    img: np.ndarray, pixel_size_um: float = 1.0, subtract_background: bool = False
) -> dict[str, float]:
    """
    计算 D4σ 光斑直径（符合 ISO 11146 标准）

    Args:
        img: 输入光强图像（2D array, 应为非负）
        pixel_size_um: 像素尺寸（微米）
        subtract_background: 是否自动扣除背景（推荐 True）

    Returns:
        dict with keys: center_x, center_y, D_x, D_y, avg_diameter, center_intensity
    """
    if img.ndim != 2:
        raise ValueError("Input image must be 2D")

    # 确保非负（数值误差可能导致极小负值）
    _img = np.maximum(img.copy().astype(np.float64), 0.0)

    if subtract_background:
        edge_pixels = np.concatenate([_img[0, :], _img[-1, :], _img[:, 0], _img[:, -1]])
        background = np.median(edge_pixels)
        _img = np.maximum(_img - background, 0.0)

    total = _img.sum()
    if total == 0 or not np.isfinite(total):
        return {
            "center_x": 0.0,
            "center_y": 0.0,
            "D_x": 0.0,
            "D_y": 0.0,
            "avg_diameter": 0.0,
            "center_intensity": 0.0,
        }

    h, w = _img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)

    cx, cy = center_of_mass_numpy(_img, x, y, 5)

    # 二阶中心矩
    mu_xx = np.sum((x - cx) ** 2 * _img) / total
    mu_yy = np.sum((y - cy) ** 2 * _img) / total

    # D4σ = 4 * σ
    Dx = 4.0 * np.sqrt(max(mu_xx, 0.0)) * pixel_size_um
    Dy = 4.0 * np.sqrt(max(mu_yy, 0.0)) * pixel_size_um
    avg_diameter = np.sqrt(Dx * Dy)

    # 中心强度（双线性插值更准，但这里用最近邻）
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    if 0 <= cy_int < h and 0 <= cx_int < w:
        center_intensity = float(_img[cy_int, cx_int])
    else:
        center_intensity = 0.0

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "D_x": float(Dx),
        "D_y": float(Dy),
        "avg_diameter": float(avg_diameter),
        "center_intensity": center_intensity,
    }


def pib_ratio(
    img: np.ndarray,
    center: tuple[float, float],
    wavelength_m: float,
    focal_length_m: float,
    aperture_diameter_m: float,
    pixel_size_m: float,
) -> tuple[float, bool]:
    """
    计算 PIB 占比 (Power In Bucket)
    以质心为圆心，计算半径为衍射极限半径的圆内能量占全部能量的比值

    Args:
        img: 输入图像（2D数组）
        center: 中心坐标 (x, y)
        wavelength_m: 波长 (米)
        focal_length_m: 焦距 (米)
        aperture_diameter_m: 入瞳直径 (米)
        pixel_size_m: 像素尺寸 (米)

    Returns:
        tuple: (pib_ratio, is_overexposed)
            - pib_ratio: PIB 占比 (0-1之间)
            - is_overexposed: 是否过曝
    """
    cx, cy = center
    h, w = img.shape

    # 计算衍射极限半径 (Airy disk radius)
    # r_airy = 1.22 * λ * f / D (物理尺寸)
    r_airy_m = 1.22 * wavelength_m * focal_length_m / aperture_diameter_m
    # 转换为像素
    r_airy_pixels = r_airy_m / pixel_size_m

    y, x = np.mgrid[0:h, 0:w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r_airy_pixels**2
    pib_intensity = img[mask].sum()
    total_intensity = img.sum()

    if total_intensity == 0:
        return 0.0, False

    # 检查是否过曝：如果中心区域像素值接近或达到最大值
    max_pixel_value = img.max()
    # 假设16位图像最大值为65535，8位为255
    bit_depth = 16 if max_pixel_value > 255 else 8
    max_possible = 65535 if bit_depth == 16 else 255
    is_overexposed = max_pixel_value >= max_possible * 0.95

    return pib_intensity / total_intensity, is_overexposed


def calculate_xy_diameters(
    image: np.ndarray, center_x: float, center_y: float, pix_size: float = 1.0
) -> dict[str, float]:
    """
    计算x和y方向的高斯直径

    Args:
        image: 输入图像（2D数组）
        center_x: 中心X坐标
        center_y: 中心Y坐标
        pix_size: 像素尺寸

    Returns:
        dict: 包含x和y方向直径的字典
            - gaussian_dia_x(um): X方向高斯直径 (μm)
            - gaussian_dia_y(um): Y方向高斯直径 (μm)
    """
    # 提取x和y方向的数据
    y_data = image[:, int(center_x)]
    x_data = image[int(center_y), :]

    # 计算x方向直径
    (mu, sigma, A, b), conv = fitting_gaussian(x_data)
    x_diameter = 2 * sigma * pix_size if not np.isnan(sigma) else 0

    # 计算y方向直径
    (mu, sigma, A, b), conv = fitting_gaussian(y_data)
    y_diameter = 2 * sigma * pix_size if not np.isnan(sigma) else 0

    return {"gaussian_dia_x(um)": x_diameter, "gaussian_dia_y(um)": y_diameter}


def _ftl_model(R: np.ndarray, I0: float, R_FL: float, q: float) -> np.ndarray:
    """
    Flat-Topped Lorentz (FTL) 光束模型

    I(R) = I0 / [1 + (R/R_FL)^q] ^ (1 + 2/q)

    Args:
        R: 径向距离数组（像素）
        I0: 中心强度
        R_FL: 特征半径（像素）
        q: 平顶阶数

    Returns:
        模型强度值数组
    """
    # 防止除零
    with np.errstate(divide="ignore", invalid="ignore"):
        result = I0 / (1 + (R / np.maximum(R_FL, 1e-10)) ** q) ** (1 + 2.0 / q)
    result[~np.isfinite(result)] = 0.0
    return result


def _sample_ray(
    image: np.ndarray,
    cx: float,
    cy: float,
    theta: float,
    max_radius: int,
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """沿单条射线采样图像强度（双线性插值）。

    Args:
        image: 输入图像 (H, W)，任意 dtype，会被转为 float64
        cx, cy: 中心坐标（像素）
        theta: 采样角度（弧度），0=右方，π/2=下方（numpy row-major）
        max_radius: 最大采样半径（像素），必须 > 0
        step: 径向采样步长（像素），必须 > 0

    Returns:
        r_arr: 径向距离数组 (0, step, 2*step, ..., max_radius)
        i_arr: 对应位置的插值强度
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    if max_radius <= 0 or step <= 0:
        return np.array([0.0]), np.array([0.0])
    cx = float(cx)
    cy = float(cy)
    n = int(np.floor(max_radius / step)) + 1
    r_arr = np.arange(n, dtype=np.float64) * step
    xp = cx + r_arr * np.cos(theta)
    yp = cy + r_arr * np.sin(theta)

    valid = (xp >= 0) & (xp < w - 1) & (yp >= 0) & (yp < h - 1)
    if not np.any(valid):
        return r_arr, np.zeros_like(r_arr)

    x0 = np.floor(xp).astype(int)
    y0 = np.floor(yp).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xp - x0
    wy = yp - y0

    I = np.zeros_like(r_arr, dtype=np.float64)
    for yy, yw in [(y0, 1.0 - wy), (y1, wy)]:
        for xx, xw in [(x0, 1.0 - wx), (x1, wx)]:
            mask = valid & (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            if np.any(mask):
                I[mask] += xw[mask] * yw[mask] * image[yy[mask], xx[mask]]
    return r_arr, I


def _fit_ftl_1d(r: np.ndarray, intensity: np.ndarray, max_r: float) -> dict:
    """对单条射线的一维强度剖面做 FTL 拟合。"""
    valid = intensity > 0
    if np.sum(valid) < 5:
        return {
            "success": False,
            "R_FL": np.nan,
            "R_FL_pixels": np.nan,
            "q": np.nan,
            "I0": np.nan,
            "R_FL_error": np.nan,
            "q_error": np.nan,
            "I0_error": np.nan,
        }

    R_fit = r[valid]
    I_fit = intensity[valid]
    I0_guess = float(max(I_fit[0], np.max(I_fit)))
    R_FL_guess = float(max_r * 0.3)
    q_guess = 4.0
    try:
        popt, pcov = curve_fit(
            _ftl_model,
            R_fit,
            I_fit,
            p0=[I0_guess, R_FL_guess, q_guess],
            bounds=([0.0, 0.0, 1.5], [np.inf, float(max_r), 50.0]),
            maxfev=10000,
        )
        I0_fit, R_FL_fit, q_fit = popt
        perr = np.sqrt(np.diag(pcov))
        return {
            "success": True,
            "R_FL": float(R_FL_fit),
            "R_FL_pixels": float(R_FL_fit),
            "q": float(q_fit),
            "I0": float(I0_fit),
            "R_FL_error": float(perr[1]),
            "q_error": float(perr[2]),
            "I0_error": float(perr[0]),
        }
    except (RuntimeError, ValueError):
        return {
            "success": False,
            "R_FL": np.nan,
            "R_FL_pixels": np.nan,
            "q": np.nan,
            "I0": np.nan,
            "R_FL_error": np.nan,
            "q_error": np.nan,
            "I0_error": np.nan,
        }


def fit_flat_topped_lorentz(
    image: np.ndarray,
    cx: float,
    cy: float,
    pixel_size: float = 1.0,
    max_radius: int | None = None,
    radial_step: float = 1.0,
    n_angles: int = 36,
) -> dict[str, Any]:
    """
    对光瞳图像进行 FTL (Flat-Topped Lorentz) 模型拟合，计算特征半径 R_FL。

    处理流程:
        1. 使用 compute_radial_profile() 计算径向剖面 Ī(R)（角向平均）
        2. 对每个角度 θ，沿射线采样并独立拟合 FTL，得到 R_FL(θ)
        3. 从角向结果统计平均、标准差、离心率
        4. 拟合全局径向剖面作为参考（向后兼容）

    Args:
        image: 去暗场后的二维光强图像
        cx: 光斑中心 X 坐标（像素），通常取包围圆圆心
        cy: 光斑中心 Y 坐标（像素）
        pixel_size: 像素尺寸（微米/像素）
        max_radius: 最大拟合半径（像素）。None 时取图像短边/2
        radial_step: 径向分箱的步长（像素），默认 1.0
        n_angles: 角度采样数（0 表示仅用角向平均，不逐角度拟合）

    Returns:
        dict 包含以下键:
            - R_FL: 角向平均特征半径 (μm)
            - R_FL_pixels: 角向平均特征半径 (像素)
            - R_FL_std: R_FL 角向标准差 (μm)
            - R_FL_min / R_FL_max: 角向极值 (μm)
            - ellipticity_from_ftl: 基于 R_FL(θ) 的椭圆度 = 长轴/短轴
            - q / I0: 全局径向平均拟合参数
            - R_FL_error / q_error / I0_error: 全局拟合标准误差
            - radial_R: 径向距离数组 (μm)
            - radial_intensity: 径向平均强度数组
            - fitted_intensity: 全局 FTL 拟合强度数组
            - angular_theta_deg: 各角度（度）
            - angular_R_FL_pixels: 各角度 R_FL（像素）
            - angular_q: 各角度 q
            - angular_I0: 各角度 I0
            - angular_success_rate: 各角度拟合成功率
            - success: 全局拟合是否成功
            - message: 状态描述
    """
    # 0. 确定 max_radius
    if max_radius is None:
        h, w = image.shape
        max_radius = int(min(h, w) / 2)
    max_radius = max(4, int(max_radius))

    # 1. 全局径向剖面（角向平均）——向后兼容
    profile = compute_radial_profile(
        image,
        cx,
        cy,
        max_radius=max_radius,
        radial_step=radial_step,
        min_pixels_per_bin=3,
    )
    if not profile["success"]:
        return {
            "success": False,
            "message": profile["message"],
            "R_FL": np.nan,
            "R_FL_pixels": np.nan,
            "q": np.nan,
            "I0": np.nan,
            "R_FL_error": np.nan,
            "q_error": np.nan,
            "I0_error": np.nan,
            "R_FL_std": np.nan,
            "R_FL_min": np.nan,
            "R_FL_max": np.nan,
            "ellipticity_from_ftl": np.nan,
            "angular_theta_deg": [],
            "angular_R_FL_pixels": [],
            "angular_q": [],
            "angular_I0": [],
            "angular_success_rate": 0.0,
            "radial_R": profile["radial_R"] * pixel_size
            if "radial_R" in profile
            else np.array([]),
            "radial_intensity": profile.get("radial_intensity", np.array([])),
            "fitted_intensity": np.array([]),
        }

    valid_mask = profile["valid_mask"]
    if np.sum(valid_mask) < 5:
        return {
            "success": False,
            "message": f"有效径向数据点不足 ({np.sum(valid_mask)} < 5)",
            "R_FL": np.nan,
            "R_FL_pixels": np.nan,
            "q": np.nan,
            "I0": np.nan,
            "R_FL_error": np.nan,
            "q_error": np.nan,
            "I0_error": np.nan,
            "R_FL_std": np.nan,
            "R_FL_min": np.nan,
            "R_FL_max": np.nan,
            "ellipticity_from_ftl": np.nan,
            "angular_theta_deg": [],
            "angular_R_FL_pixels": [],
            "angular_q": [],
            "angular_I0": [],
            "angular_success_rate": 0.0,
            "radial_R": profile["radial_R"] * pixel_size,
            "radial_intensity": profile["radial_intensity"],
            "fitted_intensity": np.array([]),
        }

    # 2. 全局（角向平均）径向拟合
    R_fit = profile["radial_R"][valid_mask]
    I_fit = profile["radial_intensity"][valid_mask]
    I0_guess = float(max(I_fit[0], np.max(image)))
    max_r = profile["max_radius"]
    R_FL_guess = float(max_r * 0.3)
    q_guess = 4.0

    try:
        popt, pcov = curve_fit(
            _ftl_model,
            R_fit,
            I_fit,
            p0=[I0_guess, R_FL_guess, q_guess],
            bounds=([0.0, 0.0, 1.5], [np.inf, float(max_r), 50.0]),
            maxfev=10000,
        )
        I0_fit, R_FL_fit, q_fit = popt
        perr = np.sqrt(np.diag(pcov))
        fitted_intensity = _ftl_model(R_fit, I0_fit, R_FL_fit, q_fit)
        success = True
        message = "拟合成功"
    except (RuntimeError, ValueError) as e:
        I0_fit, R_FL_fit, q_fit = np.nan, np.nan, np.nan
        perr = [np.nan, np.nan, np.nan]
        fitted_intensity = np.full_like(R_fit, np.nan)
        success = False
        message = f"全局拟合失败: {e}"

    # 3. 逐角度射线采样 + FTL 拟合
    angular_theta_deg = []
    angular_R_FL_pixels = []
    angular_q = []
    angular_I0 = []
    angular_R_FL_error = []
    angular_success_flags = []

    if n_angles > 0 and success:
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        for theta in thetas:
            r_arr, i_arr = _sample_ray(
                image, cx, cy, theta, max_radius, step=radial_step
            )
            result = _fit_ftl_1d(r_arr, i_arr, max_radius)
            angular_theta_deg.append(np.degrees(theta))
            angular_R_FL_pixels.append(result["R_FL_pixels"])
            angular_q.append(result["q"])
            angular_I0.append(result["I0"])
            angular_R_FL_error.append(result["R_FL_error"])
            angular_success_flags.append(result["success"])

    ang_rfl = np.array(angular_R_FL_pixels, dtype=np.float64)
    ang_q = np.array(angular_q, dtype=np.float64)
    ang_I0 = np.array(angular_I0, dtype=np.float64)
    ang_ok = np.array(angular_success_flags, dtype=bool)

    if np.any(ang_ok) and not np.all(np.isnan(ang_rfl[ang_ok])):
        valid_rfl = ang_rfl[ang_ok]
        R_FL_mean_px = float(np.nanmean(valid_rfl))
        R_FL_std_px = float(np.nanstd(valid_rfl))
        R_FL_min_px = float(np.nanmin(valid_rfl))
        R_FL_max_px = float(np.nanmax(valid_rfl))
        if R_FL_min_px > 0:
            fftl_ellipticity = R_FL_max_px / R_FL_min_px
        else:
            fftl_ellipticity = np.nan
        # 用最小二乘拟合椭圆 (R_FL(θ) = a / (1 - e*cos(θ-θ0)) 近似)
        # 简化版：主/次轴 = max/min R_FL
        success_rate = float(np.mean(ang_ok))
    else:
        R_FL_mean_px = float(R_FL_fit) if not np.isnan(R_FL_fit) else np.nan
        R_FL_std_px = np.nan
        R_FL_min_px = np.nan
        R_FL_max_px = np.nan
        fftl_ellipticity = np.nan
        success_rate = 0.0

    return {
        "R_FL": float(R_FL_mean_px * pixel_size)
        if not np.isnan(R_FL_mean_px)
        else np.nan,
        "R_FL_pixels": R_FL_mean_px,
        "q": float(q_fit) if not np.isnan(q_fit) else np.nan,
        "I0": float(I0_fit) if not np.isnan(I0_fit) else np.nan,
        "R_FL_error": float(perr[1] * pixel_size) if len(perr) > 1 else np.nan,
        "q_error": float(perr[2]) if len(perr) > 2 else np.nan,
        "I0_error": float(perr[0]) if len(perr) > 0 else np.nan,
        "R_FL_std": float(R_FL_std_px * pixel_size)
        if not np.isnan(R_FL_std_px)
        else np.nan,
        "R_FL_min": float(R_FL_min_px * pixel_size)
        if not np.isnan(R_FL_min_px)
        else np.nan,
        "R_FL_max": float(R_FL_max_px * pixel_size)
        if not np.isnan(R_FL_max_px)
        else np.nan,
        "ellipticity_from_ftl": fftl_ellipticity,
        "radial_R": profile["radial_R"] * pixel_size,
        "radial_intensity": profile["radial_intensity"],
        "fitted_intensity": fitted_intensity if success else np.array([]),
        "angular_theta_deg": angular_theta_deg,
        "angular_R_FL_pixels": angular_R_FL_pixels,
        "angular_q": angular_q,
        "angular_I0": angular_I0,
        "angular_success_rate": success_rate,
        "success": success,
        "message": message,
    }


def calculate_bpp(
    pupil_diameter_mm: float, focal_diameter_mm: float, focal_length_mm: float = 3000
) -> dict[str, float]:
    """
    计算BPP (Beam Parameter Product)

    Args:
        pupil_diameter_mm: 出瞳直径 (mm)
        focal_diameter_mm: 焦斑直径 (mm)
        focal_length_mm: 透镜焦距 (mm)

    Returns:
        dict: BPP结果
            - BPP_mm_mrad: 光束参数乘积
            - divergence_mrad: 发散角 (mrad)
    """
    w_pupil = pupil_diameter_mm / 2.0
    w_focal = focal_diameter_mm / 2.0
    f = focal_length_mm

    theta_rad = w_focal / f
    theta_mrad = theta_rad * 1000.0
    bpp_mm_mrad = w_pupil * theta_mrad

    return {
        "BPP_mm_mrad": bpp_mm_mrad,
        "divergence_mrad": theta_mrad,
    }


def calculate_m2(bpp_mm_mrad: float, wavelength_um: float) -> float:
    """
    计算M²光束质量因子

    Args:
        bpp_mm_mrad: 光束参数乘积
        wavelength_um: 波长 (微米)

    Returns:
        float: M²值
    """
    bpp_diffraction = wavelength_um / np.pi
    return bpp_mm_mrad / bpp_diffraction


def calculate_centroid(img: np.ndarray) -> tuple[float, float]:
    """
    计算图像的质心

    Args:
        img: 输入图像（2D数组）

    Returns:
        tuple: (center_x, center_y) 质心坐标
    """
    total = img.sum()
    if total == 0:
        return (0.0, 0.0)

    cy, cx = center_of_mass(img)
    try:
        cx = float(np.ravel(np.asarray(cx))[0])
        cy = float(np.ravel(np.asarray(cy))[0])
    except (TypeError, IndexError, ValueError):
        cx, cy = float(cx), float(cy)

    return (cx, cy)


def extract_beam_features(
    img: np.ndarray,
    pixel_size_um: float = 1.0,
    wavelength_m: float = 1064e-9,
    focal_length_m: float = 3.0,
    aperture_diameter_m: float = 0.1,
) -> dict[str, Any]:
    """
    提取完整的光束特征

    Args:
        img: 输入图像（2D数组）
        pixel_size_um: 像素尺寸（微米）
        wavelength_m: 波长（米）
        focal_length_m: 焦距（米）
        aperture_diameter_m: 入瞳直径（米）

    Returns:
        dict: 包含所有光束特征的字典
    """
    # D4σ特征
    d4s_features = d4sigma(img, pixel_size_um)

    # 质心
    centroid = (d4s_features["center_x"], d4s_features["center_y"])

    # PIB占比（使用衍射极限半径）
    pixel_size_m = pixel_size_um * 1e-6
    pib, is_overexposed = pib_ratio(
        img,
        centroid,
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        aperture_diameter_m=aperture_diameter_m,
        pixel_size_m=pixel_size_m,
    )

    # 高斯拟合直径
    gaussian_dia = calculate_xy_diameters(
        img, d4s_features["center_x"], d4s_features["center_y"], pixel_size_um
    )

    return {
        "centroid": centroid,
        "d4s": d4s_features,
        "pib_ratio": pib,
        "pib_overexposed": is_overexposed,
        "gaussian_diameter": gaussian_dia,
    }


def compute_encircled_energy(
    image: np.ndarray,
    cx: float,
    cy: float,
    fractions: tuple[float, ...] = (0.50, 0.80, 0.95),
) -> dict[str, Any]:
    """
    计算围困能量 (encircled energy) 曲线及关键半径。

    对图像中每个像素按距 (cx, cy) 的径向距离排序，
    计算累计能量占比，并提取指定能量占比对应的半径。

    Args:
        image: 二维光强图像。
        cx, cy: 中心坐标（像素）。
        fractions: 需提取的能量占比序列，默认 (0.50, 0.80, 0.95)。

    Returns:
        dict 包含:
            - r_fraction: dict，每个 fraction → 对应半径（像素）
            - sorted_R:   ndarray，按径向距离排序后的半径序列
            - cum_norm:   ndarray，对应的累计归一化能量 (0→1)
            - peak_intensity: float，像素最大强度
            - total_energy:   float，像素强度总和
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    y_i, x_i = np.indices((h, w))
    R = np.sqrt((x_i - cx) ** 2 + (y_i - cy) ** 2)

    sort_idx = np.argsort(R.ravel())
    sorted_R = R.ravel()[sort_idx]
    sorted_E = img.ravel()[sort_idx]

    cum_E = np.cumsum(sorted_E)
    total = cum_E[-1]
    cum_norm = cum_E / total if total > 0 else cum_E

    r_fraction: dict[str, float] = {}
    for frac in fractions:
        idx = np.searchsorted(cum_norm, frac)
        if idx < len(sorted_R):
            r_fraction[f"r{int(frac * 100)}"] = float(sorted_R[idx])
        else:
            r_fraction[f"r{int(frac * 100)}"] = np.nan

    return {
        "r_fraction": r_fraction,
        "sorted_R": sorted_R,
        "cum_norm": cum_norm,
        "peak_intensity": float(np.max(img)),
        "total_energy": float(total),
    }
