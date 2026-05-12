import os
import sys
sys.path.append('./ui')

from analysis.image.common import get_profiles, read_tiff_to_numpy
dotenv.load_dotenv('.')

exp_dir = os.environ.get('EXP_DIR', 'X:/')

# 配置swifter参数以优化性能
swifter.set_defaults(
    npartitions=os.cpu_count(),  # 分区数量，根据CPU核心数调整
    dask_threshold=50,  # 数据量阈值，超过此数量使用dask
    disable_cache=False,  # 启用缓存
    progress_bar=True  # 显示进度条
)

def process_time_columns(df: pd.DataFrame):
    """
    处理DataFrame中的时间相关列
    
    Args:
        df (pd.DataFrame): 要处理的DataFrame
    
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
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

# =============== Image Func ====================

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
    return {'shifted_img_array': np.where(shifted < 1e-3, 0, shifted)}

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

    q_array = np.asarray([q1, q2, q3, q4])
    rms = np.mean(np.sqrt((q_array - np.mean(q_array))**2))

    # four_quadrant_rms = np.sqrt((q1 + q2 + q3 + q4) / 4)
    return {
        'rms_4_quadrant': rms
    }

def piecewise_linear_func(x, l, r, la, ra, mu):
    """
    分段线性函数，用于curve_fit拟合
    - 在区间[l, r]之外为常数mu
    - 在区间[l, r]内为连接(l, la)和(r, ra)的直线
    
    参数:
    x: 自变量数组
    l: 区间左端点
    r: 区间右端点
    la: x=l时的函数值
    ra: x=r时的函数值
    mu: 区间外的常数值
    """
    # 初始化输出数组为mu
    y = np.full_like(x, mu, dtype=float)
    
    # 计算区间内的直线部分
    mask = (x >= l) & (x <= r)
    if np.any(mask):
        # 直线方程: y = la + (ra - la) / (r - l) * (x - l)
        slope = (ra - la) / (r - l) if r != l else 0
        y[mask] = la + slope * (x[mask] - l)
    
    return y

def calculate_sharpness(img:np.ndarray):
  gradient_x = np.gradient(img, axis=1)
  gradient_y = np.gradient(img, axis=0)
  gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
  sharpness = np.mean(gradient_magnitude)
  return sharpness

def calc_lr_diff(profile):
    init_mu = np.median(profile)
    indices = np.where(profile>init_mu)[0]
    p0 = [indices[0], indices[-1], np.max(profile), np.max(profile), init_mu]
    bounds = ([0, 0, 0, 0, 0], [len(profile), len(profile), np.max(profile), np.max(profile), np.max(profile)])
    try:
        params, _ = curve_fit(piecewise_linear_func, np.arange(len(profile)), profile, p0=p0, bounds=bounds)
        l, r, la, ra, mu = params
        
        #TODO nrms
        return {
            'flat_fit_diff': np.abs(ra - la),
            'flat_fit_ndiff': np.abs(ra - la) / np.abs(r - l),
            'flat_fit_diameter': r - l}
    except:
        return {
            'flat_fit_diff': np.nan,
            'flat_fit_ndiff': np.nan,
            'flat_fit_diameter': np.nan,
        }

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

def calculate_xy_diameters(image, center_x, center_y):
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

    return {'gaussian_dia_x': x_diameter, 'gaussian_dia_y': y_diameter}

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
    
    return pd.merge(df, ellipse_features, left_index=True, right_index=True).merge(circle_features, left_index=True, right_index=True)

def make_coord(img:np.ndarray):
    """
    生成坐标矩阵

    :param img: 强度分布
    :return x, y: 坐标矩阵
    """
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return x, y

def radius(intensity, center, energy=0.99):
    """
    以center为圆心，占总能量百分比为energy的圆的半径

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param center: 圆心，坐标，如(0, 0)
    :param energy: 圆内的能量比，默认0.99，取值范围0~1，常用0.5，0.865， 0.99
    :return radius: 圆的半径
    """
    x, y = make_coord(intensity)
    npix = len(x)
    dpix = x[0, 1] - x[0, 0]
    
    x0, y0 = center[0], center[1]

    power_in_circle = np.sum(intensity) * energy
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    radius = npix * dpix / 2
    radius_change = npix * dpix / 4

    for i in range(300):
        mask = (np.sign(radius - r) + 1) / 2
        power = np.sum(intensity * mask)

        if power - power_in_circle < -1e-10 * power_in_circle:
            radius += radius_change
        elif power - power_in_circle > 1e-10 * power_in_circle:
            radius -= radius_change
        else:
            break

        radius_change /= 2
        if radius_change < dpix / 50:
            break
        if i == 299:
            return np.nan

    return {'power99_radius':radius}

def calculate_strehl_ratio_with_energy_conservation(
    pupil_img,
    focus_img,
    pixel_size_pupil_um=5.5,
    pixel_size_focus_um=5.5,
    N=1/32,
    f_mm=3_000,
    wavelength_um=1.064,
    background_subtract=False,
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

def calculate_bpp_from_pupil_and_focal(
    pupil_diameter_mm,
    focal_diameter_mm,
    focal_length_mm = 3e3
):
    """
    使用出瞳和焦斑直径（单位：mm）计算 BPP 和 M²，考虑缩束比
    
    Parameters:
        pupil_diameter_mm: 出瞳 D4σ 直径（毫米）
        focal_diameter_mm: 焦平面 D4σ 直径（毫米）
        focal_length_mm: 透镜焦距（毫米）
    
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
    
    return {
        "BPP_mm_mrad": bpp_mm_mrad,
    }

# =============== 全局变量 ==========================
# TODO 一次实验的指标提取
'''
1. 出光时间
'''


# ===================================================

def process_all(axis_beam_dir, pupil_beam_dir, exp_id):
    print(exp_id)

    # 1. 读取所有 TIFF 文件
    axis_beam_img_path = Path(axis_beam_dir).glob('*.TIFF')
    pupil_beam_img_path = Path(pupil_beam_dir).glob('*.TIFF')

    axis_beam = pd.DataFrame([{'path': path} for path in axis_beam_img_path])
    pupil_beam = pd.DataFrame([{'path': path} for path in pupil_beam_img_path])

    axis_beam['img_array'] = axis_beam['path'].apply(read_tiff_to_numpy)
    pupil_beam['img_array'] = pupil_beam['path'].apply(read_tiff_to_numpy)

    # 2. 处理时间
    def process_beam_data(beam_df, denoise_method='median'):
        beam_df = process_time_columns(beam_df)
        # 降噪、计算亮度
        beam_df = process_image_data(beam_df, denoise_method)
        return beam_df

    axis_beam = process_beam_data(axis_beam, 'median')
    pupil_beam = process_beam_data(pupil_beam, 'median')

    valid_axis_beam = axis_beam[axis_beam['intensity'] > 10]
    valid_pupil_beam = pupil_beam[pupil_beam['intensity'] > 10]

    # TODO 计算出光时间

    print(valid_pupil_beam.iloc[0]['time'], valid_axis_beam.iloc[-1]['time'])

    # 3. 提取出瞳和焦斑直径
    def d4sigma_feature_extract(df: pd.DataFrame):
        d4sigma_features = df.swifter.apply(lambda x: d4sigma(x['denoise_img_array']), axis=1, result_type='expand')
        d4sigma_features['avg_sigma2'] = np.sqrt(d4sigma_features['D_x'] * d4sigma_features['D_y'])
        return pd.merge(df, d4sigma_features, left_index=True, right_index=True)

    valid_axis_beam = d4sigma_feature_extract(valid_axis_beam)
    valid_pupil_beam = d4sigma_feature_extract(valid_pupil_beam)

    # 4. 提取出瞳的均匀性指标
    uniformity_features = valid_pupil_beam.swifter.apply(
        lambda x: uniformity(x['denoise_img_array'], (x['center_x'], x['center_y'])), axis=1, result_type='expand'
    )

    valid_pupil_beam = pd.merge(valid_pupil_beam, uniformity_features, left_index=True, right_index=True)

    # 5. 提取出瞳的锐度指标
    sharpness_fit = valid_pupil_beam.swifter.apply(
        lambda x: calculate_sharpness(x['denoise_img_array']), axis=1, result_type='expand'
    )
    sharpness_fit.columns = ['sharpness_fit']
    valid_pupil_beam = pd.merge(valid_pupil_beam, sharpness_fit, left_index=True, right_index=True)

    # 6. 提取出瞳的垂直和水平投影
    xy_profile = valid_pupil_beam.swifter.apply(
        lambda x: get_profiles(
            x['denoise_img_array'], (x['center_x'], x['center_y'])), result_type='expand', axis=1
    )
    fit_features = xy_profile.swifter.apply(
        lambda x: calc_lr_diff(x['vertical']), axis=1, result_type='expand'
    )
    valid_pupil_beam = pd.merge(valid_pupil_beam, fit_features, left_index=True, right_index=True)

    # 7. 提取出轴的高斯直径
    guassian_dia = valid_axis_beam.swifter.apply(
        lambda x: calculate_xy_diameters(x['denoise_img_array'], x['center_x'], x['center_y']),
        axis=1, result_type='expand'
    )

    valid_axis_beam = pd.merge(valid_axis_beam, guassian_dia, left_index=True, right_index=True)
    
    # 8. 提取出形状指标
    valid_axis_beam = shape_feature_extract(valid_axis_beam, use_gpu=True)
    valid_pupil_beam = shape_feature_extract(valid_pupil_beam, use_gpu=True)
    
    center_axis_beam = valid_axis_beam.swifter.apply(
        lambda x: shift_to_center_fft(x['denoise_img_array'], x['border_x'], x['border_y']), axis=1, result_type='expand'
    )
    valid_axis_beam = pd.merge(valid_axis_beam, center_axis_beam, left_index=True, right_index=True)

    center_pupil_beam = valid_pupil_beam.swifter.apply(
        lambda x: shift_to_center_fft(x['denoise_img_array'], x['border_x'], x['border_y']), axis=1, result_type='expand'
    )
    valid_pupil_beam = pd.merge(valid_pupil_beam, center_pupil_beam, left_index=True, right_index=True)

    def calc_radius(x: pd.Series):
        radius_feature = radius(x['denoise_img_array'], center = (x['center_x'], x['center_y']))
        return radius_feature

    radius_feature = valid_pupil_beam.swifter.apply(calc_radius ,axis=1, result_type='expand')
    valid_pupil_beam = pd.merge(valid_pupil_beam, radius_feature, left_index=True, right_index=True)

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

    strehl_results_with_scaler = merged_beam.swifter.apply(
        lambda row: calculate_strehl_ratio_with_energy_conservation(row['shifted_img_array_pupil'], row['shifted_img_array_axis']),
        axis=1,
        result_type='expand'
    )
    strehl_results_with_scaler.columns = ['strehl_ratio', 'ideal_matched']
    merged_beam = pd.merge(merged_beam, strehl_results_with_scaler, left_index=True, right_index=True)

    bpp_results = merged_beam.swifter.apply(
    lambda row: calculate_bpp_from_pupil_and_focal(row['avg_sigma2_pupil'], row['avg_sigma2_axis']),
    axis=1, result_type='expand')

    merged_beam = pd.merge(merged_beam, bpp_results, left_index=True, right_index=True) 

    # merged_beam['path'] = merged_beam['path'].astype(str)
    # merged_beam['time'] = merged_beam['time'].astype(str)
    for c in merged_beam.select_dtypes(input=['object']).columns:
        merged_beam[c] = merged_beam[c].astype(str)
    merged_beam.to_parquet(f'{exp_dir}/temp/do/{exp_id}.parquet', compression='zstd')

    merged_beam.set_index('time', drop=True, inplace=True)
    merged_beam.to_excel(f'{exp_dir}/temp/do/{exp_id}.xlsx')
    del merged_beam, valid_axis_beam, valid_pupil_beam, axis_beam, pupil_beam

def main():
    done_exp_ids = (p.stem for p in Path(f'{exp_dir}/temp/do').glob('*.parquet'))

    do_arch_data = pd.read_excel(f"{exp_dir}/实验记录v20251230 - 加数字光学.xlsx", header=1)
    do_arch_data = do_arch_data[~do_arch_data['实验编号'].isin(done_exp_ids)]\
        .dropna(subset=['实验编号', '光轴文件夹路径', '光瞳文件夹路径'], how='any')
    
    print(len(do_arch_data))
    do_arch_data['光轴文件夹路径'] = do_arch_data['光轴文件夹路径'].str.replace('X:', exp_dir)
    do_arch_data['光瞳文件夹路径'] = do_arch_data['光瞳文件夹路径'].str.replace('X:', exp_dir)
    do_arch_data.apply(
        lambda x: process_all(x['光轴文件夹路径'], x['光瞳文件夹路径'], x['实验编号']), axis=1
    )

if __name__ == '__main__':
    main()