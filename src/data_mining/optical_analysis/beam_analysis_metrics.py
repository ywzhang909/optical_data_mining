"""
Module containing beam analysis metrics and calculations
"""

import numpy as np
import cv2
import math
from . import (
    d4sigma,
    pib_ratio,
    calculate_xy_diameters,
    calculate_bpp,
    shift_to_center_fft,
    calculate_strehl_ratio_with_energy_conservation,
    read_image_to_numpy,
    subtract_dark_field,
)


def convert_to_cv(float_image) -> np.ndarray:
    """
    将图像转换为OpenCV兼容的uint8格式
    """
    assert isinstance(float_image, np.ndarray), (
        f"{float_image} must be a numpy array, but got {type(float_image)}"
    )
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
    noise_threshold = np.max(denoised_image) * (1 / math.e)
    denoised_image = np.where(denoised_image > noise_threshold, denoised_image, 0)
    # 确保数据类型正确
    if denoised_image.dtype != np.uint8:
        denoised_image = denoised_image.astype(np.uint8)
    # 步骤 3: 去除噪声
    denoised_image = cv2.fastNlMeansDenoising(denoised_image, None, 10, 7, 21)
    # denoised_image = cv2.Canny(image, 1, 1)
    # 步骤 4: 二值化
    near_binary_img = cv2.threshold(
        denoised_image, noise_threshold, 255, cv2.THRESH_BINARY
    )[1]
    # 步骤 5: 拟合一个圆形正好包含光斑
    contours, _ = cv2.findContours(
        near_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
    else:
        x, y = np.nan, np.nan
        radius = np.nan

    return {"border_x": x, "border_y": y, "border_radius": radius}


def ellipse_fit(uint8_image):
    # 计算噪声阈值（使用30%作为阈值）
    noise_threshhold = np.max(uint8_image) * 0.3

    # 二值化处理
    binary_image = cv2.threshold(uint8_image, noise_threshhold, 255, cv2.THRESH_BINARY)[
        1
    ]

    # 查找轮廓
    try:
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert contours, "No contours found"
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        (ellipse_center_x, ellipse_center_y), (short_axis, long_axis), angle = (
            cv2.fitEllipse(largest_contour)
        )
    except (AssertionError, ValueError):
        return {
            "ellipse_center_x": np.nan,
            "ellipse_center_y": np.nan,
            "short_axis": np.nan,
            "long_axis": np.nan,
            "ellipticity": np.nan,
            "angle": np.nan,
            "uniformity": np.nan,
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
        "ellipse_center_x": ellipse_center_x,
        "ellipse_center_y": ellipse_center_y,
        "short_axis": short_axis,
        "long_axis": long_axis,
        "ellipticity": long_axis / short_axis,
        "angle": angle,
        "uniformity": uniformity,
    }


class BeamAnalysisMetrics:
    """Class to encapsulate beam analysis metrics and calculations"""

    def __init__(
        self, wavelength_nm=1064, focal_length_mm=3000, aperture_diameter_mm=100
    ):
        """
        Initialize beam analysis metrics calculator

        Args:
            wavelength_nm: Wavelength in nanometers
            focal_length_mm: Focal length in millimeters
            aperture_diameter_mm: Aperture diameter in millimeters
        """
        self.wavelength_nm = wavelength_nm
        self.focal_length_mm = focal_length_mm
        self.aperture_diameter_mm = aperture_diameter_mm

    def calculate_all_metrics(
        self,
        axis_img,
        pupil_img,
        axis_pixel_um,
        pupil_pixel_um,
        denoise_method="none",
        manual_threshold=None,
    ):
        """
        Calculate all beam analysis metrics

        Args:
            axis_img: Axis image array
            pupil_img: Pupil image array
            axis_pixel_um: Axis camera pixel size in micrometers
            pupil_pixel_um: Pupil camera pixel size in micrometers
            denoise_method: Method for dark field subtraction
            manual_threshold: Manual threshold value if denoise_method is 'manual'

        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert pixel sizes to meters
        axis_pixel_m = axis_pixel_um * 1e-6
        pupil_pixel_m = pupil_pixel_um * 1e-6

        # Denoise images
        axis_denoise, axis_black = subtract_dark_field(
            axis_img, denoise_method, manual_threshold
        )
        pupil_denoise, pupil_black = subtract_dark_field(
            pupil_img, denoise_method, manual_threshold
        )

        # Calculate D4σ features
        axis_features = d4sigma(axis_denoise, axis_pixel_um)
        pupil_features = d4sigma(pupil_denoise, pupil_pixel_um)

        # Calculate PIB ratio
        wavelength_m = self.wavelength_nm * 1e-9
        focal_length_m = self.focal_length_mm * 1e-3
        aperture_diameter_m = self.aperture_diameter_mm * 1e-3

        axis_pib, is_overexposed = pib_ratio(
            axis_denoise,
            (axis_features["center_x"], axis_features["center_y"]),
            wavelength_m=wavelength_m,
            focal_length_m=focal_length_m,
            aperture_diameter_m=aperture_diameter_m,
            pixel_size_m=axis_pixel_m,
        )

        # Calculate Gaussian fitting
        axis_gaussian = calculate_xy_diameters(
            axis_denoise,
            axis_features["center_x"],
            axis_features["center_y"],
            axis_pixel_um,
        )

        # FFT centering
        axis_shifted = shift_to_center_fft(
            axis_denoise, axis_features["center_x"], axis_features["center_y"]
        )
        pupil_shifted = shift_to_center_fft(
            pupil_denoise, pupil_features["center_x"], pupil_features["center_y"]
        )

        # Calculate Strehl ratio
        strehl, ideal_matched = calculate_strehl_ratio_with_energy_conservation(
            pupil_shifted,
            axis_shifted,
            f_m=3,
            wavelength_m=wavelength_m,
            focal_length_m=focal_length_m,
            input_pixel_size=pupil_pixel_m,
            output_pixel_size=axis_pixel_m,
        )

        # Calculate BPP
        axis_diameter_mm = axis_features["avg_diameter"] * 1e-3
        pupil_diameter_mm = pupil_features["avg_diameter"] * 1e-3

        bpp_result = calculate_bpp(
            pupil_diameter_mm, axis_diameter_mm, self.focal_length_mm
        )

        # Calculate M²
        lambda_um = self.wavelength_nm / 1000
        bpp_diffraction = lambda_um / np.pi
        m2 = bpp_result["BPP_mm_mrad"] / bpp_diffraction

        # Calculate elliptical fitting parameters
        axis_uint8 = convert_to_cv(axis_denoise)
        pupil_uint8 = convert_to_cv(pupil_denoise)

        axis_ellipse_params = ellipse_fit(axis_uint8)
        pupil_ellipse_params = ellipse_fit(pupil_uint8)

        axis_border_params = find_spot_border(axis_denoise)
        pupil_border_params = find_spot_border(pupil_denoise)

        # Compile all results
        results = {
            # D4σ metrics
            "axis_D4σ_X_um": axis_features["D_x"],
            "axis_D4σ_Y_um": axis_features["D_y"],
            "axis_avg_diameter_um": axis_features["avg_diameter"],
            "axis_center_intensity": axis_features["center_intensity"],
            "pupil_D4σ_X_um": pupil_features["D_x"],
            "pupil_D4σ_Y_um": pupil_features["D_y"],
            "pupil_avg_diameter_um": pupil_features["avg_diameter"],
            "pupil_center_intensity": pupil_features["center_intensity"],
            # PIB metrics
            "axis_PIB_ratio": axis_pib,
            "is_overexposed": is_overexposed,
            # Gaussian fitting metrics
            "axis_gaussian_diameter_X_um": axis_gaussian["gaussian_dia_x(um)"],
            "axis_gaussian_diameter_Y_um": axis_gaussian["gaussian_dia_y(um)"],
            # Elliptical fitting metrics
            "axis_ellipse_center_x": axis_ellipse_params["ellipse_center_x"],
            "axis_ellipse_center_y": axis_ellipse_params["ellipse_center_y"],
            "axis_short_axis": axis_ellipse_params["short_axis"],
            "axis_long_axis": axis_ellipse_params["long_axis"],
            "axis_ellipticity": axis_ellipse_params["ellipticity"],
            "axis_angle": axis_ellipse_params["angle"],
            "axis_uniformity": axis_ellipse_params["uniformity"],
            "pupil_ellipse_center_x": pupil_ellipse_params["ellipse_center_x"],
            "pupil_ellipse_center_y": pupil_ellipse_params["ellipse_center_y"],
            "pupil_short_axis": pupil_ellipse_params["short_axis"],
            "pupil_long_axis": pupil_ellipse_params["long_axis"],
            "pupil_ellipticity": pupil_ellipse_params["ellipticity"],
            "pupil_angle": pupil_ellipse_params["angle"],
            "pupil_uniformity": pupil_ellipse_params["uniformity"],
            # Border fitting metrics
            "axis_border_x": axis_border_params["border_x"],
            "axis_border_y": axis_border_params["border_y"],
            "axis_border_radius": axis_border_params["border_radius"],
            "pupil_border_x": pupil_border_params["border_x"],
            "pupil_border_y": pupil_border_params["border_y"],
            "pupil_border_radius": pupil_border_params["border_radius"],
            # Strehl ratio
            "strehl_ratio": strehl,
            # BPP and M² metrics
            "BPP_mm_mrad": bpp_result["BPP_mm_mrad"],
            "divergence_mrad": bpp_result["divergence_mrad"],
            "M2": m2,
            "diffraction_limit_BPP_mm_mrad": bpp_diffraction,
            # Additional intermediate values for reference
            "axis_features": axis_features,
            "pupil_features": pupil_features,
            "axis_gaussian_params": axis_gaussian,
            "axis_ellipse_params": axis_ellipse_params,
            "pupil_ellipse_params": pupil_ellipse_params,
            "axis_border_params": axis_border_params,
            "pupil_border_params": pupil_border_params,
            "bpp_params": bpp_result,
            "is_overexposed_warning": is_overexposed,
            "axis_shifted_image": axis_shifted,
            "pupil_shifted_image": pupil_shifted,
            "ideal_matched_image": ideal_matched,
        }

        return results

    def add_custom_metric(self, metric_name, calculation_function):
        """
        Dynamically add a new metric calculation function

        Args:
            metric_name: Name of the metric
            calculation_function: Function that takes (axis_img, pupil_img, params) and returns the metric value
        """
        setattr(self, metric_name, calculation_function)

    def get_supported_metrics(self):
        """
        Get list of supported metrics

        Returns:
            List of metric names
        """
        return [
            "axis_D4σ_X_um",
            "axis_D4σ_Y_um",
            "axis_avg_diameter_um",
            "axis_center_intensity",
            "pupil_D4σ_X_um",
            "pupil_D4σ_Y_um",
            "pupil_avg_diameter_um",
            "pupil_center_intensity",
            "axis_PIB_ratio",
            "is_overexposed",
            "axis_gaussian_diameter_X_um",
            "axis_gaussian_diameter_Y_um",
            # Elliptical fitting metrics
            "axis_ellipse_center_x",
            "axis_ellipse_center_y",
            "axis_short_axis",
            "axis_long_axis",
            "axis_ellipticity",
            "axis_angle",
            "axis_uniformity",
            "pupil_ellipse_center_x",
            "pupil_ellipse_center_y",
            "pupil_short_axis",
            "pupil_long_axis",
            "pupil_ellipticity",
            "pupil_angle",
            "pupil_uniformity",
            # Border fitting metrics
            "axis_border_x",
            "axis_border_y",
            "axis_border_radius",
            "pupil_border_x",
            "pupil_border_y",
            "pupil_border_radius",
            # Other metrics
            "strehl_ratio",
            "BPP_mm_mrad",
            "divergence_mrad",
            "M2",
            "diffraction_limit_BPP_mm_mrad",
        ]


def calculate_custom_metric_example(axis_img, pupil_img, params):
    """
    Example of how to implement a custom metric
    This would typically be implemented by the user
    """
    # Example: Peak intensity ratio
    axis_peak = np.max(axis_img)
    pupil_peak = np.max(pupil_img)
    return axis_peak / pupil_peak if pupil_peak != 0 else 0
