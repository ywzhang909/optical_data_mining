"""
Module containing beam analysis metrics and calculations
"""

import numpy as np
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
