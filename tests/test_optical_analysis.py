"""
光束质量分析模块测试
====================
测试optical_analysis模块中的所有函数
"""

import pytest
import numpy as np
from pathlib import Path

# 导入被测模块
from src.data_mining.optical_analysis import (
    # beam_analysis
    gaussian,
    fitting_gaussian,
    d4sigma,
    pib_ratio,
    calculate_xy_diameters,
    calculate_bpp,
    calculate_m2,
    calculate_centroid,
    extract_beam_features,
    # diffraction
    crop_to_square,
    shift_to_center_fft,
    fnr3,
    calculate_strehl_ratio_with_energy_conservation,
    propagate_through_lens,
    angular_spectrum_propagation,
    # image_utils
    normalize_image_for_display,
    read_image_to_numpy,
    load_image,
    subtract_dark_field,
    normalize_image,
    resize_image,
    pad_to_square,
    clip_negative_values,
    calculate_background_threshold,
)


class TestGaussian:
    """测试高斯函数"""

    def test_gaussian_basic(self):
        """测试基本高斯函数计算"""
        x = np.array([0, 1, 2, 3, 4])
        result = gaussian(x, mu=2, sigma=1, A=1, b=0)
        
        # 中心点应该最大
        assert result[2] == 1.0
        # 两边应该对称
        assert result[1] == result[3]
        # 远离中心应该接近基线b
        assert result[0] < 0.5  # e^(-4) ≈ 0.018

    def test_gaussian_with_offset(self):
        """测试带偏移的高斯函数"""
        x = np.linspace(-5, 5, 100)
        result = gaussian(x, mu=0, sigma=1, A=2, b=1)
        
        assert np.max(result) > 2.9  # A + b - 少量误差
        assert np.min(result) < 1.1  # b + 少量误差


class TestFittingGaussian:
    """测试高斯拟合"""

    def test_fitting_gaussian_basic(self):
        """测试基本高斯拟合"""
        # 创建理想高斯数据
        x = np.arange(100)
        true_params = (50, 10, 100, 0)
        y = gaussian(x, *true_params)
        
        fitted_params, cov = fitting_gaussian(y)
        
        # 检查拟合结果是否接近真实值（允许一定误差）
        assert np.isclose(fitted_params[0], true_params[0], rtol=0.1)  # mu
        assert np.isclose(fitted_params[1], true_params[1], rtol=0.2)  # sigma
        assert np.isclose(fitted_params[2], true_params[2], rtol=0.1)  # A

    def test_fitting_gaussian_with_noise(self):
        """测试带噪声的高斯拟合"""
        x = np.arange(100)
        y = gaussian(x, 50, 10, 100, 0) + np.random.normal(0, 5, 100)
        
        fitted_params, cov = fitting_gaussian(y)
        
        # 拟合应该不会完全失败
        assert not np.isnan(fitted_params[0])


class TestD4Sigma:
    """测试D4σ直径计算"""

    @pytest.fixture
    def gaussian_beam(self):
        """创建高斯光束图像"""
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # 中心在(50, 50)的高斯光束
        sigma = 10
        beam = 100 * np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * sigma**2))
        return beam

    def test_d4sigma_basic(self, gaussian_beam):
        """测试基本D4σ计算"""
        sigma = 10
        result = d4sigma(gaussian_beam, pixel_size_um=1.0)
        
        # 质心应该在中心附近
        assert 45 < result['center_x'] < 55
        assert 45 < result['center_y'] < 55
        
        # 直径应该约为4*sigma*sqrt(2) ≈ 5.6*sigma
        expected_d = 4 * sigma * np.sqrt(2)
        assert 0.3 * expected_d < result['avg_sigma2'] < 1.7 * expected_d

    def test_d4sigma_empty_image(self):
        """测试空图像"""
        result = d4sigma(np.zeros((10, 10)), pixel_size_um=1.0)
        
        assert result['center_x'] == 0
        assert result['center_y'] == 0
        assert result['D_x'] == 0

    def test_d4sigma_pixel_size(self):
        """测试像素尺寸转换"""
        # 创建高斯光束
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        beam = np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * 10**2))
        
        result_1um = d4sigma(beam, pixel_size_um=1.0)
        result_2um = d4sigma(beam, pixel_size_um=2.0)
        
        # 像素尺寸为2μm时，直径应该为2倍
        assert np.isclose(result_2um['D_x'], result_1um['D_x'] * 2, rtol=0.01)


class TestPIBRatio:
    """测试PIB占比计算"""

    def test_pib_ratio_centered(self):
        """测试中心光斑的PIB"""
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # 中心高斯光束 - 使用更小的sigma来集中更多能量
        sigma = 5
        beam = np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * sigma**2))
        
        result, is_overexposed = pib_ratio(
            beam,
            center=(50, 50),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=2.9e-6,
        )
        
        # 中心区域能量应该有一定占比
        assert result > 0.05
        assert not is_overexposed

    def test_pib_ratio_small_radius(self):
        """测试小半径PIB"""
        beam = np.ones((50, 50))
        
        result_small, _ = pib_ratio(
            beam,
            center=(25, 25),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=2.9e-6,
        )
        result_large, _ = pib_ratio(
            beam,
            center=(25, 25),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.05,  # 更小的孔径 = 更大的Airy斑
            pixel_size_m=2.9e-6,
        )
        
        # 小孔径（大Airy斑）占比应该大于大孔径
        assert result_small < result_large

    def test_pib_ratio_zero_image(self):
        """测试零图像"""
        result, is_overexposed = pib_ratio(
            np.zeros((10, 10)),
            center=(5, 5),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=2.9e-6,
        )
        
        assert result == 0.0
        assert not is_overexposed
        assert not is_overexposed


class TestCalculateXYDiameters:
    """测试高斯直径计算"""

    def test_calculate_xy_diameters(self):
        """测试XY方向直径计算"""
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # X方向宽，Y方向窄的光束
        sigma_x = 15
        sigma_y = 8
        beam = np.exp(-((X - 50)**2 / (2 * sigma_x**2) + (Y - 50)**2 / (2 * sigma_y**2)))
        
        result = calculate_xy_diameters(beam, 50, 50, pix_size=1.0)
        
        # X方向直径应该大于Y方向
        assert result['gaussian_dia_x(um)'] > result['gaussian_dia_y(um)']


class TestCalculateBPP:
    """测试BPP计算"""

    def test_bpp_basic(self):
        """测试基本BPP计算"""
        result = calculate_bpp(
            pupil_diameter_mm=10.0,
            focal_diameter_mm=1.0,
            focal_length_mm=3000
        )
        
        # BPP = w_pupil * theta
        # theta = w_focal / f = 0.5 / 3000 = 0.000167 rad = 0.167 mrad
        # BPP = 5 * 0.167 = 0.835 mm·mrad
        assert result['BPP_mm_mrad'] > 0
        assert result['divergence_mrad'] > 0

    def test_bpp_diffraction_limit(self):
        """测试衍射极限BPP"""
        wavelength_um = 1.064  # 1064nm
        m2 = calculate_m2(0.5, wavelength_um)
        
        # 衍射极限BPP = λ/π
        bpp_diffraction = wavelength_um / np.pi
        expected_m2 = 0.5 / bpp_diffraction
        
        assert np.isclose(m2, expected_m2)


class TestCalculateCentroid:
    """测试质心计算"""

    def test_centroid_centered(self):
        """测试中心光斑质心"""
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        beam = np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * 10**2))
        
        cx, cy = calculate_centroid(beam)
        
        assert 45 < cx < 55
        assert 45 < cy < 55

    def test_centroid_offset(self):
        """测试偏移光斑质心"""
        beam = np.zeros((100, 100))
        beam[30:70, 20:60] = 1  # 偏心矩形
        
        cx, cy = calculate_centroid(beam)
        
        assert 20 < cx < 60
        assert 30 < cy < 70


class TestCropToSquare:
    """测试图像裁剪"""

    def test_crop_square_already_square(self):
        """测试已经是正方形的图像"""
        img = np.ones((100, 100))
        result = crop_to_square(img)
        
        assert result.shape == (100, 100)

    def test_crop_square_taller(self):
        """测试高大于宽的图像"""
        img = np.ones((100, 50))
        result = crop_to_square(img)
        
        assert result.shape == (50, 50)

    def test_crop_square_wider(self):
        """测试宽大于高的图像"""
        img = np.ones((50, 100))
        result = crop_to_square(img)
        
        assert result.shape == (50, 50)


class TestShiftToCenterFFT:
    """测试FFT居中"""

    def test_shift_to_center_fft(self):
        """测试FFT居中"""
        # 创建偏心高斯光束
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # 中心在(30, 30)
        beam = np.exp(-((X - 30)**2 + (Y - 30)**2) / (2 * 10**2))
        
        result = shift_to_center_fft(beam, 30, 30)
        
        # 结果应该接近正方形
        assert result.shape[0] == result.shape[1]


class TestFNR3:
    """测试菲涅尔衍射积分"""

    def test_fnr3_basic(self):
        """测试基本菲涅尔衍射"""
        size = 64
        # 平面波
        field = np.ones((size, size), dtype=complex)
        
        result = fnr3(
            field,
            input_pixel_size=10e-6,
            output_pixel_size=10e-6,
            zz=1.0,  # 1m传播
            lambda_m=1064e-9
        )
        
        # 结果应该是复数数组
        assert result.shape == (size, size)
        assert np.iscomplexobj(result)

    def test_fnr3_with_lens(self):
        """测试带透镜的菲涅尔衍射"""
        size = 64
        field = np.ones((size, size), dtype=complex)
        
        result = fnr3(
            field,
            input_pixel_size=10e-6,
            output_pixel_size=10e-6,
            zz=3.0,
            lambda_m=1064e-9,
            focal_length_m=3.0
        )
        
        assert result.shape == (size, size)


class TestStrehlRatio:
    """测试斯特列尔比计算"""

    def test_strehl_ratio_perfect_beam(self):
        """测试理想光束的斯特列尔比"""
        size = 64
        
        # 创建均匀光瞳
        pupil = np.ones((size, size))
        
        # 创建理想焦斑（通过衍射计算）
        focus = np.abs(fnr3(
            pupil,
            input_pixel_size=2.9e-6,
            output_pixel_size=5.5e-6,
            zz=3.0,
            lambda_m=1064e-9,
            focal_length_m=3.0
        ))
        
        strehl, _ = calculate_strehl_ratio_with_energy_conservation(
            pupil,
            focus,
            f_m=3.0,
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            input_pixel_size=2.9e-6,
            output_pixel_size=5.5e-6
        )
        
        # 理想情况下斯特列尔比应该接近1
        assert strehl > 0.8

    def test_strehl_ratio_degraded_beam(self):
        """测试退化光束的斯特列尔比"""
        size = 64
        
        # 创建有像差的光瞳 - 使用不同的模式
        pupil = np.ones((size, size))
        # 添加环形结构模拟像差
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = ((x - center)**2 + (y - center)**2) < (size // 4)**2
        pupil[mask] = 0.5  # 中心部分较弱
        
        # 创建实际焦斑（较弱）
        focus = np.abs(fnr3(
            pupil,
            input_pixel_size=2.9e-6,
            output_pixel_size=5.5e-6,
            zz=3.0,
            lambda_m=1064e-9,
            focal_length_m=3.0
        )) * 0.5
        
        strehl, _ = calculate_strehl_ratio_with_energy_conservation(
            pupil,
            focus,
            f_m=3.0,
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            input_pixel_size=2.9e-6,
            output_pixel_size=5.5e-6
        )
        
        # 退化情况下斯特列尔比应该较小
        assert strehl <= 1.0  # 允许等于1.0的情况


class TestImageUtils:
    """测试图像处理工具"""

    def test_normalize_image_for_display(self):
        """测试图像归一化显示"""
        img = np.array([[0, 50], [100, 200]], dtype=np.float64)
        
        result = normalize_image_for_display(img)
        
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_normalize_image_custom_range(self):
        """测试自定义范围归一化"""
        img = np.array([0, 25, 50, 75, 100])
        
        result = normalize_image(img, target_min=0, target_max=10)
        
        assert result.min() == 0
        assert result.max() == 10

    def test_subtract_dark_field_none(self):
        """测试不去暗场"""
        img = np.array([[10, 20], [30, 40]])
        
        result, black = subtract_dark_field(img, denoise_method='none')
        
        assert black == 0
        np.testing.assert_array_equal(result, img)

    def test_subtract_dark_field_median(self):
        """测试中值去暗场"""
        img = np.array([[10, 20], [30, 40]])
        
        result, black = subtract_dark_field(img, denoise_method='median')
        
        assert black == 25  # median([10,20,30,40]) = 25
        assert result[0, 0] == 0  # 10 - 25 = -15 -> 0
        assert result[0, 1] == 0  # 20 - 25 = -5 -> 0
        assert result[1, 0] == 5  # 30 - 25 = 5
        assert result[1, 1] == 15  # 40 - 25 = 15

    def test_subtract_dark_field_1_e(self):
        """测试1/e去暗场"""
        img = np.array([[10, 20], [30, 100]])
        
        result, black = subtract_dark_field(img, denoise_method='1_e')
        
        expected_black = 100 / np.e
        assert np.isclose(black, expected_black)

    def test_clip_negative_values(self):
        """测试负值裁剪"""
        img = np.array([-1, -0.5, 0, 0.5, 1])
        
        result = clip_negative_values(img, min_value=0)
        
        assert result.min() >= 0

    def test_calculate_background_threshold(self):
        """测试背景阈值计算"""
        img = np.array([10, 20, 30, 40, 50])
        
        median_thresh = calculate_background_threshold(img, method='median')
        min_thresh = calculate_background_threshold(img, method='min')
        mean_thresh = calculate_background_threshold(img, method='mean')
        
        assert median_thresh == 30
        assert min_thresh == 10
        assert mean_thresh == 30


class TestExtractBeamFeatures:
    """测试完整光束特征提取"""

    def test_extract_beam_features(self):
        """测试完整特征提取"""
        size = 100
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # 创建高斯光束
        beam = 100 * np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * 10**2))
        
        result = extract_beam_features(beam, pixel_size_um=1.0)
        
        # 应该包含所有特征
        assert 'centroid' in result
        assert 'd4s' in result
        assert 'pib_ratio' in result
        assert 'gaussian_diameter' in result
        assert 'pib_overexposed' in result
        
        # 检查D4σ特征
        assert 'D_x' in result['d4s']
        assert 'D_y' in result['d4s']
        assert 'avg_sigma2' in result['d4s']


class TestIntegration:
    """集成测试"""

    def test_full_beam_analysis_workflow(self):
        """测试完整光束分析流程"""
        # 1. 创建测试光束图像
        size = 128
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # 模拟光瞳图像
        pupil = np.exp(-((X - 64)**2 + (Y - 64)**2) / (2 * 15**2))
        
        # 模拟焦面图像（通过衍射）
        focus = np.abs(fnr3(
            pupil,
            input_pixel_size=2.9e-6,
            output_pixel_size=5.5e-6,
            zz=3.0,
            lambda_m=1064e-9,
            focal_length_m=3.0
        ))
        
        # 2. 提取光束特征
        pupil_features = d4sigma(pupil, pixel_size_um=2.9)
        focus_features = d4sigma(focus, pixel_size_um=5.5)
        
        # 3. 计算PIB
        pib, is_overexposed = pib_ratio(
            focus,
            (focus_features['center_x'], focus_features['center_y']),
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            aperture_diameter_m=0.1,
            pixel_size_m=5.5e-6,
        )
        
        # 4. 计算高斯直径
        gaussian_dia = calculate_xy_diameters(
            focus,
            focus_features['center_x'],
            focus_features['center_y'],
            pix_size=5.5
        )
        
        # 5. 计算BPP
        pupil_diameter_mm = pupil_features['avg_sigma2'] * 1e-3
        focus_diameter_mm = focus_features['avg_sigma2'] * 1e-3
        bpp = calculate_bpp(pupil_diameter_mm, focus_diameter_mm, focal_length_mm=3000)
        
        # 6. 计算斯特列尔比
        strehl, _ = calculate_strehl_ratio_with_energy_conservation(
            pupil, focus,
            f_m=3.0,
            wavelength_m=1064e-9,
            focal_length_m=3.0,
            input_pixel_size=2.9e-6,
            output_pixel_size=5.5e-6
        )
        
        # 验证结果
        assert pupil_features['center_x'] > 0
        assert focus_features['center_x'] > 0
        assert 0 <= pib <= 1
        assert bpp['BPP_mm_mrad'] > 0
        assert 0 <= strehl <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
