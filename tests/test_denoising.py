#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像降噪模块测试
"""

import numpy as np
import pytest
import cv2
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_mining.image.denoising import (
    bm3d_denoise,
    gaussian_denoise,
    median_denoise,
    nlmeans_denoise,
    bilateral_denoise,
    wavelet_denoise,
    total_variation_denoise,
    wiener_denoise,
    denoise,
    batch_denoise,
    estimate_noise_sigma,
    evaluate_denoising,
    DenoiseMethod,
    DenoiseResult,
    QualityMetrics,
    BM3D_AVAILABLE,
)


@pytest.fixture
def clean_image():
    size = 128
    x = np.arange(size) - (size // 2)
    y = np.arange(size) - (size // 2)
    xx, yy = np.meshgrid(x, y)
    sigma = 10.0
    img = 255 * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return img.astype(np.uint8)


@pytest.fixture
def noisy_image(clean_image):
    noise = np.random.normal(0, 25, clean_image.shape).astype(np.int16)
    noisy = clean_image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.fixture
def noisy_image_float(clean_image):
    noise = np.random.normal(0, 0.1, clean_image.shape)
    return np.clip(clean_image.astype(np.float32) / 255.0 + noise, 0, 1)


@pytest.fixture
def color_noisy_image(clean_image):
    color_img = np.stack([clean_image, clean_image, clean_image], axis=2)
    noise = np.random.normal(0, 25, color_img.shape).astype(np.int16)
    noisy = color_img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


class TestDenoiseResult:
    def test_creation(self, clean_image):
        result = DenoiseResult(
            image=clean_image,
            method="Test",
            parameters={"param1": 1},
            noise_estimate=25.0
        )
        assert isinstance(result.image, np.ndarray)
        assert result.method == "Test"
        assert result.noise_estimate == 25.0

    def test_array_conversion(self, clean_image):
        result = DenoiseResult(image=clean_image, method="Test", parameters={})
        arr = np.array(result)
        assert np.array_equal(arr, clean_image)


@pytest.mark.skipif(not BM3D_AVAILABLE, reason="bm3d not installed")
class TestBM3DDenoise:
    def test_uint8(self, noisy_image):
        result = bm3d_denoise(noisy_image, sigma_psd=25)
        assert isinstance(result, DenoiseResult)
        assert result.method == "BM3D"
        assert result.image.shape == noisy_image.shape

    def test_float(self, noisy_image_float):
        result = bm3d_denoise(noisy_image_float, sigma_psd=0.1)
        assert result.image.shape == noisy_image_float.shape

    def test_color(self, color_noisy_image):
        result = bm3d_denoise(color_noisy_image, sigma_psd=25)
        assert result.image.shape == color_noisy_image.shape


class TestGaussianDenoise:
    def test_basic(self, noisy_image):
        result = gaussian_denoise(noisy_image, kernel_size=5, sigma=1.5)
        assert isinstance(result, DenoiseResult)
        assert result.method == "Gaussian"
        assert result.image.shape == noisy_image.shape

    def test_even_kernel(self, noisy_image):
        result = gaussian_denoise(noisy_image, kernel_size=4, sigma=1.5)
        assert result.parameters["kernel_size"] == 5

    def test_reduces_noise(self, noisy_image):
        result = gaussian_denoise(noisy_image, kernel_size=5, sigma=2.0)
        noise_var_before = np.var(noisy_image.astype(np.float32))
        noise_var_after = np.var(result.image.astype(np.float32))
        assert noise_var_after < noise_var_before


class TestMedianDenoise:
    def test_basic(self, noisy_image):
        result = median_denoise(noisy_image, kernel_size=5)
        assert result.method == "Median"

    def test_salt_pepper(self):
        img = np.ones((100, 100), dtype=np.uint8) * 128
        img[::10, ::10] = 0
        img[5::10, 5::10] = 255
        result = median_denoise(img, kernel_size=5)
        assert np.abs(result.image.mean() - 128) < 10


class TestNLMeansDenoise:
    def test_gray(self, noisy_image):
        result = nlmeans_denoise(noisy_image, h=10)
        assert result.method == "NLMeans"

    def test_color(self, color_noisy_image):
        result = nlmeans_denoise(color_noisy_image, h=10)
        assert result.image.shape == color_noisy_image.shape


class TestBilateralDenoise:
    def test_basic(self, noisy_image):
        result = bilateral_denoise(noisy_image, d=9, sigma_color=75, sigma_space=75)
        assert result.method == "Bilateral"


@pytest.mark.skipif(True, reason="PyWavelets optional")
class TestWaveletDenoise:
    def test_basic(self, noisy_image):
        result = wavelet_denoise(noisy_image, wavelet='db1', level=2)
        assert result.method == "Wavelet"


class TestTotalVariationDenoise:
    def test_basic(self, noisy_image):
        result = total_variation_denoise(noisy_image, weight=0.1, max_iter=50)
        assert result.method == "TotalVariation"


class TestWienerDenoise:
    def test_basic(self, noisy_image):
        result = wiener_denoise(noisy_image, kernel_size=5)
        assert result.method == "Wiener"


class TestDenoiseInterface:
    def test_string_method(self, noisy_image):
        result = denoise(noisy_image, 'gaussian', kernel_size=5)
        assert result.method == "Gaussian"

    def test_enum_method(self, noisy_image):
        result = denoise(noisy_image, DenoiseMethod.MEDIAN, kernel_size=5)
        assert result.method == "Median"

    def test_invalid_method(self, noisy_image):
        with pytest.raises(ValueError):
            denoise(noisy_image, 'invalid_method')


class TestBatchDenoise:
    def test_basic(self, noisy_image, clean_image):
        images = [noisy_image, clean_image, noisy_image.copy()]
        results = batch_denoise(images, 'gaussian', kernel_size=5)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, DenoiseResult)


class TestNoiseEstimation:
    def test_mad(self, noisy_image):
        sigma = estimate_noise_sigma(noisy_image, method='mad')
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_std(self, noisy_image):
        sigma = estimate_noise_sigma(noisy_image, method='std')
        assert isinstance(sigma, float)
        assert sigma > 0


class TestDenoisingEvaluation:
    def test_with_original(self, clean_image, noisy_image):
        denoised = cv2.GaussianBlur(noisy_image, (5, 5), 1.5)
        metrics = evaluate_denoising(clean_image, denoised, noisy_image)
        assert isinstance(metrics, QualityMetrics)
        if metrics.psnr:
            assert 0 < metrics.psnr < 100

    def test_without_original(self, noisy_image):
        denoised = cv2.GaussianBlur(noisy_image, (5, 5), 1.5)
        metrics = evaluate_denoising(None, denoised, noisy_image)
        assert isinstance(metrics, QualityMetrics)


class TestDenoiseMethod:
    def test_enum_values(self):
        assert DenoiseMethod.BM3D.value == 'bm3d'
        assert DenoiseMethod.GAUSSIAN.value == 'gaussian'

    def test_from_string(self):
        assert DenoiseMethod.from_string('gaussian') == DenoiseMethod.GAUSSIAN
        with pytest.raises(ValueError):
            DenoiseMethod.from_string('invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
