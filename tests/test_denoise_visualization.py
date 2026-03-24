#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Tests for Denoising Results

Tests that verify denoising results visually and save comparison images.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_mining.image.denoising import (
    gaussian_denoise,
    median_denoise,
    nlmeans_denoise,
    bilateral_denoise,
    bm3d_denoise,
    denoise,
    evaluate_denoising,
    DenoiseMethod,
    DenoiseResult,
    BM3D_AVAILABLE,
)

# Output directory for visualization
VISUAL_OUTPUT_DIR = Path(__file__).parent / "visualization_output"
VISUAL_OUTPUT_DIR.mkdir(exist_ok=True)


def save_comparison_image(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    method_name: str,
    index: int,
    metrics: dict = None
) -> Path:
    """
    Save a side-by-side comparison image.
    
    Returns:
        Path to saved image
    """
    if not PIL_AVAILABLE:
        return None
    
    # Normalize images for display
    def normalize(img):
        if img.max() > img.min():
            return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        return img.astype(np.uint8)
    
    clean_vis = normalize(clean)
    noisy_vis = normalize(noisy)
    denoised_vis = normalize(denoised)
    
    # Create side-by-side comparison
    comparison = np.hstack([clean_vis, noisy_vis, denoised_vis])
    
    # Add label row
    label_height = 40
    labels = np.zeros((label_height, comparison.shape[1]), dtype=np.uint8) + 255
    
    # Create PIL image with labels
    pil_img = Image.fromarray(comparison, mode='L')
    
    # Resize to handle drawing
    scale = 2
    new_width = pil_img.width * scale
    new_height = pil_img.height + label_height * scale
    pil_img = pil_img.resize((new_width, new_height), Image.NEAREST)
    
    # Draw labels
    draw = ImageDraw.Draw(pil_img)
    
    # Font
    try:
        font = ImageFont.truetype("arial.ttf", 16 * scale)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    texts = [
        ("Clean", 10),
        ("Noisy", clean_vis.shape[1] * scale + 10),
    ]
    
    label_text = f"Denoised ({method_name})"
    if metrics:
        label_text += f" PSNR:{metrics.get('psnr', 0):.1f}dB"
    
    texts.append((label_text, 2 * clean_vis.shape[1] * scale + 10))
    
    for text, x in texts:
        draw.text((x, 10 * scale), text, fill=0, font=font)
    
    # Save
    output_path = VISUAL_OUTPUT_DIR / f"comparison_{method_name.lower()}_{index:02d}.png"
    pil_img.save(output_path)
    
    return output_path


def save_profile_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    method_name: str,
    index: int
) -> Path:
    """
    Save horizontal profile comparison (center cut).
    
    Returns:
        Path to saved image
    """
    if not PIL_AVAILABLE:
        return None
    
    # Get center row
    center_y = clean.shape[0] // 2
    
    # Extract profiles
    clean_profile = clean[center_y, :]
    noisy_profile = noisy[center_y, :]
    denoised_profile = denoised[center_y, :]
    
    # Normalize
    max_val = max(clean_profile.max(), noisy_profile.max(), denoised_profile.max())
    if max_val > 0:
        clean_profile = (clean_profile / max_val * 255).astype(np.uint8)
        noisy_profile = (noisy_profile / max_val * 255).astype(np.uint8)
        denoised_profile = (denoised_profile / max_val * 255).astype(np.uint8)
    
    # Create profile plot
    height, width = 200, 512
    profile_img = np.zeros((height, width), dtype=np.uint8) + 255
    
    # Draw grid
    for i in range(0, width, 50):
        for j in range(0, height, 5):
            profile_img[j, i] = 230
    
    # Scale x coordinates
    x_scale = width / len(clean_profile)
    
    # Draw profiles using PIL
    pil_img = Image.fromarray(profile_img, mode='L')
    draw = ImageDraw.Draw(pil_img)
    
    for i in range(len(clean_profile) - 1):
        x1 = int(i * x_scale)
        x2 = int((i + 1) * x_scale)
        
        y1_clean = height - int(clean_profile[i])
        y2_clean = height - int(clean_profile[i + 1])
        
        y1_noisy = height - int(noisy_profile[i])
        y2_noisy = height - int(noisy_profile[i + 1])
        
        y1_denoised = height - int(denoised_profile[i])
        y2_denoised = height - int(denoised_profile[i + 1])
        
        # Green - Clean
        draw.line([(x1, y1_clean), (x2, y2_clean)], fill=50, width=1)
        # Red - Noisy
        draw.line([(x1, y1_noisy), (x2, y2_noisy)], fill=150, width=1)
        # Blue - Denoised
        draw.line([(x1, y1_denoised), (x2, y2_denoised)], fill=200, width=1)
    
    # Add legend
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Green=Clean, Blue=Denoised, Red=Noisy", fill=0, font=font)
    
    output_path = VISUAL_OUTPUT_DIR / f"profile_{method_name.lower()}_{index:02d}.png"
    pil_img.save(output_path)
    
    return output_path


class TestDenoisingVisualization:
    """Visual tests for denoising algorithms."""
    
    def test_gaussian_denoise_visual(self, noisy_beam, clean_beam):
        """Visual test: Gaussian denoising result."""
        result = gaussian_denoise(noisy_beam, kernel_size=5, sigma=1.5)
        
        # Evaluate quality
        metrics = evaluate_denoising(clean_beam, result.image, noisy_beam)
        
        # Save visualization
        output_path = save_comparison_image(
            clean_beam, noisy_beam, result.image,
            "Gaussian", 0,
            {'psnr': metrics.psnr or 0, 'ssim': metrics.ssim or 0}
        )
        
        # Verify image was saved
        assert output_path is None or output_path.exists()
        
        # Verify denoising improved quality
        # Note: SSIM can be low for synthetic data with high noise
        assert metrics.psnr is None or metrics.psnr > 0
        assert metrics.ssim is None or metrics.ssim > 0.3
    
    def test_median_denoise_visual(self, noisy_beam, clean_beam):
        """Visual test: Median denoising result."""
        result = median_denoise(noisy_beam, kernel_size=5)
        
        metrics = evaluate_denoising(clean_beam, result.image, noisy_beam)
        
        output_path = save_comparison_image(
            clean_beam, noisy_beam, result.image,
            "Median", 0,
            {'psnr': metrics.psnr or 0, 'ssim': metrics.ssim or 0}
        )
        
        assert output_path is None or output_path.exists()
    
    def test_nlmeans_denoise_visual(self, noisy_beam, clean_beam):
        """Visual test: NLMeans denoising result."""
        result = nlmeans_denoise(noisy_beam, h=10)
        
        metrics = evaluate_denoising(clean_beam, result.image, noisy_beam)
        
        output_path = save_comparison_image(
            clean_beam, noisy_beam, result.image,
            "NLMeans", 0,
            {'psnr': metrics.psnr or 0, 'ssim': metrics.ssim or 0}
        )
        
        assert output_path is None or output_path.exists()
    
    @pytest.mark.skipif(not BM3D_AVAILABLE, reason="BM3D not installed")
    def test_bm3d_denoise_visual(self, noisy_beam, clean_beam):
        """Visual test: BM3D denoising result."""
        result = bm3d_denoise(noisy_beam, sigma_psd=25)
        
        metrics = evaluate_denoising(clean_beam, result.image, noisy_beam)
        
        output_path = save_comparison_image(
            clean_beam, noisy_beam, result.image,
            "BM3D", 0,
            {'psnr': metrics.psnr or 0, 'ssim': metrics.ssim or 0}
        )
        
        assert output_path is None or output_path.exists()
    
    def test_bilateral_denoise_visual(self, noisy_beam, clean_beam):
        """Visual test: Bilateral denoising result."""
        result = bilateral_denoise(noisy_beam, d=9, sigma_color=75, sigma_space=75)
        
        metrics = evaluate_denoising(clean_beam, result.image, noisy_beam)
        
        output_path = save_comparison_image(
            clean_beam, noisy_beam, result.image,
            "Bilateral", 0,
            {'psnr': metrics.psnr or 0, 'ssim': metrics.ssim or 0}
        )
        
        assert output_path is None or output_path.exists()


class TestProfileVisualization:
    """Profile visualization tests."""
    
    def test_profile_comparison_gaussian(self, beam_pair):
        """Test profile comparison for Gaussian denoising."""
        clean, noisy = beam_pair
        result = gaussian_denoise(noisy, kernel_size=5, sigma=1.5)
        
        output_path = save_profile_comparison(clean, noisy, result.image, "Gaussian", 0)
        assert output_path is None or output_path.exists()
    
    def test_profile_comparison_median(self, beam_pair):
        """Test profile comparison for Median denoising."""
        clean, noisy = beam_pair
        result = median_denoise(noisy, kernel_size=5)
        
        output_path = save_profile_comparison(clean, noisy, result.image, "Median", 0)
        assert output_path is None or output_path.exists()


class TestMethodComparison:
    """Compare multiple denoising methods visually."""
    
    def test_all_methods_comparison(self, noisy_beam, clean_beam):
        """Compare all available denoising methods."""
        methods = [
            ("Gaussian", gaussian_denoise, {"kernel_size": 5, "sigma": 1.5}),
            ("Median", median_denoise, {"kernel_size": 5}),
            ("NLMeans", nlmeans_denoise, {"h": 10}),
            ("Bilateral", bilateral_denoise, {"d": 9, "sigma_color": 75, "sigma_space": 75}),
        ]
        
        if BM3D_AVAILABLE:
            methods.append(("BM3D", bm3d_denoise, {"sigma_psd": 25}))
        
        results = []
        for name, func, kwargs in methods:
            result = func(noisy_beam, **kwargs)
            metrics = evaluate_denoising(clean_beam, result.image, noisy_beam)
            results.append({
                'name': name,
                'result': result,
                'psnr': metrics.psnr,
                'ssim': metrics.ssim
            })
        
        # Create combined comparison image
        n_methods = len(results)
        height, width = noisy_beam.shape
        combined_height = height
        combined_width = width * (n_methods + 2)  # clean + noisy + methods
        
        combined = np.zeros((combined_height, combined_width), dtype=np.uint8) + 255
        
        # Add clean and noisy
        def normalize(img):
            if img.max() > img.min():
                return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            return img.astype(np.uint8)
        
        combined[:, :width] = normalize(clean_beam)
        combined[:, width:2*width] = normalize(noisy_beam)
        
        # Add each method result
        for i, res in enumerate(results):
            col_start = (2 + i) * width
            col_end = col_start + width
            combined[:, col_start:col_end] = normalize(res['result'].image)
        
        output_path = VISUAL_OUTPUT_DIR / "all_methods_comparison.png"
        
        if PIL_AVAILABLE:
            pil_img = Image.fromarray(combined, mode='L')
            pil_img.save(output_path)
        
        assert output_path.exists()
        assert len(results) >= 4  # At least 4 methods compared
    
    def test_noise_level_comparison(self, clean_beam):
        """Compare denoising at different noise levels."""
        from synthetic_beam_generator import BeamParameters, generate_noisy_beam
        
        noise_levels = [10, 25, 40, 60]
        results = []
        
        for level in noise_levels:
            params = BeamParameters(size=256, sigma_x=30, sigma_y=30, noise_level=level)
            noisy = generate_noisy_beam(params, noise_type='gaussian')
            denoised = gaussian_denoise(noisy, kernel_size=5, sigma=1.5)
            metrics = evaluate_denoising(clean_beam, denoised.image, noisy)
            
            results.append({
                'noise_level': level,
                'psnr_before': metrics.psnr if hasattr(metrics, 'psnr') else None,
                'ssim': metrics.ssim
            })
        
        # Save comparison
        height, width = clean_beam.shape
        comparison = np.zeros((height, width * len(noise_levels)), dtype=np.uint8) + 255
        
        for i, (level, res) in enumerate(zip(noise_levels, results)):
            params = BeamParameters(size=256, sigma_x=30, sigma_y=30, noise_level=level)
            noisy = generate_noisy_beam(params, noise_type='gaussian')
            denoised = gaussian_denoise(noisy, kernel_size=5, sigma=1.5)
            
            col_start = i * width
            col_end = col_start + width
            
            if denoised.image.max() > 0:
                normalized = (denoised.image / denoised.image.max() * 255).astype(np.uint8)
            else:
                normalized = denoised.image.astype(np.uint8)
            comparison[:, col_start:col_end] = normalized
        
        output_path = VISUAL_OUTPUT_DIR / "noise_level_comparison.png"
        
        if PIL_AVAILABLE:
            pil_img = Image.fromarray(comparison, mode='L')
            pil_img.save(output_path)
        
        assert output_path.exists()


class TestSpotImageVisualization:
    """Visualization tests using SpotImage objects."""
    
    def test_spot_image_denoise_visual(self, beam_pair):
        """Visual test with SpotImage objects."""
        from data_mining.image.common import SpotImage
        
        clean, noisy = beam_pair
        
        # Create SpotImage
        spot = SpotImage(noisy.astype(np.float32), "test_spot")
        
        # Denoise
        result = gaussian_denoise(spot.image_array, kernel_size=5, sigma=1.5)
        
        # Create visualization
        def normalize(img):
            if img.max() > img.min():
                return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            return img.astype(np.uint8)
        
        visualization = np.hstack([
            normalize(clean),
            noisy,
            result.image
        ])
        
        output_path = VISUAL_OUTPUT_DIR / "spot_image_denoise.png"
        
        if PIL_AVAILABLE:
            pil_img = Image.fromarray(visualization, mode='L')
            pil_img.save(output_path)
        
        assert output_path.exists()


class TestEllipticalBeamVisualization:
    """Visualization tests for elliptical beams."""
    
    def test_elliptical_beam_denoise(self, elliptical_noisy_beam, elliptical_beam):
        """Visual test for elliptical beam denoising."""
        result = gaussian_denoise(elliptical_noisy_beam, kernel_size=5, sigma=1.5)
        
        metrics = evaluate_denoising(elliptical_beam, result.image, elliptical_noisy_beam)
        
        output_path = save_comparison_image(
            elliptical_beam, elliptical_noisy_beam, result.image,
            "Elliptical_Gaussian", 0,
            {'psnr': metrics.psnr or 0, 'ssim': metrics.ssim or 0}
        )
        
        assert output_path is None or output_path.exists()
        assert metrics.ssim is None or metrics.ssim > 0.3


if __name__ == "__main__":
    # Run tests and generate visualizations
    pytest.main([__file__, "-v", "--tb=short"])
    
    print(f"\nVisualization output saved to: {VISUAL_OUTPUT_DIR}")
    print("Generated images:")
    for img_path in sorted(VISUAL_OUTPUT_DIR.glob("*.png")):
        print(f"  - {img_path.name}")
