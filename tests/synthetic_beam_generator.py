#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Beam Image Generator

Generates realistic synthetic laser beam images for testing denoising algorithms.
Based on beam analysis parameters from optical measurements.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class BeamParameters:
    """Parameters for synthetic beam generation"""
    size: int = 256
    center_x: float = 128.0
    center_y: float = 128.0
    sigma_x: float = 20.0
    sigma_y: float = 20.0
    amplitude: float = 255.0
    rotation: float = 0.0  # degrees
    background: float = 0.0
    noise_level: float = 0.0


def generate_gaussian_beam(params: BeamParameters) -> np.ndarray:
    """
    Generate a Gaussian beam profile.
    
    Args:
        params: BeamParameters containing beam characteristics
        
    Returns:
        2D numpy array of the beam profile
    """
    size = params.size
    x = np.arange(size) - params.center_x
    y = np.arange(size) - params.center_y
    xx, yy = np.meshgrid(x, y)
    
    # Apply rotation
    if params.rotation != 0:
        theta = np.deg2rad(params.rotation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xx_rot = cos_t * xx + sin_t * yy
        yy_rot = -sin_t * xx + cos_t * yy
        xx, yy = xx_rot, yy_rot
    
    # Gaussian intensity profile
    intensity = params.amplitude * np.exp(
        -(xx**2 / (2 * params.sigma_x**2) + yy**2 / (2 * params.sigma_y**2))
    )
    
    # Add background
    intensity = intensity + params.background
    
    return intensity.astype(np.float32)


def add_poisson_noise(image: np.ndarray, photon_level: float = 1000.0) -> np.ndarray:
    """
    Add Poisson (shot) noise to simulate photon noise.
    """
    # Normalize to photon counts
    photon_image = image / image.max() * photon_level
    # Add Poisson noise
    noisy = np.random.poisson(photon_image).astype(np.float32)
    # Scale back
    noisy = noisy / photon_level * image.max()
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add Gaussian (thermal) noise to simulate sensor noise.
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """
    Add salt and pepper noise.
    """
    noisy = image.copy()
    # Salt
    salt_mask = np.random.random(image.shape) < amount / 2
    noisy[salt_mask] = 255
    # Pepper
    pepper_mask = np.random.random(image.shape) < amount / 2
    noisy[pepper_mask] = 0
    return noisy


def generate_clean_beam(params: BeamParameters) -> np.ndarray:
    """Generate a clean beam without noise."""
    return generate_gaussian_beam(params)


def generate_noisy_beam(params: BeamParameters, noise_type: str = 'gaussian') -> np.ndarray:
    """
    Generate a beam with specified noise.
    
    Args:
        params: BeamParameters
        noise_type: 'gaussian', 'poisson', 'salt_pepper', or 'mixed'
        
    Returns:
        Noisy beam as uint8 array
    """
    clean = generate_gaussian_beam(params)
    
    if noise_type == 'gaussian':
        noisy = add_gaussian_noise(clean, sigma=params.noise_level)
    elif noise_type == 'poisson':
        noisy = add_poisson_noise(clean, photon_level=max(100, 1000/params.noise_level))
    elif noise_type == 'salt_pepper':
        noisy = add_salt_pepper_noise(clean.astype(np.uint8), amount=params.noise_level/255)
    elif noise_type == 'mixed':
        noisy = add_gaussian_noise(clean, sigma=params.noise_level * 0.5)
        noisy = add_salt_pepper_noise(noisy, amount=params.noise_level/1000)
    else:
        noisy = add_gaussian_noise(clean, sigma=params.noise_level)
    
    return noisy


def save_image_numpy(path: Path, image: np.ndarray) -> None:
    """Save numpy array as PNG using PIL."""
    if PIL_AVAILABLE:
        # Normalize to 0-255
        if image.max() <= 1.0:
            img_data = (image * 255).astype(np.uint8)
        else:
            img_data = image.astype(np.uint8)
        
        # Handle grayscale vs RGB
        if len(img_data.shape) == 2:
            img = Image.fromarray(img_data, mode='L')
        else:
            img = Image.fromarray(img_data)
        
        img.save(path)


class SyntheticBeamGenerator:
    """
    Generator for synthetic beam images with various parameters.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = output_dir or Path(__file__).parent / "test_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_batch(
        self, 
        count: int = 5,
        size: int = 256,
        noise_level: float = 25.0,
        save: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[BeamParameters]]:
        """
        Generate a batch of beam images with varying parameters.
        
        Returns:
            Tuple of (clean_images, noisy_images, parameters)
        """
        clean_images = []
        noisy_images = []
        all_params = []
        
        for i in range(count):
            # Vary parameters for each beam
            params = BeamParameters(
                size=size,
                center_x=size // 2 + np.random.uniform(-10, 10),
                center_y=size // 2 + np.random.uniform(-10, 10),
                sigma_x=np.random.uniform(15, 35),
                sigma_y=np.random.uniform(15, 35),
                amplitude=np.random.uniform(200, 255),
                rotation=np.random.uniform(0, 45) if i % 3 == 0 else 0,
                background=np.random.uniform(0, 10),
                noise_level=noise_level
            )
            
            clean = generate_clean_beam(params)
            noisy = generate_noisy_beam(params, noise_type='gaussian')
            
            clean_images.append(clean)
            noisy_images.append(noisy)
            all_params.append(params)
            
            if save:
                # Save clean beam
                save_image_numpy(
                    self.output_dir / f"clean_beam_{i:02d}.png",
                    clean
                )
                # Save noisy beam
                save_image_numpy(
                    self.output_dir / f"noisy_beam_{i:02d}.png",
                    noisy
                )
        
        return clean_images, noisy_images, all_params
    
    def generate_realistic_beam_set(
        self,
        count: int = 10,
        size: int = 256
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a set of realistic beam pairs (clean, noisy).
        
        Returns:
            List of (clean, noisy) image tuples
        """
        beams = []
        
        # Generate different beam types
        for i in range(count):
            # Determine beam type based on index
            beam_type = i % 4
            
            if beam_type == 0:
                # Circular Gaussian beam
                params = BeamParameters(
                    size=size,
                    sigma_x=np.random.uniform(20, 40),
                    sigma_y=np.random.uniform(20, 40),
                    noise_level=25.0
                )
            elif beam_type == 1:
                # Elliptical beam
                params = BeamParameters(
                    size=size,
                    sigma_x=np.random.uniform(30, 50),
                    sigma_y=np.random.uniform(15, 25),
                    rotation=np.random.uniform(-30, 30),
                    noise_level=25.0
                )
            elif beam_type == 2:
                # Off-center beam
                params = BeamParameters(
                    size=size,
                    center_x=size * 0.4,
                    center_y=size * 0.6,
                    sigma_x=30,
                    sigma_y=30,
                    noise_level=30.0
                )
            else:
                # High noise beam
                params = BeamParameters(
                    size=size,
                    sigma_x=25,
                    sigma_y=25,
                    noise_level=40.0
                )
            
            clean = generate_clean_beam(params)
            noisy = generate_noisy_beam(params, noise_type='gaussian')
            
            beams.append((clean, noisy))
            
            # Save individual images
            save_image_numpy(self.output_dir / f"beam_{i:02d}_clean.png", clean)
            save_image_numpy(self.output_dir / f"beam_{i:02d}_noisy.png", noisy)
        
        return beams


# Fixtures for pytest
def get_synthetic_beam_fixtures():
    """Get pre-generated synthetic beam fixtures."""
    generator = SyntheticBeamGenerator()
    return generator.generate_realistic_beam_set(count=5)


if __name__ == "__main__":
    # Generate test images when run directly
    generator = SyntheticBeamGenerator()
    
    print("Generating synthetic beam test images...")
    
    # Generate batch of varying beams
    clean, noisy, params = generator.generate_batch(count=10, size=256)
    print(f"Generated {len(clean)} beam pairs")
    
    # Generate realistic beam set
    beams = generator.generate_realistic_beam_set(count=10, size=256)
    print(f"Generated {len(beams)} realistic beam pairs")
    
    print(f"Images saved to: {generator.output_dir}")
    print("Generated images:")
    for img_path in sorted(generator.output_dir.glob("*.png")):
        print(f"  - {img_path.name}")
