#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration file for pytest.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
SRC_PATH = Path(__file__).parent.parent / "data_mining"
sys.path.insert(0, str(SRC_PATH))

# Also add the image module path
IMAGE_PATH = SRC_PATH / "image"
sys.path.insert(0, str(IMAGE_PATH))

# Add the experiment_analysis module path
EXPERIMENT_ANALYSIS_PATH = SRC_PATH / "experiment_analysis"
sys.path.insert(0, str(EXPERIMENT_ANALYSIS_PATH))

# Add the tests directory path for synthetic_beam_generator
TESTS_PATH = Path(__file__).parent
sys.path.insert(0, str(TESTS_PATH))

# =============================================================================
# Fixtures - Synthetic Beam Images
# =============================================================================

import numpy as np
import pytest
from synthetic_beam_generator import (
    SyntheticBeamGenerator,
    BeamParameters,
    generate_clean_beam,
    generate_noisy_beam,
)


@pytest.fixture(scope="session")
def beam_generator():
    """Session-scoped beam generator for all tests."""
    return SyntheticBeamGenerator()


@pytest.fixture(scope="session")
def synthetic_beams(beam_generator):
    """Generate synthetic beam pairs for the entire test session."""
    beams = beam_generator.generate_realistic_beam_set(count=5, size=256)
    return beams


@pytest.fixture
def clean_beam():
    """Generate a clean circular Gaussian beam."""
    params = BeamParameters(size=256, sigma_x=30, sigma_y=30, amplitude=255)
    return generate_clean_beam(params)


@pytest.fixture
def noisy_beam():
    """Generate a noisy circular Gaussian beam."""
    params = BeamParameters(size=256, sigma_x=30, sigma_y=30, amplitude=255, noise_level=25)
    return generate_noisy_beam(params, noise_type='gaussian')


@pytest.fixture
def elliptical_beam():
    """Generate a clean elliptical beam."""
    params = BeamParameters(
        size=256, 
        sigma_x=40, 
        sigma_y=20, 
        amplitude=255,
        rotation=30
    )
    return generate_clean_beam(params)


@pytest.fixture
def elliptical_noisy_beam():
    """Generate a noisy elliptical beam."""
    params = BeamParameters(
        size=256, 
        sigma_x=40, 
        sigma_y=20, 
        amplitude=255,
        rotation=30,
        noise_level=25
    )
    return generate_noisy_beam(params, noise_type='gaussian')


@pytest.fixture
def beam_pair():
    """Generate a single clean/noisy beam pair."""
    params = BeamParameters(size=256, sigma_x=25, sigma_y=25, amplitude=255, noise_level=30)
    clean = generate_clean_beam(params)
    noisy = generate_noisy_beam(params, noise_type='mixed')
    return clean, noisy


@pytest.fixture
def high_noise_beam():
    """Generate a beam with high noise level."""
    params = BeamParameters(size=256, sigma_x=30, sigma_y=30, amplitude=255, noise_level=50)
    return generate_noisy_beam(params, noise_type='gaussian')


@pytest.fixture
def low_noise_beam():
    """Generate a beam with low noise level."""
    params = BeamParameters(size=256, sigma_x=30, sigma_y=30, amplitude=255, noise_level=10)
    return generate_noisy_beam(params, noise_type='gaussian')
