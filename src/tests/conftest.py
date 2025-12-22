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