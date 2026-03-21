#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration file for pytest.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))
