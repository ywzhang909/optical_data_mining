#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run all tests
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests using pytest."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to the project root directory
    original_cwd = Path.cwd()
    try:
        # Change to project root
        import os
        os.chdir(project_root)
        
        # Run pytest on the tests directory
        print("Running tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", "src/tests", "-v"
        ], cwd=project_root)
        
        return result.returncode
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)