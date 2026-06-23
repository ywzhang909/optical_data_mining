"""
Pytest root conftest.

The upstream ``zernike`` package depends on ``h5py`` which has a broken
installation in the current environment.  This conftest installs a minimal mock
so that test collection does not crash.
"""

import sys
from unittest.mock import MagicMock

# ── h5py mock (broken in this environment) ────────────────────────────
_h5py_mock = MagicMock()
_h5py_mock.__version__ = "3.12.1"
sys.modules["h5py"] = _h5py_mock
sys.modules["h5py._errors"] = MagicMock()
sys.modules["h5py.h5t"] = MagicMock()
sys.modules["h5py.h5s"] = MagicMock()
sys.modules["h5py.h5i"] = MagicMock()
sys.modules["h5py.h5g"] = MagicMock()
sys.modules["h5py.h5f"] = MagicMock()
sys.modules["h5py.h5d"] = MagicMock()
sys.modules["h5py.h5p"] = MagicMock()
sys.modules["h5py.h5a"] = MagicMock()
sys.modules["h5py._objects"] = MagicMock()
