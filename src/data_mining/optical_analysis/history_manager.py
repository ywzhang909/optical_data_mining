"""
History manager for AO analysis results
Manages saving and loading of analysis results with timestamps
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import hashlib
from PIL import Image
import numpy as np
import pandas as pd


class HistoryManager:
    def __init__(self, base_dir=".history"):
        """
        Initialize history manager with base directory

        Args:
            base_dir: Base directory for history storage, defaults to '.history'
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_session_dir = None

    def start_new_session(self, session_name=None):
        """
        Start a new analysis session with timestamp-based directory

        Args:
            session_name: Optional custom session name

        Returns:
            Path: Session directory path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name:
            session_dir_name = f"{timestamp}_{session_name}"
        else:
            session_dir_name = timestamp

        self.current_session_dir = self.base_dir / session_dir_name
        self.current_session_dir.mkdir(exist_ok=True)

        return self.current_session_dir

    def save_analysis_results(
        self, results_dict, axis_image=None, pupil_image=None, params=None
    ):
        """
        Save analysis results, images, and parameters to the current session

        Args:
            results_dict: Dictionary containing analysis results
            axis_image: Axis image (numpy array or PIL Image)
            pupil_image: Pupil image (numpy array or PIL Image)
            params: Dictionary containing analysis parameters
        """
        if not self.current_session_dir:
            raise ValueError("No active session. Call start_new_session() first.")

        # Create subdirectories for different data types
        images_dir = self.current_session_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Save axis image
        if axis_image is not None:
            axis_path = images_dir / "axis_image.png"
            if isinstance(axis_image, np.ndarray):
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(np.uint8(axis_image))
                pil_image.save(axis_path)
            elif isinstance(axis_image, Image.Image):
                axis_image.save(axis_path)
            else:
                # Assume it's a file-like object
                with open(axis_path, "wb") as f:
                    f.write(axis_image.read())

        # Save pupil image
        if pupil_image is not None:
            pupil_path = images_dir / "pupil_image.png"
            if isinstance(pupil_image, np.ndarray):
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(np.uint8(pupil_image))
                pil_image.save(pupil_path)
            elif isinstance(pupil_image, Image.Image):
                pupil_image.save(pupil_path)
            else:
                # Assume it's a file-like object
                with open(pupil_path, "wb") as f:
                    f.write(pupil_image.read())

        # Save analysis parameters
        if params:
            params_path = self.current_session_dir / "params.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._convert_numpy_types(params), f, indent=2, ensure_ascii=False
                )

        # Save results
        results_path = self.current_session_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                self._convert_numpy_types(results_dict), f, indent=2, ensure_ascii=False
            )

        # Create a summary CSV file
        df = pd.DataFrame(
            {
                "Parameter": list(results_dict.keys()),
                "Value": [str(v) for v in results_dict.values()],
            }
        )
        summary_path = self.current_session_dir / "summary.csv"
        df.to_csv(summary_path, index=False)

        # Save version information
        version_info = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "data_files": [
                str(file.name)
                for file in self.current_session_dir.iterdir()
                if file.is_file()
            ],
        }
        version_path = self.current_session_dir / "version.json"
        with open(version_path, "w", encoding="utf-8") as f:
            json.dump(self._convert_numpy_types(version_info), f, indent=2)

    def list_sessions(self):
        """
        List all available sessions in the history directory

        Returns:
            List of session directories
        """
        sessions = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                sessions.append(item)
        return sorted(sessions, key=lambda x: x.name, reverse=True)

    def load_session_results(self, session_dir):
        """
        Load analysis results from a specific session directory

        Args:
            session_dir: Session directory path

        Returns:
            Dictionary containing session data
        """
        session_path = Path(session_dir)

        data = {"session_dir": session_path, "results": {}, "params": {}, "images": {}}

        # Load results
        results_path = session_path / "results.json"
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                data["results"] = json.load(f)

        # Load parameters
        params_path = session_path / "params.json"
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                data["params"] = json.load(f)

        # Load version info
        version_path = session_path / "version.json"
        if version_path.exists():
            with open(version_path, "r", encoding="utf-8") as f:
                data["version"] = json.load(f)

        # Load images
        images_dir = session_path / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tiff",
                    ".tif",
                ]:
                    # We store the path, not the actual image data, to avoid memory issues
                    data["images"][img_file.stem] = str(img_file)

        return data

    def get_session_summary(self, session_dir):
        """
        Get a summary of a session (parameters and results)

        Args:
            session_dir: Session directory path

        Returns:
            Dictionary with session summary
        """
        session_data = self.load_session_results(session_dir)

        summary = {
            "session_name": session_dir.name,
            "saved_at": session_data.get("version", {}).get("saved_at", "Unknown"),
            "parameters": session_data.get("params", {}),
            "results": session_data.get("results", {}),
        }

        return summary

    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy data types to native Python types for JSON serialization

        Args:
            obj: Object to convert

        Returns:
            Converted object with native Python types
        """
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):  # Handle numpy boolean specifically
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            # Convert ndarray to list recursively
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.complexfloating):
            # Convert complex numbers to a JSON-serializable format (real and imaginary parts)
            complex_val = complex(obj)
            return {"real": complex_val.real, "imag": complex_val.imag}
        elif hasattr(obj, "item"):  # Handles scalar numpy types
            try:
                return self._convert_numpy_types(obj.item())
            except (ValueError, AttributeError):
                # If .item() fails, return the object as is
                return obj
        elif isinstance(obj, complex):
            # Handle native Python complex numbers
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (bool, int, float, str, type(None))):
            # These types are already JSON serializable
            return obj
        else:
            # For any other type, try to convert to string as fallback
            try:
                return str(obj)
            except:
                # If all conversion attempts fail, return None
                return None
