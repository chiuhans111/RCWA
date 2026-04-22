import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Any, Union, Dict, List

class GoldenManager:
    """
    Consolidated manager for numerical and visual 'Golden' baselines.
    Handles storage, verification, and explicit updating of ground-truth results.
    """
    def __init__(self, golden_dir: str = None):
        if golden_dir is None:
            # Default to tests/goldens
            base_dir = os.path.dirname(os.path.abspath(__file__))
            golden_dir = os.path.join(base_dir, "goldens")
        
        self.golden_dir = golden_dir
        self.plot_dir = os.path.join(self.golden_dir, "plots")
        
        # Ensure directories exist
        os.makedirs(self.golden_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
            
        # Check for update flag in CLI arguments
        self.update_mode = "--update" in sys.argv

    def verify(self, test_name: str, actual_data: Any, rtol: float = 1e-10, atol: float = 1e-10):
        """
        Verifies actual numerical data against a stored JSON baseline.
        In update mode, overwrites the baseline with current results.
        """
        path = os.path.join(self.golden_dir, f"{test_name}.json")
        actual_clean = self._sanitize(actual_data)
        
        if self.update_mode or not os.path.exists(path):
            self._save_json(path, actual_clean)
            return True
            
        expected_data = self._load_json(path)
        self._compare(actual_clean, expected_data, rtol, atol, test_name)
        return True

    def verify_image(self, test_name: str, fig: plt.Figure, mse_threshold: float = 1e-5):
        """
        Verifies a Matplotlib figure against a stored PNG baseline.
        Standardizes DPI, size, and layout for cross-environment reproducibility.
        """
        path = os.path.join(self.plot_dir, f"{test_name}.png")
        actual_pixels = self._prepare_figure_pixels(fig)
        
        if self.update_mode or not os.path.exists(path):
            fig.savefig(path, dpi=100)
            return True
            
        expected_pixels = mpimg.imread(path).astype(float)
        mse = np.mean((actual_pixels - expected_pixels)**2)
        
        if mse > mse_threshold:
            raise AssertionError(
                f"Visual regression detected in '{test_name}.png' (MSE={mse:.6f}).\n"
                f"Review baseline at: {path}"
            )
        return True

    # --- Private Helpers ---

    def _prepare_figure_pixels(self, fig: plt.Figure) -> np.ndarray:
        """Standardizes figure and extracts normalized RGBA pixel buffer."""
        fig.set_size_inches(6, 4)
        fig.set_dpi(100)
        fig.tight_layout()
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255.0

    def _save_json(self, path: str, data: Any):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def _load_json(self, path: str) -> Any:
        with open(path, 'r') as f:
            return json.load(f)

    def _sanitize(self, data: Any) -> Any:
        """Recursively converts NumPy types to native Python types for JSON compatibility."""
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._sanitize(v) for v in data]
        elif isinstance(data, np.ndarray):
            return self._sanitize(data.tolist())
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (complex, np.complex128, np.complex64)):
            return [float(data.real), float(data.imag)]
        return data

    def _compare(self, actual: Any, expected: Any, rtol: float, atol: float, context: str):
        """Recursively compares data structures with numerical tolerance."""
        if isinstance(actual, dict):
            if not isinstance(expected, dict):
                raise AssertionError(f"Type mismatch at {context}: expected dict, got {type(expected)}")
            for k, v in actual.items():
                if k not in expected:
                    raise AssertionError(f"Missing key '{k}' in golden baseline: {context}")
                self._compare(v, expected[k], rtol, atol, f"{context}.{k}")
        elif isinstance(actual, (list, tuple)):
            if not isinstance(expected, list):
                raise AssertionError(f"Type mismatch at {context}: expected list, got {type(expected)}")
            if len(actual) != len(expected):
                raise AssertionError(f"Length mismatch at {context}: expected {len(expected)}, got {len(actual)}")
            for i, (a, e) in enumerate(zip(actual, expected)):
                self._compare(a, e, rtol, atol, f"{context}[{i}]")
        elif isinstance(actual, (float, int)):
            if not np.allclose(actual, expected, rtol=rtol, atol=atol):
                diff = abs(actual - expected)
                raise AssertionError(f"Value mismatch at {context}:\n  Expected: {expected}\n  Actual:   {actual}\n  Diff:     {diff}")
        else:
            if actual != expected:
                raise AssertionError(f"Literal mismatch at {context}: {actual} != {expected}")
