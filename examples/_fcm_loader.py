"""Import helper for the legacy FCM filename, which contains hyphens."""

import importlib.util
from pathlib import Path


def load_fcm_module():
    """Load the legacy FCM module (whose import has no side effects)."""
    path = Path(__file__).resolve().parents[1] / "fcm-timeseries-continuous-univariant.py"
    spec = importlib.util.spec_from_file_location("fcm_timeseries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
