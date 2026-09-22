"""Import helper for the legacy FCM filename, which contains hyphens."""

import contextlib
import importlib.util
import io
from pathlib import Path


def load_fcm_module():
    """Load the legacy module without printing its historical demonstration."""
    path = Path(__file__).resolve().parents[1] / "fcm-timeseries-continuous-univariant.py"
    spec = importlib.util.spec_from_file_location("fcm_timeseries", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module
