"""Finite-safe forecasting metrics used for selection and evaluation."""

from __future__ import annotations

import numpy as np


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and percentage sMAPE.

    For sMAPE, each contribution is ``200*|y-yhat|/(|y|+|yhat|)``. If both
    values are zero the contribution is defined as zero, keeping the result
    finite and avoiding an arbitrary epsilon in the denominator.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.size != prediction.size:
        raise ValueError("y_true and y_pred must be equally sized 1-D arrays.")
    if truth.size == 0 or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metric inputs must be non-empty and finite.")

    errors = truth - prediction
    denominator = np.abs(truth) + np.abs(prediction)
    contributions = np.zeros_like(denominator)
    nonzero = denominator > 0.0
    contributions[nonzero] = 200.0 * np.abs(errors[nonzero]) / denominator[nonzero]
    result = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "smape": float(np.mean(contributions)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("metric calculation produced a non-finite value.")
    return result
