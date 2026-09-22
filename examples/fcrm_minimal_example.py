"""Minimal leakage-safe one-step FCRM example."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcrm_timeseries import FuzzyCRegression, make_lagged_supervised


series = np.array([0.1, 0.2, 0.3, 0.18, 0.35, 0.32, 0.42, 0.38, 0.5, 0.46])
x, y = make_lagged_supervised(series, n_lags=2)
model = FuzzyCRegression(n_clusters=2, n_lags=2, random_state=0).fit(x, y)

print("Coefficients [intercept, lag_2, lag_1]:")
print(model.coef_)
print("Training memberships:")
print(model.memberships_)
print("One-step forecast:", model.predict_next(series))
