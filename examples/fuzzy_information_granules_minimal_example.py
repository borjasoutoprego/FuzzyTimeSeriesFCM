"""Minimal FCM information-granules forecasting example."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzy_information_granules import FuzzyInformationGranules


series = np.array([10, 11, 12, 13, 25, 26, 27, 28, 18, 19, 20, 21], dtype=float)
model = FuzzyInformationGranules(window_size=3, n_clusters=2, m=2.0).fit(series)

print("Information granules:")
print(model.granules)
print("One-step granular forecast:", model.predict(series))
