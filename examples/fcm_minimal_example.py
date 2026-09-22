"""Minimal Fuzzy C-Means fuzzification example."""

import numpy as np

from _fcm_loader import load_fcm_module


series = np.array([10, 11, 12, 13, 25, 26, 27, 28], dtype=float)
FuzzyTimeSeriesFCM = load_fcm_module().FuzzyTimeSeriesFCM
model = FuzzyTimeSeriesFCM(n_clusters=2, m=2.0)
model.fit_fcm(series)
labels = model.fuzzify()

print("Centroids:", model.centers)
print("Memberships (clusters x observations):")
print(model.u)
print("Fuzzy labels:", labels)
