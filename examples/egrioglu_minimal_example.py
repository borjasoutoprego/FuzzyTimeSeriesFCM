"""Minimal Egrioglu FTS-NN forecasting example."""

import numpy as np

from _fcm_loader import load_fcm_module


series = np.array([10, 11, 12, 13, 25, 26, 27, 28], dtype=float)
FuzzyTimeSeriesFCM = load_fcm_module().FuzzyTimeSeriesFCM
model = FuzzyTimeSeriesFCM(n_clusters=2, m=2.0)
model.fit_fcm(series)
labels = model.fuzzify()
model.train_nn()

print("Fuzzy labels:", labels)
print("One-step Egrioglu forecast:", model.predict_nn(labels[-1]))
