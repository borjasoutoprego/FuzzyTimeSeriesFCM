"""Minimal Cheng first-order fuzzy-rule forecasting example."""

import numpy as np

from _fcm_loader import load_fcm_module


def main():
    series = np.array([10, 11, 12, 13, 25, 26, 27, 28], dtype=float)
    model = load_fcm_module().FuzzyTimeSeriesFCM(2, m=2.0, random_state=0).fit_fcm(series)
    print("Fuzzy rules:", model.build_rules())
    print("One-step Cheng forecast:", model.predict_one_step(series, method="cheng"))


if __name__ == "__main__":
    main()
