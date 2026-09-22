"""Minimal Fuzzy C-Means fuzzification example."""

import numpy as np

from _fcm_loader import load_fcm_module


def main():
    series = np.array([10, 11, 12, 13, 25, 26, 27, 28], dtype=float)
    model = load_fcm_module().FuzzyTimeSeriesFCM(2, m=2.0, random_state=0).fit_fcm(series)
    print("Centroids:", model.centers)
    print("Memberships (clusters x observations):")
    print(model.u)
    print("Fuzzy labels:", model.fuzzify())


if __name__ == "__main__":
    main()
