"""Minimal FCM information-granules forecasting example."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzy_information_granules import FuzzyInformationGranules


def main():
    series = np.array([10, 11, 12, 13, 25, 26, 27, 28, 18, 19, 20, 21], dtype=float)
    model = FuzzyInformationGranules(3, 2, m=2.0, random_state=0).fit(series)
    print("Information granules:")
    print(model.granules)
    print("One-step granular forecast:", model.predict(series))


if __name__ == "__main__":
    main()
