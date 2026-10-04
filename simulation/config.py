"""Experimental design and time-ordered data splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DGP_NAMES = ("E1", "E2", "E3", "E4")
SAMPLE_SIZES = (50, 100, 500)
NOISE_STDS = (0.25, 0.50)
GAMMA_VALUES = (1.0, 5.0, 20.0)
FUZZIFIER = 2.0
CLUSTER_CANDIDATES = (2, 3, 4, 5, 6)
GRANULAR_WINDOWS = {
    50: (3, 5, 8),
    100: (3, 5, 8, 12, 25),
    500: (3, 5, 8, 12, 25),
}
FCRM_GATING_METHODS = ("fuzzy_centers", "multinomial")
AR_ORDERS = {"E1": 1, "E2": 2, "E3": 2, "E4": 2}
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
DEFAULT_BURN_IN = 200
DEFAULT_MASTER_SEED = 20261004


@dataclass(frozen=True)
class Scenario:
    """One DGP and its experimental parameter values."""

    dgp: str
    T: int
    sigma: float
    gamma: float | None = None

    def __post_init__(self):
        if self.dgp not in DGP_NAMES:
            raise ValueError(f"dgp must be one of {DGP_NAMES}.")
        if self.T not in SAMPLE_SIZES:
            raise ValueError(f"T must be one of {SAMPLE_SIZES}.")
        if self.sigma not in NOISE_STDS:
            raise ValueError(f"sigma must be one of {NOISE_STDS}.")
        if self.dgp == "E4":
            if self.gamma not in GAMMA_VALUES:
                raise ValueError(f"E4 gamma must be one of {GAMMA_VALUES}.")
        elif self.gamma is not None:
            raise ValueError("gamma applies only to E4.")


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological 60/20/20 split, with half-open source index ranges."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_range: tuple[int, int]
    validation_range: tuple[int, int]
    test_range: tuple[int, int]


def split_series(series: np.ndarray) -> TemporalSplit:
    """Split a finite series chronologically into 60%, 20%, and 20%.

    Train and validation sizes use floor rounding; any remainder is assigned to
    test. The split contains no shuffle and no overlap.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or values.size < 5 or not np.all(np.isfinite(values)):
        raise ValueError("series must be one-dimensional, finite, and have >= 5 values.")
    train_end = int(np.floor(values.size * TRAIN_FRACTION))
    validation_end = train_end + int(np.floor(values.size * VALIDATION_FRACTION))
    if train_end < 2 or validation_end <= train_end or validation_end >= values.size:
        raise ValueError("series is too short for a non-empty 60/20/20 split.")
    return TemporalSplit(
        train=values[:train_end].copy(),
        validation=values[train_end:validation_end].copy(),
        test=values[validation_end:].copy(),
        train_range=(0, train_end),
        validation_range=(train_end, validation_end),
        test_range=(validation_end, values.size),
    )


def scenario_slug(scenario: Scenario) -> str:
    """Stable filename-safe name for one scenario."""
    sigma = f"{int(round(scenario.sigma * 100)):03d}"
    suffix = f"_gamma{int(scenario.gamma):02d}" if scenario.gamma is not None else ""
    return f"{scenario.dgp}_T{scenario.T}_sigma{sigma}{suffix}"
