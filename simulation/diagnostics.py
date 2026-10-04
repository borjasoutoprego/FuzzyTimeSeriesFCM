"""FCRM regime, coefficient, and smooth-transition diagnostics."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score


TRUE_REGIME_COEFFICIENTS = np.asarray(
    [[0.0, 0.82, 0.05], [0.0, -0.45, 0.30]], dtype=float
)


def match_two_regime_coefficients(estimated, true=TRUE_REGIME_COEFFICIENTS):
    """Match two estimated coefficient rows to true regimes by total L2 error."""
    coefficients = np.asarray(estimated, dtype=float)
    truth = np.asarray(true, dtype=float)
    if coefficients.shape != (2, truth.shape[1]):
        raise ValueError("two-regime coefficient matching requires exactly c=2.")
    distances = np.linalg.norm(coefficients[:, None, :] - truth[None, :, :], axis=2)
    choices = list(permutations(range(2)))
    assignment = min(choices, key=lambda match: sum(distances[i, match[i]] for i in range(2)))
    return {
        "estimated_cluster_to_true_regime": list(assignment),
        "distance_matrix": distances,
        "total_distance": float(sum(distances[i, assignment[i]] for i in range(2))),
    }


def nearest_true_regime_by_coefficients(estimated, true=TRUE_REGIME_COEFFICIENTS):
    """Assign each of c estimated dynamics to its closest one of two truths."""
    coefficients = np.asarray(estimated, dtype=float)
    truth = np.asarray(true, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != truth.shape[1]:
        raise ValueError("estimated coefficients have an incompatible shape.")
    distances = np.linalg.norm(coefficients[:, None, :] - truth[None, :, :], axis=2)
    return np.argmin(distances, axis=1).astype(int), distances


def adjusted_rand(true_labels, estimated_labels):
    """Return an ARI float, with arbitrary cluster labels handled by sklearn."""
    truth = np.asarray(true_labels)
    estimate = np.asarray(estimated_labels)
    if truth.ndim != 1 or estimate.ndim != 1 or truth.size != estimate.size:
        raise ValueError("label vectors must be one-dimensional and equally sized.")
    return float(adjusted_rand_score(truth, estimate))


def gating_transition_estimate(gating_weights, estimated_coefficients):
    """Aggregate gating mass for the estimated dynamics nearest true regime 2."""
    weights = np.asarray(gating_weights, dtype=float)
    mapping, distances = nearest_true_regime_by_coefficients(estimated_coefficients)
    if weights.ndim != 2 or weights.shape[1] != mapping.size:
        raise ValueError("gating_weights must have one column per coefficient row.")
    second_regime = mapping == 1
    if not np.any(second_regime):
        # If every estimated dynamic is closest to regime 1, the model assigns
        # no gating mass to regime 2; the corresponding transition estimate is 0.
        return np.zeros(weights.shape[0], dtype=float), mapping, distances
    return np.sum(weights[:, second_regime], axis=1), mapping, distances


def correlations(true_values, estimated_values):
    """Pearson and Spearman correlations, using null for constant vectors."""
    truth = np.asarray(true_values, dtype=float)
    estimate = np.asarray(estimated_values, dtype=float)
    if truth.ndim != 1 or estimate.ndim != 1 or truth.size != estimate.size:
        raise ValueError("correlation vectors must be one-dimensional and equally sized.")
    if truth.size < 2 or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("correlation inputs must be finite and contain at least two values.")
    if np.ptp(truth) == 0.0 or np.ptp(estimate) == 0.0:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(truth, estimate)[0, 1])
    spearman = float(np.corrcoef(rankdata(truth), rankdata(estimate))[0, 1])
    return {
        "pearson": pearson if np.isfinite(pearson) else None,
        "spearman": spearman if np.isfinite(spearman) else None,
    }
