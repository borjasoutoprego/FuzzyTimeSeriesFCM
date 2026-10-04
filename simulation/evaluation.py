"""Chronological validation, model selection, refitting, and rolling forecasts."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

from fuzzy_information_granules import FuzzyInformationGranules
from fcrm_timeseries import FuzzyCRegression

from .config import CLUSTER_CANDIDATES, FCRM_GATING_METHODS, FUZZIFIER
from .metrics import calculate_metrics


@lru_cache(maxsize=1)
def load_fcm_class():
    """Load the legacy hyphenated FCM module used by Cheng and Egrioglu."""
    path = Path(__file__).resolve().parents[1] / "fcm-timeseries-continuous-univariant.py"
    spec = importlib.util.spec_from_file_location("fcm_timeseries_simulation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the FCM implementation at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FuzzyTimeSeriesFCM


def rolling_one_step(initial_history, future, predict):
    """Predict each target before revealing and appending its actual value."""
    history = np.asarray(initial_history, dtype=float).tolist()
    targets = np.asarray(future, dtype=float)
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ValueError("future must be a finite one-dimensional array.")
    forecasts = np.empty(targets.size, dtype=float)
    for i, target in enumerate(targets):
        forecast = float(predict(history))
        if not np.isfinite(forecast):
            raise FloatingPointError(f"non-finite one-step forecast at offset {i}.")
        forecasts[i] = forecast
        history.append(float(target))
    return forecasts


def _fcm_validation_predictions(train, validation, c, method, m, random_state):
    model_class = load_fcm_class()
    model = model_class(c, m=m, random_state=random_state).fit_fcm(train)
    labels = model.fuzzify()
    if method == "cheng":
        model.build_rules(labels)
    elif method == "egrioglu":
        model.train_nn(labels)
    else:
        raise ValueError("method must be 'cheng' or 'egrioglu'.")
    forecasts = rolling_one_step(
        train,
        validation,
        lambda history: model.predict_one_step(history, method=method),
    )
    return forecasts


def select_fcm_clusters(
    train,
    validation,
    *,
    method,
    candidates=CLUSTER_CANDIDATES,
    m=FUZZIFIER,
    seed_for_candidate,
):
    """Choose c by rolling validation MAE; this API has no test argument."""
    train = np.asarray(train, dtype=float)
    validation = np.asarray(validation, dtype=float)
    scores = {}
    failures = {}
    forecasts_by_c = {}
    for c in candidates:
        try:
            forecasts = _fcm_validation_predictions(
                train, validation, c, method, m, seed_for_candidate(c)
            )
            scores[int(c)] = calculate_metrics(validation, forecasts)["mae"]
            forecasts_by_c[int(c)] = forecasts
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            failures[int(c)] = f"{type(error).__name__}: {error}"
    if not scores:
        raise RuntimeError(f"all {method} FCM candidates failed: {failures}")
    best_c = min(scores, key=lambda c: (scores[c], c))
    return {
        "selected_c": best_c,
        "selected_mae": scores[best_c],
        "scores": scores,
        "failures": failures,
        "validation_predictions": forecasts_by_c[best_c],
    }


def select_granular_parameters(
    train,
    validation,
    *,
    cluster_candidates=CLUSTER_CANDIDATES,
    window_candidates,
    m=FUZZIFIER,
    seed_for_candidate,
):
    """Select the joint (c, L) candidate using only train and validation."""
    train = np.asarray(train, dtype=float)
    validation = np.asarray(validation, dtype=float)
    candidates = {}
    failures = {}
    for c in cluster_candidates:
        for window in window_candidates:
            key = (int(c), int(window))
            try:
                model = FuzzyInformationGranules(
                    window, c, m=m, random_state=seed_for_candidate(c, window)
                ).fit(train)
                forecasts = rolling_one_step(
                    train, validation, lambda history: model.predict(history)
                )
                candidates[key] = {
                    "mae": calculate_metrics(validation, forecasts)["mae"],
                    "forecasts": forecasts,
                }
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures[key] = f"{type(error).__name__}: {error}"
    if not candidates:
        raise RuntimeError(f"all FCM-Granular candidates failed: {failures}")
    best_key = min(candidates, key=lambda key: (candidates[key]["mae"], key[0], key[1]))
    return {
        "selected_c": best_key[0],
        "selected_L": best_key[1],
        "selected_mae": candidates[best_key]["mae"],
        "scores": {key: value["mae"] for key, value in candidates.items()},
        "failures": failures,
        "validation_predictions": candidates[best_key]["forecasts"],
    }


def select_fcrm_clusters(
    train,
    validation,
    *,
    cluster_candidates=CLUSTER_CANDIDATES,
    n_lags=2,
    m=FUZZIFIER,
    seed_for_candidate,
):
    """Select c independently for each FCRM gate using validation MAE.

    Each candidate FCRM regression fit is shared by both gating methods. Both
    gates therefore see identical memberships, coefficients, lags, and data.
    """
    train = np.asarray(train, dtype=float)
    validation = np.asarray(validation, dtype=float)
    scores = {gating: {} for gating in FCRM_GATING_METHODS}
    failures = {gating: {} for gating in FCRM_GATING_METHODS}
    forecasts = {gating: {} for gating in FCRM_GATING_METHODS}
    for c in cluster_candidates:
        try:
            model = FuzzyCRegression(
                n_clusters=c,
                n_lags=n_lags,
                m=m,
                random_state=seed_for_candidate(c),
            ).fit_series(train)
            for gating in FCRM_GATING_METHODS:
                try:
                    model.fit_gating(method=gating)
                    values = rolling_one_step(
                        train,
                        validation,
                        lambda history, gate=gating: model.predict_next(
                            history, gating=gate, method="dominant"
                        ),
                    )
                    scores[gating][int(c)] = calculate_metrics(validation, values)["mae"]
                    forecasts[gating][int(c)] = values
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    failures[gating][int(c)] = f"{type(error).__name__}: {error}"
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            message = f"{type(error).__name__}: {error}"
            for gating in FCRM_GATING_METHODS:
                failures[gating][int(c)] = message
    selected = {}
    for gating in FCRM_GATING_METHODS:
        if not scores[gating]:
            raise RuntimeError(f"all FCRM {gating} candidates failed: {failures[gating]}")
        best_c = min(scores[gating], key=lambda c: (scores[gating][c], c))
        selected[gating] = {
            "selected_c": best_c,
            "selected_mae": scores[gating][best_c],
            "scores": scores[gating],
            "failures": failures[gating],
            "validation_predictions": forecasts[gating][best_c],
        }
    return {
        "selected": selected,
        "candidate_scores": scores,
        "candidate_failures": failures,
        "candidate_forecasts": forecasts,
    }


def fit_ar(series, order):
    """Fit the known-order AR benchmark by least squares with an intercept."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or values.size <= order or order < 1:
        raise ValueError("series must contain more observations than the AR order.")
    design = np.ones((values.size - order, order + 1), dtype=float)
    for lag in range(1, order + 1):
        design[:, lag] = values[order - lag:values.size - lag]
    coefficients, _, _, _ = np.linalg.lstsq(design, values[order:], rcond=None)
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("AR least-squares fit produced non-finite coefficients.")
    return coefficients


def predict_ar(history, coefficients):
    """One-step AR prediction from the latest observed lag values."""
    coefficients = np.asarray(coefficients, dtype=float)
    order = coefficients.size - 1
    values = np.asarray(history, dtype=float)
    if values.size < order:
        raise ValueError("history is shorter than the AR order.")
    newest_first = values[-1:-order - 1:-1]
    return float(coefficients[0] + np.dot(coefficients[1:], newest_first))


def rolling_lag_vectors(initial_history, future, n_lags):
    """Construct each one-step lag vector, revealing actual values in order."""
    history = np.asarray(initial_history, dtype=float).tolist()
    rows = []
    for target in np.asarray(future, dtype=float):
        if len(history) < n_lags:
            raise ValueError("history is shorter than n_lags.")
        rows.append(history[-1:-n_lags - 1:-1])
        history.append(float(target))
    return np.asarray(rows, dtype=float)
