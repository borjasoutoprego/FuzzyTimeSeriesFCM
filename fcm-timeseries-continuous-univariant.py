"""FCM-based Cheng and Egrioglu one-step time-series forecasting.

``fit_fcm`` learns cluster centres only from its input.  ``fuzzify_new`` is
inference-only: it assigns observations to those fixed centres and never
changes the fitted model.  This distinction is important for temporal
validation and test evaluation.
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from sklearn.neural_network import MLPClassifier


class FuzzyTimeSeriesFCM:
    """First-order FCM fuzzy time-series model.

    Cheng rules and the Egrioglu neural network are trained from labels of
    observed training values. A one-step prediction uses the label of the
    last *already observed* value, never the label of its target.
    """

    def __init__(self, n_clusters, m=2, random_state=None):
        self.c = n_clusters
        self.m = m
        self.random_state = random_state
        self.centers = None
        self.u = None
        self.labels = None

    def _validate_parameters(self):
        if not isinstance(self.c, (int, np.integer)) or self.c < 2:
            raise ValueError("n_clusters must be an integer greater than or equal to 2.")
        if not np.isfinite(self.m) or self.m <= 1:
            raise ValueError("m must be finite and greater than 1.")

    @staticmethod
    def _as_series(X, name="X"):
        values = np.asarray(X, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional series.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        return values

    def _check_fitted(self):
        if self.centers is None:
            raise RuntimeError("Fit FCM before fuzzifying or predicting new observations.")

    def fit_fcm(self, X):
        """Fit FCM on an observed training series and return ``self``."""
        self._validate_parameters()
        values = self._as_series(X)
        if values.size < self.c:
            raise ValueError("X must contain at least n_clusters observations.")
        seed = None
        if self.random_state is not None:
            seed = int(np.random.default_rng(self.random_state).integers(0, 2**31 - 1))
        centers, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
            values.reshape(1, -1), c=self.c, m=self.m, error=0.005,
            maxiter=1000, seed=seed,
        )
        order = np.argsort(centers[:, 0])
        self.centers = centers[order, 0]
        self.u = memberships[order]
        self.labels = np.argmax(self.u, axis=0) + 1
        return self

    def fuzzify(self):
        """Return labels for the series used in ``fit_fcm`` (compatibility API)."""
        self._check_fitted()
        return self.labels.copy()

    def fuzzify_new(self, X, return_memberships=False):
        """Assign new observed values to fixed centres without refitting.

        When ``return_memberships`` is true, return ``(labels, memberships)``;
        memberships has shape ``(n_clusters, n_observations)``.
        """
        self._check_fitted()
        values = self._as_series(X)
        memberships, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            values.reshape(1, -1), self.centers.reshape(-1, 1), self.m,
            error=0.005, maxiter=1000,
        )
        labels = np.argmax(memberships, axis=0) + 1
        if return_memberships:
            return labels, memberships
        return labels

    predict_labels = fuzzify_new

    def build_rules(self, labels=None):
        """Build first-order Cheng transitions from labels of observed values."""
        self._check_fitted()
        labels = self.labels if labels is None else np.asarray(labels)
        if labels.ndim != 1 or labels.size < 2:
            raise ValueError("At least two observed labels are required to build rules.")
        if np.any((labels < 1) | (labels > self.c)):
            raise ValueError("labels must be state labels between 1 and n_clusters.")
        self.rules = {state: set() for state in range(1, self.c + 1)}
        for previous, current in zip(labels[:-1], labels[1:]):
            self.rules[int(previous)].add(int(current))
        return self.rules

    def predict_cheng(self, last_label):
        """Forecast from an already known state using fitted Cheng rules."""
        if not hasattr(self, "rules"):
            raise RuntimeError("Build Cheng rules before predicting.")
        if not isinstance(last_label, (int, np.integer)) or not 1 <= last_label <= self.c:
            raise ValueError("last_label must be a valid fuzzy state label.")
        next_labels = self.rules[last_label]
        if not next_labels:
            return float(self.centers[last_label - 1])
        return float(np.mean([self.centers[label - 1] for label in next_labels]))

    def train_nn(self, labels=None):
        """Train Egrioglu's state-transition NN from training labels only."""
        self._check_fitted()
        labels = self.labels if labels is None else np.asarray(labels)
        if labels.ndim != 1 or labels.size < 2:
            raise ValueError("At least two observed labels are required to train the NN.")
        x, y = labels[:-1].reshape(-1, 1), labels[1:]
        if np.unique(y).size < 2:
            raise ValueError("The NN needs at least two target states in the training labels.")
        self.nn = MLPClassifier(
            hidden_layer_sizes=(10,), activation="logistic", solver="lbfgs",
            max_iter=1000, random_state=self.random_state,
        )
        self.nn.fit(x, y)
        return self

    def predict_nn(self, last_label):
        """Forecast from an already known state with the fitted Egrioglu NN."""
        if not hasattr(self, "nn"):
            raise RuntimeError("Train the NN before predicting.")
        if not isinstance(last_label, (int, np.integer)) or not 1 <= last_label <= self.c:
            raise ValueError("last_label must be a valid fuzzy state label.")
        label = int(self.nn.predict([[last_label]])[0])
        return float(self.centers[label - 1])

    def predict_one_step(self, observed_history, method="cheng"):
        """One-step forecast using only the last value in ``observed_history``."""
        values = self._as_series(observed_history, "observed_history")
        last_label = int(self.fuzzify_new(values[-1:])[0])
        if method == "cheng":
            return self.predict_cheng(last_label)
        if method in {"nn", "egrioglu"}:
            return self.predict_nn(last_label)
        raise ValueError("method must be 'cheng', 'nn', or 'egrioglu'.")


def grid_search_fcm(train, validation, c_values=range(2, 8),
                    m_values=(1.5, 1.8, 2.0, 2.2, 2.5), method="cheng",
                    random_state=None):
    """Select FCM parameters on a chronological validation partition.

    Candidates are fitted and have rules/the NN trained only on ``train``.
    Validation is evaluated rolling one-step; test data are intentionally
    absent from this API.
    """
    train = FuzzyTimeSeriesFCM._as_series(train, "train")
    validation = FuzzyTimeSeriesFCM._as_series(validation, "validation")
    if train.size < 2:
        raise ValueError("train must contain at least two observations.")
    results = []
    for c in c_values:
        for m in m_values:
            model = FuzzyTimeSeriesFCM(c, m, random_state=random_state).fit_fcm(train)
            labels = model.fuzzify()
            if method == "cheng":
                model.build_rules(labels)
            elif method in {"nn", "egrioglu"}:
                model.train_nn(labels)
            else:
                raise ValueError("method must be 'cheng', 'nn', or 'egrioglu'.")
            history, forecasts = train.tolist(), []
            for target in validation:
                forecasts.append(model.predict_one_step(history, method=method))
                history.append(float(target))
            results.append({"c": c, "m": m,
                            "mse": float(np.mean((validation - forecasts) ** 2))})
    if not results:
        raise ValueError("c_values and m_values must contain at least one candidate.")
    return min(results, key=lambda result: result["mse"]), results


if __name__ == "__main__":
    series = np.array([13055, 13563, 13867, 14696, 15460, 15311, 15603], dtype=float)
    best, _ = grid_search_fcm(series[:5], series[5:], c_values=(2,), method="cheng")
    print("Validation selection:", best)
