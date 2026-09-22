"""Fuzzy C-Regression Models (FCRM) for univariate time series.

FCM assigns an observation using its distance to a centroid.  FCRM instead
assigns it using its squared residual under every regression model.  In this
module a cluster can therefore be interpreted as an estimated dynamic regime.
"""

from __future__ import annotations

import warnings

import numpy as np


def make_lagged_supervised(series, n_lags):
    """Convert a univariate series into lagged regression data.

    Each row is ordered from the oldest to the most recent lag, so for
    ``n_lags=2`` the pairs are ``[x_1, x_2] -> x_3``,
    ``[x_2, x_3] -> x_4``, etc.  Consequently, column ``j`` represents
    ``lag_(n_lags-j)``.  This ordering is retained in :attr:`coef_`.
    """
    if not isinstance(n_lags, (int, np.integer)) or n_lags < 1:
        raise ValueError("n_lags must be a positive integer.")
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("series must contain only finite values.")
    if values.size <= n_lags:
        raise ValueError("series must contain more values than n_lags.")

    x = np.asarray([values[i - n_lags:i] for i in range(n_lags, values.size)])
    return x, values[n_lags:].copy()


class FuzzyCRegression:
    """Fuzzy C-Regression Model estimated by alternating weighted least squares.

    The fitted objective is

    ``J = sum_i sum_k u[i, k]**m * (y[i] - x[i] @ phi[k])**2``.

    ``memberships_`` has shape ``(n_observations, n_clusters)``: rows are
    supervised temporal observations and columns are regression regimes.
    Cluster numbering is arbitrary and may be permuted between fits.
    """

    def __init__(self, n_clusters=2, m=2.0, n_lags=2, max_iter=1000,
                 tol=1e-5, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.m = m
        self.n_lags = n_lags
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

    def _validate_parameters(self):
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 2:
            raise ValueError("n_clusters must be an integer greater than or equal to 2.")
        if not np.isfinite(self.m) or self.m <= 1:
            raise ValueError("m must be finite and greater than 1.")
        if not isinstance(self.n_lags, (int, np.integer)) or self.n_lags < 1:
            raise ValueError("n_lags must be a positive integer.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive.")
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer.")

    def _validate_xy(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_lags:
            raise ValueError("X must have shape (n_observations, n_lags).")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be one-dimensional and have one value per row of X.")
        if x.shape[0] < self.n_lags + 1:
            raise ValueError("At least n_lags + 1 supervised observations are required.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")
        return x, y

    @staticmethod
    def _design_matrix(x):
        return np.column_stack((np.ones(x.shape[0]), x))

    def _fit_coefficients(self, design, y, memberships):
        coefficients = np.empty((self.n_clusters, design.shape[1]))
        for cluster in range(self.n_clusters):
            weights = memberships[:, cluster] ** self.m
            if not np.all(np.isfinite(weights)) or weights.sum() <= np.finfo(float).eps:
                raise FloatingPointError("A regression cluster has numerically zero total weight.")
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_y = y * np.sqrt(weights)
            coefficient, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
            if rank < design.shape[1]:
                warnings.warn(
                    "Weighted least squares is rank deficient; using its minimum-norm solution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if not np.all(np.isfinite(coefficient)):
                raise FloatingPointError("Weighted least squares produced non-finite coefficients.")
            coefficients[cluster] = coefficient
        return coefficients

    def _memberships_from_residuals(self, residuals):
        distances = np.abs(residuals)
        if not np.all(np.isfinite(distances)):
            raise FloatingPointError("Residuals must be finite to update memberships.")
        memberships = np.zeros_like(distances)
        zero = distances <= np.finfo(float).eps
        for row in range(distances.shape[0]):
            if np.any(zero[row]):
                memberships[row, zero[row]] = 1.0 / zero[row].sum()
                continue
            scaled = distances[row] / distances[row].min()
            powers = scaled ** (2.0 / (self.m - 1.0))
            memberships[row] = 1.0 / powers
            memberships[row] /= memberships[row].sum()
        return memberships

    def _fit_once(self, x, y, rng):
        design = self._design_matrix(x)
        memberships = rng.dirichlet(np.ones(self.n_clusters), size=x.shape[0])
        objective_history = []
        converged = False
        for iteration in range(1, self.max_iter + 1):
            coefficients = self._fit_coefficients(design, y, memberships)
            residuals = y[:, None] - design @ coefficients.T
            updated = self._memberships_from_residuals(residuals)
            objective = float(np.sum((updated ** self.m) * residuals ** 2))
            if not np.isfinite(objective):
                raise FloatingPointError("FCRM objective became non-finite.")
            objective_history.append(objective)
            change = np.max(np.abs(updated - memberships))
            memberships = updated
            if change < self.tol:
                converged = True
                break

        coefficients = self._fit_coefficients(design, y, memberships)
        residuals = y[:, None] - design @ coefficients.T
        memberships = self._memberships_from_residuals(residuals)
        objective = float(np.sum((memberships ** self.m) * residuals ** 2))
        return {
            "coef": coefficients,
            "residuals": residuals,
            "memberships": memberships,
            "objective": objective,
            "objective_history": objective_history,
            "converged": converged,
            "n_iter": iteration,
        }

    def fit(self, X, y):
        """Fit FCRM to arbitrary supervised data with exactly ``n_lags`` features.

        ``n_init`` independent membership initializations are evaluated and the
        solution with the lowest final objective is retained.  This reduces the
        risk of returning a poor local FCRM solution while retaining fully
        reproducible results when ``random_state`` is set.
        """
        self._validate_parameters()
        x, y = self._validate_xy(X, y)
        rng = np.random.default_rng(self.random_state)
        solutions = [self._fit_once(x, y, rng) for _ in range(self.n_init)]
        best = min(solutions, key=lambda solution: solution["objective"])
        self.coef_ = best["coef"]
        self.residuals_ = best["residuals"]
        self.memberships_ = best["memberships"]
        self.labels_ = np.argmax(self.memberships_, axis=1)
        self.objective_ = best["objective"]
        self.objective_history_ = best["objective_history"]
        self.converged_ = best["converged"]
        self.n_iter_ = best["n_iter"]
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = tuple(
            f"lag_{lag}" for lag in range(self.n_lags, 0, -1)
        )
        return self

    def fit_series(self, series):
        """Build lagged data from ``series`` and fit the model."""
        x, y = make_lagged_supervised(series, self.n_lags)
        return self.fit(x, y)

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the model before requesting predictions.")

    def _validate_predict_x(self, X):
        self._check_fitted()
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.n_lags:
            raise ValueError("X must have shape (n_observations, n_lags).")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values.")
        return x

    def predict_by_cluster(self, X):
        """Return one forecast per regression, shape ``(n_observations, c)``."""
        x = self._validate_predict_x(X)
        return self._design_matrix(x) @ self.coef_.T

    def memberships_for(self, X, y):
        """Calculate FCRM memberships from observed responses (diagnostic use)."""
        x = self._validate_predict_x(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.size != x.shape[0] or not np.all(np.isfinite(y)):
            raise ValueError("y must be a finite one-dimensional array matching X.")
        return self._memberships_from_residuals(y[:, None] - self.predict_by_cluster(x))

    def predict(self, X, memberships, method="dominant"):
        """Combine per-cluster forecasts using supplied, already available memberships.

        ``method='dominant'`` implements the PF.4 defuzzification (argmax
        regime). ``method='weighted'`` is supplied as a separate fuzzy option.
        Memberships are required because they depend on a response residual and
        cannot be inferred from future targets without leakage.
        """
        forecasts = self.predict_by_cluster(X)
        memberships = np.asarray(memberships, dtype=float)
        if memberships.shape != forecasts.shape:
            raise ValueError("memberships must have shape (n_observations, n_clusters).")
        if not np.all(np.isfinite(memberships)) or np.any(memberships < 0):
            raise ValueError("memberships must be finite and non-negative.")
        row_sums = memberships.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Each membership row must sum to one.")
        if method == "dominant":
            return forecasts[np.arange(forecasts.shape[0]), np.argmax(memberships, axis=1)]
        if method == "weighted":
            return np.sum(memberships * forecasts, axis=1)
        raise ValueError("method must be 'dominant' or 'weighted'.")

    def predict_next(self, history, method="dominant"):
        """Forecast the next value using only the observed history.

        The final observed value is used to obtain the current residual-based
        membership; that membership selects or weights the forecast for the
        next instant.  Thus no future response is used.
        """
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or values.size < self.n_lags + 1:
            raise ValueError("history must contain at least n_lags + 1 finite values.")
        if not np.all(np.isfinite(values)):
            raise ValueError("history must contain only finite values.")
        current_x = values[-self.n_lags - 1:-1].reshape(1, -1)
        current_u = self.memberships_for(current_x, values[-1:])
        next_x = values[-self.n_lags:].reshape(1, -1)
        return self.predict(next_x, current_u, method=method)[0]
