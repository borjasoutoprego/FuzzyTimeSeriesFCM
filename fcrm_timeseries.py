"""Fuzzy C-Regression Models (FCRM) for univariate time series.

FCRM memberships are calculated from regression residuals.  They are used to
fit the regime-specific autoregressions and, separately, to train a gating
model.  Forecasts use gating weights calculated from observed lag vectors; a
future target is never needed to obtain its weights.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


_GATING_METHODS = ("fuzzy_centers", "multinomial")


def make_lagged_supervised(series, n_lags):
    """Return ``X, y`` in the PDF's newest-to-oldest lag order.

    For a target ``X_t``, a row of ``X`` is
    ``[X_(t-1), X_(t-2), ..., X_(t-n_lags)]`` and the matching response is
    ``X_t``.  Thus the first regression coefficient after the intercept is
    the coefficient of lag 1.
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

    x = np.asarray([
        [values[t - lag] for lag in range(1, n_lags + 1)]
        for t in range(n_lags, values.size)
    ])
    return x, values[n_lags:].copy()


class FuzzyCRegression:
    """FCRM estimated by alternating weighted least squares.

    The objective is ``sum_t sum_k u[t, k]**m * residual[t, k]**2``.
    ``memberships_`` are response-residual memberships used by FCRM.  Gating
    weights (``omega``) are instead calculated from the lag predictors and are
    used to forecast without observing the target.
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
        """Solve each weighted least-squares problem without regularization."""
        coefficients = np.empty((self.n_clusters, design.shape[1]))
        for cluster in range(self.n_clusters):
            weights = memberships[:, cluster] ** self.m
            if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
                raise FloatingPointError(
                    "A regression cluster has numerically zero total weight."
                )
            sqrt_weights = np.sqrt(weights)
            weighted_design = design * sqrt_weights[:, None]
            weighted_y = y * sqrt_weights
            coefficient, _, rank, _ = np.linalg.lstsq(
                weighted_design, weighted_y, rcond=None
            )
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
        """Apply the PDF update; squared residuals give the same formula.

        The PDF uses ``(|e_tk| / |e_tj|)**(2 / (m - 1))``.  In log space
        this is a softmax of ``-2 * log(|e_tk|) / (m - 1)``, which is the
        same update expressed using squared residuals and avoids overflow.
        Exact zero residuals share all membership equally, as required by the
        limiting formula.
        """
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim != 2 or residuals.shape[1] != self.n_clusters:
            raise ValueError("residuals must have shape (n_observations, n_clusters).")
        if not np.all(np.isfinite(residuals)):
            raise FloatingPointError("Residuals must be finite to update memberships.")

        memberships = np.zeros_like(residuals)
        exponent = 2.0 / (self.m - 1.0)
        for row, errors in enumerate(np.abs(residuals)):
            exact_zeros = errors == 0.0
            if np.any(exact_zeros):
                memberships[row, exact_zeros] = 1.0 / exact_zeros.sum()
                continue
            log_weights = -exponent * np.log(errors)
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            memberships[row] = weights / weights.sum()
        return memberships

    @staticmethod
    def _objective(memberships, residuals, m):
        value = float(np.sum((memberships ** m) * (residuals ** 2)))
        if not np.isfinite(value):
            raise FloatingPointError("FCRM objective became non-finite.")
        return value

    def _fit_once(self, x, y, rng):
        design = self._design_matrix(x)
        memberships = rng.random((x.shape[0], self.n_clusters))
        memberships /= memberships.sum(axis=1, keepdims=True)
        objective_history = []
        converged = False
        coefficients = None
        residuals = None

        for iteration in range(1, self.max_iter + 1):
            coefficients = self._fit_coefficients(design, y, memberships)
            residuals = y[:, None] - design @ coefficients.T
            updated = self._memberships_from_residuals(residuals)
            objective_history.append(self._objective(updated, residuals, self.m))
            change = float(np.max(np.abs(updated - memberships)))
            memberships = updated

            if change < self.tol:
                # Refit against the memberships that will be stored.  Check
                # the fixed-point change once more after this refit so the
                # final memberships and coefficients describe the same state.
                final_coefficients = self._fit_coefficients(
                    design, y, memberships
                )
                final_residuals = y[:, None] - design @ final_coefficients.T
                refined = self._memberships_from_residuals(final_residuals)
                if np.max(np.abs(refined - memberships)) < self.tol:
                    coefficients = final_coefficients
                    residuals = final_residuals
                    converged = True
                    break
                memberships = refined

        # Ensure beta was estimated using the exact U that is returned, and
        # residuals/objective are computed from those returned coefficients.
        coefficients = self._fit_coefficients(design, y, memberships)
        residuals = y[:, None] - design @ coefficients.T
        objective = self._objective(memberships, residuals, self.m)
        objective_history.append(objective)

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
        """Fit FCRM to supervised lag vectors and responses."""
        self._validate_parameters()
        x, y = self._validate_xy(X, y)
        rng = np.random.default_rng(self.random_state)
        solutions = [self._fit_once(x, y, rng) for _ in range(self.n_init)]
        self.init_objectives_ = np.asarray(
            [solution["objective"] for solution in solutions]
        )
        best_index = int(np.argmin(self.init_objectives_))
        best = solutions[best_index]

        self.coef_ = best["coef"]
        self.residuals_ = best["residuals"]
        self.memberships_ = best["memberships"]
        self.labels_ = np.argmax(self.memberships_, axis=1)
        self.objective_ = best["objective"]
        self.objective_history_ = np.asarray(best["objective_history"])
        self.converged_ = best["converged"]
        self.n_iter_ = best["n_iter"]
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = tuple(
            f"lag_{lag}" for lag in range(1, self.n_lags + 1)
        )
        self.X_fit_ = x.copy()
        self.y_fit_ = y.copy()
        # Gating models are fitted separately and must match this FCRM fit.
        self._gating_models_ = {}
        for name in ("gating_centers_", "gating_coef_", "gating_m_", "gating_method_"):
            if hasattr(self, name):
                delattr(self, name)
        return self

    def fit_series(self, series):
        """Prepare PDF-ordered lags from a univariate series and fit FCRM."""
        x, y = make_lagged_supervised(series, self.n_lags)
        return self.fit(x, y)

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the FCRM model before requesting predictions.")

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
        """Return the K regression forecasts, shape ``(n_observations, K)``."""
        x = self._validate_predict_x(X)
        return self._design_matrix(x) @ self.coef_.T

    def memberships_for(self, X, y):
        """Return FCRM residual memberships for observed ``(X, y)`` pairs.

        This response-dependent diagnostic is not used to forecast.  Forecast
        memberships are obtained with :meth:`predict_gating_memberships`.
        """
        x = self._validate_predict_x(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.size != x.shape[0] or not np.all(np.isfinite(y)):
            raise ValueError("y must be a finite one-dimensional array matching X.")
        return self._memberships_from_residuals(
            y[:, None] - self.predict_by_cluster(x)
        )

    def fit_gating(self, method="fuzzy_centers", m_g=None):
        """Fit or construct one of the two PDF gating methods.

        ``fuzzy_centers`` computes membership-weighted lag centers.  The
        ``multinomial`` option fits a reference-category softmax by minimizing
        cross-entropy against the fuzzy FCRM memberships, without hard labels.
        """
        self._check_fitted()
        if method not in _GATING_METHODS:
            raise ValueError("method must be 'fuzzy_centers' or 'multinomial'.")
        if method == "fuzzy_centers":
            gating_m = self.m if m_g is None else float(m_g)
            if not np.isfinite(gating_m) or gating_m <= 1:
                raise ValueError("m_g must be finite and greater than 1.")
            weights = self.memberships_ ** self.m
            totals = weights.sum(axis=0)
            if np.any(totals <= 0):
                raise FloatingPointError("A gating center has numerically zero total weight.")
            centers = (weights.T @ self.X_fit_) / totals[:, None]
            self.gating_centers_ = centers
            self.gating_m_ = gating_m
            self._gating_models_[method] = {
                "centers": centers.copy(), "m_g": gating_m
            }
        else:
            if m_g is not None:
                raise ValueError("m_g applies only to gating='fuzzy_centers'.")
            design = self._design_matrix(self.X_fit_)
            targets = self.memberships_
            n_non_reference = self.n_clusters - 1
            parameter_shape = (n_non_reference, design.shape[1])

            def loss_and_gradient(flat_parameters):
                coefficients = flat_parameters.reshape(parameter_shape)
                logits = design @ coefficients.T
                logits = np.column_stack((logits, np.zeros(design.shape[0])))
                log_probabilities = logits - logsumexp(
                    logits, axis=1, keepdims=True
                )
                probabilities = np.exp(log_probabilities)
                loss = -float(np.sum(targets * log_probabilities))
                gradient = (probabilities[:, :n_non_reference] -
                            targets[:, :n_non_reference]).T @ design
                return loss, gradient.ravel()

            result = minimize(
                loss_and_gradient,
                np.zeros(int(np.prod(parameter_shape))),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
                raise FloatingPointError("Multinomial gating optimization was non-finite.")
            if not result.success:
                raise RuntimeError(
                    f"Multinomial gating optimization failed: {result.message}"
                )
            coefficients = result.x.reshape(parameter_shape)
            self.gating_coef_ = coefficients
            self._gating_models_[method] = {"coef": coefficients.copy()}

        self.gating_method_ = method
        return self

    @staticmethod
    def _fuzzy_distance_memberships(distances, m_g):
        """Fuzzy c-means membership formula applied to gating distances."""
        memberships = np.zeros_like(distances, dtype=float)
        exponent = 2.0 / (m_g - 1.0)
        for row, row_distances in enumerate(distances):
            zeros = row_distances == 0.0
            if np.any(zeros):
                memberships[row, zeros] = 1.0 / zeros.sum()
                continue
            log_weights = -exponent * np.log(row_distances)
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            memberships[row] = weights / weights.sum()
        return memberships

    def predict_gating_memberships(self, X, method="fuzzy_centers"):
        """Return ``omega`` for lag vectors using the selected gating model."""
        x = self._validate_predict_x(X)
        if method not in _GATING_METHODS:
            raise ValueError("method must be 'fuzzy_centers' or 'multinomial'.")
        if method not in self._gating_models_:
            self.fit_gating(method)
        gating_model = self._gating_models_[method]

        if method == "fuzzy_centers":
            distances = np.linalg.norm(
                x[:, None, :] - gating_model["centers"][None, :, :], axis=2
            )
            return self._fuzzy_distance_memberships(
                distances, gating_model["m_g"]
            )

        design = self._design_matrix(x)
        logits = design @ gating_model["coef"].T
        logits = np.column_stack((logits, np.zeros(x.shape[0])))
        log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
        return np.exp(log_probabilities)

    def predict_regime(self, X, gating="fuzzy_centers"):
        """Return the dominant gating regime for each supplied lag vector."""
        return np.argmax(self.predict_gating_memberships(X, method=gating), axis=1)

    def predict(self, X, gating="fuzzy_centers", method="weighted"):
        """Combine cluster forecasts with gating weights.

        ``gating`` selects ``fuzzy_centers`` or ``multinomial`` independently
        from ``method``, which selects ``weighted`` or ``dominant`` prediction.
        """
        if method not in ("weighted", "dominant"):
            raise ValueError("method must be 'weighted' or 'dominant'.")
        x = self._validate_predict_x(X)
        weights = self.predict_gating_memberships(x, method=gating)
        forecasts = self.predict_by_cluster(x)
        if method == "weighted":
            return np.sum(weights * forecasts, axis=1)
        return forecasts[np.arange(forecasts.shape[0]), np.argmax(weights, axis=1)]

    def predict_next(self, history, gating="fuzzy_centers", method="weighted"):
        """Forecast one step using only the latest ``n_lags`` observed values."""
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or values.size < self.n_lags:
            raise ValueError("history must contain at least n_lags finite values.")
        if not np.all(np.isfinite(values)):
            raise ValueError("history must contain only finite values.")
        lag_vector = values[-1:-self.n_lags - 1:-1].reshape(1, -1)
        return self.predict(lag_vector, gating=gating, method=method)[0]
