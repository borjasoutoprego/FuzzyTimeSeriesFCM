import numpy as np
import unittest

from fcrm_timeseries import FuzzyCRegression, make_lagged_supervised


def two_regime_data(seed=7, n_per_regime=160):
    rng = np.random.default_rng(seed)
    x_left = rng.uniform(-4.0, -0.5, n_per_regime)
    x_right = rng.uniform(0.5, 4.0, n_per_regime)
    x = np.concatenate([x_left, x_right])[:, None]
    y = np.concatenate([1.0 + 2.0 * x_left, -1.0 - 1.5 * x_right])
    y += rng.normal(0.0, 0.03, y.size)
    return x, y


class TestFuzzyCRegression(unittest.TestCase):
    def test_memberships_and_dimensions(self):
        x, y = two_regime_data()
        model = FuzzyCRegression(n_clusters=2, n_lags=1, random_state=11).fit(x, y)
        self.assertEqual(model.memberships_.shape, (x.shape[0], 2))
        self.assertEqual(model.coef_.shape, (2, 2))
        self.assertEqual(model.labels_.shape, (x.shape[0],))
        self.assertTrue(np.all((model.memberships_ >= 0) & (model.memberships_ <= 1)))
        self.assertTrue(np.allclose(model.memberships_.sum(axis=1), 1.0))
        self.assertEqual(model.predict_by_cluster(x[:3]).shape, (3, 2))

    def test_recovers_two_simple_regressions_up_to_permutation(self):
        x, y = two_regime_data()
        model = FuzzyCRegression(n_clusters=2, n_lags=1, random_state=3, tol=1e-7).fit(x, y)
        expected = np.array([[1.0, 2.0], [-1.0, -1.5]])
        direct = np.linalg.norm(model.coef_ - expected)
        swapped = np.linalg.norm(model.coef_ - expected[::-1])
        self.assertLess(min(direct, swapped), 0.25)

    def test_two_lags_and_out_of_sample_prediction(self):
        rng = np.random.default_rng(4)
        series = np.zeros(180)
        for t in range(2, series.size):
            series[t] = 0.3 + 0.65 * series[t - 1] - 0.2 * series[t - 2] + rng.normal(0, 0.05)
        x, y = make_lagged_supervised(series, n_lags=2)
        split = 120
        model = FuzzyCRegression(n_clusters=2, n_lags=2, random_state=9).fit(x[:split], y[:split])
        forecasts = model.predict_by_cluster(x[split:])
        memberships = model.memberships_for(x[split:], y[split:])
        dominant = model.predict(x[split:], memberships, method="dominant")
        self.assertEqual(model.coef_.shape, (2, 3))
        self.assertEqual(forecasts.shape, (x.shape[0] - split, 2))
        self.assertEqual(dominant.shape, (x.shape[0] - split,))
        self.assertTrue(np.isfinite(model.predict_next(series[:130])))

    def test_reproducible_with_random_state(self):
        x, y = two_regime_data()
        first = FuzzyCRegression(n_clusters=2, n_lags=1, random_state=42).fit(x, y)
        second = FuzzyCRegression(n_clusters=2, n_lags=1, random_state=42).fit(x, y)
        self.assertTrue(np.allclose(first.coef_, second.coef_))
        self.assertTrue(np.allclose(first.memberships_, second.memberships_))
