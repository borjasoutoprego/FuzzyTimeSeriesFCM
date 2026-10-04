import copy
import unittest

import numpy as np

from fcrm_timeseries import FuzzyCRegression, make_lagged_supervised


def two_regime_data(seed=7, n_per_regime=80):
    rng = np.random.default_rng(seed)
    x_left = rng.uniform(-4.0, -0.5, n_per_regime)
    x_right = rng.uniform(0.5, 4.0, n_per_regime)
    x = np.concatenate([x_left, x_right])[:, None]
    y = np.concatenate([1.0 + 2.0 * x_left, -1.0 - 1.5 * x_right])
    y += rng.normal(0.0, 0.03, y.size)
    return x, y


def autoregressive_series(seed=17, size=90):
    rng = np.random.default_rng(seed)
    values = np.zeros(size)
    for t in range(2, size):
        values[t] = (
            0.3 + 0.65 * values[t - 1] - 0.2 * values[t - 2]
            + rng.normal(0.0, 0.05)
        )
    return values


class TestFuzzyCRegression(unittest.TestCase):
    def test_lagged_data_uses_pdf_order(self):
        x, y = make_lagged_supervised([10, 20, 30, 40], n_lags=2)
        self.assertTrue(np.array_equal(x, [[20, 10], [30, 20]]))
        self.assertTrue(np.array_equal(y, [30, 40]))

    def test_memberships_dimensions_and_objective_match_stored_solution(self):
        x, y = two_regime_data()
        model = FuzzyCRegression(2, n_lags=1, random_state=11).fit(x, y)
        self.assertEqual(model.memberships_.shape, (x.shape[0], 2))
        self.assertEqual(model.residuals_.shape, (x.shape[0], 2))
        self.assertEqual(model.coef_.shape, (2, 2))
        self.assertEqual(model.labels_.shape, (x.shape[0],))
        self.assertTrue(np.all((model.memberships_ >= 0) & (model.memberships_ <= 1)))
        self.assertTrue(np.allclose(model.memberships_.sum(axis=1), 1.0))
        calculated_residuals = y[:, None] - model._design_matrix(x) @ model.coef_.T
        self.assertTrue(np.allclose(model.residuals_, calculated_residuals))
        recalculated_memberships = model._memberships_from_residuals(
            model.residuals_
        )
        self.assertTrue(
            np.max(np.abs(model.memberships_ - recalculated_memberships))
            < model.tol
        )
        calculated_objective = np.sum(model.memberships_ ** model.m * calculated_residuals ** 2)
        self.assertAlmostEqual(model.objective_, calculated_objective, places=11)
        self.assertEqual(model.objective_history_[-1], model.objective_)
        self.assertEqual(model.predict_by_cluster(x[:3]).shape, (3, 2))

    def test_objective_history_is_nonincreasing_and_fit_converges(self):
        x, y = two_regime_data()
        model = FuzzyCRegression(
            2, n_lags=1, random_state=3, tol=1e-6, max_iter=2000
        ).fit(x, y)
        differences = np.diff(model.objective_history_)
        self.assertTrue(np.all(differences <= 1e-9))
        self.assertTrue(model.converged_)
        self.assertGreater(model.n_iter_, 0)
        self.assertLessEqual(model.n_iter_, model.max_iter)
        self.assertEqual(model.objective_history_[-1], model.objective_)

        exhausted = FuzzyCRegression(
            2, n_lags=1, random_state=3, tol=1e-14, max_iter=1, n_init=1
        ).fit(x, y)
        self.assertFalse(exhausted.converged_)
        self.assertEqual(exhausted.n_iter_, exhausted.max_iter)
        self.assertEqual(exhausted.objective_history_[-1], exhausted.objective_)

    def test_recovers_two_simple_regressions_up_to_permutation(self):
        x, y = two_regime_data()
        model = FuzzyCRegression(
            n_clusters=2, n_lags=1, random_state=3, tol=1e-7
        ).fit(x, y)
        expected = np.array([[1.0, 2.0], [-1.0, -1.5]])
        direct = np.linalg.norm(model.coef_ - expected)
        swapped = np.linalg.norm(model.coef_ - expected[::-1])
        self.assertLess(min(direct, swapped), 0.25)

    def test_random_state_is_reproducible_and_n_init_selects_best(self):
        x, y = two_regime_data()
        first = FuzzyCRegression(
            n_clusters=2, n_lags=1, random_state=42, n_init=4
        ).fit(x, y)
        second = FuzzyCRegression(
            n_clusters=2, n_lags=1, random_state=42, n_init=4
        ).fit(x, y)
        self.assertTrue(np.allclose(first.coef_, second.coef_))
        self.assertTrue(np.allclose(first.memberships_, second.memberships_))
        self.assertTrue(np.array_equal(first.init_objectives_, second.init_objectives_))
        self.assertAlmostEqual(first.objective_, np.min(first.init_objectives_))

    def test_residual_zero_and_small_nonzero_treatment(self):
        model = FuzzyCRegression(n_clusters=2, n_lags=1, m=2)
        memberships = model._memberships_from_residuals(
            np.array([[0.0, 2.0], [0.0, 0.0], [1e-12, 2.0]])
        )
        self.assertTrue(np.all(np.isfinite(memberships)))
        self.assertTrue(np.allclose(memberships.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(memberships[0], [1.0, 0.0]))
        self.assertTrue(np.allclose(memberships[1], [0.5, 0.5]))
        self.assertGreater(memberships[2, 0], 0.999999)

    def test_predict_by_cluster_returns_all_autoregressions(self):
        series = autoregressive_series()
        model = FuzzyCRegression(2, n_lags=2, random_state=3).fit_series(series[:65])
        predictions = model.predict_by_cluster([[0.2, 0.1], [0.4, 0.2]])
        self.assertEqual(predictions.shape, (2, 2))

    def test_prediction_modes_for_both_gating_methods(self):
        series = autoregressive_series()
        model = FuzzyCRegression(2, n_lags=2, random_state=5).fit_series(series[:65])
        x = np.array([[series[64], series[63]], [series[63], series[62]]])
        for gating in ("fuzzy_centers", "multinomial"):
            for method in ("weighted", "dominant"):
                with self.subTest(gating=gating, method=method):
                    values = model.predict(x, gating=gating, method=method)
                    self.assertEqual(values.shape, (2,))
                    self.assertTrue(np.all(np.isfinite(values)))

    def test_predict_next_all_four_gating_and_combination_variants(self):
        series = autoregressive_series()
        model = FuzzyCRegression(2, n_lags=2, random_state=5).fit_series(series[:65])
        for gating in ("fuzzy_centers", "multinomial"):
            coefficients_before = model.coef_.copy()
            memberships_before = model.memberships_.copy()
            model.fit_gating(gating)
            self.assertTrue(np.array_equal(model.coef_, coefficients_before))
            self.assertTrue(np.array_equal(model.memberships_, memberships_before))
            for method in ("weighted", "dominant"):
                with self.subTest(gating=gating, method=method):
                    result = model.predict_next(
                        series[:65], gating=gating, method=method
                    )
                    self.assertTrue(np.isfinite(result))

    def test_gating_weights_and_predictions_match_manual_formulas(self):
        x = np.array([[0.0], [1.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 2.5, 6.0])
        model = FuzzyCRegression(
            n_clusters=2, n_lags=1, random_state=0, n_init=1
        ).fit(x, y)
        model.memberships_ = np.array([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]
        ])
        model.coef_ = np.array([[1.0, 2.0], [10.0, 4.0]])
        model.fit_gating("fuzzy_centers", m_g=2.0)

        query = np.array([[1.5]])
        self.assertEqual(model.gating_centers_.shape, (2, 1))
        self.assertTrue(np.allclose(model.gating_centers_[:, 0], [0.5, 3.5]))
        weights = model.predict_gating_memberships(query, "fuzzy_centers")
        self.assertTrue(np.allclose(weights, [[0.8, 0.2]]))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))

        cluster_predictions = model.predict_by_cluster(query)
        self.assertTrue(np.allclose(cluster_predictions, [[4.0, 16.0]]))
        self.assertAlmostEqual(model.predict(query, method="weighted")[0], 6.4)
        self.assertAlmostEqual(model.predict(query, method="dominant")[0], 4.0)
        self.assertEqual(model.predict_regime(query)[0], 0)

    def test_zero_distance_in_fuzzy_center_gating(self):
        x = np.array([[0.0], [1.0], [3.0], [4.0]])
        model = FuzzyCRegression(2, n_lags=1, random_state=1, n_init=1).fit(
            x, [1.0, 2.0, 2.5, 6.0]
        )
        model.memberships_ = np.array([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]
        ])
        model.fit_gating("fuzzy_centers", m_g=2.0)
        weights = model.predict_gating_memberships(
            [[0.5], [2.0]], method="fuzzy_centers"
        )
        self.assertTrue(np.allclose(weights, [[1.0, 0.0], [0.5, 0.5]]))

    def test_multinomial_soft_targets_reference_category_and_reproducibility(self):
        x = np.array([[-2.0], [-0.5], [0.7], [2.0]])
        y = np.array([-1.0, 0.0, 1.0, 2.0])
        memberships = np.array([
            [0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]
        ])
        soft_model = FuzzyCRegression(
            2, n_lags=1, random_state=8, n_init=1
        ).fit(x, y)
        soft_model.memberships_ = memberships.copy()
        soft_model.fit_gating("multinomial")
        soft_probabilities = soft_model.predict_gating_memberships(x, "multinomial")

        self.assertEqual(soft_model.gating_coef_.shape, (1, 2))
        self.assertEqual(soft_model.gating_method_, "multinomial")
        self.assertTrue(np.all(soft_probabilities >= 0))
        self.assertTrue(np.allclose(soft_probabilities.sum(axis=1), 1.0))
        logits = soft_model._design_matrix(x) @ soft_model.gating_coef_.T
        manual = np.column_stack((np.exp(logits[:, 0]), np.ones(x.shape[0])))
        manual /= manual.sum(axis=1, keepdims=True)
        self.assertTrue(np.allclose(soft_probabilities, manual))

        hard_model = copy.deepcopy(soft_model)
        hard_model.memberships_ = np.eye(2)[np.argmax(memberships, axis=1)]
        hard_model._gating_models_ = {}
        hard_model.fit_gating("multinomial")
        self.assertGreater(
            np.linalg.norm(soft_model.gating_coef_ - hard_model.gating_coef_), 1e-3
        )

        repeated = FuzzyCRegression(
            2, n_lags=1, random_state=8, n_init=1
        ).fit(x, y)
        repeated.memberships_ = memberships.copy()
        repeated.fit_gating("multinomial")
        self.assertTrue(np.allclose(soft_model.gating_coef_, repeated.gating_coef_))

    def test_multinomial_gating_returns_all_k_memberships(self):
        x = np.array([[-2.0], [-0.5], [0.7], [2.0], [3.5]])
        y = np.array([-1.0, 0.0, 1.0, 2.0, -0.3])
        model = FuzzyCRegression(
            n_clusters=3, n_lags=1, random_state=12, n_init=2
        ).fit(x, y)
        model.memberships_ = np.array([
            [0.7, 0.2, 0.1],
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.2, 0.2, 0.6],
        ])
        model.fit_gating("multinomial")
        weights = model.predict_gating_memberships(x, method="multinomial")
        self.assertEqual(weights.shape, (x.shape[0], 3))
        self.assertEqual(model.gating_coef_.shape, (2, 2))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))

    def test_future_changes_do_not_change_gating_or_forecasts(self):
        series = autoregressive_series(size=100)
        cutoff = 65
        observed_history = series[:cutoff].copy()
        model = FuzzyCRegression(2, n_lags=2, random_state=5).fit_series(
            observed_history
        )
        changed_series = series.copy()
        changed_series[cutoff:] = np.linspace(1e6, -1e6, series.size - cutoff)
        changed_history = changed_series[:cutoff].copy()
        for gating in ("fuzzy_centers", "multinomial"):
            for method in ("weighted", "dominant"):
                before_weights = model.predict_gating_memberships(
                    observed_history[-2:], method=gating
                )
                after_weights = model.predict_gating_memberships(
                    changed_history[-2:], method=gating
                )
                before_prediction = model.predict_next(
                    observed_history, gating=gating, method=method
                )
                after_prediction = model.predict_next(
                    changed_history, gating=gating, method=method
                )
                self.assertTrue(np.array_equal(before_weights, after_weights))
                self.assertEqual(before_prediction, after_prediction)

    def test_future_prediction_does_not_call_response_memberships(self):
        series = autoregressive_series(size=70)
        model = FuzzyCRegression(2, n_lags=2, random_state=13).fit_series(series[:60])

        def fail_if_called(*args, **kwargs):
            raise AssertionError("response-residual memberships cannot generate a forecast")

        model.memberships_for = fail_if_called
        prediction = model.predict_next(
            series[:60], gating="fuzzy_centers", method="dominant"
        )
        self.assertTrue(np.isfinite(prediction))

    def test_predict_weighted_and_dominant_follow_gating_weights(self):
        x = np.array([[0.0], [1.0], [3.0], [4.0]])
        model = FuzzyCRegression(2, n_lags=1, random_state=0, n_init=1).fit(
            x, [1.0, 2.0, 2.5, 6.0]
        )
        model.memberships_ = np.array([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]
        ])
        model.coef_ = np.array([[1.0, 2.0], [10.0, 4.0]])
        model.fit_gating("fuzzy_centers", m_g=2.0)
        query = np.array([[1.5]])
        weights = model.predict_gating_memberships(query)
        individual = model.predict_by_cluster(query)
        self.assertAlmostEqual(
            model.predict(query, method="weighted")[0],
            np.sum(weights[0] * individual[0]),
        )
        self.assertAlmostEqual(
            model.predict(query, method="dominant")[0],
            individual[0, np.argmax(weights[0])],
        )

    def test_invalid_gating_and_prediction_methods_are_rejected(self):
        x, y = two_regime_data()
        model = FuzzyCRegression(
            2, n_lags=1, random_state=1, n_init=1
        ).fit(x, y)
        with self.assertRaises(ValueError):
            model.fit_gating("other")
        with self.assertRaises(ValueError):
            model.predict([[1.0]], gating="other")
        with self.assertRaises(ValueError):
            model.predict([[1.0]], method="other")


if __name__ == "__main__":
    unittest.main()
