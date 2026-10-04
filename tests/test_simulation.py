import inspect
import unittest

import numpy as np
from scipy.special import expit

from simulation.config import Scenario, split_series
from simulation.dgp import derive_seed, generate_series
from simulation.diagnostics import (
    adjusted_rand,
    correlations,
    match_two_regime_coefficients,
    regime_membership_transition_estimate,
)
from simulation.evaluation import (
    rolling_one_step,
    select_fcm_clusters,
    select_fcrm_clusters,
    select_granular_parameters,
)
from simulation.experiment import _lstar_diagnostics, _setar_diagnostics
from simulation.metrics import calculate_metrics
from simulation.statistics import (
    friedman_and_nemenyi,
    read_counterfactual_metrics,
    selection_frequencies,
)


class TestSimulationDGP(unittest.TestCase):
    def test_all_dgps_are_finite_correct_length_and_reproducible(self):
        scenarios = [
            Scenario("E1", 50, 0.25),
            Scenario("E2", 50, 0.50),
            Scenario("E3", 100, 0.25),
            Scenario("E4", 500, 0.50, 20.0),
        ]
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                first = generate_series(scenario, seed=54321, burn_in=17)
                second = generate_series(scenario, seed=54321, burn_in=17)
                self.assertEqual(first.series.size, scenario.T)
                self.assertTrue(np.all(np.isfinite(first.series)))
                self.assertTrue(np.array_equal(first.series, second.series))
                if scenario.dgp == "E3":
                    self.assertEqual(first.true_regime.size, scenario.T)
                    self.assertTrue(set(np.unique(first.true_regime)).issubset({0, 1}))
                if scenario.dgp == "E4":
                    self.assertEqual(first.G_true.size, scenario.T)
                    self.assertTrue(np.all((first.G_true >= 0.0) & (first.G_true <= 1.0)))

    def test_burnin_is_removed_from_the_retained_sample(self):
        scenario = Scenario("E2", 50, 0.25)
        no_burn = generate_series(scenario, seed=17, burn_in=0).series
        with_burn = generate_series(scenario, seed=17, burn_in=8).series
        self.assertTrue(np.array_equal(with_burn[:42], no_burn[8:]))

    def test_setar_regime_matches_the_threshold_and_lstar_g_is_preserved(self):
        setar = generate_series(Scenario("E3", 100, 0.25), seed=42, burn_in=200)
        expected_regime = (setar.series[:-1] > 0.0).astype(np.int8)
        self.assertTrue(np.array_equal(setar.true_regime[1:], expected_regime))

        gamma = 5.0
        lstar = generate_series(Scenario("E4", 100, 0.25, gamma), seed=42, burn_in=200)
        expected_g = expit(gamma * lstar.series[:-1])
        self.assertTrue(np.allclose(lstar.G_true[1:], expected_g))

    def test_seed_derivation_is_stable_and_distinct_by_scenario_and_replication(self):
        scenario = Scenario("E4", 100, 0.25, 5.0)
        same = derive_seed(20261004, scenario, 3, stream=0)
        self.assertEqual(same, derive_seed(20261004, scenario, 3, stream=0))
        self.assertNotEqual(same, derive_seed(20261004, scenario, 4, stream=0))
        self.assertNotEqual(same, derive_seed(20261004, Scenario("E4", 100, 0.25, 20.0), 3, stream=0))


class TestSimulationEvaluation(unittest.TestCase):
    def test_chronological_60_20_20_split_for_t50(self):
        values = np.arange(50, dtype=float)
        split = split_series(values)
        self.assertEqual((split.train.size, split.validation.size, split.test.size), (30, 10, 10))
        self.assertTrue(np.array_equal(split.train, values[:30]))
        self.assertTrue(np.array_equal(split.validation, values[30:40]))
        self.assertTrue(np.array_equal(split.test, values[40:]))
        self.assertEqual(split.train_range, (0, 30))
        self.assertEqual(split.validation_range, (30, 40))
        self.assertEqual(split.test_range, (40, 50))

    def test_rolling_one_step_appends_actual_observations(self):
        observed = []

        def predict(history):
            observed.append(tuple(history))
            return history[-1]

        forecast = rolling_one_step([10.0, 11.0], [100.0, 101.0], predict)
        self.assertTrue(np.array_equal(forecast, [11.0, 100.0]))
        self.assertEqual(observed, [(10.0, 11.0), (10.0, 11.0, 100.0)])

    def test_selection_interfaces_cannot_access_test_partition(self):
        for function in (select_fcm_clusters, select_granular_parameters, select_fcrm_clusters):
            with self.subTest(function=function.__name__):
                self.assertNotIn("test", inspect.signature(function).parameters)


class TestSimulationMetricsAndDiagnostics(unittest.TestCase):
    def test_metrics_include_zero_cases_without_nan(self):
        scores = calculate_metrics([0.0, 2.0, -1.0], [0.0, 1.0, -2.0])
        self.assertAlmostEqual(scores["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(2.0 / 3.0))
        self.assertTrue(np.isfinite(scores["smape"]))
        self.assertAlmostEqual(calculate_metrics([0.0], [0.0])["smape"], 0.0)

    def test_selection_frequencies_preserve_joint_granular_candidates(self):
        rows = [
            {"dgp": "E1", "T": 50, "sigma": 0.25, "gamma": None,
             "method_variant": "FCM_Granular", "replication": 1,
             "selected_c": 2, "selected_L": 3},
            {"dgp": "E1", "T": 50, "sigma": 0.25, "gamma": None,
             "method_variant": "FCM_Granular", "replication": 2,
             "selected_c": 3, "selected_L": 5},
        ]
        frequencies = selection_frequencies(rows)
        joint = {
            row["value"]: row["count"]
            for row in frequencies if row["parameter"] == "selected_c_L"
        }
        self.assertEqual(joint, {"2,3": 1, "3,5": 1})

    def test_counterfactual_metrics_are_read_for_a_separate_summary(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            result = {
                "metrics": [
                    {"evaluation_split": "test", "result_group": "core", "method_variant": "AR"},
                    {"evaluation_split": "test", "result_group": "counterfactual", "method_variant": "FCRM_TRUE"},
                    {"evaluation_split": "validation", "result_group": "counterfactual", "method_variant": "FCRM_RV"},
                ]
            }
            Path(directory, "rep.json").write_text(json.dumps(result), encoding="utf-8")
            rows = read_counterfactual_metrics(directory)
        self.assertEqual([row["method_variant"] for row in rows], ["FCRM_TRUE"])

    def test_friedman_significance_triggers_nemenyi_comparisons(self):
        rows = []
        for replication in range(1, 9):
            for method, base in (("A", 0.1), ("B", 1.0), ("C", 2.0)):
                rows.append({
                    "dgp": "E1",
                    "T": 50,
                    "sigma": 0.25,
                    "gamma": None,
                    "replication": replication,
                    "method_variant": method,
                    "mae": base + replication * 0.001,
                    "rmse": base + replication * 0.001,
                    "smape": base + replication * 0.001,
                })
        friedman, nemenyi = friedman_and_nemenyi(rows)
        self.assertEqual(len(friedman), 3)
        self.assertTrue(all(row["significant"] for row in friedman))
        self.assertEqual(len(nemenyi), 9)
        self.assertTrue(any(row["significant"] for row in nemenyi))

    def test_two_regime_coefficient_matching_handles_cluster_permutation(self):
        estimated = np.asarray([[0.0, -0.45, 0.30], [0.0, 0.82, 0.05]])
        match = match_two_regime_coefficients(estimated)
        self.assertEqual(match["estimated_cluster_to_true_regime"], [1, 0])
        self.assertAlmostEqual(match["total_distance"], 0.0)
        self.assertAlmostEqual(adjusted_rand([0, 0, 1, 1], [3, 3, 8, 8]), 1.0)

    def test_lstar_correlations_and_constant_input_handling(self):
        self.assertAlmostEqual(correlations([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])["pearson"], 1.0)
        self.assertAlmostEqual(correlations([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])["spearman"], 1.0)
        self.assertEqual(correlations([1.0, 1.0], [0.0, 1.0]), {"pearson": None, "spearman": None})

    def test_lstar_membership_with_no_cluster_near_regime_two_is_finite(self):
        memberships = np.asarray([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
        coefficients = np.asarray([
            [0.0, 0.8, 0.05],
            [0.0, 0.7, 0.04],
            [0.0, 0.75, 0.03],
        ])
        estimate, mapping, _ = regime_membership_transition_estimate(
            memberships, coefficients
        )
        self.assertTrue(np.array_equal(mapping, [0, 0, 0]))
        self.assertTrue(np.array_equal(estimate, [0.0, 0.0]))
        self.assertEqual(correlations([0.2, 0.8], estimate), {"pearson": None, "spearman": None})

    def test_lstar_c2_uses_one_to_one_coefficient_match_and_c_gt_2_is_auxiliary(self):
        class Model:
            def __init__(self, coefficients, weights):
                self.coef_ = coefficients
                self.memberships = weights

            def memberships_for(self, lag_vectors, responses):
                self.responses_seen = np.asarray(responses).copy()
                return self.memberships[:len(lag_vectors)]

            def predict_gating_memberships(self, lag_vectors, method):
                return self.memberships[:len(lag_vectors)]

        c2_coefficients = np.asarray([
            [0.0, 0.82, 0.05],
            [0.0, 0.40, 0.08],
        ])
        c2_weights = np.asarray([
            [0.90, 0.10],
            [0.20, 0.80],
            [0.55, 0.45],
        ])
        c2_model = Model(c2_coefficients, c2_weights)
        c3_coefficients = np.asarray([
            [0.0, 0.82, 0.05],
            [0.0, 0.75, 0.04],
            [0.0, 0.70, 0.06],
        ])
        c3_weights = np.full((3, 3), 1.0 / 3.0)
        c3_model = Model(c3_coefficients, c3_weights)
        result = _lstar_diagnostics(
            models={
                2: c2_model,
                3: c3_model,
            },
            selected_by_gate={"fuzzy_centers": 3, "multinomial": 3},
            G_true=np.asarray([0.1, 0.2, 0.3, 0.4, 0.1, 0.5, 0.9]),
            train_validation=np.asarray([0.1, -0.1, 0.2, -0.2]),
            test=np.asarray([0.3, -0.3, 0.4]),
            test_start=4,
        )
        c2 = result["models"]["fuzzy_centers_c2"]
        self.assertEqual(c2["diagnostic_scope"], "pdf_defined_c2")
        self.assertEqual(c2["estimated_cluster_to_true_regime"], [0, 1])
        self.assertTrue(np.allclose(c2["G_estimated_test"], [0.10, 0.80, 0.45]))
        self.assertTrue(np.array_equal(c2_model.responses_seen, [0.3, -0.3, 0.4]))

        c3 = result["models"]["fuzzy_centers_c_selected"]
        self.assertEqual(
            c3["diagnostic_scope"],
            "auxiliary_c_gt_2_nearest_coefficient_aggregation",
        )
        self.assertIn("G_estimated_auxiliary_test", c3)
        self.assertNotIn("G_estimated_test", c3)
        self.assertNotIn("correlations", c3)

    def test_setar_test_membership_diagnostic_is_posthoc_but_forecast_uses_gate(self):
        class Model:
            coef_ = np.asarray([[0.0, -0.45, 0.30], [0.0, 0.82, 0.05]])
            memberships_ = np.tile([0.5, 0.5], (6, 1))
            labels_ = np.arange(6) % 2

            def predict_by_cluster(self, lag_vectors):
                return np.asarray([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])

            def predict_gating_memberships(self, lag_vectors, method):
                return np.asarray([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])

            def memberships_for(self, lag_vectors, responses):
                self.responses_seen = np.asarray(responses).copy()
                return np.asarray([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])

        model = Model()
        recorded = {}

        def record(method, variant, gate, group, split, y_true, y_pred,
                   *, selected_c, start_index):
            recorded[variant] = np.asarray(y_pred).copy()

        true_regime = np.asarray([0, 1] * 5 + [0])
        result = _setar_diagnostics(
            models={2: model},
            selected_by_gate={"fuzzy_centers": 2, "multinomial": 2},
            true_regime=true_regime,
            train_validation=np.arange(8, dtype=float),
            test=np.asarray([0.2, -0.2, 0.3]),
            test_start=8,
            predictions_by_gate_and_c={},
            record=record,
        )

        diagnostic = result["models"]["fuzzy_centers_c2"]
        self.assertTrue(np.array_equal(model.responses_seen, [0.2, -0.2, 0.3]))
        self.assertAlmostEqual(diagnostic["test_membership_ari"], 1.0)
        self.assertTrue(np.array_equal(recorded["FCRM_fuzzy_centers_c2"], [10.0, 21.0, 12.0]))
        self.assertFalse(np.array_equal(recorded["FCRM_fuzzy_centers_c2"], [20.0, 11.0, 22.0]))


if __name__ == "__main__":
    unittest.main()
