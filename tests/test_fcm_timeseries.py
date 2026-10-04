import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load_fcm_module():
    path = Path(__file__).resolve().parents[1] / "fcm-timeseries-continuous-univariant.py"
    spec = importlib.util.spec_from_file_location("fcm_timeseries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fcm_class():
    return load_fcm_module().FuzzyTimeSeriesFCM


class TestFuzzyTimeSeriesFCM(unittest.TestCase):
    def setUp(self):
        self.series = np.array([10, 11, 12, 13, 25, 26, 27, 28], dtype=float)
        self.model = load_fcm_class()(n_clusters=2, m=2.0, random_state=4)
        self.model.fit_fcm(self.series)
        self.labels = self.model.fuzzify()

    def test_fcm_fuzzification(self):
        self.assertEqual(self.model.centers.shape, (2,))
        self.assertEqual(self.model.u.shape, (2, self.series.size))
        self.assertEqual(self.labels.shape, (self.series.size,))
        self.assertTrue(np.allclose(self.model.u.sum(axis=0), 1.0))
        self.assertTrue(np.all(np.diff(self.model.centers) >= 0))

    def test_cheng_rules_and_prediction(self):
        rules = self.model.build_rules()
        prediction = self.model.predict_cheng(self.labels[-1])
        self.assertEqual(set(rules), {1, 2})
        self.assertTrue(np.isfinite(prediction))

    def test_egrioglu_prediction(self):
        self.model.train_nn()
        prediction = self.model.predict_nn(self.labels[-1])
        self.assertTrue(np.isfinite(prediction))
        self.assertGreaterEqual(prediction, self.model.centers.min())
        self.assertLessEqual(prediction, self.model.centers.max())

    def test_new_fuzzification_does_not_refit(self):
        centers = self.model.centers.copy()
        validation_values = [14.0, 30.0]
        labels, memberships = self.model.fuzzify_new(
            validation_values, return_memberships=True
        )
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(memberships.shape, (2, 2))
        self.assertTrue(np.allclose(memberships.sum(axis=0), 1.0))
        self.assertTrue(np.allclose(self.model.centers, centers))

    def test_grid_search_is_insensitive_to_test_values(self):
        train = np.array([1, 2, 3, 10, 11, 12], dtype=float)
        validation = np.array([13, 14, 15], dtype=float)
        test_a = np.array([16, 17, 18], dtype=float)
        test_b = np.array([1e6, -1e6, 1e6], dtype=float)
        module = load_fcm_module()
        first = module.grid_search_fcm(train, validation, c_values=(2,), m_values=(2.0,), random_state=3)
        second = module.grid_search_fcm(train, validation, c_values=(2,), m_values=(2.0,), random_state=3)
        self.assertEqual(first, second)
        self.assertFalse(np.array_equal(test_a, test_b))

    def test_validation_first_forecast_uses_last_train_value(self):
        self.model.build_rules()
        expected = self.model.predict_cheng(self.model.fuzzify_new([self.series[-1]])[0])
        actual = self.model.predict_one_step(self.series, method="cheng")
        self.assertEqual(expected, actual)
