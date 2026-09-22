import unittest

import numpy as np

from fuzzy_information_granules import FuzzyInformationGranules


class TestFuzzyInformationGranules(unittest.TestCase):
    def test_fit_and_one_step_prediction(self):
        series = np.array([10, 11, 12, 13, 25, 26, 27, 28, 18, 19, 20, 21], dtype=float)
        model = FuzzyInformationGranules(window_size=3, n_clusters=2, m=2.0, random_state=1).fit(series)
        prediction = model.predict(series)
        self.assertEqual(model.granules.shape, (series.size - 3, 6))
        self.assertEqual(model.memberships.shape, (2, series.size - 3))
        self.assertEqual(model.labels.shape, (series.size - 3,))
        self.assertTrue(np.allclose(model.memberships.sum(axis=0), 1.0))
        self.assertTrue(np.isfinite(prediction))
        self.assertTrue(np.allclose(model.last_memberships.sum(), 1.0))

    def test_last_window_is_not_used_to_fit_rules(self):
        series = np.arange(1.0, 12.0)
        model = FuzzyInformationGranules(3, 2, random_state=5).fit(series)
        self.assertEqual(model.windows.shape[0], series.size - 3)
        self.assertTrue(np.array_equal(model.targets_, series[3:]))
        self.assertFalse(any(np.array_equal(window, series[-3:]) for window in model.windows))

    def test_new_window_prediction_keeps_fitted_centers(self):
        series = np.array([1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11], dtype=float)
        model = FuzzyInformationGranules(3, 2, random_state=3).fit(series)
        centers = model.centers.copy()
        prediction = model.predict(np.r_[series, 12.0])
        self.assertTrue(np.isfinite(prediction))
        self.assertTrue(np.allclose(model.centers, centers))

    def test_future_values_do_not_change_training_fit(self):
        train = np.array([1, 2, 3, 7, 8, 9, 4, 5, 6], dtype=float)
        first = FuzzyInformationGranules(3, 2, random_state=9).fit(train)
        second = FuzzyInformationGranules(3, 2, random_state=9).fit(train)
        self.assertTrue(np.allclose(first.centers, second.centers))
        self.assertTrue(np.allclose(first.memberships, second.memberships))
