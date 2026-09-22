import unittest

import numpy as np

from fuzzy_information_granules import FuzzyInformationGranules


class TestFuzzyInformationGranules(unittest.TestCase):
    def test_fit_and_one_step_prediction(self):
        series = np.array([10, 11, 12, 13, 25, 26, 27, 28, 18, 19, 20, 21], dtype=float)
        model = FuzzyInformationGranules(window_size=3, n_clusters=2, m=2.0).fit(series)
        prediction = model.predict(series)
        self.assertEqual(model.granules.shape, (series.size - 2, 6))
        self.assertEqual(model.memberships.shape, (2, series.size - 2))
        self.assertEqual(model.labels.shape, (series.size - 2,))
        self.assertTrue(np.allclose(model.memberships.sum(axis=0), 1.0))
        self.assertTrue(np.isfinite(prediction))
        self.assertTrue(np.allclose(model.last_memberships.sum(), 1.0))
