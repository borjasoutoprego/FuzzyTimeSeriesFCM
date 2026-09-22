import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np


def load_fcm_class():
    path = Path(__file__).resolve().parents[1] / "fcm-timeseries-continuous-univariant.py"
    spec = importlib.util.spec_from_file_location("fcm_timeseries", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module.FuzzyTimeSeriesFCM


class TestFuzzyTimeSeriesFCM(unittest.TestCase):
    def setUp(self):
        self.series = np.array([10, 11, 12, 13, 25, 26, 27, 28], dtype=float)
        self.model = load_fcm_class()(n_clusters=2, m=2.0)
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
