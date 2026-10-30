import importlib.util
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('climbing', Path(__file__).resolve().parents[1] / 'climbingModel/climbing.py')
climbing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(climbing)


class ClimbingTests(unittest.TestCase):
    def load(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            frame.to_csv(path, index=False)
            return climbing.load_climber_data(path)

    def frame(self, ages):
        return pd.DataFrame({'height': 175, 'weight': 70, 'age': ages, 'sex': 0, 'grades_max': 50, 'grades_mean': 40, 'country': 'SWE'})

    def test_age_boundaries_match_labels(self):
        result = self.load(self.frame([24, 25, 34, 35, 44, 45, 54, 55]))
        self.assertEqual(list(result['age_band']), ['<25', '25-34', '25-34', '35-44', '35-44', '45-54', '45-54', '55+'])

    def test_invalid_measurements_do_not_create_infinite_bmi(self):
        frame = self.frame([20, 21, 22, 23])
        frame['height'] = [175, 0, -1, np.inf]
        result = self.load(frame)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result['bmi']).all())

    def test_missing_schema_is_explained(self):
        with self.assertRaisesRegex(ValueError, 'Missing columns'):
            self.load(pd.DataFrame({'height': [175]}))

    def test_sparse_and_constant_groups_do_not_fit_invalid_lines(self):
        for x, y in [([20], [50]), ([20, 20], [40, 50])]:
            slope, _, _ = climbing.scatter_with_trend(pd.Series(x), pd.Series(y))
            self.assertTrue(np.isnan(slope))
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
