from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import InputError, windows, FEATURES
from random_forest import load_forest, predict_forest, score_forest


class RandomForestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forest = load_forest()
        cls.report = json.loads((ROOT / 'reports/random_forest_report.json').read_text())

    def test_model_identity_and_report_integrity(self):
        self.assertEqual(len(self.forest['trees']), 100)
        self.assertEqual(self.forest['feature_names'], FEATURES)
        digest = hashlib.sha256((ROOT / 'model/trained_forest.json').read_bytes()).hexdigest()
        self.assertEqual(digest, self.report['forest_sha256'])
        tree_digest = hashlib.sha256((ROOT / 'model/trained_tree.json').read_bytes()).hexdigest()
        self.assertEqual(tree_digest, self.report['original_tree_sha256'])

    def test_reproduce_report_from_exported_model_and_real_recordings(self):
        actual = np.zeros((2, 2), dtype=int)
        for path in (ROOT / 'data/processed/test').glob('*_belt.csv'):
            for window in windows(pd.read_csv(path), require_labels=True):
                predicted = int(score_forest(self.forest, window['features']) >= self.forest['decision_threshold'])
                actual[window['target'], predicted] += 1
        expected = self.report['splits']['reused_test']['random_forest']['window_metrics']['confusion_matrix']
        np.testing.assert_array_equal(actual, expected)

    def test_inference_uses_sensor_values_and_ignores_forged_labels(self):
        source = pd.read_csv(ROOT / 'examples/fall_window.csv')
        original = predict_forest(self.forest, source)
        changed = source.copy()
        changed['frame_activity_id'] = 7
        changed['subject_id'] = 999
        changed['trial_activity_id'] = 6
        self.assertEqual(original, predict_forest(self.forest, changed))
        self.assertTrue(all(0 <= row['fall_score'] <= 1 for row in original))

    def test_invalid_sensor_inputs_are_rejected(self):
        frame = pd.read_csv(ROOT / 'examples/walking_window.csv')
        with self.assertRaises(InputError):
            predict_forest(self.forest, frame.iloc[:1])
        frame.loc[1, 'time_s'] = frame.loc[0, 'time_s']
        with self.assertRaises(InputError):
            predict_forest(self.forest, frame)


if __name__ == '__main__':
    unittest.main(verbosity=2)
