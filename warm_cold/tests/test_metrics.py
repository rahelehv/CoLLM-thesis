import math
import unittest

import numpy as np

from warm_cold.metrics import subset_metrics, uAUC_me


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.users = np.array([1, 1, 1, 2, 2, 3, 4, 4])
        self.labels = np.array([1, 0, 1, 1, 1, 1, 1, 0])
        self.logits = np.array([0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

    def test_primary_uauc_is_manual_mean_over_users_with_both_classes(self):
        result = subset_metrics(self.users, self.labels, self.logits)
        self.assertAlmostEqual(result["uauc"], 1.0)
        self.assertEqual(result["n_unique_users"], 4)
        self.assertEqual(result["n_computed_users"], 2)
        self.assertEqual(result["n_skipped_single_class"], 1)
        self.assertEqual(result["n_single_interaction"], 1)

    def test_primary_uauc_skips_single_class_and_single_interaction(self):
        users = np.array([1, 1, 2, 3, 3, 4])
        labels = np.array([1, 1, 1, 1, 1, 1])
        logits = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
        result = subset_metrics(users, labels, logits)
        self.assertTrue(math.isnan(result["uauc"]))
        self.assertEqual(result["n_computed_users"], 0)
        self.assertEqual(result["n_skipped_single_class"], 2)
        self.assertEqual(result["n_single_interaction"], 2)

    def test_primary_uauc_manual_example(self):
        users = np.array([10, 10, 20, 20, 30])
        labels = np.array([1, 1, 1, 0, 1])
        logits = np.array([0.9, 0.4, 0.8, 0.2, 0.7])
        result = subset_metrics(users, labels, logits)
        self.assertAlmostEqual(result["uauc"], 1.0)
        self.assertEqual(result["n_computed_users"], 1)
        self.assertEqual(result["n_skipped_single_class"], 1)
        self.assertEqual(result["n_single_interaction"], 1)

    def test_official_function_is_nan_propagating(self):
        official, computed, aucs = uAUC_me(self.users, self.logits, self.labels)
        self.assertTrue(math.isnan(official))
        self.assertEqual(len(computed), 3)
        self.assertEqual(aucs.shape[0], 3)
        self.assertTrue(np.isnan(aucs).any())

    def test_parity_with_repo_function_when_importable(self):
        try:
            from minigpt4.tasks.rec_base_task import uAUC_me as official_uauc
        except Exception as exc:
            self.skipTest(f"official uAUC_me not importable in this environment: {exc}")

        got = uAUC_me(self.users, self.logits, self.labels)
        expected = official_uauc(self.users, self.logits, self.labels)
        self.assertTrue(np.allclose(got[0], expected[0], equal_nan=True))
        self.assertEqual(list(got[1]), list(expected[1]))
        self.assertTrue(np.allclose(got[2], expected[2], equal_nan=True))


if __name__ == "__main__":
    unittest.main()
