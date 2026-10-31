import ast
import math
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import uauc
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

    def test_primary_and_official_agree_when_auc_is_defined(self):
        users = np.array([1, 1, 2, 2, 3, 3])
        labels = np.array([1, 0, 1, 0, 1, 1])
        logits = np.array([0.9, 0.1, 0.2, 0.8, 0.7, 0.6])
        result = subset_metrics(users, labels, logits)
        self.assertAlmostEqual(result["uauc"], 0.5)
        self.assertAlmostEqual(result["uauc_official"], 0.5)

    def test_all_single_class_subset_returns_nan_for_both_metrics(self):
        users = np.array([1, 1, 2, 2])
        labels = np.array([1, 1, 0, 0])
        logits = np.array([0.9, 0.8, 0.7, 0.6])
        result = subset_metrics(users, labels, logits)
        self.assertTrue(math.isnan(result["uauc"]))
        self.assertTrue(math.isnan(result["uauc_official"]))
        self.assertEqual(result["n_computed_users"], 0)

    def test_empty_subset_returns_nan(self):
        result = subset_metrics(np.array([]), np.array([]), np.array([]))
        self.assertEqual(result["n"], 0)
        self.assertTrue(math.isnan(result["uauc"]))
        self.assertTrue(math.isnan(result["uauc_official"]))


class UAUCAggregationTests(unittest.TestCase):
    def test_nanmean_averages_defined_values(self):
        value = uauc._nanmean_auc([0.9, np.nan, 0.7])
        self.assertAlmostEqual(float(value), 0.8)

    def test_all_nan_returns_nan_without_runtime_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = uauc._nanmean_auc([np.nan, np.nan])
        self.assertTrue(math.isnan(value))
        self.assertFalse(
            any(issubclass(item.category, RuntimeWarning) for item in caught),
            [str(item.message) for item in caught],
        )

    def test_empty_input_returns_nan(self):
        self.assertTrue(math.isnan(uauc._nanmean_auc([])))

    def test_uAUC_me_ignores_nan_per_user_values(self):
        users = np.array([1, 1, 2, 2, 3, 3])
        labels = np.array([1, 0, 1, 0, 1, 0])
        logits = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        with mock.patch("uauc.roc_auc_score", side_effect=[0.9, np.nan, 0.7]):
            value, computed, per_user = uAUC_me(
                users, logits, labels, verbose=False
            )
        self.assertAlmostEqual(float(value), 0.8)
        self.assertEqual(list(computed), [1, 2, 3])
        self.assertEqual(per_user.shape[0], 3)
        self.assertAlmostEqual(per_user[0], 0.9)
        self.assertTrue(np.isnan(per_user[1]))
        self.assertAlmostEqual(per_user[2], 0.7)

    def test_uAUC_me_all_nan_returns_nan_without_runtime_warning(self):
        users = np.array([1, 1, 2, 2])
        labels = np.array([1, 0, 1, 0])
        logits = np.array([0.9, 0.1, 0.8, 0.2])
        with mock.patch("uauc.roc_auc_score", return_value=np.nan):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value, computed, per_user = uAUC_me(
                    users, logits, labels, verbose=False
                )
        self.assertTrue(math.isnan(value))
        self.assertEqual(list(computed), [1, 2])
        self.assertEqual(per_user.shape[0], 2)
        self.assertFalse(
            any(issubclass(item.category, RuntimeWarning) for item in caught),
            [str(item.message) for item in caught],
        )

    def test_uAUC_me_empty_input_returns_nan(self):
        value, computed, per_user = uAUC_me(
            np.array([]), np.array([]), np.array([]), verbose=False
        )
        self.assertTrue(math.isnan(value))
        self.assertEqual(computed, [])
        self.assertEqual(per_user.shape[0], 0)

    def test_uAUC_me_preserves_single_interaction_skip(self):
        users = np.array([1, 2, 2, 3, 3])
        labels = np.array([1, 1, 0, 1, 1])
        logits = np.array([0.9, 0.8, 0.2, 0.7, 0.6])
        with mock.patch("uauc.roc_auc_score", side_effect=[1.0, np.nan]):
            value, computed, per_user = uAUC_me(
                users, logits, labels, verbose=False
            )
        self.assertAlmostEqual(float(value), 1.0)
        self.assertEqual(list(computed), [2, 3])
        self.assertEqual(per_user.shape[0], 2)

    def test_warm_cold_uses_shared_implementation(self):
        self.assertIs(uAUC_me, uauc.uAUC_me)

    def test_standard_task_imports_shared_implementation(self):
        try:
            from minigpt4.tasks.rec_base_task import uAUC_me as task_uauc
        except Exception as exc:
            if os.environ.get("COLLM_REQUIRE_OFFICIAL_PARITY") == "1":
                self.fail(f"standard task import failed: {exc}")
            self.skipTest(f"standard task not importable in this environment: {exc}")

        self.assertIs(task_uauc, uauc.uAUC_me)

    def test_standard_task_source_wires_shared_implementation(self):
        source_path = Path(__file__).resolve().parents[2] / "minigpt4/tasks/rec_base_task.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports_shared = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "uauc"
            and any(alias.name == "uAUC_me" for alias in node.names)
            for node in ast.walk(tree)
        )
        self.assertTrue(imports_shared)
        self.assertNotIn("def uAUC_me", source)


if __name__ == "__main__":
    unittest.main()
