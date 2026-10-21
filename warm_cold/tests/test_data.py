import os
import unittest

import numpy as np
import pandas as pd

from warm_cold.data import (
    DEFAULT_THRESHOLD,
    PAPER_THRESHOLD,
    WarmColdError,
    build_masks,
    load_warm_cold,
)


def synthetic_frames():
    train = pd.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 3],
            "iid": [10, 10, 10, 20, 20, 30],
        }
    )
    test = pd.DataFrame(
        {
            "uid": [1, 1, 2, 3, 4],
            "iid": [10, 20, 20, 30, 10],
            "label": [1, 0, 1, 1, 0],
        }
    )
    return train, test


class DataTests(unittest.TestCase):
    def test_default_is_paper_threshold(self):
        self.assertEqual(DEFAULT_THRESHOLD, 3)
        self.assertEqual(PAPER_THRESHOLD, 3)

    def test_masks_use_both_user_and_item_counts(self):
        train, test = synthetic_frames()
        warm, cold, user_counts, item_counts = build_masks(train, test, threshold=3)
        self.assertEqual(int(warm.sum()), 1)
        self.assertEqual(int(cold.sum()), 4)
        self.assertTrue(warm[0])
        self.assertFalse(cold[0])
        self.assertEqual(int(user_counts[0]), 3)
        self.assertEqual(int(item_counts[0]), 3)
        self.assertTrue(bool((warm | cold).all()))
        self.assertFalse(bool((warm & cold).any()))

    def test_cold_is_exact_complement(self):
        train, test = synthetic_frames()
        warm, cold, _, _ = build_masks(train, test, threshold=3)
        np.testing.assert_array_equal(cold, ~warm)

    def test_invalid_threshold_rejected(self):
        train, test = synthetic_frames()
        with self.assertRaises(WarmColdError):
            build_masks(train, test, threshold=0)

    def test_missing_column_rejected(self):
        train, test = synthetic_frames()
        with self.assertRaises(WarmColdError):
            build_masks(train.drop(columns=["iid"]), test, threshold=3)

    def test_real_counts_when_data_dir_is_available(self):
        data_dir = os.environ.get("COLLM_ML1M_DIR")
        if not data_dir:
            self.skipTest("COLLM_ML1M_DIR is not set")
        data = load_warm_cold(data_dir)
        self.assertEqual(data.stats["n_test"], 7331)
        self.assertEqual(data.stats["n_warm"], 3682)
        self.assertEqual(data.stats["n_cold"], 3649)


if __name__ == "__main__":
    unittest.main()
