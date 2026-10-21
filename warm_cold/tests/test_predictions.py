import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from warm_cold.predictions import (
    PredictionError,
    read_predictions,
    write_predictions,
)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame(
            {
                "uid": [1, 2, 3],
                "iid": [10, 20, 30],
                "label": [1, 0, 1],
            }
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "pred.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, **overrides):
        values = {
            "row_idx": np.array([0, 1, 2]),
            "uid": np.array([1, 2, 3]),
            "iid": np.array([10, 20, 30]),
            "label": np.array([1, 0, 1]),
            "logit": np.array([0.9, 0.1, 0.8]),
        }
        values.update(overrides)
        write_predictions(self.path, **values)

    def test_round_trip_full_coverage(self):
        self.write()
        table = read_predictions(self.path, self.test)
        np.testing.assert_array_equal(table.row_idx, [0, 1, 2])
        np.testing.assert_array_equal(table.users, [1, 2, 3])
        np.testing.assert_array_equal(table.labels, [1, 0, 1])
        np.testing.assert_allclose(table.logits, [0.9, 0.1, 0.8])

    def test_unordered_rows_are_aligned_by_row_idx(self):
        self.write(
            row_idx=np.array([2, 0, 1]),
            uid=np.array([3, 1, 2]),
            iid=np.array([30, 10, 20]),
            label=np.array([1, 1, 0]),
            logit=np.array([0.8, 0.9, 0.1]),
        )
        table = read_predictions(self.path, self.test)
        np.testing.assert_array_equal(table.row_idx, [0, 1, 2])
        np.testing.assert_allclose(table.logits, [0.9, 0.1, 0.8])

    def test_duplicate_row_idx_rejected(self):
        self.write(row_idx=np.array([0, 0, 2]))
        with self.assertRaises(PredictionError):
            read_predictions(self.path, self.test)

    def test_partial_coverage_rejected_by_default(self):
        self.write(
            row_idx=np.array([0, 1]),
            uid=np.array([1, 2]),
            iid=np.array([10, 20]),
            label=np.array([1, 0]),
            logit=np.array([0.9, 0.1]),
        )
        with self.assertRaises(PredictionError):
            read_predictions(self.path, self.test)
        table = read_predictions(self.path, self.test, require_full=False)
        self.assertEqual(table.row_idx.size, 2)

    def test_id_mismatch_rejected(self):
        self.write(uid=np.array([1, 99, 3]))
        with self.assertRaises(PredictionError):
            read_predictions(self.path, self.test)

    def test_nan_logit_rejected_on_write(self):
        with self.assertRaises(PredictionError):
            self.write(logit=np.array([0.9, np.nan, 0.8]))

    def test_non_integer_row_idx_rejected(self):
        self.write()
        df = pd.read_csv(self.path)
        df["row_idx"] = df["row_idx"].astype(float)
        df.loc[0, "row_idx"] = 0.5
        df.to_csv(self.path, index=False)
        with self.assertRaises(PredictionError):
            read_predictions(self.path, self.test)


if __name__ == "__main__":
    unittest.main()
