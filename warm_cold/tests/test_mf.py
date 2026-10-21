import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from warm_cold.models.mf import load_mf_weights, predict_mf, score_mf


class MFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = self.tmp.name
        self.checkpoint = os.path.join(self.tmp.name, "mf.pth")
        self.user_weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float32
        )
        self.item_weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float32
        )
        torch.save(
            {
                "user_embedding.weight": self.user_weight,
                "item_embedding.weight": self.item_weight,
            },
            self.checkpoint,
        )
        pd.DataFrame({"uid": [0, 1, 2, 3], "iid": [0, 1, 2, 0], "label": [1, 0, 1, 0]}).to_pickle(
            os.path.join(self.data_dir, "train_ood2.pkl")
        )
        pd.DataFrame({"uid": [0, 1, 2, 3], "iid": [0, 1, 2, 1], "label": [1, 0, 1, 0]}).to_pickle(
            os.path.join(self.data_dir, "test_ood2.pkl")
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_score_is_exact_dot_product(self):
        uid = np.array([0, 1, 2, 3])
        iid = np.array([0, 1, 2, 1])
        got = score_mf(uid, iid, self.user_weight, self.item_weight)
        expected = np.array([1.0, 1.0, 3.0, 0.0])
        np.testing.assert_allclose(got, expected)

    def test_load_weights(self):
        user_weight, item_weight = load_mf_weights(self.checkpoint)
        np.testing.assert_allclose(user_weight.numpy(), self.user_weight.numpy())
        np.testing.assert_allclose(item_weight.numpy(), self.item_weight.numpy())

    def test_predict_mf_writes_full_csv_and_meta(self):
        out = os.path.join(self.tmp.name, "pred_mf.csv")
        meta = predict_mf(self.data_dir, self.checkpoint, out, threshold=3, seed=42)
        self.assertEqual(meta["data"]["n_test"], 4)
        self.assertTrue(os.path.isfile(out))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "pred_mf.meta.json")))
        pred = pd.read_csv(out)
        self.assertEqual(list(pred.columns), ["row_idx", "uid", "iid", "label", "logit"])
        np.testing.assert_array_equal(pred["row_idx"].to_numpy(), np.arange(4))
        np.testing.assert_allclose(pred["logit"].to_numpy(), [1.0, 1.0, 3.0, 0.0])


if __name__ == "__main__":
    unittest.main()
