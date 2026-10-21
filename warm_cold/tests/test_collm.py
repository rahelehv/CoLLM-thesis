import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from warm_cold.models import collm
from warm_cold.models.collm import (
    MODEL_SPECS,
    CollmExportError,
    mica_lora_b_stats,
    predict_collm,
    resume_plan,
    validate_mica_config,
    validate_mica_lora_b,
    validate_single_process,
)


class CollmTests(unittest.TestCase):
    def test_model_specs_match_the_two_eval_configs(self):
        self.assertEqual(MODEL_SPECS["collm-lora"].arch, "mini_gpt4rec_v2")
        self.assertFalse(MODEL_SPECS["collm-lora"].mica)
        self.assertEqual(
            MODEL_SPECS["collm-lora"].cfg_path,
            Path("train_configs/collm_eval_mf_ood.yaml"),
        )
        self.assertEqual(MODEL_SPECS["collm-mica"].arch, "mini_gpt4rec_mica")
        self.assertTrue(MODEL_SPECS["collm-mica"].mica)
        self.assertEqual(
            MODEL_SPECS["collm-mica"].cfg_path,
            Path("train_configs/collm_mica_eval_mf_ood.yaml"),
        )

    def test_single_process_validation(self):
        validate_single_process({"distributed": False, "world_size": 1}, environ={})
        with self.assertRaises(CollmExportError):
            validate_single_process({"distributed": True, "world_size": 1}, environ={})
        with self.assertRaises(CollmExportError):
            validate_single_process({"distributed": False, "world_size": 2}, environ={})
        with self.assertRaises(CollmExportError):
            validate_single_process(
                {"distributed": False, "world_size": 1}, environ={"RANK": "1"}
            )
        with self.assertRaises(CollmExportError):
            validate_single_process(
                {"distributed": False, "world_size": 1}, environ={"WORLD_SIZE": "2"}
            )

    def test_resume_plan_requires_completed_batches(self):
        self.assertEqual(resume_plan(0, 10, 4), (0, False))
        self.assertEqual(resume_plan(8, 10, 4), (2, False))
        self.assertEqual(resume_plan(10, 10, 4), (3, True))
        with self.assertRaises(CollmExportError):
            resume_plan(7, 10, 4)
        with self.assertRaises(CollmExportError):
            resume_plan(11, 10, 4)

    def test_mica_config_assertions(self):
        good = {
            "arch": "mini_gpt4rec_mica",
            "lora_config": {"use_lora": True, "mica": True},
        }
        validate_mica_config(good)
        with self.assertRaises(CollmExportError):
            validate_mica_config({"arch": "mini_gpt4rec_v2", "lora_config": good["lora_config"]})
        with self.assertRaises(CollmExportError):
            validate_mica_config(
                {"arch": "mini_gpt4rec_mica", "lora_config": {"use_lora": True, "mica": False}}
            )

    def test_mica_lora_b_must_be_nonzero(self):
        good = {
            "a.lora_B": torch.tensor([[1.0, 2.0]]),
            "b.lora_B": torch.tensor([[0.0, 3.0]]),
            "a.lora_A": torch.zeros(2, 2),
        }
        stats = mica_lora_b_stats(good)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["nonzero"], 2)

        class FakeModel:
            mica_patched_modules = 2

            def named_parameters(self):
                return good.items()

        result = validate_mica_lora_b(FakeModel())
        self.assertEqual(result["patched"], 2)

        bad = {"a.lora_B": torch.zeros(1, 2)}
        bad_stats = mica_lora_b_stats(bad)
        self.assertEqual(bad_stats["zero_names"], ["a.lora_B"])
        self.assertEqual(bad_stats["nonzero"], 0)

        class BadModel:
            mica_patched_modules = 1

            def named_parameters(self):
                return bad.items()

        with self.assertRaises(CollmExportError):
            validate_mica_lora_b(BadModel())


class _FakeDataset(Dataset):
    def __init__(self, ann_paths):
        self.n = 5

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"UserID": index, "label": index % 2}

    def collater(self, samples):
        return default_collate(samples)


class _FakeModel:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate_for_samples_v2(self, samples):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        user = samples["UserID"].float()
        return {"logits": user + 0.25}


class CollmResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        train = pd.DataFrame({"uid": [0, 1, 2], "iid": [0, 1, 2], "label": [1, 1, 0]})
        test = pd.DataFrame(
            {
                "uid": [0, 1, 2, 3, 4],
                "iid": [0, 1, 2, 3, 4],
                "label": [1, 0, 1, 0, 1],
            }
        )
        train.to_pickle(self.root / "train_ood2.pkl")
        test.to_pickle(self.root / "test_ood2.pkl")
        self.cfg = SimpleNamespace(
            model_cfg=SimpleNamespace(
                arch="mini_gpt4rec_v2",
                ckpt=str(self.root / "stage2.pth"),
                llama_model=str(self.root / "vicuna"),
                prompt_path=str(self.root / "prompt.txt"),
                rec_config=SimpleNamespace(
                    pretrained_path=str(self.root / "mf.pth"),
                    user_num=0,
                    item_num=0,
                ),
            ),
            run_cfg=SimpleNamespace(
                batch_size_eval=2,
                device="cpu",
                distributed=False,
                world_size=1,
                mode="v2",
            ),
        )
        self.cfg_path = self.root / "eval.yaml"
        self.spec = MODEL_SPECS["collm-lora"]

    def tearDown(self):
        self.tmp.cleanup()

    def _modules(self):
        pkg = types.ModuleType("minigpt4")
        datasets = types.ModuleType("minigpt4.datasets")
        datasets_datasets = types.ModuleType("minigpt4.datasets.datasets")
        rec_datasets = types.ModuleType("minigpt4.datasets.datasets.rec_datasets")
        rec_datasets.MoiveOOData = _FakeDataset
        return {
            "minigpt4": pkg,
            "minigpt4.datasets": datasets,
            "minigpt4.datasets.datasets": datasets_datasets,
            "minigpt4.datasets.datasets.rec_datasets": rec_datasets,
        }

    def _run_with_fake_stack(self, model):
        stack = ExitStack()
        stack.enter_context(mock.patch.dict(sys.modules, self._modules()))
        stack.enter_context(
            mock.patch.object(
                collm,
                "load_collm_config",
                return_value=(self.cfg, self.spec, self.cfg_path),
            )
        )
        stack.enter_context(
            mock.patch.object(
                collm,
                "build_collm_model",
                side_effect=lambda *args, **kwargs: (
                    model,
                    self.cfg,
                    self.spec,
                    self.cfg_path,
                    None,
                ),
            )
        )
        return stack

    def test_partial_checkpoint_is_resumed_from_completed_batch(self):
        out = self.root / "pred_collm_lora.csv"
        first_model = _FakeModel(fail_on_call=3)
        with self._run_with_fake_stack(first_model):
            with self.assertRaises(RuntimeError):
                predict_collm(
                    "collm-lora",
                    str(self.root),
                    str(out),
                    batch_size=2,
                    checkpoint_every=2,
                    device="cpu",
                )

        partial = out.with_suffix(".partial.csv")
        self.assertTrue(partial.is_file())
        self.assertEqual(len(pd.read_csv(partial)), 4)
        self.assertFalse(out.exists())

        second_model = _FakeModel()
        with self._run_with_fake_stack(second_model):
            meta = predict_collm(
                "collm-lora",
                str(self.root),
                str(out),
                batch_size=2,
                checkpoint_every=2,
                device="cpu",
            )

        self.assertEqual(meta["n_rows"], 5)
        self.assertEqual(second_model.calls, 1)
        self.assertFalse(partial.exists())
        self.assertTrue(out.is_file())
        self.assertEqual(len(pd.read_csv(out)), 5)

if __name__ == "__main__":
    unittest.main()
