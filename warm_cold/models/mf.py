"""Matrix-factorization predictor.

The score is the exact dot product used by ``MatrixFactorization.forward``
in ``minigpt4/models/rec_base_models.py:86-90``. Loading the weights directly
keeps the CPU scorer usable without importing the LAVIS/transformers stack.
"""

from pathlib import Path

import numpy as np
import torch

from warm_cold.data import DEFAULT_THRESHOLD, load_split
from warm_cold.predictions import read_predictions, write_predictions
from warm_cold.provenance import (
    checkpoint_info,
    environment_info,
    git_commit,
    sha256_file,
    write_json,
)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mf_weights(path: str):
    checkpoint = _torch_load(path)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"MF checkpoint is not a state dict: {path}")

    state = {str(k).removeprefix("module."): v for k, v in checkpoint.items()}
    for key in ("user_embedding.weight", "item_embedding.weight"):
        if key not in state:
            raise ValueError(f"MF checkpoint missing '{key}': {path}")
        if not torch.is_tensor(state[key]) or state[key].ndim != 2:
            raise ValueError(f"MF checkpoint '{key}' is not a 2D tensor: {path}")
    if state["user_embedding.weight"].shape[1] != state["item_embedding.weight"].shape[1]:
        raise ValueError("MF user/item embedding sizes differ")

    return (
        state["user_embedding.weight"].float(),
        state["item_embedding.weight"].float(),
    )


def score_mf(uid, iid, user_weight, item_weight, batch_size: int = 8192):
    uid = np.asarray(uid, dtype=np.int64)
    iid = np.asarray(iid, dtype=np.int64)
    if uid.size != iid.size:
        raise ValueError("uid and iid lengths differ")
    if uid.size and int(uid.max()) >= user_weight.shape[0]:
        raise ValueError(f"uid out of range: max={int(uid.max())}, rows={user_weight.shape[0]}")
    if iid.size and int(iid.max()) >= item_weight.shape[0]:
        raise ValueError(f"iid out of range: max={int(iid.max())}, rows={item_weight.shape[0]}")

    uid_t = torch.as_tensor(uid, dtype=torch.long)
    iid_t = torch.as_tensor(iid, dtype=torch.long)
    chunks = []
    with torch.no_grad():
        for start in range(0, uid.size, batch_size):
            stop = min(start + batch_size, uid.size)
            user_embedding = user_weight[uid_t[start:stop]]
            item_embedding = item_weight[iid_t[start:stop]]
            chunks.append(torch.mul(user_embedding, item_embedding).sum(dim=-1))
    if not chunks:
        return np.array([], dtype=np.float32)
    return torch.cat(chunks).cpu().numpy()


def predict_mf(
    data_dir: str,
    checkpoint_path: str,
    out_path: str,
    train_file: str = "train_ood2.pkl",
    test_file: str = "test_ood2.pkl",
    batch_size: int = 8192,
    threshold: int = DEFAULT_THRESHOLD,
    seed: int = 42,
) -> dict:
    test = load_split(data_dir, test_file)
    train = load_split(data_dir, train_file)
    user_weight, item_weight = load_mf_weights(checkpoint_path)

    uid = test["uid"].to_numpy()
    iid = test["iid"].to_numpy()
    label = test["label"].to_numpy()
    logits = score_mf(uid, iid, user_weight, item_weight, batch_size=batch_size)

    row_idx = np.arange(len(test), dtype=np.int64)
    write_predictions(out_path, row_idx, uid, iid, label, logits)
    read_predictions(out_path, test, require_full=True)

    meta_path = str(Path(out_path).with_suffix(".meta.json"))
    meta = {
        "model": "MF",
        "prediction_file": str(out_path),
        "checkpoint": checkpoint_info(checkpoint_path),
        "config": None,
        "threshold": int(threshold),
        "seed": int(seed),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "data": {
            "data_dir": str(data_dir),
            "train_file": str(Path(data_dir) / train_file),
            "train_sha256": sha256_file(str(Path(data_dir) / train_file)),
            "test_file": str(Path(data_dir) / test_file),
            "test_sha256": sha256_file(str(Path(data_dir) / test_file)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
    }
    write_json(meta_path, meta)
    return meta
