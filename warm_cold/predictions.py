"""Prediction CSV contract and validation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("row_idx", "uid", "iid", "label", "logit")


class PredictionError(ValueError):
    """Raised when a prediction CSV violates the contract."""


@dataclass(frozen=True)
class PredictionTable:
    row_idx: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    logits: np.ndarray


def write_predictions(path: str, row_idx, uid, iid, label, logit) -> None:
    columns = {
        "row_idx": np.asarray(row_idx, dtype=np.int64),
        "uid": np.asarray(uid),
        "iid": np.asarray(iid),
        "label": np.asarray(label),
        "logit": np.asarray(logit, dtype=np.float64),
    }
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise PredictionError(f"prediction columns have different lengths: {lengths}")
    if not np.all(np.isfinite(columns["logit"])):
        raise PredictionError("logits contain NaN/Inf")
    if not set(np.unique(columns["label"])).issubset({0, 1, 0.0, 1.0}):
        raise PredictionError("labels must be binary 0/1")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    pd.DataFrame(columns).to_csv(tmp, index=False)
    tmp.replace(out)


def read_predictions(path: str, test: pd.DataFrame, require_full: bool = True) -> PredictionTable:
    pred_path = Path(path)
    if not pred_path.is_file():
        raise PredictionError(f"prediction file not found: {pred_path}")
    pred = pd.read_csv(pred_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in pred.columns]
    if missing:
        raise PredictionError(f"{pred_path} lacks columns: {missing} (need {list(REQUIRED_COLUMNS)})")
    if len(pred) == 0:
        raise PredictionError(f"{pred_path} is empty")

    raw_idx = pd.to_numeric(pred["row_idx"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(raw_idx)):
        raise PredictionError(f"{pred_path}: row_idx must be finite integers")
    idx = raw_idx.astype(np.int64)
    if not np.all(raw_idx == idx):
        raise PredictionError(f"{pred_path}: row_idx must be integer positions")
    if np.unique(idx).size != idx.size:
        raise PredictionError(f"{pred_path}: duplicated row_idx")

    n = len(test)
    if ((idx < 0) | (idx >= n)).any():
        raise PredictionError(f"{pred_path}: row_idx out of range [0, {n})")

    order = np.argsort(idx)
    idx_sorted = idx[order]
    if require_full:
        if idx.size != n or not np.array_equal(idx_sorted, np.arange(n)):
            raise PredictionError(
                f"{pred_path}: full test coverage required; got {idx.size}/{n} rows"
            )

    test_uid = test["uid"].to_numpy()
    test_iid = test["iid"].to_numpy()
    test_label = test["label"].to_numpy()

    for name, test_values in (("uid", test_uid), ("iid", test_iid), ("label", test_label)):
        values = pred[name].to_numpy()[order]
        expected = test_values[idx_sorted]
        if not np.array_equal(values, expected):
            mismatch = np.flatnonzero(values != expected)
            first = int(mismatch[0]) if mismatch.size else 0
            raise PredictionError(
                f"{pred_path}: column '{name}' mismatch at row_idx={int(idx_sorted[first])} "
                f"(pred={values[first]} vs test={expected[first]})"
            )

    labels = test_label[idx_sorted]
    if not set(np.unique(labels)).issubset({0, 1}):
        raise PredictionError(f"{pred_path}: labels must be binary 0/1")

    raw_logits = pd.to_numeric(pred["logit"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(raw_logits)):
        raise PredictionError(f"{pred_path}: logits contain NaN/Inf")

    return PredictionTable(
        row_idx=idx_sorted,
        users=test_uid[idx_sorted],
        labels=labels.astype(np.int64),
        logits=raw_logits[order].astype(np.float64),
    )
