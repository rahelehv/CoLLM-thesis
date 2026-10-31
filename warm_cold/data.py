"""Data loading and V-B2 warm/cold mask construction.

The paper's V-B2 rule is:

    warm = count_train(uid) >= THRESHOLD AND count_train(iid) >= THRESHOLD
    cold = exact complement

The default threshold is 3, matching the existing scorer and the paper.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 3
PAPER_THRESHOLD = 3
PAPER_THRESHOLD_SOURCE = (
    "docs/CoLLM_paper.pdf p. 2336, Sec. V-B2: the warm subset comprises "
    "interactions between users and items that have appeared at least three "
    "times in the training dataset; existing default at "
    "eval_warm_cold_vb2.py:240"
)


class WarmColdError(ValueError):
    """Raised when the data or masks violate the V-B2 contract."""


@dataclass(frozen=True)
class WarmColdData:
    train: pd.DataFrame
    test: pd.DataFrame
    warm: np.ndarray
    cold: np.ndarray
    user_counts: np.ndarray
    item_counts: np.ndarray
    threshold: int
    warm_row_idx: np.ndarray
    cold_row_idx: np.ndarray

    @property
    def stats(self) -> dict:
        n_test = int(len(self.test))
        n_warm = int(self.warm.sum())
        n_cold = int(self.cold.sum())
        return {
            "n_train": int(len(self.train)),
            "n_test": n_test,
            "n_warm": n_warm,
            "n_cold": n_cold,
            "warm_pct": 100.0 * n_warm / n_test if n_test else float("nan"),
            "cold_pct": 100.0 * n_cold / n_test if n_test else float("nan"),
            "threshold": int(self.threshold),
        }


def load_split(data_dir: str, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.is_file():
        raise WarmColdError(f"missing pickle: {path} (data-dir={data_dir})")
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise WarmColdError(f"{path} did not unpickle to a DataFrame")
    for col in ("uid", "iid", "label"):
        if col not in df.columns:
            raise WarmColdError(f"{path} lacks required column '{col}'")
    if df[["uid", "iid", "label"]].isna().any().any():
        raise WarmColdError(f"{path} contains null uid/iid/label values")
    labels = pd.unique(df["label"])
    if not set(np.asarray(labels)).issubset({0, 1, 0.0, 1.0}):
        raise WarmColdError(f"{path} labels must be binary 0/1")
    return df.reset_index(drop=True)


def build_masks(train: pd.DataFrame, test: pd.DataFrame, threshold: int = DEFAULT_THRESHOLD):
    if not isinstance(threshold, int) or threshold < 1:
        raise WarmColdError(f"threshold must be an integer >= 1, got {threshold!r}")
    if len(train) == 0:
        raise WarmColdError("training split is empty")
    if len(test) == 0:
        raise WarmColdError("test split is empty")

    for col in ("uid", "iid"):
        if col not in train.columns:
            raise WarmColdError(f"train lacks required column '{col}'")
    for col in ("uid", "iid", "label"):
        if col not in test.columns:
            raise WarmColdError(f"test lacks required column '{col}'")

    user_counts_series = train.groupby("uid", sort=False).size()
    item_counts_series = train.groupby("iid", sort=False).size()

    if not test["uid"].isin(user_counts_series.index).any():
        raise WarmColdError("no test uid matches any train uid (dtype mismatch between splits?)")
    if not test["iid"].isin(item_counts_series.index).any():
        raise WarmColdError("no test iid matches any train iid (dtype mismatch between splits?)")

    user_counts = (
        test["uid"].map(user_counts_series).fillna(0).astype(np.int64).to_numpy()
    )
    item_counts = (
        test["iid"].map(item_counts_series).fillna(0).astype(np.int64).to_numpy()
    )
    warm = (user_counts >= threshold) & (item_counts >= threshold)
    cold = ~warm
    validate_masks(warm, cold, user_counts, item_counts, threshold, len(test))
    return warm, cold, user_counts, item_counts


def validate_masks(warm, cold, user_counts, item_counts, threshold, total):
    warm = np.asarray(warm, dtype=bool)
    cold = np.asarray(cold, dtype=bool)
    if int(warm.sum() + cold.sum()) != int(total):
        raise WarmColdError(
            f"warm_count + cold_count != total "
            f"({int(warm.sum())} + {int(cold.sum())} != {int(total)})"
        )
    if bool((warm & cold).any()):
        raise WarmColdError("warm and cold overlap (warm & cold != empty)")
    if bool(warm.any()):
        if int(user_counts[warm].min()) < threshold:
            raise WarmColdError(f"a warm row violates user count >= {threshold}")
        if int(item_counts[warm].min()) < threshold:
            raise WarmColdError(f"a warm row violates item count >= {threshold}")


def load_warm_cold(
    data_dir: str,
    train_file: str = "train_ood2.pkl",
    test_file: str = "test_ood2.pkl",
    threshold: int = DEFAULT_THRESHOLD,
) -> WarmColdData:
    train = load_split(data_dir, train_file)
    test = load_split(data_dir, test_file)
    warm, cold, user_counts, item_counts = build_masks(train, test, threshold)
    return WarmColdData(
        train=train,
        test=test,
        warm=warm,
        cold=cold,
        user_counts=user_counts,
        item_counts=item_counts,
        threshold=threshold,
        warm_row_idx=np.flatnonzero(warm),
        cold_row_idx=np.flatnonzero(cold),
    )


def threshold_warning(threshold: int) -> Optional[str]:
    if threshold != PAPER_THRESHOLD:
        return (
            f"threshold={threshold} differs from the paper V-B2 default "
            f"{PAPER_THRESHOLD}; default was not changed"
        )
    return None
