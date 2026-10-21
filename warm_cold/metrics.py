"""AUC and UAUC metrics for V-B2.

Primary UAUC (the thesis table):
    mean of per-user AUCs over users with BOTH classes in the evaluated
    subset. Users with one interaction and users with a single class are
    skipped because their AUC is undefined. If no user has both classes,
    UAUC is NaN.

Secondary UAUC:
    the original repo ``uAUC_me`` semantics, vendored below. It averages
    the per-user AUC array with ``np.mean``; a single NaN poisons the
    aggregate. It is reported only as a diagnostic.
"""

import math
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

VENDORED_UAUC_SOURCE = "minigpt4/tasks/rec_base_task.py:25-67 (uAUC_me)"


def uAUC_me(user, predict, label):
    """Vendored numeric logic of the repo's ``uAUC_me``.

    The original prints and timing calls are omitted; the AUC computation,
    grouping, skip rules, and NaN-propagating mean are preserved.
    """
    predict = np.asarray(predict).squeeze()
    label = np.asarray(label).squeeze()
    user = np.asarray(user)

    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    computed_u = []

    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        k += 1

    auc = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for ui, pre_and_true in candidates_dict.items():
            pre_i, label_i = pre_and_true
            try:
                ui_auc = roc_auc_score(label_i, pre_i)
                auc.append(ui_auc)
                computed_u.append(ui)
            except Exception:
                continue

    auc_for_user = np.array(auc)
    uauc = auc_for_user.mean()
    return uauc, computed_u, auc_for_user


def per_user_aucs(user, predict, label):
    """Compute primary per-user AUCs, explicitly skipping undefined users."""
    user = np.asarray(user)
    predict = np.asarray(predict, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)

    unique_users = np.unique(user)
    computed_users = []
    computed_aucs = []
    n_single_interaction = 0
    n_single_class = 0
    n_nonfinite = 0

    for uid in unique_users:
        mask = user == uid
        y = label[mask]
        p = predict[mask]
        if y.size < 2:
            n_single_interaction += 1
            continue
        if np.unique(y).size < 2:
            n_single_class += 1
            continue
        try:
            value = roc_auc_score(y, p)
        except ValueError:
            n_single_class += 1
            continue
        if not np.isfinite(value):
            n_nonfinite += 1
            continue
        computed_users.append(uid)
        computed_aucs.append(float(value))

    return {
        "computed_users": computed_users,
        "aucs": np.asarray(computed_aucs, dtype=np.float64),
        "n_unique_users": int(unique_users.size),
        "n_computed_users": int(len(computed_users)),
        "n_skipped_single_class": int(n_single_class),
        "n_single_interaction": int(n_single_interaction),
        "n_skipped_nonfinite": int(n_nonfinite),
    }


def uauc_summary(user, predict, label):
    primary = per_user_aucs(user, predict, label)
    if primary["aucs"].size:
        uauc = float(np.mean(primary["aucs"]))
    else:
        uauc = float("nan")

    official_uauc, official_computed, official_aucs = uAUC_me(user, predict, label)
    return {
        "uauc": uauc,
        "uauc_official": float(official_uauc),
        "n_unique_users": primary["n_unique_users"],
        "n_computed_users": primary["n_computed_users"],
        "n_skipped_single_class": primary["n_skipped_single_class"],
        "n_single_interaction": primary["n_single_interaction"],
        "n_skipped_nonfinite": primary["n_skipped_nonfinite"],
        "n_official_computed_users": int(len(official_computed)),
        "per_user_auc": primary["aucs"],
        "official_per_user_auc": np.asarray(official_aucs),
    }


def subset_metrics(users, labels, logits):
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    users = np.asarray(users)

    if labels.size == 0:
        return {
            "n": 0,
            "auc": float("nan"),
            "auc_note": "empty subset",
            "uauc": float("nan"),
            "uauc_official": float("nan"),
            "uauc_note": "empty subset",
            "n_unique_users": 0,
            "n_computed_users": 0,
            "n_skipped_single_class": 0,
            "n_single_interaction": 0,
            "n_skipped_nonfinite": 0,
            "n_official_computed_users": 0,
            "per_user_auc": np.array([]),
            "official_per_user_auc": np.array([]),
        }

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            auc = float(roc_auc_score(labels, logits))
        auc_note = ""
    except ValueError as exc:
        auc = float("nan")
        auc_note = f"single-class subset ({exc})"

    summary = uauc_summary(users, logits, labels)
    uauc_note = "" if summary["n_computed_users"] > 0 else "no user has both classes"
    return {
        "n": int(labels.size),
        "auc": auc,
        "auc_note": auc_note,
        "uauc": summary["uauc"],
        "uauc_official": summary["uauc_official"],
        "uauc_note": uauc_note,
        "n_unique_users": summary["n_unique_users"],
        "n_computed_users": summary["n_computed_users"],
        "n_skipped_single_class": summary["n_skipped_single_class"],
        "n_single_interaction": summary["n_single_interaction"],
        "n_skipped_nonfinite": summary["n_skipped_nonfinite"],
        "n_official_computed_users": summary["n_official_computed_users"],
        "per_user_auc": summary["per_user_auc"],
        "official_per_user_auc": summary["official_per_user_auc"],
    }


def fmt(value) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.4f}"


def json_safe(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value
