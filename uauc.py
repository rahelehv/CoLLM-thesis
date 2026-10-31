"""Shared user-level AUC (UAUC) implementation.

Both the standard evaluation task and the V-B2 warm/cold evaluator import
``uAUC_me`` from this module so that per-user results and NaN handling cannot
drift between the two paths.
"""

import time

import numpy as np
from sklearn.metrics import roc_auc_score


def _nanmean_auc(auc_for_user):
    """Average defined per-user AUCs; return NaN when none are defined."""
    values = np.asarray(auc_for_user, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.float64(np.nan)
    return np.nanmean(values)


def uAUC_me(user, predict, label, verbose=True):
    user = np.asarray(user)
    predict = np.asarray(predict).squeeze()
    label = np.asarray(label).squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)  # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        k += 1
    if verbose:
        print("only one interaction users:", only_one_interaction)
    auc = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except Exception:
            only_one_class += 1

    auc_for_user = np.array(auc, dtype=float)
    n_eligible = len(candidates_dict)
    n_nan = int(np.isnan(auc_for_user).sum())
    n_valid = int(auc_for_user.shape[0] - n_nan)
    if verbose:
        print("eligible users:", n_eligible, "computed user:", auc_for_user.shape[0],
              "can not users:", only_one_class, "valid auc:", n_valid, "nan auc:", n_nan)
    uauc = _nanmean_auc(auc_for_user)
    if verbose:
        if n_valid == 0:
            print("uauc for validation Cost:", time.time()-start_time,
                  'uauc: nan (no defined per-user AUC)')
        else:
            print("uauc for validation Cost:", time.time()-start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user
