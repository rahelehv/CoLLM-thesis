"""
Focused local tests for UAUC aggregation in the shared `uauc.py` implementation
used by minigpt4/tasks/rec_base_task.py (no GPU, no LAVIS runtime deps needed).

Covers:
  1. mixture of valid per-user AUC values and NaN
  2. all per-user AUC values NaN
  3. single-interaction users excluded from the user population
  4. multi-interaction users with one label class -> NaN retained (not fabricated)
  5. users with both label classes -> finite per-user AUC
  6. compute_dcg unchanged (independent reference formula); global-AUC path untouched

The module is loaded with stubbed `minigpt4` package init and `timm` because those
import paths pull in LAVIS deps (decord/webdataset/iopath) irrelevant to these
pure-numpy functions. The source file itself is executed unmodified.

Run:  python -u unit_test_uauc.py
"""

import importlib.machinery
import importlib.util
import os
import sys
import types
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message="Only one class is present in y_true")

ROOT = os.path.dirname(os.path.abspath(__file__))
PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS  %s" % name)
    else:
        FAIL += 1
        print("FAIL  %s  %s" % (name, detail))


def _stub_pkg(name, path=None):
    m = types.ModuleType(name)
    m.__path__ = [path] if path else []
    m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    m.__spec__.submodule_search_locations = list(m.__path__)
    return m


def load_rec_base_task():
    timm, timm_models, timm_hub = _stub_pkg("timm"), _stub_pkg("timm.models"), _stub_pkg("timm.models.hub")
    timm.models, timm_models.hub = timm_models, timm_hub
    sys.modules.update({"timm": timm, "timm.models": timm_models, "timm.models.hub": timm_hub})

    for name in ("minigpt4", "minigpt4.common", "minigpt4.tasks", "minigpt4.datasets"):
        sys.modules[name] = _stub_pkg(name, os.path.join(ROOT, *name.split(".")))

    # rec_base_task imports prepare_sample; not used by the functions under test.
    data_utils = _stub_pkg("minigpt4.datasets.data_utils")
    data_utils.prepare_sample = lambda *a, **k: None
    sys.modules["minigpt4.datasets.data_utils"] = data_utils

    path = os.path.join(ROOT, "minigpt4", "tasks", "rec_base_task.py")
    spec = importlib.util.spec_from_file_location("minigpt4.tasks.rec_base_task", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def single_class_returns_nan():
    """Current sklearn (>=1.6) warns and returns nan; older versions raise ValueError."""
    try:
        return bool(np.isnan(roc_auc_score([1, 1], [0.2, 0.3])))
    except ValueError:
        return False


def ref_dcg(user, predict, label):
    """Independent restatement of compute_dcg (no shared code with the repo)."""
    out = []
    for u in np.unique(user):
        m = user == u
        p, y = predict[m], label[m]
        if m.sum() == 1 or np.unique(y).size < 2:
            continue
        order = np.argsort(-p)
        ranked = y[order]
        dcg = sum(ranked[i] / np.log2(i + 2.0) for i in range(ranked.size))
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(int(y.sum())))
        out.append(dcg / idcg)
    return float(np.mean(out))


def main():
    mod = load_rec_base_task()
    uAUC_me, compute_dcg = mod.uAUC_me, mod.compute_dcg
    nan_retained = single_class_returns_nan()
    print("sklearn single-class behavior: %s" % ("nan (1.6+)" if nan_retained else "raises (<=1.5)"))

    # 1/3/4/5: mixture of valid AUCs, a single-interaction user, and single-class users
    user = np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 4])
    predict = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.8, 0.5, 0.3, 0.7])
    label = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0])
    uauc, ids, per_user = uAUC_me(user, predict, label)
    check("mixture: 4 eligible multi-interaction users", len(ids) == (4 if nan_retained else 2),
          "ids=%s" % ids)
    check("mixture: single-interaction user 3 excluded", 3 not in list(ids))
    check("both-class users 0 and 2 are finite and exact",
          per_user[np.where(np.array(ids) == 0)[0][0]] == 1.0 and
          per_user[np.where(np.array(ids) == 2)[0][0]] == 0.0,
          "per_user=%s" % per_user)
    if nan_retained:
        check("single-class users 1 and 4 kept as NaN, not fabricated",
              np.isnan(per_user[np.where(np.array(ids) == 1)[0][0]]) and
              np.isnan(per_user[np.where(np.array(ids) == 4)[0][0]]),
              "per_user=%s" % per_user)
        check("mixture: uauc is nanmean over valid values only", np.isclose(uauc, 0.5),
              "uauc=%r" % uauc)
        check("mixture: old np.mean would have been NaN", np.isnan(per_user.mean()))
    else:
        check("mixture: uauc over the two both-class users", np.isclose(uauc, 0.5),
              "uauc=%r" % uauc)

    # 2: every eligible user single-class -> all NaN
    user2 = np.array([0, 0, 1, 1, 1, 2])
    pred2 = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1])
    lab2 = np.array([1, 1, 0, 0, 0, 1])
    uauc2, ids2, per2 = uAUC_me(user2, pred2, lab2)
    check("all-NaN: result is NaN, not a numeric score", np.isnan(uauc2), "uauc=%r" % uauc2)
    check("all-NaN: every eligible user retained", len(ids2) == (2 if nan_retained else 0),
          "ids=%s" % ids2)
    check("all-NaN: no non-NaN per-user value", not np.any(~np.isnan(per2)) if per2.size else True)

    # 3: only single-interaction users -> empty population, NaN (no crash)
    uauc3, ids3, per3 = uAUC_me(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]),
                                np.array([0, 1, 0]))
    check("all-single-interaction: result is NaN", np.isnan(uauc3), "uauc=%r" % uauc3)
    check("all-single-interaction: empty user list/array", len(ids3) == 0 and per3.size == 0)

    # 6: NDCG unchanged (independent formula), and uAUC_me does not mutate inputs
    u = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.9, 0.5, 0.1, 0.2, 0.8, 0.4])
    y = np.array([1, 0, 1, 0, 1, 0])
    ndcg, ndcg_ids, _ = compute_dcg(u, p, y)
    check("NDCG matches independent reference", np.isclose(ndcg, ref_dcg(u, p, y)),
          "got=%r ref=%r" % (ndcg, ref_dcg(u, p, y)))
    check("NDCG keeps both eligible users", len(ndcg_ids) == 2)
    p_copy, y_copy = p.copy(), y.copy()
    uAUC_me(u, p, y)
    check("uAUC_me leaves predictions/labels unmodified",
          np.array_equal(p, p_copy) and np.array_equal(y, y_copy))
    # global AUC is a direct roc_auc_score call in evaluation(); unchanged by this patch.
    check("global AUC reference unchanged", np.isclose(roc_auc_score(y, p), 2.0 / 3.0),
          "got=%r" % roc_auc_score(y, p))

    print("\n%d passed, %d failed" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
