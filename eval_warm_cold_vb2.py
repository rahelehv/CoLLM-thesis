#!/usr/bin/env python
"""V-B2 warm/cold scorer for CoLLM (ML-1M first pass). CPU-only. No training.

Paper rule (CoLLM v3, Sec. V-B2), implemented exactly here and nowhere else:

    warm = (count_train(uid) >= THRESHOLD) AND (count_train(iid) >= THRESHOLD)
    cold = all remaining test interactions (exact complement)

Legacy flags in the pickles (`not_cold`, `warm`, `cold`) encode a DIFFERENT
criterion (unique-membership, >= 1) and are IGNORED by this script.

Data flow (one full-test prediction pass per model, split applied on CPU)::

    train_ood2.pkl -> user/item counts -> warm_mask / cold_mask
    test_ood2.pkl + pred_<model>.csv (row_idx,uid,iid,label,logit)
        -> Warm AUC/UAUC + Cold AUC/UAUC (existing implementations only)

Prediction CSV contract (produced on Kaggle, consumed here):
    columns: row_idx,uid,iid,label,logit
    row_idx: positional index into test_ood2.pkl AFTER reset_index(drop=True),
             covering a subset or all rows, with no duplicates.
    uid,iid,label must match the test row at row_idx (validated, hard error).
    logit: raw "Yes"-logit (CoLLM/TALLRec) or MF matching score (pre-sigmoid).

Metrics (NOT reimplemented):
    AUC  = sklearn.metrics.roc_auc_score (same as everywhere in this repo)
    UAUC = minigpt4.tasks.rec_base_task.uAUC_me (direct import, semantics kept)

Kaggle runbook (ML-1M):
    1. CPU:  python eval_warm_cold_vb2.py --data-dir "$COLLM_ML1M_DIR"
              # mask-only mode: prints split stats + validations, no preds needed
    2. MF predictions (CPU): load MatrixFactorization with the SAME .pth that
       rec_config.pretrained_path points to (${oc.env:COLLM_MF_REC_PTH}),
       score full test_ood2.pkl in row order, write pred_mf.csv per contract.
       Minimal snippet (run from repo root, CPU is fine):
         import os, pandas as pd, numpy as np, torch, omegaconf
         from minigpt4.models.rec_model import MatrixFactorization
         d = os.environ["COLLM_ML1M_DIR"]
         test = pd.read_pickle(os.path.join(d, "test_ood2.pkl")).reset_index(drop=True)
         n_u = int(max(pd.read_pickle(os.path.join(d, "train_ood2.pkl"))["uid"].max(),
                       test["uid"].max())) + 1
         n_i = int(max(pd.read_pickle(os.path.join(d, "train_ood2.pkl"))["iid"].max(),
                       test["iid"].max())) + 1
         cfg = omegaconf.OmegaConf.create({"user_num": n_u, "item_num": n_i,
                                           "embedding_size": 256})
         m = MatrixFactorization(cfg)
         m.load_state_dict(torch.load(os.environ["COLLM_MF_REC_PTH"],
                                      map_location="cpu", weights_only=False))
         m.eval()
         with torch.no_grad():
             logits = m(torch.tensor(test["uid"].to_numpy()).long(),
                        torch.tensor(test["iid"].to_numpy()).long()).numpy()
         pd.DataFrame({"row_idx": np.arange(len(test)), "uid": test["uid"].to_numpy(),
                       "iid": test["iid"].to_numpy(), "label": test["label"].to_numpy(),
                       "logit": np.asarray(logits).reshape(-1)}).to_csv("pred_mf.csv", index=False)
       (embedding_size must match the trained .pth; 256 for the ML-1M recipe.)
    3. TALLRec predictions (GPU): torchrun train_collm_*.py with the Stage-1
       config (prompts/tallrec_movie.txt, model.ckpt=${oc.env:COLLM_STAGE1_CKPT},
       evaluate True), collecting generate_for_samples_v2 "Yes"-logits in
       dataset order -> pred_tallrec.csv per contract. No retraining.
    4. CoLLM predictions (GPU): same with collm_eval_mf_ood.yaml semantics
       (prompts/collm_movie.txt, model.ckpt=${oc.env:COLLM_STAGE2_CKPT} merged
       via merge_stage_ckpts.py) -> pred_collm.csv. No retraining.
       Run LLM inferences SEQUENTIALLY (one model resident at a time).
    5. CPU:  python eval_warm_cold_vb2.py --data-dir "$COLLM_ML1M_DIR" \\
               --mf-pred pred_mf.csv --tallrec-pred pred_tallrec.csv \\
               --collm-pred pred_collm.csv --out-dir "$COLLM_OUTPUT_DIR/vb2_out"
              # prints the paper table, writes results.csv + results.json

Exit status: 0 on success (NaN metrics are reported, not fatal); 1 on any
validation failure (counts, disjointness, complement, alignment).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from minigpt4.tasks.rec_base_task import uAUC_me
except ImportError as e:  # keep semantics: reuse, never vendor a copy
    print(
        "ERROR: could not import uAUC_me from minigpt4.tasks.rec_base_task.\n"
        "Run this script from the CoLLM repo root with the repo importable "
        "(e.g. PYTHONPATH=<repo> python eval_warm_cold_vb2.py ...).",
        file=sys.stderr,
    )
    raise

MODELS = (("MF", "mf"), ("TALLRec", "tallrec"), ("CoLLM", "collm"))

REQUIRED_PRED_COLS = ("row_idx", "uid", "iid", "label", "logit")


def fail(msg):
    print("VALIDATION FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print("WARNING: " + msg, file=sys.stderr)


def load_split(data_dir, fname):
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        fail("missing pickle: %s (data-dir=%s)" % (path, data_dir))
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        fail("%s did not unpickle to a DataFrame" % path)
    for col in ("uid", "iid", "label"):
        if col not in df.columns:
            fail("%s lacks required column '%s'" % (path, col))
    return df.reset_index(drop=True)


def build_masks(train, test, threshold):
    """Paper V-B2 masks. Legacy not_cold/warm/cold columns are ignored."""
    user_counts = train.groupby("uid").size()
    item_counts = train.groupby("iid").size()
    if len(user_counts) == 0 or len(item_counts) == 0:
        fail("empty training counts (train has no rows?)")
    # ID-space sanity: silent dtype mismatch (int vs str) would map to NaN->0.
    if not test["uid"].isin(user_counts.index).any() and (test["uid"].size > 0):
        fail("no test uid matches any train uid (dtype mismatch between splits?)")
    if not test["iid"].isin(item_counts.index).any() and (test["iid"].size > 0):
        fail("no test iid matches any train iid (dtype mismatch between splits?)")
    u = test["uid"].map(user_counts).fillna(0).astype(np.int64).to_numpy()
    v = test["iid"].map(item_counts).fillna(0).astype(np.int64).to_numpy()
    warm = (u >= threshold) & (v >= threshold)
    cold = ~warm
    return warm, cold, u, v


def validate_masks(warm, cold, u_counts, v_counts, threshold, total):
    if int(warm.sum() + cold.sum()) != int(total):
        fail("warm_count + cold_count != total (%d + %d != %d)"
             % (int(warm.sum()), int(cold.sum()), int(total)))
    if bool((warm & cold).any()):
        fail("warm and cold overlap (warm & cold != empty)")
    if warm.any():
        if int(u_counts[warm].min()) < threshold or int(v_counts[warm].min()) < threshold:
            fail("a warm row violates count >= %d" % threshold)
    # cold is defined as the exact complement; nothing further to check.


def load_predictions(path, test, tag):
    """Load one pred CSV, validate it against the test frame. Returns aligned arrays."""
    if not os.path.isfile(path):
        fail("prediction file not found for %s: %s" % (tag, path))
    pred = pd.read_csv(path)
    missing = [c for c in REQUIRED_PRED_COLS if c not in pred.columns]
    if missing:
        fail("%s (%s) lacks columns: %s (need %s)"
             % (path, tag, missing, list(REQUIRED_PRED_COLS)))
    if pred["row_idx"].duplicated().any():
        fail("%s (%s) has duplicated row_idx" % (path, tag))
    idx = pred["row_idx"].to_numpy()
    try:
        idx = idx.astype(np.int64)
    except (TypeError, ValueError):
        fail("%s (%s): row_idx must be integer positions" % (path, tag))
    n = len(test)
    if ((idx < 0) | (idx >= n)).any():
        fail("%s (%s): row_idx out of range [0, %d)" % (path, tag, n))
    tu = test["uid"].to_numpy()
    ti = test["iid"].to_numpy()
    tl = test["label"].to_numpy()
    # align by row_idx explicitly (pred file may be unordered, must be a subset w/o dups)
    order = np.argsort(idx)
    idx_s = idx[order]
    for col in ("uid", "iid", "label"):
        pv = pred[col].to_numpy()[order]
        tv = {"uid": tu, "iid": ti, "label": tl}[col][idx_s]
        if not np.array_equal(np.asarray(pv), np.asarray(tv)):
            bad = int(np.argmax(np.asarray(pv) != np.asarray(tv)))
            fail("%s (%s): column '%s' mismatch at row_idx=%d "
                 "(pred=%s vs test=%s); pred rows must match test_ood2.pkl order/IDs)"
                 % (path, tag, col, int(idx_s[bad]), pv[bad], tv[bad]))
    labels = tl[idx_s].astype(np.int64)
    logits = pred["logit"].to_numpy()[order].astype(np.float64)
    users = tu[idx_s]
    if not set(np.unique(labels)).issubset({0, 1}):
        fail("%s (%s): labels must be binary 0/1" % (path, tag))
    if not np.all(np.isfinite(logits)):
        fail("%s (%s): logits contain NaN/Inf" % (path, tag))
    return idx_s, users, labels, logits


def subset_metrics(users, labels, logits):
    """AUC + UAUC on one subset. Returns dict with values or NaN + reasons."""
    out = {}
    try:
        out["auc"] = float(roc_auc_score(labels, logits))
        out["auc_note"] = ""
    except ValueError as e:
        out["auc"] = float("nan")
        out["auc_note"] = "single-class subset (%s)" % e
    try:
        uauc, computed_u, _ = uAUC_me(
            np.asarray(users), np.asarray(logits), np.asarray(labels))
        out["uauc"] = float(uauc)  # nan if no valid user; reported, not imputed
        out["n_valid_users"] = int(len(computed_u))
        out["uauc_note"] = "" if out["n_valid_users"] > 0 else "no valid users for UAUC"
    except Exception as e:  # never invent a number
        out["uauc"] = float("nan")
        out["n_valid_users"] = 0
        out["uauc_note"] = "uAUC_me raised: %s" % e
    out["n"] = int(len(labels))
    return out


def fmt(v):
    return "NaN" if (v is None or (isinstance(v, float) and math.isnan(v))) else "%.4f" % v


def _json_safe(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="V-B2 (CoLLM v3) warm/cold scorer, ML-1M. CPU-only, no training.")
    ap.add_argument("--data-dir", default=os.environ.get("COLLM_ML1M_DIR", ""),
                    help="dir with train_ood2.pkl + test_ood2.pkl "
                         "(default: $COLLM_ML1M_DIR)")
    ap.add_argument("--train-file", default="train_ood2.pkl")
    ap.add_argument("--test-file", default="test_ood2.pkl")
    ap.add_argument("--threshold", type=int, default=3,
                    help="paper V-B2 count threshold (default 3, do not change)")
    ap.add_argument("--mf-pred", default=None, help="pred CSV for MF (contract columns)")
    ap.add_argument("--tallrec-pred", default=None, help="pred CSV for TALLRec")
    ap.add_argument("--collm-pred", default=None, help="pred CSV for CoLLM")
    ap.add_argument("--out-dir", default=os.environ.get("COLLM_OUTPUT_DIR", "vb2_out"),
                    help="where results.csv/results.json go (default: $COLLM_OUTPUT_DIR or ./vb2_out)")
    ap.add_argument("--dump-masks", action="store_true",
                    help="also save warm_row_idx.npy / cold_row_idx.npy")
    args = ap.parse_args(argv)

    if not args.data_dir:
        fail("no data dir: pass --data-dir or set COLLM_ML1M_DIR")
    if args.threshold != 3:
        warn("threshold=%d differs from the paper V-B2 (>= 3)" % args.threshold)

    train = load_split(args.data_dir, args.train_file)
    test = load_split(args.data_dir, args.test_file)
    total = len(test)
    if total == 0:
        fail("test split is empty")
    if len(train) == 0:
        fail("train split is empty")

    warm, cold, u_counts, v_counts = build_masks(train, test, args.threshold)
    validate_masks(warm, cold, u_counts, v_counts, args.threshold, total)

    n_warm, n_cold = int(warm.sum()), int(cold.sum())
    print("Total test interactions: %d" % total)
    print("Warm interactions: %d (%.2f%%)" % (n_warm, 100.0 * n_warm / total))
    print("Cold interactions: %d (%.2f%%)" % (n_cold, 100.0 * n_cold / total))
    print("Threshold: train count >= %d for BOTH user and item" % args.threshold)

    pred_paths = {"MF": args.mf_pred, "TALLRec": args.tallrec_pred, "CoLLM": args.collm_pred}
    rows, details = [], {}
    for name, _ in MODELS:
        path = pred_paths[name]
        if not path:
            print("%-8s no prediction file given (--%s-pred); skipped"
                  % (name, name.lower()))
            rows.append((name, None, None, None, None))
            details[name] = {"pred_file": None, "skipped": True}
            continue
        idx_s, users, labels, logits = load_predictions(path, test, name)
        # map full-test boolean masks onto the (possibly subset) pred rows
        w = warm[idx_s]
        c = cold[idx_s]
        if int(w.sum() + c.sum()) != len(idx_s):
            fail("%s: mask coverage != pred rows" % name)
        wm = subset_metrics(users[w], labels[w], logits[w])
        cm = subset_metrics(users[c], labels[c], logits[c])
        for m, note_key in (("auc", "auc_note"), ("uauc", "uauc_note")):
            for res, t in ((wm, "warm"), (cm, "cold")):
                if res[note_key]:
                    warn("%s %s %s: %s" % (name, t, m.upper(), res[note_key]))
        print("%-8s Warm AUC=%s Warm UAUC=%s (n=%d, users=%d) | Cold AUC=%s Cold UAUC=%s (n=%d, users=%d)"
              % (name, fmt(wm["auc"]), fmt(wm["uauc"]), wm["n"], wm["n_valid_users"],
                 fmt(cm["auc"]), fmt(cm["uauc"]), cm["n"], cm["n_valid_users"]))
        rows.append((name, wm["auc"], wm["uauc"], cm["auc"], cm["uauc"]))
        details[name] = {"pred_file": path, "skipped": False,
                         "n_pred_rows": int(len(idx_s)),
                         "n_warm": int(wm["n"]), "n_cold": int(cm["n"]),
                         "n_valid_users_warm": int(wm["n_valid_users"]),
                         "n_valid_users_cold": int(cm["n_valid_users"]),
                         "warm_auc": wm["auc"], "warm_uauc": wm["uauc"],
                         "cold_auc": cm["auc"], "cold_uauc": cm["uauc"]}

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    res_csv = os.path.join(out_dir, "results.csv")
    pd.DataFrame(rows, columns=["Model", "Warm AUC", "Warm UAUC", "Cold AUC", "Cold UAUC"])\
        .to_csv(res_csv, index=False)
    payload = {
        "experiment": "CoLLM v3 Sec. V-B2 warm/cold (ML-1M)",
        "criterion": "warm = (train user count >= T) AND (train item count >= T); "
                     "cold = complement",
        "threshold": int(args.threshold),
        "data_dir": args.data_dir,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "n_train": int(len(train)),
        "n_test": total,
        "n_warm": n_warm,
        "n_cold": n_cold,
        "warm_pct": 100.0 * n_warm / total,
        "cold_pct": 100.0 * n_cold / total,
        "notes": "legacy not_cold/warm/cold columns ignored; "
                 "UAUC via minigpt4.tasks.rec_base_task.uAUC_me (unmodified); "
                 "NDCG not computed (not required by V-B2)",
        "models": details,
    }
    res_json = os.path.join(out_dir, "results.json")
    with open(res_json, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    if args.dump_masks:
        np.save(os.path.join(out_dir, "warm_row_idx.npy"), np.flatnonzero(warm))
        np.save(os.path.join(out_dir, "cold_row_idx.npy"), np.flatnonzero(cold))
    print("Saved: %s, %s" % (res_csv, res_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
