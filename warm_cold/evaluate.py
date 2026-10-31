"""Score full-test V-B2 prediction CSVs and write the paper table."""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from warm_cold.data import (
    DEFAULT_THRESHOLD,
    PAPER_THRESHOLD,
    PAPER_THRESHOLD_SOURCE,
    WarmColdError,
    load_warm_cold,
    threshold_warning,
)
from warm_cold.metrics import fmt, json_safe, subset_metrics
from warm_cold.predictions import PredictionError, read_predictions
from warm_cold.provenance import copy_freeze_file, write_json

ACCEPTANCE = {
    "mf": {
        "overall": 0.649366087184,
        "warm": 0.727298048078,
        "cold": 0.543353024549,
    },
    "collm-lora": {"overall": 0.7434193394},
    "collm-mica": {"overall": 0.7419987742},
}

RESULT_COLUMNS = (
    "Model",
    "Warm AUC",
    "Warm UAUC",
    "Warm UAUC (official)",
    "Warm Unique Users",
    "Warm Computed Users",
    "Warm Skipped (single-class)",
    "Warm Single-Interaction Users",
    "Cold AUC",
    "Cold UAUC",
    "Cold UAUC (official)",
    "Cold Unique Users",
    "Cold Computed Users",
    "Cold Skipped (single-class)",
    "Cold Single-Interaction Users",
    "Overall AUC",
)


def canonical_label(label: str) -> str:
    return label.strip().lower().replace("_", "-").replace(" ", "-")


def parse_predictions(items):
    parsed = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--pred must be LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not label.strip() or not path.strip():
            raise ValueError(f"--pred must be LABEL=PATH, got {item!r}")
        parsed.append((label.strip(), path.strip()))
    return parsed


def _check(name, expected, actual, tolerance):
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return {
            "metric": name,
            "expected": expected,
            "actual": actual,
            "passed": False,
            "reason": "actual is NaN",
        }
    delta = abs(float(actual) - float(expected))
    return {
        "metric": name,
        "expected": expected,
        "actual": float(actual),
        "delta": delta,
        "passed": delta <= tolerance,
        "reason": "" if delta <= tolerance else f"delta {delta:.12g} > {tolerance}",
    }


def acceptance_checks(label, overall_auc, warm_auc, cold_auc, tolerance):
    key = canonical_label(label)
    expected = ACCEPTANCE.get(key)
    if expected is None:
        return None
    checks = [_check("overall_auc", expected["overall"], overall_auc, tolerance)]
    if "warm" in expected:
        checks.append(_check("warm_auc", expected["warm"], warm_auc, tolerance))
    if "cold" in expected:
        checks.append(_check("cold_auc", expected["cold"], cold_auc, tolerance))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score full-test V-B2 prediction CSVs (CPU only, no training)."
    )
    parser.add_argument("--data-dir", default=os.environ.get("COLLM_ML1M_DIR", ""))
    parser.add_argument("--train-file", default="train_ood2.pkl")
    parser.add_argument("--test-file", default="test_ood2.pkl")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="prediction CSV; repeat for each model",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.environ.get("COLLM_OUTPUT_DIR", ""), "vb2_out"),
    )
    parser.add_argument("--dump-masks", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--no-acceptance", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_dir:
        print("ERROR: pass --data-dir or set COLLM_ML1M_DIR", file=sys.stderr)
        return 2
    warning = threshold_warning(args.threshold)
    if warning:
        print("WARNING: " + warning, file=sys.stderr)

    try:
        data = load_warm_cold(
            args.data_dir,
            train_file=args.train_file,
            test_file=args.test_file,
            threshold=args.threshold,
        )
        predictions = parse_predictions(args.pred)
    except (WarmColdError, ValueError, PredictionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stats = data.stats
    print(f"Total test interactions: {stats['n_test']}")
    print(f"Warm interactions: {stats['n_warm']} ({stats['warm_pct']:.2f}%)")
    print(f"Cold interactions: {stats['n_cold']} ({stats['cold_pct']:.2f}%)")
    print(f"Threshold: train count >= {data.threshold} for BOTH user and item")
    if data.threshold != PAPER_THRESHOLD:
        print("WARNING: threshold differs from paper V-B2 default; default unchanged")

    rows = []
    details = {}
    acceptance = {"tolerance": args.tolerance, "passed": True, "models": {}}

    for label, path in predictions:
        try:
            pred = read_predictions(path, data.test, require_full=True)
        except PredictionError as exc:
            print(f"ERROR: {label}: {exc}", file=sys.stderr)
            return 2

        idx = pred.row_idx
        warm = data.warm[idx]
        cold = data.cold[idx]
        if int(warm.sum() + cold.sum()) != len(idx):
            print(f"ERROR: {label}: mask coverage != prediction rows", file=sys.stderr)
            return 2

        overall_auc = float(roc_auc_score(pred.labels, pred.logits))
        wm = subset_metrics(pred.users[warm], pred.labels[warm], pred.logits[warm])
        cm = subset_metrics(pred.users[cold], pred.labels[cold], pred.logits[cold])

        for subset_name, metrics in (("warm", wm), ("cold", cm)):
            if metrics["uauc_note"]:
                print(f"WARNING: {label} {subset_name} UAUC: {metrics['uauc_note']}", file=sys.stderr)

        row = {
            "Model": label,
            "Warm AUC": wm["auc"],
            "Warm UAUC": wm["uauc"],
            "Warm UAUC (official)": wm["uauc_official"],
            "Warm Unique Users": wm["n_unique_users"],
            "Warm Computed Users": wm["n_computed_users"],
            "Warm Skipped (single-class)": wm["n_skipped_single_class"],
            "Warm Single-Interaction Users": wm["n_single_interaction"],
            "Cold AUC": cm["auc"],
            "Cold UAUC": cm["uauc"],
            "Cold UAUC (official)": cm["uauc_official"],
            "Cold Unique Users": cm["n_unique_users"],
            "Cold Computed Users": cm["n_computed_users"],
            "Cold Skipped (single-class)": cm["n_skipped_single_class"],
            "Cold Single-Interaction Users": cm["n_single_interaction"],
            "Overall AUC": overall_auc,
        }
        rows.append(row)
        details[label] = {
            "pred_file": path,
            "overall_auc": overall_auc,
            "warm": wm,
            "cold": cm,
        }

        checks = None
        if not args.no_acceptance:
            checks = acceptance_checks(label, overall_auc, wm["auc"], cm["auc"], args.tolerance)
            if checks is None:
                print(f"SKIP acceptance gate for unknown label {label!r}")
            else:
                passed = all(check["passed"] for check in checks)
                acceptance["passed"] = acceptance["passed"] and passed
                acceptance["models"][label] = checks
                if passed:
                    print(f"PASS acceptance gate: {label}")
                else:
                    print(f"FAIL acceptance gate: {label}")
                    for check in checks:
                        if not check["passed"]:
                            print(
                                f"  FAIL {check['metric']}: expected {check['expected']} "
                                f"actual {check['actual']} ({check['reason']})"
                            )
                    print(
                        "  Check row order, checkpoint, prompt/config, and raw Yes-logit extraction."
                    )

        print(
            f"{label}: Warm AUC={fmt(wm['auc'])} Warm UAUC={fmt(wm['uauc'])} "
            f"Cold AUC={fmt(cm['auc'])} Cold UAUC={fmt(cm['uauc'])} "
            f"Overall AUC={fmt(overall_auc)}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(results_csv, index=False)

    payload = {
        "experiment": "CoLLM v3 Sec. V-B2 warm/cold (ML-1M)",
        "criterion": "warm = (train user count >= T) AND (train item count >= T); cold = complement",
        "threshold": int(data.threshold),
        "threshold_default": int(DEFAULT_THRESHOLD),
        "threshold_source": PAPER_THRESHOLD_SOURCE,
        "data_dir": args.data_dir,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "stats": stats,
        "models": details,
        "acceptance": acceptance,
    }
    write_json(str(out_dir / "results.json"), json_safe(payload))

    if args.dump_masks:
        np.save(out_dir / "warm_row_idx.npy", data.warm_row_idx)
        np.save(out_dir / "cold_row_idx.npy", data.cold_row_idx)

    freeze_copy = copy_freeze_file(str(out_dir))
    if freeze_copy:
        print(f"Copied {freeze_copy}")

    print(f"Saved: {results_csv}, {out_dir / 'results.json'}")
    if not args.no_acceptance and not acceptance["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
