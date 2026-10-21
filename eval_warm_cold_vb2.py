#!/usr/bin/env python
"""Compatibility wrapper for the V-B2 warm/cold scorer.

The implementation now lives in ``warm_cold.evaluate``. This wrapper preserves
the old CLI flags (``--mf-pred``, ``--tallrec-pred``, ``--collm-pred``) and
delegates to the reusable scorer, which also enforces full prediction coverage.
"""

import argparse
import os
import sys

from warm_cold.evaluate import main as evaluate_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper: V-B2 warm/cold scorer for ML-1M (CPU only)."
    )
    parser.add_argument("--data-dir", default=os.environ.get("COLLM_ML1M_DIR", ""))
    parser.add_argument("--train-file", default="train_ood2.pkl")
    parser.add_argument("--test-file", default="test_ood2.pkl")
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--mf-pred", default=None)
    parser.add_argument("--tallrec-pred", default=None)
    parser.add_argument("--collm-pred", default=None)
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("COLLM_OUTPUT_DIR", "vb2_out"),
    )
    parser.add_argument("--dump-masks", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    forwarded = [
        "--data-dir",
        args.data_dir,
        "--train-file",
        args.train_file,
        "--test-file",
        args.test_file,
        "--threshold",
        str(args.threshold),
        "--out-dir",
        args.out_dir,
    ]
    if args.dump_masks:
        forwarded.append("--dump-masks")
    for label, path in (
        ("MF", args.mf_pred),
        ("TALLRec", args.tallrec_pred),
        ("CoLLM", args.collm_pred),
    ):
        if path:
            forwarded.extend(["--pred", f"{label}={path}"])
    print(
        "NOTE: eval_warm_cold_vb2.py delegates to warm_cold.evaluate; "
        "results.csv uses the new per-subset schema.",
        file=sys.stderr,
    )
    return evaluate_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
