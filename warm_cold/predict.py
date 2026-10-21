"""Prediction CLI for the V-B2 warm/cold pipeline.

Current predictors: MF. CoLLM-LoRA and CoLLM-MICA are added separately.
"""

import argparse
import os
import sys
from pathlib import Path

from warm_cold.data import DEFAULT_THRESHOLD, WarmColdError, threshold_warning
from warm_cold.predictions import PredictionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate full-test V-B2 prediction CSVs.")
    parser.add_argument("--model", required=True, choices=["mf"])
    parser.add_argument("--data-dir", default=os.environ.get("COLLM_ML1M_DIR", ""))
    parser.add_argument("--train-file", default="train_ood2.pkl")
    parser.add_argument("--test-file", default="test_ood2.pkl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--mf-ckpt", default=os.environ.get("COLLM_MF_REC_PTH", ""))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    warning = threshold_warning(args.threshold)
    if warning:
        print("WARNING: " + warning, file=sys.stderr)
    if not args.data_dir:
        print("ERROR: pass --data-dir or set COLLM_ML1M_DIR", file=sys.stderr)
        return 2

    try:
        if args.model == "mf":
            if not args.mf_ckpt:
                print("ERROR: pass --mf-ckpt or set COLLM_MF_REC_PTH", file=sys.stderr)
                return 2
            from warm_cold.models.mf import predict_mf

            meta = predict_mf(
                data_dir=args.data_dir,
                checkpoint_path=args.mf_ckpt,
                out_path=args.out,
                train_file=args.train_file,
                test_file=args.test_file,
                batch_size=args.batch_size,
                threshold=args.threshold,
                seed=args.seed,
            )
            print(f"Wrote {args.out} ({meta['data']['n_test']} rows)")
            print(f"Wrote {Path(args.out).with_suffix('.meta.json')}")
            return 0
    except (WarmColdError, PredictionError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"ERROR: unsupported model {args.model!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
