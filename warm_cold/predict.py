"""Prediction CLI for the V-B2 warm/cold pipeline."""

import argparse
import os
import sys
from pathlib import Path

from warm_cold.data import DEFAULT_THRESHOLD, WarmColdError, threshold_warning
from warm_cold.predictions import PredictionError

COLLM_MODELS = ("collm-lora", "collm-mica")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate full-test V-B2 prediction CSVs.")
    parser.add_argument("--model", required=True, choices=("mf",) + COLLM_MODELS)
    parser.add_argument("--data-dir", default=os.environ.get("COLLM_ML1M_DIR", ""))
    parser.add_argument("--train-file", default="train_ood2.pkl")
    parser.add_argument("--test-file", default="test_ood2.pkl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mf-ckpt", default=os.environ.get("COLLM_MF_REC_PTH", ""))
    parser.add_argument("--cfg-path", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_message(meta: dict, out: str) -> None:
    n_rows = meta.get("n_rows") or meta.get("data", {}).get("n_test")
    if n_rows is not None:
        print(f"Wrote {out} ({n_rows} rows)")
    else:
        print(f"Wrote {out}")
    print(f"Wrote {Path(out).with_suffix('.meta.json')}")


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
                batch_size=args.batch_size or 8192,
                threshold=args.threshold,
                seed=args.seed,
            )
            _write_message(meta, args.out)
            return 0

        from warm_cold.models.collm import predict_collm

        meta = predict_collm(
            model_name=args.model,
            data_dir=args.data_dir,
            out_path=args.out,
            cfg_path=args.cfg_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            overwrite=args.overwrite,
            threshold=args.threshold,
            seed=args.seed,
            train_file=args.train_file,
            test_file=args.test_file,
        )
        _write_message(meta, args.out)
        return 0
    except ImportError as exc:
        print(
            "ERROR: CoLLM export requires the full Kaggle environment "
            f"(transformers/peft/omegaconf/minigpt4): {exc}",
            file=sys.stderr,
        )
        return 2
    except (WarmColdError, PredictionError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if args.model in COLLM_MODELS:
            print(f"ERROR: {args.model}: {exc}", file=sys.stderr)
            return 2
        raise

    print(f"ERROR: unsupported model {args.model!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
