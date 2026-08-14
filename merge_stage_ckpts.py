"""Merge the stage-1 (LoRA) and stage-2 (CIE/llama_proj) checkpoints.

The stage-1 checkpoint saves only the LoRA adapters (frozen in stage 2, so not
re-saved there). The stage-2 checkpoint saves only the trained llama_proj
mapping layer. Phase 4 (evaluation) needs BOTH, but the model builder loads a
single ``model.ckpt``. This produces one file containing the union, so a plain
eval run (model.ckpt=<merged file>) uses the correct frozen LoRA + CIE weights.

Usage:
    python merge_stage_ckpts.py \\
        --stage1-ckpt /path/to/stage1/checkpoint_best.pth \\
        --stage2-ckpt /path/to/stage2/checkpoint_best.pth \\
        --output /path/to/merged_checkpoint.pth

Stage-2 keys take precedence on any collision.
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True, help="stage-1 LoRA checkpoint")
    ap.add_argument("--stage2-ckpt", required=True, help="stage-2 CIE checkpoint")
    ap.add_argument("--output", required=True, help="merged output .pth path")
    args = ap.parse_args()

    for path in (args.stage1_ckpt, args.stage2_ckpt):
        if not os.path.isfile(path):
            raise FileNotFoundError("checkpoint not found: {}".format(path))

    ck1 = torch.load(args.stage1_ckpt, map_location="cpu")
    ck2 = torch.load(args.stage2_ckpt, map_location="cpu")

    if "model" not in ck1 or "model" not in ck2:
        raise ValueError("both checkpoints must contain a 'model' key")

    merged = dict(ck1["model"])
    merged.update(ck2["model"])

    out = {
        "model": merged,
        "epoch": ck2.get("epoch", ck1.get("epoch", 0)),
        "config": ck2.get("config", ck1.get("config")),
    }
    n1, n2 = len(ck1["model"]), len(ck2["model"])
    overlap = set(ck1["model"]) & set(ck2["model"])
    print("stage-1 keys: {} | stage-2 keys: {} | merged: {} | overlap: {}".format(
        n1, n2, len(merged), len(overlap)))
    print("stage-1 sample keys:", sorted(ck1["model"])[:3])
    print("stage-2 sample keys:", sorted(ck2["model"])[:3])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print("saved merged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()