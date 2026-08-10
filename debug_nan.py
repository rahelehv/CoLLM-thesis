import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

import minigpt4.models as models  # noqa: F401  (registers model classes), mirrors train script

from minigpt4.common.config import Config
from minigpt4.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser(description="NaN-source isolation for stage-1 training")
    parser.add_argument(
        "--cfg-path",
        default="train_configs/minigpt4rec_pretrain_ood_cc.yaml",
        help="path to configuration file.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config (key=value).",
    )
    return parser.parse_args()


def build_samples(device, b, ann):
    # Mirror MoiveOOData.__getitem__ (no-history branch, minigpt4/datasets/datasets/rec_datasets.py:304)
    # using real train_ood2 rows, so the batch has exactly the fields recprompt_wrap_v2 touches
    # (notably TargetItemTitle, which it indexes for every sample at minigpt4rec_v2.py:518).
    rows = ann.iloc[:b]
    return {
        "UserID": torch.tensor(rows["uid"].values, dtype=torch.long, device=device),
        "TargetItemID": torch.tensor(rows["iid"].values, dtype=torch.long, device=device),
        "TargetItemTitle": [str(t).strip(" ") for t in rows["title"].values],
        "label": torch.tensor(rows["label"].values.astype(np.int64), dtype=torch.long, device=device),
    }


def scan_params(tag, named_params):
    total_nan = total_inf = 0
    max_abs = 0.0
    worst = ""
    for name, p in named_params:
        if p is None or not isinstance(p, torch.Tensor):
            continue
        d = p.detach().float()
        n = int(torch.isnan(d).sum())
        i = int(torch.isinf(d).sum())
        total_nan += n
        total_inf += i
        ma = float(d.abs().max())
        if ma > max_abs:
            max_abs, worst = ma, name
    print("scan {:<18} max|w|={:.4e} nan={} inf={} worst={}".format(tag, max_abs, total_nan, total_inf, worst))


def run_case(model, device, ann, name, grad_check, use_lora, train_mode, b):
    torch.cuda.empty_cache()
    model.llama_model.model.gradient_checkpointing = bool(grad_check)
    model.use_lora = use_lora
    model.set_mode("v2")
    model.train() if train_mode else model.eval()

    nan_layers = []
    hooks = []

    def make_output_hook(i):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor) and h.numel() and bool(torch.isnan(h).any()):
                nan_layers.append(i)

        return hook

    for i, layer in enumerate(model.llama_model.model.layers):
        hooks.append(layer.register_forward_hook(make_output_hook(i)))

    samples = build_samples(device, b, ann)
    loss_val = float("nan")
    logits_max = float("nan")
    try:
        grad_ctx = (
            torch.enable_grad()
            if train_mode
            else torch.no_grad()
        )
        with grad_ctx, torch.cuda.amp.autocast(enabled=True):
            outputs = model(samples)
            loss_t = outputs["loss"]
            loss_val = float(loss_t.item())
    finally:
        for hk in hooks:
            hk.remove()

    bad = math.isnan(loss_val) or math.isinf(loss_val)
    print(
        "case={:<28} grad_ckpt={:<5} lora={:<5} train={:<5} b={}  loss={!s:<10} nan={} first_nan_layer={}".format(
            name,
            str(grad_check).upper(),
            str(use_lora).upper(),
            str(train_mode).upper(),
            b,
            loss_val,
            bad,
            nan_layers[0] if nan_layers else "-",
        )
    )
    del samples
    torch.cuda.empty_cache()
    return bad, (nan_layers[0] if nan_layers else None)


def main():
    cfg = Config(parse_args())

    data_dir = cfg.datasets_cfg.movie_ood.path
    train_ = pd.read_pickle(data_dir + "train_ood2.pkl")
    valid_ = pd.read_pickle(data_dir + "valid_ood2.pkl")
    test_ = pd.read_pickle(data_dir + "test_ood2.pkl")
    user_num = max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1
    item_num = max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1
    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    print("user_num={} item_num={}".format(user_num, item_num))

    device = torch.device("cuda")
    model = registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)
    model.to(device)
    # base_model.device is a read-only property derived from the first parameter,
    # so after .to(device) it already reads as 'cuda' and Rec2Base.maybe_autocast
    # (which compares it against cpu) enables fp16 autocast as during real training.

    # guarantee the input-requires-grad path exists for every grad-checkpoint case
    def _make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.llama_model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    print("== weight sanity scan (fp16 checkpoint as loaded) ==")
    for tag, params in [
        ("llama(base)", model.llama_model.named_parameters()),
        ("lora", model.llama_model_lora.named_parameters()),
        ("llama_proj", model.llama_proj.named_parameters()),
        ("rec_encoder", model.rec_encoder.named_parameters()),
    ]:
        scan_params(tag, params)

    print("== forward matrix (loss via forward_v2, autocast fp16 like training) ==")
    cases = [
        ("A train ckpt OFF", False, True, True, 1),
        ("B train ckpt ON (current)", True, True, True, 2),
        ("C train ckpt ON no-lora", True, False, True, 1),
        ("D eval ckpt OFF", False, True, False, 2),
    ]
    results = [run_case(model, device, train_, *case) for case in cases]

    print("== verdict ==")
    for case, res in zip(cases, results):
        print("  {} -> {}".format(case[0], "NaN" if res[0] else "finite"))


if __name__ == "__main__":
    main()