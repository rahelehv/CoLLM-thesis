import argparse
import copy
import math
import os
import random

import numpy as np
import pandas as pd
import torch

import minigpt4.models as models  # noqa: F401  (registers model classes), mirrors train script

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.models.modeling_llama import apply_rotary_pos_emb


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


def stat(tag, t):
    x = t.detach()
    n = int(torch.isnan(x).sum())
    i = int(torch.isinf(x).sum())
    lo = float(x.min()) if x.numel() else float("nan")
    hi = float(x.max()) if x.numel() else float("nan")
    print(
        "    {:<22} dtype={:<6} shape={:<20} nan={} inf={} min={:.5e} max={:.5e}".format(
            tag, str(x.dtype), "x".join(str(s) for s in x.shape), n, i, lo, hi
        )
    )


def replicate_inputs(model, device, ann, b=2):
    # Reconstruct the exact inputs_embeds / attention_mask / position_ids that forward_v2
    # feeds the Llama stack, using the real tallrec prompt (which injects per-sample
    # TargetItemTitle -> samples within a batch differ in length -> left padding is active).
    model.eval()
    model.set_mode("v2")
    samples = build_samples(device, b, ann)

    captured = {}

    def cap_hook(module, inp, out):
        captured["prompt_ids"] = inp[0].detach().cpu().clone()

    hk = model.llama_model.model.embed_tokens.register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            prompt = random.choice(model.prompt_with_p([5, 5, 5, 1]))
            sample_embeds, atts_samples = model.prompt_based_encode_v2(prompt, samples)
            model.llama_tokenizer.padding_side = "right"
            ans_ = {1: model.pos_ans[0], 0: model.neg_ans[0]}
            text = [ans_[int(t)] for t in samples["label"]]
            to_regress_tokens = model.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=model.max_txt_len,
                add_special_tokens=False,
            ).to(device)
            to_regress_embeds = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
            raw_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
    finally:
        hk.remove()

    bsz, seq, _ = inputs_embeds.shape
    mask4d = model.llama_model.model._prepare_decoder_attention_mask(
        raw_mask, (bsz, seq), inputs_embeds, 0
    )
    position_ids = torch.arange(seq, device=device).unsqueeze(0).view(-1, seq)

    prompt_ids = captured["prompt_ids"]
    print("== input construction inspection (real tallrec path) ==")
    print("  prompt selected (prefix: {!r})".format(str(prompt)[:160]))
    print("  prompt contains ID/title tags:", [t for t in ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<TargetItemTitle>"] if t in prompt])
    print("  vocab_size =", model.llama_model.config.vocab_size,
          " prompt max id =", int(prompt_ids.max()),
          " answer max id =", int(to_regress_tokens.input_ids.max()),
          " unk tokens in prompt =", int((prompt_ids == model.llama_tokenizer.unk_token_id).sum()))
    print("  prompt[0] decoded: {!r}".format(model.llama_tokenizer.decode(prompt_ids[0])[:200]))
    stat("sample_embeds", sample_embeds)
    stat("to_regress_embeds", to_regress_embeds)
    stat("inputs_embeds", inputs_embeds)
    stat("raw attention_mask", raw_mask)
    stat("prepared mask4d", mask4d)
    print("  raw_mask pad count: {} of {}".format(int((raw_mask == 0).sum()), raw_mask.numel()))
    return inputs_embeds, mask4d, position_ids, raw_mask


def _layer0_steps(layer0, h, mask4d, position_ids, tag, raw_mask=None):
    # step-by-step clone of LlamaDecoderLayer.forward inside LlamaAttention, recording
    # finite-ness at every intermediate so the first NaN/Inf op is identified.
    bsz, q_len, _ = h.shape
    nh, hd = layer0.self_attn.num_heads, layer0.self_attn.head_dim
    steps = {}
    with torch.no_grad():
        steps["h_in"] = h
        h_ln = layer0.input_layernorm(h)
        steps["after_input_ln"] = h_ln
        q0 = layer0.self_attn.q_proj(h_ln)
        k0 = layer0.self_attn.k_proj(h_ln)
        v0 = layer0.self_attn.v_proj(h_ln)
        q = q0.view(bsz, q_len, nh, hd).transpose(1, 2)
        k = k0.view(bsz, q_len, nh, hd).transpose(1, 2)
        v = v0.view(bsz, q_len, nh, hd).transpose(1, 2)
        steps["q_proj"], steps["k_proj"], steps["v_proj"] = q0, k0, v0
        cos, sin = layer0.self_attn.rotary_emb(v, seq_len=q_len)
        steps["rotary_cos"], steps["rotary_sin"] = cos, sin
        qabs = q.abs().amax(dim=(0, 1, 3))
        steps["q_absmax_per_pos"] = qabs
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        steps["q_rot"], steps["k_rot"] = q, k
        qrotabs = q.abs().amax(dim=(0, 1, 3))
        steps["q_rot_absmax_per_pos"] = qrotabs
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        steps["attn_qk"] = qk
        qk_m = torch.max(
            qk + mask4d, torch.tensor(torch.finfo(qk.dtype).min, device=qk.device)
        )
        steps["attn_qk_masked"] = qk_m
        probs = torch.softmax(qk_m, dim=-1, dtype=torch.float32).to(q.dtype)
        steps["attn_softmax"] = probs
        av = torch.matmul(probs, v).transpose(1, 2).reshape(bsz, q_len, -1)
        steps["attn_context"] = av
        o = layer0.self_attn.o_proj(av)
        steps["o_proj"] = o
        h2 = o + steps["h_in"]
        steps["after_attn_residual"] = h2
        h_ln2 = layer0.post_attention_layernorm(h2)
        steps["after_post_ln"] = h_ln2
        steps["mlp_out"] = layer0.mlp(h_ln2)
        steps["layer0_out"] = steps["mlp_out"] + h2
    print("  -- {} layer-0 steps --".format(tag))
    for name, t in steps.items():
        stat(name, t)
    print("    position_ids =", position_ids.view(-1).cpu().tolist())
    print("    max|q| per position =", [float(x) for x in steps["q_absmax_per_pos"].cpu()])
    print("    max|q_rot| per position =", [float(x) for x in steps["q_rot_absmax_per_pos"].cpu()])
    if raw_mask is not None:
        pad_pos = [int(i) for i in range(raw_mask.size(1)) if int(raw_mask[0, i]) == 0]
        print("    padded positions (sample 0) =", pad_pos)
        qrot_n = steps["q_rot"][:, :, pad_pos, :].abs()
        if qrot_n.numel():
            print(
                "    q_rot max over padded positions = {:.3e}".format(float(qrot_n.max()))
            )
    return steps


def probe_layer0(model, device, inputs_embeds, mask4d, position_ids, raw_mask=None):
    layer0 = model.llama_model.model.layers[0]
    bsz, seq, _ = inputs_embeds.shape

    print("-- module identity (is this repo's modeling_llama running?) --")
    attn = layer0.self_attn
    print("  self_attn class: {}".format(type(attn)))
    print("  rotary_emb class: {}".format(type(attn.rotary_emb)))
    print("  rotary_emb module: {}".format(attn.rotary_emb.__class__.__module__))
    print("  q/k/v/o proj type: {}".format(type(attn.q_proj)))
    print("  cfg rope_scaling/rope_type: {}/{}".format(
        getattr(model.llama_model.config, "rope_scaling", None),
        getattr(model.llama_model.config, "rope_type", None)))

    print("-- mask variant quick checks (fp16 layer0 out) --")
    variants = {
        "real mask": layer0(inputs_embeds, attention_mask=mask4d, position_ids=position_ids)[0],
        "zero mask": layer0(inputs_embeds, position_ids=position_ids)[0],
        "real mask fp32cast": layer0(
            inputs_embeds, attention_mask=mask4d.float(), position_ids=position_ids
        )[0],
    }
    for name, out in variants.items():
        stat("layer0_out: " + name, out)

    _layer0_steps(layer0, inputs_embeds, mask4d, position_ids, "FP16", raw_mask=raw_mask)

    layer0_f32 = copy.deepcopy(layer0).float().to(device)
    pos_ids_t = position_ids.long()
    _layer0_steps(
        layer0_f32,
        inputs_embeds.float(),
        mask4d.float(),
        pos_ids_t,
        "FP32",
        raw_mask=raw_mask,
    )

    print("-- canary: bounded random input through the same code path --")
    rot = layer0.self_attn.rotary_emb
    print("    inv_freq head/tail =", rot.inv_freq[:5].cpu().tolist(), rot.inv_freq[-3:].cpu().tolist())
    print("    max_seq_len_cached =", rot.max_seq_len_cached,
          " cos_cached min/max = {:.3f}/{:.3f}".format(
              float(rot.cos_cached.min()), float(rot.cos_cached.max())))
    h_fake = torch.randn(1, seq, 64, device=device, dtype=torch.float16).abs() * (1.0 / 128) + 0.0001
    cos_f, sin_f = rot(h_fake.transpose(1, 2), seq_len=seq)
    q_f = torch.randn(1, 32, seq, 128, device=device) * 0.1
    q_f_rot, _ = apply_rotary_pos_emb(q_f, q_f, cos_f.float(), sin_f.float(), position_ids.long())
    print("    canary max|q_rot|  = {:.3e}  (must be ~<1.0; if ~1e35 -> code path broken)".format(
        float(q_f_rot.abs().max())))


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

    inputs_embeds, mask4d, position_ids, raw_mask = replicate_inputs(model, device, train_)
    print("== case E: layer-0 step-by-step fp16 vs fp32 ==")
    probe_layer0(model, device, inputs_embeds, mask4d, position_ids, raw_mask)


if __name__ == "__main__":
    main()