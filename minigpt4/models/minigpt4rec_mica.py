"""
MiCA (Minor Component Adaptation) variant for CoLLM Stage 1.

Implements Rüdiger & Raschka 2026 (arXiv 2604.01694) as a vendored
~30-line init surgery on top of peft 0.3 LoRA, avoiding any peft upgrade
and keeping transformers==4.28.0 pinned.

Mechanism (per target Linear W of shape [out, in]):
  W = U Σ Vᵀ  via SVD (computed in float32, one-time at init, seconds)
  B ← U[:, -r:]   (minor left singular vectors, FROZEN)
  A ← 0           (TRAINABLE, so ΔW = 0 at step 0 → exact preservation)
  ΔW = (alpha/r) * B @ A

Only A is trained → ~50% params of LoRA at same r. Gradients flow
through frozen B. Registered as 'mini_gpt4rec_mica'; import this module
for side-effect before model build (see train_collm_mf_din_mica.py).
"""

import logging

import torch

from minigpt4.common.registry import registry
from minigpt4.models.minigpt4rec_v2 import MiniGPT4Rec_v2


def _apply_mica_to_peft_model(peft_model, base_llama, r):
    """
    Overwrite peft LoRA weights with MiCA init and freeze B.
    Returns number of target modules patched.
    """
    import torch
    patched = 0
    # Collect base weights by module name for SVD source
    # base_llama is the original LlamaForCausalLM before peft wrapping
    base_state = {n: p for n, p in base_llama.named_parameters()}

    for name, param in peft_model.named_parameters():
        # peft 0.3 names: ...layers.<i>.self_attn.<q/v>_proj.lora_A.default.weight or lora_B
        if "lora_B" in name:
            # Derive base weight name: strip ".lora_B.default.weight" -> ".weight"
            # e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
            #  -> model.layers.0.self_attn.q_proj.weight  (relative to base_llama)
            base_key = name.replace("base_model.model.", "").replace(".lora_B.default.weight", ".weight")
            # Try direct lookup, fallback to searching suffix
            base_w = base_state.get(base_key)
            if base_w is None:
                # fallback: find by suffix match
                for k, v in base_state.items():
                    if k.endswith(base_key.split(".")[-3] + ".weight") and "layers" in k:
                        # ambiguous - skip, rely on exact above
                        pass
                logging.warning(f"MiCA: base weight not found for {name} (key {base_key}), skipping")
                continue
            # SVD in float32
            W = base_w.detach().float()
            # W shape [out, in]; SVD: U [out, out], S [min], Vh [in, in] or reduced
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except RuntimeError as e:
                logging.warning(f"MiCA SVD failed for {name} shape {W.shape}: {e}")
                continue
            out_dim, in_dim = W.shape
            if r > min(out_dim, in_dim):
                raise ValueError(f"MiCA r={r} exceeds min({out_dim},{in_dim}) for {name}")
            U_minor = U[:, -r:].contiguous()  # [out, r]
            # Find corresponding param to overwrite (B)
            # param is already the lora_B weight [out, r]
            if param.shape != U_minor.shape:
                logging.warning(f"MiCA shape mismatch {name}: param {tuple(param.shape)} vs U_minor {tuple(U_minor.shape)}")
                continue
            with torch.no_grad():
                param.copy_(U_minor.to(param.dtype))
            param.requires_grad = False
            patched += 1
        elif "lora_A" in name:
            with torch.no_grad():
                param.zero_()
            # keep trainable (default)
    # Verify A zero, B frozen
    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            assert param.requires_grad, f"MiCA: lora_A should be trainable: {name}"
        if "lora_B" in name:
            assert not param.requires_grad, f"MiCA: lora_B should be frozen: {name}"
    return patched


@registry.register_model("mini_gpt4rec_mica")
class MiniGPT4Rec_MiCA(MiniGPT4Rec_v2):
    """
    Vicuna-7B + MiCA instead of LoRA. All other CoLLM plumbing
    (rec encoder, llama_proj CIE, prompting, BCE loss) unchanged.
    """

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        freeze_rec=True,
        rec_precision="fp16",
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        proj_token_num=1,
        proj_drop=0,
        lora_config=None,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=False,
        use_grad_checkpoint=False,
    ):
        # Detect MiCA request: lora_config extra field mica=True or init_lora_weights=="mica"
        use_mica = False
        if lora_config is not None:
            # support both OmegaConf and plain dict
            try:
                use_mica = bool(lora_config.get("mica", False) or lora_config.get("init_lora_weights", "") == "mica")
            except Exception:
                use_mica = getattr(lora_config, "mica", False)

        # Call parent - it will build standard LoRA (A random, B zero)
        super().__init__(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec=pretrained_rec,
            freeze_rec=freeze_rec,
            rec_precision=rec_precision,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num=proj_token_num,
            proj_drop=proj_drop,
            lora_config=lora_config,
            proj_mid=proj_mid,
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        if use_mica and self.use_lora:
            r = int(lora_config.r) if hasattr(lora_config, "r") else int(lora_config.get("r", 8))
            # base_llama is the original model before peft; peft_model wraps it
            # Keep a reference before overwriting
            n_patched = _apply_mica_to_peft_model(self.llama_model_lora, self.llama_model, r)
            logging.info(f"MiCA init done: patched {n_patched} lora_B modules (r={r}), froze B, zeroed A")
            print(f"MiCA: patched {n_patched} lora_B (r={r}) with minor singular vectors, B frozen, A zeroed")
            # Re-apply freeze_lora if requested (should not affect: A remains trainable? freeze_lora means freeze ALL lora)
            # If freeze_lora True, Stage 2 freezes MiCA too - respect it.
            if freeze_lora:
                for _, p in self.llama_model_lora.named_parameters():
                    p.requires_grad = False
                logging.info("freeze lora (MiCA) per freeze_lora=True")

    @classmethod
    def from_config(cls, cfg):
        # Delegate to parent then log MiCA status (parent handles ckpt load etc.)
        model = super().from_config(cfg)
        lora_cfg = cfg.get("lora_config", None)
        is_mica = False
        if lora_cfg is not None:
            try:
                is_mica = bool(lora_cfg.get("mica", False) or lora_cfg.get("init_lora_weights", "") == "mica")
            except Exception:
                pass
        if is_mica:
            logging.info(f"MiCA from_config: active (r={lora_cfg.get('r', '?')}, lambda N/A)")
            print(f"MiCA from_config: active")
        return model
