"""
CIE variants for CoLLM (Option C).

This module adds NEW model architectures without touching
minigpt4/models/minigpt4rec_v2.py or minigpt4/models/__init__.py.

- CIE-G (mini_gpt4rec_cie_g): affine + gated ReLU-MLP. Tests the
  "ReLU destroys MF dot-product geometry" hypothesis. The baseline
  llama_proj is Linear(256->2560) -> ReLU -> Linear(2560->4096*proj_token_num).
  CIE-G keeps that MLP as a *gated* residual alongside a pure affine path
  that preserves dot-product structure: output = affine(x) + sigmoid(gate)*MLP(x).

Checkpoint keys are disjoint from the baseline (llama_proj.affine.*,
llama_proj.mlp.*, llama_proj.gate) so strict=False loading is fine and
merge_stage_ckpts.py still works (LoRA ckpt + CIE ckpt have no overlap).

CIE-X (mini_gpt4rec_cie_x) will be added to this same file later.
"""

import logging

import torch
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.rec_model import disabled_train
from minigpt4.models.minigpt4rec_v2 import MiniGPT4Rec_v2


class CIE_G_Proj(nn.Module):
    """
    Gated CIE mapping (Option C Phase 1).

    Two parallel paths from the collaborative embedding (dim = emb_size):
      affine: Linear(emb -> hidden*proj_token_num)  — preserves geometry
      mlp:    Linear(emb -> emb*proj_mid) -> ReLU -> Linear(emb*proj_mid -> hidden*proj_token_num)

    Output = affine(x) + sigmoid(gate) * mlp(x), where gate is a learnable
    scalar (init 0.0 -> sigmoid 0.5, balanced). With gate ~ 0 the model is
    near-affine; training can increase gate to blend in non-linearity if useful.

    Input x shape: (..., emb_size). Output shape: (..., hidden*proj_token_num)
    — identical to the baseline llama_proj, so encode_recdata_v2's reshape
    logic (batch, -1, proj_token_num, hidden) continues to work unchanged.
    """

    def __init__(self, emb_size: int, proj_mid: int, hidden_size: int, proj_token_num: int):
        super().__init__()
        self.affine = nn.Linear(emb_size, hidden_size * proj_token_num)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * proj_mid),
            nn.ReLU(),
            nn.Linear(emb_size * proj_mid, hidden_size * proj_token_num),
        )
        # learnable scalar gate logit; sigmoid(gate) in (0,1)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.affine(x) + torch.sigmoid(self.gate) * self.mlp(x)


@registry.register_model("mini_gpt4rec_cie_g")
class MiniGPT4Rec_CIE_G(MiniGPT4Rec_v2):
    """
    CIE-G variant of MiniGPT4Rec_v2.

    Reuses all of MiniGPT4Rec_v2 (rec encoder, Llama, LoRA, prompt handling,
    forward_v2/generate_for_samples_v2, from_config ckpt loading, etc.) and
    only swaps the CIE mapping layer (llama_proj) for CIE_G_Proj.

    New files only — minigpt4/models/__init__.py is not touched; the new
    entry script train_collm_mf_din_cie.py imports this module so the
    @registry.register_model decorator runs before model construction.
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
        # Call the parent to set up rec_encoder, llama_model(+LoRA), tokenizer,
        # prompt_list, max_txt_len, etc. Pass freeze_proj=False so the parent
        # does not freeze-and-eval the *old* Sequential before we replace it;
        # we re-apply the requested freeze after swapping.
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
            freeze_proj=False,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        # Swap the CIE mapping layer for the gated variant when the MF branch
        # is active (the only branch used in this thesis). The 'prompt'-model
        # branches are left as-is and are not used for MF.
        if self.rec_encoder is not None and "prompt" not in rec_model and self.llama_proj is not None:
            emb_size = self.rec_encoder.config.embedding_size
            hidden_size = self.llama_model.config.hidden_size
            mid = int(proj_mid)
            old_keys = len(list(self.llama_proj.parameters())) if hasattr(self.llama_proj, "parameters") else 0
            self.llama_proj = CIE_G_Proj(emb_size, mid, hidden_size, self.proj_token_num)
            logging.info(
                "CIE-G: replaced llama_proj (was %d param tensors) with CIE_G_Proj "
                "(affine + gated MLP, emb=%d mid=%d hidden=%d proj_token_num=%d, gate init 0.0 -> sigmoid 0.5).",
                old_keys, emb_size, mid, hidden_size, self.proj_token_num,
            )
            print(
                f"CIE-G llama_proj: affine Linear({emb_size}->{hidden_size * self.proj_token_num}) + "
                f"gated MLP Linear({emb_size}->{emb_size*mid})->ReLU->Linear({emb_size*mid}->{hidden_size * self.proj_token_num}), "
                f"gate=0.0 (sigmoid 0.5)"
            )

        # Re-apply freeze_proj as originally requested, now on the new module
        if freeze_proj and self.llama_proj is not None:
            for _, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("!!!! freeze llama_proj (CIE-G)...")
        elif self.llama_proj is not None:
            # ensure the new CIE-G params are trainable when freeze_proj is False
            # (stage 2: this is the only trainable module)
            for _, param in self.llama_proj.named_parameters():
                param.requires_grad = True
