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

CIE-X (mini_gpt4rec_cie_x): joint fusion via 4-head self-attention over the
collaborative sequence (user+history+target) + FFN, then affine to LLM space.
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


@registry.register_model("mini_gpt4rec_cie_x")
class MiniGPT4Rec_CIE_X(MiniGPT4Rec_v2):
    """
    CIE-X variant: query-based cross-attention fusion (Option C Phase 2).

    Collaborative embeddings (user + history + target) jointly attend before
    mapping to the LLM space. Each position queries the full collaborative
    context via multi-head self-attention, so every token is context-aware
    before the final affine projection. Tests "does joint fusion help?" vs.
    the per-ID independent mapping of the baseline / CIE-G.

    Architecture (all in 256-d collaborative space, then affine to 4096):
      seq_raw = [user_raw (1), interacted_raw (max_items), target_raw (1)]  # (B, S, 256)
      seq = seq_raw + MultiheadAttention(LayerNorm(seq_raw), mask)           # residual
      seq = seq + FFN(LayerNorm(seq))  # FFN: Linear(256->2560)->ReLU->Linear(2560->256)
      fused_* = split(seq)  ->  llama_proj (Linear 256->4096*proj_token_num)

    The per-ID token budget stays at proj_token_num=1 (same as baseline/CIE-G),
    so prompt placeholder counts are unchanged; only *how* the 256-d
    collaborative vectors are contextualised before the LLM injection changes.
    Checkpoint keys are disjoint from baseline (cie_fusion_* + llama_proj) so
    strict=False and merge_stage_ckpts.py remain correct.
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

        if self.rec_encoder is not None and "prompt" not in rec_model:
            emb_size = self.rec_encoder.config.embedding_size
            hidden_size = self.llama_model.config.hidden_size
            mid = int(proj_mid)
            # Fusion: 4-head self-attention over the collaborative sequence (user+history+target)
            self.cie_fusion_attn = nn.MultiheadAttention(emb_size, num_heads=4, batch_first=True)
            self.cie_fusion_norm1 = nn.LayerNorm(emb_size)
            self.cie_fusion_norm2 = nn.LayerNorm(emb_size)
            self.cie_fusion_ff = nn.Sequential(
                nn.Linear(emb_size, emb_size * mid),
                nn.ReLU(),
                nn.Linear(emb_size * mid, emb_size),
            )
            # Final affine mapping after fusion (no ReLU, preserves fused geometry)
            self.llama_proj = nn.Linear(emb_size, hidden_size * self.proj_token_num)
            logging.info(
                "CIE-X: built fusion (MHA 4-head, FFN %d->%d->%d, LNx2) + final Linear(%d->%d).",
                emb_size, emb_size * mid, emb_size, emb_size, hidden_size * self.proj_token_num,
            )
            print(f"CIE-X fusion: MHA(256, 4-head, batch_first) + FFN(256->{emb_size*mid}->256) + Linear(256->{hidden_size * self.proj_token_num})")

        # Re-apply freeze as requested — now over fusion + final proj
        cie_modules = []
        for attr in ("cie_fusion_attn", "cie_fusion_norm1", "cie_fusion_norm2", "cie_fusion_ff", "llama_proj"):
            mod = getattr(self, attr, None)
            if mod is not None:
                cie_modules.append(mod)
        if freeze_proj:
            for mod in cie_modules:
                for _, p in mod.named_parameters():
                    p.requires_grad = False
                if hasattr(mod, "eval"):
                    mod.eval()
                    mod.train = disabled_train
            logging.info("!!!! freeze CIE-X fusion + llama_proj...")
        else:
            for mod in cie_modules:
                for _, p in mod.named_parameters():
                    p.requires_grad = True

    def encode_recdata_v2(self, sample, ids_order=None):
        if self.rec_encoder is None:
            return None, None
        device = sample["UserID"].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to("cpu")

        with self.maybe_autocast():
            batch_size = sample["UserID"].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":
                user_embeds_raw = self.rec_encoder.seq_encoder(sample["sas_seq"]).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                user_embeds_raw = self.rec_encoder.all_encode(
                    sample["UserID"], sample["TargetItemID"], sample["sas_seq"][:, -10:]
                ).unsqueeze(-2)
            else:
                user_embeds_raw = self.rec_encoder.user_encoder(
                    sample["UserID"], all_users=all_user_embeds
                ).unsqueeze(-2)
            target_raw = self.rec_encoder.item_encoder(
                sample["TargetItemID"], all_items=all_item_embeds
            ).unsqueeze(-2)

            # Joint-fusion path: only when history is present and ids_order covers all three
            if "InteractedItemIDs_pad" in sample.keys() and ids_order is not None and len(ids_order) == 3:
                interacted_raw = self.rec_encoder.item_encoder(
                    sample["InteractedItemIDs_pad"], all_items=all_item_embeds
                )  # (B, max_items, 256)

                # Build sequence [user, history..., target] and padding mask for history
                seq_raw = torch.cat([user_embeds_raw, interacted_raw, target_raw], dim=1)  # (B, S, 256)
                max_items = interacted_raw.shape[1]
                pad_mask_interacted = sample["InteractedItemIDs_pad"] == self.rec_encoder.padding_index  # (B, max_items)
                pad_mask_seq = torch.cat(
                    [
                        torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                        pad_mask_interacted,
                        torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                    ],
                    dim=1,
                )  # (B, S), True = ignore

                # Fusion: pre-norm attention + residual, then pre-norm FFN + residual
                # (LayerNorm and MHA/FFN run inside autocast, same as parent)
                seq_norm = self.cie_fusion_norm1(seq_raw)
                attn_out, _ = self.cie_fusion_attn(
                    seq_norm, seq_norm, seq_norm, key_padding_mask=pad_mask_seq
                )
                seq_fused = seq_raw + attn_out
                seq_fused_norm = self.cie_fusion_norm2(seq_fused)
                ff_out = self.cie_fusion_ff(seq_fused_norm)
                seq_fused = seq_fused + ff_out

                # Split back to per-entity fused embeddings
                fused_user = seq_fused[:, 0:1, :]  # (B,1,256)
                fused_interacted = seq_fused[:, 1 : 1 + max_items, :]  # (B,max_items,256)
                fused_target = seq_fused[:, -1:, :]  # (B,1,256)

                # Final affine mapping to LLM space (proj_token_num tokens per collaborative embedding)
                user_llama = self.llama_proj(fused_user).reshape(
                    batch_size, -1, self.proj_token_num, hidden_size
                )
                target_llama = self.llama_proj(fused_target).reshape(
                    batch_size, -1, self.proj_token_num, hidden_size
                )
                interacted_llama = self.llama_proj(fused_interacted).reshape(
                    batch_size, -1, self.proj_token_num, hidden_size
                )

                merged_embeds = [user_llama, interacted_llama, target_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds, dim=1)
                idx_flag = torch.ones_like(sample["InteractedItemIDs_pad"])
                idx_flag = torch.where(
                    sample["InteractedItemIDs_pad"] == self.rec_encoder.padding_index, 0, idx_flag
                )
                idx_flag = [
                    torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device),
                    idx_flag,
                    torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device),
                ]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag, dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)

                sample_embeds_llama = {
                    "User_emb": user_llama.reshape(batch_size, -1, hidden_size),
                    "TargetItem_emb": target_llama.reshape(batch_size, -1, hidden_size),
                    "InteractedItems_embs": interacted_llama.reshape(batch_size, -1, hidden_size),
                    "merged_embs": merged_embeds[idx_nopad[:, 0], idx_nopad[:, 1]].reshape(-1, hidden_size),
                }
            else:
                # Fallback: no history (or non-standard prompt) — per-ID affine mapping
                user_llama = self.llama_proj(user_embeds_raw).reshape(
                    batch_size, -1, self.proj_token_num, hidden_size
                )
                target_llama = self.llama_proj(target_raw).reshape(
                    batch_size, -1, self.proj_token_num, hidden_size
                )
                sample_embeds_llama = {
                    "User_emb": user_llama.reshape(batch_size, -1, hidden_size),
                    "TargetItem_emb": target_llama.reshape(batch_size, -1, hidden_size),
                    "InteractedItems_embs": None,
                    "merged_embs": None,
                }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama
