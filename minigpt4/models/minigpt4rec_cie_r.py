"""
CIE-R (relational-consistency regularization) variant for CoLLM.

Baseline architecture (rec encoder, llama_proj MLP, LoRA, prompting) is kept
EXACTLY as in minigpt4rec_v2.py. The only change is an auxiliary training-time
loss that penalizes divergence between the relational (pairwise cosine)
structure of the raw MF embeddings and their projected LLM-space counterparts:

    L_total = BCE + lambda * MSE_offdiag( S_raw, S_proj )

where S_* = cosine similarity matrices over all non-pad collaborative tokens
[user; history...; target] in the batch (distance-wise RKD, Park et al. 2019,
arXiv 1904.5068). Gradients flow ONLY into llama_proj (the rec encoder is
frozen and detached). No new inference-time parameters are introduced, so
checkpoint keys stay 'llama_proj.*' and merge_stage_ckpts.py works unchanged.

Rationale (post Option-C): capacity and fusion-architecture changes to the CIE
all failed (Control/CIE-G/CIE-X); this adds supervisory SIGNAL to the already-
validated mapping instead. Revives the authors' dormant consistency idea
(consitence_loss in minigpt4rec_v2.py, call site commented out) in a principled
distance-wise form.

Registered as 'mini_gpt4rec_cie_r'. Import this module for registry side-effect
before building the model (see train_collm_mf_din_cie_r.py).
"""

import logging

import torch
import torch.nn.functional as F

from minigpt4.common.registry import registry
from minigpt4.models.minigpt4rec_v2 import MiniGPT4Rec_v2


def cie_relational_loss(raw_tokens, proj_tokens):
    """
    Distance-wise relational consistency between two embedding spaces.

    raw_tokens:  (N, d_raw)  raw MF embeddings (will be detached internally)
    proj_tokens: (N, d_proj) their llama_proj outputs (keep grad)

    Returns scalar MSE between off-diagonal entries of the two pairwise
    cosine-similarity matrices computed in L2-normalized float32 space.
    """
    R = F.normalize(raw_tokens.detach().float(), dim=-1)
    P = F.normalize(proj_tokens.float(), dim=-1)
    S_raw = R @ R.T
    S_proj = P @ P.T
    n = S_raw.shape[0]
    if n < 2:
        return torch.zeros((), device=S_raw.device)
    off = ~torch.eye(n, dtype=torch.bool, device=S_raw.device)
    return F.mse_loss(S_proj[off], S_raw[off])


@registry.register_model("mini_gpt4rec_cie_r")
class MiniGPT4Rec_CIE_R(MiniGPT4Rec_v2):
    """
    Baseline MiniGPT4Rec_v2 + lambda-weighted relational consistency loss.

    llama_proj is NOT swapped: the baseline Linear->ReLU->Linear mapping stays,
    per the Option-C finding that architecture changes do not help. Only the
    training objective changes.
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
        cie_reg_lambda=0.0,
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
            freeze_proj=freeze_proj,
            use_grad_checkpoint=use_grad_checkpoint,
        )
        self.cie_reg_lambda = float(cie_reg_lambda or 0.0)
        logging.info(
            "CIE-R ready: llama_proj unchanged (baseline MLP), "
            "cie_reg_lambda=%g, loss = BCE + %g * MSE_offdiag(S_raw, S_proj)",
            self.cie_reg_lambda, self.cie_reg_lambda,
        )
        print(
            f"CIE-R: cie_reg_lambda={self.cie_reg_lambda} "
            f"(llama_proj untouched; relational-consistency reg active in training only)"
        )

    @classmethod
    def from_config(cls, cfg):
        model = super().from_config(cfg)
        model.cie_reg_lambda = float(cfg.get("cie_reg_lambda", 0.0) or 0.0)
        logging.info("CIE-R from_config: cie_reg_lambda=%g", model.cie_reg_lambda)
        print(f"CIE-R from_config: cie_reg_lambda={model.cie_reg_lambda}")
        return model

    def _batch_relation_loss(self, samples):
        """
        Recompute raw MF embeddings for this batch (encoder is frozen; cheap),
        project them with the CURRENT llama_proj (fresh graph -> gradients reach
        it), and return the distance-wise relational MSE over non-pad tokens.
        Returns None when the batch carries no history field (non-ID prompt).
        """
        hist_ids = samples.get("InteractedItemIDs_pad")
        if hist_ids is None or self.rec_encoder is None:
            return None

        with self.maybe_autocast():
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            user_raw = self.rec_encoder.user_encoder(
                samples["UserID"], all_users=all_user_embeds
            ).unsqueeze(1)                                                    # (B,1,E)
            target_raw = self.rec_encoder.item_encoder(
                samples["TargetItemID"], all_items=all_item_embeds
            ).unsqueeze(1)                                                    # (B,1,E)
            hist_raw = self.rec_encoder.item_encoder(
                hist_ids, all_items=all_item_embeds
            )                                                                 # (B,L,E)

        B, L, E = hist_raw.shape
        pad = hist_ids == self.rec_encoder.padding_index                      # (B,L)
        raw = torch.cat([user_raw, hist_raw, target_raw], dim=1)              # (B,S,E)
        keep = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=raw.device),
                ~pad,
                torch.ones(B, 1, dtype=torch.bool, device=raw.device),
            ],
            dim=1,
        )                                                                     # (B,S)

        proj = self.llama_proj(raw)                                           # (B,S,H*ptn)
        S_tot = keep.shape[1]
        H = proj.shape[-1] // max(self.proj_token_num, 1)

        keep_exp = (
            keep.unsqueeze(-1)
            .expand(B, S_tot, self.proj_token_num)
            .reshape(-1)                                                      # (B*S*ptn,)
        )
        raw_flat = (
            raw.detach().unsqueeze(2)
            .expand(B, S_tot, self.proj_token_num, E)
            .reshape(-1, E)[keep_exp]
        )                                                                     # (N,E)
        proj_sel = proj.reshape(B * S_tot * self.proj_token_num, H)[keep_exp]  # (N,H)

        n_sel = raw_flat.shape[0]
        if proj_sel.shape[0] != n_sel:
            raise RuntimeError(
                f"CIE-R token alignment broken: raw {n_sel} vs proj {proj_sel.shape[0]}"
            )
        if n_sel < 2:
            return None

        return cie_relational_loss(raw_flat, proj_sel)

    def forward_v2(self, samples):
        out = super().forward_v2(samples)
        if self.training and self.cie_reg_lambda > 0 and self.rec_encoder is not None:
            l_rel = self._batch_relation_loss(samples)
            if l_rel is not None:
                out["loss"] = out["loss"] + self.cie_reg_lambda * l_rel.to(out["loss"].dtype)
        return out
