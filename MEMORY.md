# CoLLM Thesis Project - Status

## Phase 1: MF recommender training - DONE
- baseline_train_mf_ood.py trained successfully on ML-1M.
- Result: best_valid_auc=0.676, epoch 467, early-stopped.
- Output .pth file saved locally on my machine (not in repo/Kaggle) - needs to be re-uploaded to Kaggle when Phase 3 starts.
- Added: warning suppression, throttled logging, checkpoint/resume, env-var paths (COLLM_ML1M_DIR, COLLM_OUTPUT_DIR).

## Phase 2: LoRA + Vicuna text-only (paper Stage 1) - DIAGNOSTIC DONE, READY FOR FULL TRAINING RUN
- Entrypoint: train_collm_mf_din.py --cfg-path=train_configs/collm_pretrain_mf_ood.yaml
- Env vars needed: COLLM_LLAMA_DIR, COLLM_ML1M_DIR, COLLM_OUTPUT_DIR
- Vicuna-7B v0 weights: HF mirror mvsoom/pandagpt-vicuna-v0-7b (~18GB, download via huggingface_hub.snapshot_download on Kaggle each fresh session).
- Fixed: CUDA OOM (gradient checkpointing + empty_cache + reduced eval batch), grad_fn RuntimeError (LoRA+checkpointing interaction), and the ROOT CAUSE of NaN loss: transformers version mismatch. Kaggle pip install had pulled transformers v4.50 (incompatible with this repo's 4.28-era modeling_llama.py fork), corrupting rotary embeddings.
- REQUIRED KAGGLE SETUP (every fresh session, in this exact order):
  1. pip install torch pandas numpy scikit-learn pyyaml omegaconf iopath decord webdataset bitsandbytes (NO transformers, NO peft here)
  2. pip install "transformers==4.28.0" --no-deps
  3. pip install peft==0.3.0 --no-deps
  4. Patch tokenizers version check: find transformers/dependency_versions_table.py and replace the "tokenizers": "tokenizers>=...,<0.14" line with "tokenizers": "tokenizers" (bypasses an artificial version gate; this repo never uses the fast tokenizer, confirmed safe by code audit).
  5. Restart kernel/session after the patch (Python caches the import).
  6. Verify: import transformers, peft; should show 4.28.0 and 0.3.0 with no ImportError.
- debug_nan.py exists at repo root as a diagnostic tool (5 forward-pass cases + weight scan + layer-0 step trace + canary) - reusable if similar issues appear in Phase 3.
- NEXT STEP: run the full training command above and let it run to completion (~3-6h estimated for 80 epochs on single T4, fp16, batch 4, grad checkpointing on).

## Not yet started
- Phase 3: CIE module (mapping layer) tuning, using Phase 1 + Phase 2 outputs.
- Phase 4: Evaluation against paper's Table II.
- LightGCN/SASRec baseline scripts: same logging/checkpoint improvements as MF are planned but deferred until actually needed.

## Kaggle setup notes
- 2x T4 GPUs, 30 GPU-hrs/week quota, ~9hr session limit.
- Repo forked to: https://github.com/rahelehv/CoLLM-thesis
- /kaggle/working/ resets between sessions - Vicuna download (~18GB, few min) and dataset unzip must be redone each fresh session unless session persists.