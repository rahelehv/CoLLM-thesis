# CoLLM Thesis Project - Status

## Phase 1: MF recommender training - DONE
- baseline_train_mf_ood.py trained successfully on ML-1M.
- Result: best_valid_auc=0.676, epoch 467, early-stopped.
- Output committed to repo: weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth (~4MB). On Kaggle: /kaggle/working/CoLLM/weights/... ; used as model.rec_config.pretrained_path in stage 2 via env var COLLM_MF_REC_PTH.
- Added: warning suppression, throttled logging, checkpoint/resume, env-var paths (COLLM_ML1M_DIR, COLLM_OUTPUT_DIR).

## Phase 2: LoRA + Vicuna text-only (paper Stage 1) - DONE
- Entrypoint: train_collm_mf_din.py --cfg-path=train_configs/collm_pretrain_mf_ood.yaml
- Result: best_valid_auc=0.662, best epoch 10, early-stopped after 10 evaluated epochs. Total 5h08m (eval_freq=2 cut it from ~8.5h).
- Checkpoint (~17MB LoRA-only): backed up as Kaggle Model collm-stage1-lora-checkpoint AND locally on my machine.
- Env vars: COLLM_LLAMA_DIR, COLLM_ML1M_DIR, COLLM_OUTPUT_DIR.
- Fixed: CUDA OOM (gradient checkpointing + empty_cache + reduced eval batch), grad_fn RuntimeError (LoRA+checkpointing interaction), and ROOT CAUSE of NaN loss: transformers version mismatch. Kaggle pip had pulled transformers v4.50 (incompatible with this repo's 4.28-era modeling_llama.py fork), corrupting rotary embeddings.
- REQUIRED KAGGLE SETUP (every fresh session, in this exact order):
  1. pip install torch pandas numpy scikit-learn pyyaml omegaconf iopath decord webdataset bitsandbytes (NO transformers, NO peft here)
  2. pip install "transformers==4.28.0" --no-deps
  3. pip install peft==0.3.0 --no-deps
  4. Patch tokenizers version check: find transformers/dependency_versions_table.py and replace the "tokenizers": "tokenizers>=...,<0.14" line with "tokenizers": "tokenizers" (bypasses an artificial version gate; this repo never uses the fast tokenizer, confirmed safe by code audit).
  5. Restart kernel/session after the patch (Python caches the import).
  6. Verify: import transformers, peft; should show 4.28.0 and 0.3.0 with no ImportError.
- debug_nan.py exists at repo root as a diagnostic tool (5 forward-pass cases + weight scan + layer-0 step trace + canary).
- NOTE: the automatic Kaggle Dataset sync (kaggle_checkpoint_sync.py / collm-stage1-checkpoints) did NOT reliably push; fell back to manual download from the session Output panel. Prefer manual backup for stage-2 too.

## Phase 3: CIE module tuning (paper Stage 2) - DONE
- Entrypoint: train_collm_mf_din.py --cfg-path=train_configs/collm_finetune_mf_ood.yaml
- Config: freeze_rec=True, freeze_proj=False, freeze_lora=True, prompt=prompts/collm_movie.txt (WITH IDs), model.ckpt=${oc.env:COLLM_STAGE1_CKPT}, rec_config.pretrained_path=${oc.env:COLLM_MF_REC_PTH}, output_dir=${COLLM_OUTPUT_DIR}/collm-stage2.
- Result: best_valid_auc=0.7257, best epoch 28, early-stopped after 10 evaluated epochs with no improvement. Total 9h22m.
- Clear improvement over stage-1 (0.662) -> confirms the paper's claim that collaborative info via CIE over frozen MF + frozen LoRA helps.
- Checkpoint: /kaggle/working/collm_logs/collm-stage2/*/checkpoint_best.pth (~44MB). NOT YET BACKED UP (TODO: download locally + create Kaggle Model collm-stage2-cie-checkpoint).
- Fixed during Phase 3: in-place autograd error "a leaf Variable that requires grad is being used in an in-place operation" at minigpt4rec_v2.py:550 (recprompt_wrap_v2 ID-embedding merge). Cause: our grad-checkpoint input-requires-grad hook makes the embed output a grad-requiring LEAF, and in-place indexed assignment on a leaf raises; stage-1 never hit it because tallrec prompt has no ID tags. Fix: .clone() the embed output before the indexed assignment (minigpt4rec_v2.py:545, commit 219438f). Verified via repro that grad still flows into llama_proj (detach would have silently killed it).

## Phase 4: Final Evaluation (Table II comparison) - DONE
- Config: train_configs/collm_eval_mf_ood.yaml, merged checkpoint (stage-1 LoRA + stage-2 CIE via merge_stage_ckpts.py).
- Results on ML-1M "test" split (compare to paper's Table II CoLLM-MF row: AUC=0.7295, UAUC=0.6875, NDCG=0.8714):
  - AUC = 0.7434 (paper: 0.7295) -- we exceed the paper's number
  - NDCG = 0.8663 (paper: 0.8714) -- very close
  - UAUC = nan (not computable due to single-interaction users in our split; paper reports 0.6875)
- Also ran on "valid" split: AUC=0.7257, NDCG=0.8646 (consistent with training-time best_valid_auc).
- Total eval time: 45m54s.
- CONCLUSION: full CoLLM pipeline (MF -> LoRA text-only -> CIE mapping) successfully reproduced and validated against the paper, with results in the same range as reported (even exceeding on AUC).
- Fixed during Phase 4: rec builders unconditionally built test_warm/test_cold on evaluate_only=True (requiring missing test_warm_cold_ood2.pkl). Threaded run.test_splits from task.build_datasets into the builders so warm/cold are only built when requested (commit 8d10dc3).

## Project status: Baseline reproduction COMPLETE
All 4 phases done. Next step (not started): literature review / improvement ideas for the thesis's novel contribution, building on this validated baseline.

## Not yet started
- LightGCN/SASRec baseline scripts: same logging/checkpoint improvements as MF are planned but deferred until actually needed.
- Full report: TODO -- a separate detailed report (not in MEMORY.md) will be requested next; MEMORY.md stays short/factual.

## Kaggle setup notes
- 2x T4 GPUs, 30 GPU-hrs/week quota, ~9hr session limit.
- Repo forked to: https://github.com/rahelehv/CoLLM-thesis
- /kaggle/working/ resets between sessions - Vicuna download (~18GB, few min) and dataset unzip must be redone each fresh session unless session persists.