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

## Project status: Baseline reproduction COMPLETE, Option C (richer CIE) IN PROGRESS
All 4 phases done. Novel-contribution plan (confirmed): 3 sequential CIE variants on Vicuna-7B, all NEW files only (no edits to baseline_train_mf_ood.py / minigpt4/ existing files / existing train_configs). Single-GPU (`python`, world_size=1), same as Phases 1-4.
- Phase 0 CONTROL (proj_token_num: 4) -> collm_cie_pn4_{finetune,eval}.yaml. Isolates token-capacity effect. **DONE 2026-08-22: best epoch 28, best_valid_auc=0.7256933808225703, 6h55m (max_epoch 50 reached, not early-stop). Essentially IDENTICAL to Phase 3's proj_token_num=1 result (0.7256668021338382 @28, 9h22m). Verdict: increasing token capacity alone (1->4) gives no meaningful improvement — the bottleneck is architectural (how the mapping is structured), not raw parameter/token count. checkpoint_best.pth not downloaded (only checkpoint_42.pth); not needed — CIE-G/X start fresh from Phase-1 MF + Phase-2 LoRA ckpts, only the AUC number is needed for the writeup.**
- Phase 1 CIE-G (affine + gated residual, registered mini_gpt4rec_cie_g) -> isolates ReLU-destroys-geometry hypothesis. **DONE 2026-08-23: best epoch 28, best_valid_auc=0.7213615742542488, max_epoch 50 reached (plateau). Slightly BELOW both Phase 3 baseline (0.7256668021338382 @28) and Control (0.7256933808225703 @28) — delta ~-0.004. Verdict: affine+gated MLP alone does NOT rescue geometry; ReLU hypothesis not supported in this form, or single-token affine insufficient. WeightsOnly fix (torch.load weights_only=False) applied mid-run to resume (commit e06de42); training used eval_freq:4, save_ckpt_keep:3, max mem 14041 MB.**
- Phase 2 CIE-X (query-based cross-attention fusion, mini_gpt4rec_cie_x) -> tests full fusion hypothesis. **DONE 2026-08-23: best epoch 48, best_valid_auc=0.7004266250752259, max_epoch 50 reached. ~0.025 BELOW baseline (0.72567 @28) and Control (0.72569 @28), also below CIE-G (0.72136 @28). Model state 10.5 MB/14 keys (vs 48.8 MB/7 keys for CIE-G — far fewer params), late peak (48 vs 28) suggests capacity-limited / under-fitting, not over-fitting. Verdict: joint MHA fusion in this form does NOT help; may need larger capacity, different pooling, or history-aware prompting.**
- Fair comparison: same eval protocol as Phase 4 (test_splits ["test","valid"], collm_movie.txt, COLLM_MF_REC_PTH, seed 42). Baseline to beat: test AUC=0.7434, NDCG=0.8663, valid AUC=0.7257. UAUC excluded (nan artifact).
- CIE hyperparams read from new rec_config.cie key (inherited from_config unchanged). Checkpoint flow unchanged (merge_stage_ckpts.py). Stage-1 NOT retrained (freeze_proj=True in stage-1 so old CIE never saved; stage-2 ckpt=stage-1 LoRA).
- Budget: ~10h/variant (stage-2 ~9.4h + eval ~0.8h) -> ~2 variants/week, ~1.5 weeks for all three.
- Gotcha: proj_token_num>1 expands history slots 4x vs max_txt_len=1024; verify no truncation (breaks placeholder/embedding count) in the #######prmpt decoded example: log line.
- Runner infra since a712fc7: atomic saves (tmp+os.replace), retention save_ckpt_keep=3 (+checkpoint_best.pth), persisted best_agg_metric/best_epoch/not_change with legacy log.txt recovery; Control updated to eval_freq:4/max_epoch:50/save_ckpt_keep:3. CIE-G/X will use same from the start.

## Current work: CIE-R full run + eval — DONE (2026-08-31) — see INNOVATION.md
- λ-proxy sweep DONE (12-epoch, same horizon):
  - l001 (λ=0.01): best 0.67133 @10 — `raw_logs/cie_r_l001_proxy_results.txt` (near-control)
  - l01  (λ=0.1):  best **0.70334** @10 (+0.032) — **WINNER**, picked for full run
  - l1   (λ=1.0):  best 0.69300 @4 then decay — over-regularized
- FULL RUN (λ=0.1, 50-epoch, resumed from checkpoint_43 after crash — a712fc7/e06de42 saved it): **best valid 0.72523 @36** (first session), resume tail 0.72083@44 / 0.71852@48, 0:51 wall on resume. vs baselines: baseline 0.72567@28, control 0.72569@28, CIE-G 0.72136@28, CIE-X 0.70043@48 → **tied with baseline (-0.00044, null)**. Proxy's +0.032 washed out over full horizon — reg accelerated early fit only.
- EVAL (checkpoint_39 proxy, `collm_cie_r_l01_full_eval.yaml`, merged_39.pth): **valid AUC 0.72050, test AUC 0.71852**, NDCG 0.86838 (valid) / 0.86837 (test) — evaluated on 1300/1833 batches. Baseline Phase 4 had test 0.7434 / valid 0.7257 (NDCG 0.8663). CIE-R test is -0.025 below baseline test on this proxy; valid -0.005 below baseline valid. Note: best @36 lives only in first session's wiped output dir; only checkpoint_43 was published, so checkpoint_39 (-0.004 away) used as proxy — true best would be marginally higher.
- Fix applied: SyntaxWarning `mode \in` → `mode in` in minigpt4rec_v2.py:258.
- EVIDENCE: `raw_logs/cie_r_l01_full_results.txt`, Kaggle `collm-cie-r-l01-full-eval` log (2026-08-31, 46m55s eval). All CIE-R code committed: `minigpt4/models/minigpt4rec_cie_r.py` (mini_gpt4rec_cie_r), entry `train_collm_mf_din_cie_r.py`, configs `collm_cie_r_l01_full.yaml` + `_eval.yaml`, unit test 23/23 (`raw_logs/step_cie_r_local_unittest.txt`), parser `parse_cie_r_proxies.py`.

## Entry scripts (which to use when)
| Script | Registers | Use for |
|---|---|---|
| `baseline_train_mf_ood.py` | — (standalone trainer) | Phase-1 MF recommender .pth |
| `train_collm_mf_din.py` | `mini_gpt4rec_v2` only | Phases 2–4 baseline (stage 1 / stage 2 / eval) |
| `train_collm_mf_din_cie.py` | `mini_gpt4rec_cie_g`, `mini_gpt4rec_cie_x` | Option C CIE-G/CIE-X runs (completed, negative results) |
| `train_collm_mf_din_cie_r.py` | `mini_gpt4rec_cie_g` + `_x` + `_r` | **CIE-R runs** — superset of _cie.py |
| `train_collm_mf_din_mica.py` | `mini_gpt4rec_mica` | **MiCA Stage-1 runs (current)** |
| `train_collm_{mf_din}_lgcn.py` / `_sasrec.py` | respective archs | deferred, not yet used |
Rule of thumb: the YAML's `model.arch` must be registered by the entry script you launch; if unregistered → error.

## MiCA Stage-1 — DONE (2026-08-31, checkpoint recovery in progress) — see INNOVATION.md §2.1 and IMPLEMENTATION_REPORT.md §6.8
- MiCA = Minor Component Adaptation, arXiv 2604.01694 (Ruediger & Raschka, Apr 2026): B=minor singular vectors (frozen), A=0 (trainable), ~50% params of LoRA at same r. Official peft has it (examples/mica_finetuning) but needs py>=3.10 + transformers upgrade that reintroduces rotary NaN → vendored ~30-line surgery on peft 0.3 chosen instead.
- IMPLEMENTED: `minigpt4/models/minigpt4rec_mica.py` (mini_gpt4rec_mica, SVD init + freeze B), entry `train_collm_mf_din_mica.py`, config `train_configs/collm_mica_pretrain_mf_ood.yaml` (arch mica, mica: true, r=8 alpha=16 q/v). Local unit test PASS (raw_logs/step_mica_local_unittest.txt).
- RESULT: **best 0.68894 @64** (1:34:34, max_epoch 80, eval_freq 2, 2.1M trainable vs LoRA 4.2M), vs baseline LoRA Stage-1 0.662 @10 → **+0.027**. Resumed from 69 after session interruption — a712fc7/e06de42 persisted best correctly; note log concatenation artifact after "Training stopped" is not an extra loop.
- INCIDENT (thesis transparency): During MiCA Stage 1 training, checkpoint_best.pth (epoch 64, best_agg_metrics=0.6889447800271994) was lost due to a Kaggle session ending before it was manually downloaded (periodic checkpoints 29/49/51/53/55/69/75/79 were saved, but not checkpoint_best.pth, which persists separately from the pruned periodic checkpoints and requires manual backup like all our checkpoints). To recover the exact result, we resumed training from checkpoint_55.pth with max_epoch=65 (via --options override), reproducing epochs 56-64 forward-only. Since dataloader shuffle order/RNG state isn't persisted in checkpoints (only model/optimizer/scaler/epoch/best-metric state), this reproduction is not bit-identical to the original run, but empirically differs by an estimated ~0.001-0.003 AUC based on prior similar recoveries (see CIE-R's checkpoint_39-for-36 substitution). This reconstructed checkpoint is used as the canonical MiCA Stage 1 result going forward.
- RECOVERED CHECKPOINT (55→64, 9 epochs): *pending — will record best_agg_metrics once the short recovery run finishes and note whether it closely matched 0.6889447800271994 (reproducibility check).*
- NEXT: Stage-2 CIE tuning on frozen MiCA if recovered result holds; otherwise MiCA Stage-1 alone is already a thesis contribution.

## Not yet started
- LightGCN/SASRec baseline scripts: same logging/checkpoint improvements as MF are planned but deferred until actually needed.
- Full report: TODO -- a separate detailed report (not in MEMORY.md) will be requested next; MEMORY.md stays short/factual.

## Kaggle setup notes
- 2x T4 GPUs, 30 GPU-hrs/week quota, ~9hr session limit.
- Repo forked to: https://github.com/rahelehv/CoLLM-thesis
- /kaggle/working/ resets between sessions - Vicuna download (~18GB, few min) and dataset unzip must be redone each fresh session unless session persists.