# Q/A — Session Documentation

## Q1 — Vicuna-7B weights provenance (PandaGPT vs FastChat apply_delta)

### Question
> Verify exactly how Vicuna-7B weights were obtained for the CoLLM implementation. Inspect `PrepareVicuna.md` (delta + LLaMA + FastChat `apply_delta` workflow) versus the `mvsoom/pandagpt-vicuna-v0-7b` download via `snapshot_download(allow_patterns="pretrained_ckpt/vicuna_ckpt/7b_v0/*")`. Determine what `7b_v0/` contains (base vs delta vs merged), where it came from, whether `snapshot_download` merges/converts/quantizes, and whether the statement "I did not run FastChat/apply_delta myself because I obtained an already-prepared Vicuna v0-7B checkpoint from the PandaGPT repository" is accurate. Separate verified facts from inference, for v0-7B only, including CoLLM reproducibility implications. Do not infer from filenames alone.

### Answer
**Verified facts:**
- Local `PrepareVicuna.md` requires: LLaMA base + `lmsys/vicuna-7b-delta-v0` + `pip install FastChat@v0.1.10` + `python -m fastchat.model.apply_delta --base ... --target ... --delta ...` to produce the final working weight.
- `lmsys/vicuna-7b-delta-v0` model card states: "This 'delta model' cannot be used directly. Users have to apply it on top of the original LLaMA weights."
- `FastChat@v0.1.10:fastchat/model/apply_delta.py` does `param.data += delta.state_dict()[name]` in `torch.float16`, then `save_pretrained(target)`; no quantization.
- `mvsoom/pandagpt-vicuna-v0-7b:pretrained_ckpt/vicuna_ckpt/7b_v0/` contains 13.5 GB full-model files: `config.json`, `generation_config.json`, `pytorch_model-00001-of-00002.bin` (9.98 GB), `pytorch_model-00002-of-00002.bin` (3.5 GB), `pytorch_model.bin.index.json`, `special_tokens_map.json`, `tokenizer.model`, `tokenizer_config.json`. Pickle imports show `torch.HalfStorage` (FP16).
- PandaGPT `pretrained_ckpt/README.md` (identical in GitHub `yxuansu/PandaGPT` and HF mirror) defines `./vicuna_ckpt/7b_v0/` as the `--target` output of `apply_delta`, not the delta input. PandaGPT `code/model/openllama.py` loads it directly via `LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)` with no runtime merge, same as CoLLM `minigpt4rec_v2.py:115-129`.
- `snapshot_download()` per `huggingface_hub` docs only downloads/caches via `hf_hub_download()` internally; `allow_patterns` filters, `local_dir` changes destination. No merge, conversion, or quantization.
- `mvsoom` is a third-party single-commit (`3133fdb` "Upload folder using huggingface_hub") mirror, not `lmsys`/`yxuansu`/`openllmplayground`. Delta vs mvsoom configs/indexes differ (`bos/eos/pad`, `_name_or_path`, `transformers_version` 4.28 vs 4.32, `total_size` 13476855808 with `rotary_emb.inv_freq` vs 13476847616 without), proving they are not byte-copies, though filenames/sizes alone are indistinguishable.

**Conclusion:**
```text
Original LLaMA-7B HF + lmsys/vicuna-7b-delta-v0
      ↓ FastChat apply_delta (param += delta, fp16)
Final working Vicuna v0-7B  <-- mvsoom 7b_v0 corresponds HERE
```
The download is FP16 full-model at the merged-output stage. INT8/INT4/NF4 are load-time quantizations (e.g. CoLLM `load_in_8bit`), not present here.

**Statement verdict:** Mostly supported but imprecise on provenance. Accurate rewrite: "I did not run `apply_delta` myself. I `snapshot_download`ed `pretrained_ckpt/vicuna_ckpt/7b_v0/*` (FP16 full model) from third-party mirror `mvsoom/pandagpt-vicuna-v0-7b` (commit `3133fdb`) at the conventional `apply_delta --target` path and loaded it directly as `model.llama_model`. No merge/conversion/quantization was performed by the download. Byte-identity with a locally merged output was not independently verified."

**Reproducibility:** Method-correct for CoLLM (expects merged Vicuna), but exact reproduction caveat remains (third-party merge, transformers version drift, no hash proof). For thesis, disclose repo + commit + patterns + this note; strictest reproduction re-runs `apply_delta`.

---

## Q2 — Hyperparameter/configuration history forensics

### Question
> For every important training configuration trace initial value, every change, final runtime value, exact reason with source (commit diff/message, log, error/OOM log, code, paper/README), and grade reason as explicitly documented / strongly supported / inference / unknown. Cover max_epoch (200→80), batch sizes (train 16→4, eval 64→16→8→4, OOM/gradient-checkpointing/single-GPU/Vicuna-FP16/train-vs-eval memory), LR, weight decay, LoRA (r/alpha/targets/dropout/freeze_lora/freeze_proj), MF (embedding_size/LR/WD/batch/patience/epochs/negatives), sequence/prompt (max length/history/truncation), memory (grad-checkpoint/AMP/single-vs-distributed/low_resource/device_8bit/quant), checkpointing/eval (save freq/eval freq/save-best/resume/patience). Classify each as PAPER/ORIGINAL, HARDWARE-DRIVEN, DEBUGGING/STABILITY, EXPERIMENTAL, REPRODUCTION/COMPATIBILITY, or UNKNOWN. Do not assume final value means optimal; distinguish "used" vs "selected because better." Produce table + narrative answering what was paper-dictated vs Kaggle-driven vs experimental.

### Answer
**Method:** `IMPLEMENTATION_REPORT.md §§1-6`, `MEMORY.md`, `git log -p` for `collm_pretrain_mf_ood.yaml`/`baseline_train_mf_ood.py`/`minigpt4rec_v2.py`/`runner_base.py`, final YAMLs, `raw_logs/` (Phase 1 MF, Phase 2/3 official logs). Logs prove final runtime values only; OOM narratives come from commit messages/report, no raw OOM log survives (per report §1 caveat).

| Parameter | Initial | Changes | Final | Reason / Evidence | Classification |
|---|---|---|---|---|---|
| `max_epoch` (LLM) | 200 (`66ddd4d`) | 200→80 in `5805f4d` | 80 | Budget/quota (report §4.1). Commit msg silent. Logs: bests @10 (S1), @28 (S2), @64 (MiCA-S1); 80 is ceiling never hit for baseline. Do not claim optimal. | HARDWARE-DRIVEN (explicit in report) |
| `batch_size_train` | 16 | 16→4 in `5805f4d` | 4 | CUDA OOM single-T4 7B-fp16 (report §4.2; `b68d8cd` "fix OOM"). | HARDWARE-DRIVEN (strongly supported, no OOM log) |
| `batch_size_eval` | 64 | 64→16 (`5805f4d`) →8 (`b68d8cd`) →4 (`ee4d3e8`) | 4 | Same OOM; eval still OOM at 8 (report §4.2; `ee4d3e8` msg). | HARDWARE-DRIVEN (explicit) |
| `init_lr/min_lr/warmup_lr/sched/warmup_steps` | 1e-3/8e-5/1e-5/`linear_warmup_cosine_lr`/200 | none numerically | same (logs `init_lr:0.001`) | Inherited; reason undocumented. | PAPER/ORIGINAL, reason unknown |
| `weight_decay` (LLM) | 1e-3 (`#0.05` alt commented) | comment-only | 1e-3 | Inherited; why not 0.05 unexplained. | PAPER, unknown |
| LoRA `r/alpha/targets/dropout/use_lora` | 8/16/`[q_proj,v_proj]`/0.05/True | none numerically | same | Paper method; alternatives only in comments, never run for baseline. | PAPER, unknown |
| `freeze_rec/proj/lora` | `66ddd4d` proj False (mislabeled); authors fixed in `2aab75c` to S1 True/False, S2 False/True | retained | S1 True/True/False, S2 True/False/True | Paper division of labor (report §5.1). | PAPER |
| MF `embedding_size/lr/wd` | 256/1e-3/1e-4 (grid + filename `d256lr-0.001wd0.0001`) | grid reduced to single point, values same (`6f793e3`) | same | Paper known-good; no sweep. "Used" ≠ "proven better". | PAPER |
| MF `batch_size` | 1024 | 1024→2048 (`6f793e3`) | 2048 | "Faster on small ML-1M" (report §3.2). No speed A/B log. | REPRODUCTION/COMPATIBILITY (explicit) |
| MF `patience` | 50 | 50→100 (`6f793e3`) | 100 (stop @567, best @467; 567−467=100) | Best lands late (log proves lateness); "50 would miss it" is inference (no 50-run). | DEBUGGING/STABILITY (lateness strongly supported) |
| MF `epoch/eval_epoch` | 5000/1 | none | same | Ceiling never reached. | PAPER |
| MF negatives/threshold | BCE + `train_ood2.pkl[label]` | none | same | From preprocessing; not tuned here. | UNKNOWN (inherited data) |
| `max_txt_len` + truncation | 1024 + `truncation,max_length` (`v2.py:417,534`) | none | 1024 | Inherited; why 1024 undocumented. | PAPER, unknown |
| `prompt_path` | `rec_alignment.txt` | → tallrec/collm in author `2aab75c` | tallrec S1, collm S2 | Paper stage design. | PAPER |
| history length | `min(max_len,10)` (`rec_datasets.py:269,370,484`) | none in fork | 10 | Author choice; why 10 undocumented. | PAPER, unknown |
| `proj_token_num/mid/drop` | 1/10/0 | 4 only in Control (null +0.00003) | 1/10/0 baseline | Paper default; 4 experimental rejected. | PAPER (4 EXPERIMENTAL) |
| `use_grad_checkpoint` | absent/False | +True (`b68d8cd`) | True | OOM fix, numerically lossless vs rejected 8-bit (report §4.2). | HARDWARE-DRIVEN (explicit) |
| `amp`/fp16 | True/`float16` | none | same | Paper default. | PAPER |
| `distributed/world_size/device` | True/1/cuda | True→False (`5805f4d`) | False/1/cuda ("Not using distributed") | DDP replicates 7B, no memory saving + NCCL overhead (report §1.3). | HARDWARE-DRIVEN (explicit in report) |
| `low_resource/device_8bit` | defaults False/0 | never set in MF YAMLs | False/0 | Considered/rejected bnb0.37 fragile + mixes numerics (report §4.2). | PAPER (stay fp16, rejection explicit) |
| `empty_cache` | commented | uncommented (`ee4d3e8`) | active | OOM hygiene. | HARDWARE/DEBUGGING (explicit) |
| `early_stop_epochs` (LLM) | absent (`if not_change>20`) | +20 (`69d369f`) →10 (`a78b2ac`) | 10 evaluated epochs (~20 real, `eval_freq=2`) | Preserve ~20-epoch patience under `eval_freq=2` (msg "Keep patience equivalent"). | REPRODUCTION/COMPATIBILITY (explicit) |
| `eval_freq` | every epoch | +2 (`6fdbbe8`) | 2 | 1300-batch validation dominant cost; 8.5h→5h08m (report §4.9). | HARDWARE-DRIVEN (explicit) |
| `save_ckpt_freq/resume` | absent | +20 + resume (`69d369f`) | 20 | 9h cap + `/kaggle/working` wipe; lost 8.5h run (report §4.8-4.9). | HARDWARE-DRIVEN (explicit) |
| ckpt filter | saved 13 GB backbone (bug) | `f3f3ab2` keep only `requires_grad` + exclude non-lora `llama_model` | 17 MB LoRA / 44 MB CIE | peft `state_dict` vs `named_parameters` mismatch (report §4.7). | DEBUGGING/STABILITY (explicit) |
| auto-sync | absent | added then abandoned (`6fdbbe8`, report §4.9) | manual backup | Kaggle API unreliable. | HARDWARE, negative result |
| `iters_per_epoch/log_freq/workers/seed` | 50/absent/4/42 | +`log_freq:50` | 50/50/4/42 | Throttle; rest inherited unknown. | `log_freq` REPRODUCTION, rest PAPER unknown |

**Narrative:** All paper numerics retained (zero numeric changes in fork for LR/WD/LoRA/sequence/MF-core). Changes cluster into Kaggle T4/quota/persistence (batches, 80 ceiling, checkpoint/eval cadence, MF batch/patience, stability fixes) plus parallel experimental variants (new files only, baseline untouched). "Used" does not imply "proven better" (MF 2048, patience 100, epoch 80 have no A/B proof).

---

## Q3 — Early stopping, patience, eval_freq, max_epoch implementation

### Question
> Trace actual Python code (not names/YAML) for MF (`patience:100, eval_epoch:1, epoch:5000`) and LLM S1/S2 (`max_epoch:80, eval_freq:2, early_stop_epochs:10`): where read/stored, what increments counter, units (epochs vs evals), `eval_epoch`/`eval_freq` effect, exact stop condition, `max_epoch` interaction, why MF stopped at 567 with best 467/100. Quote snippets with paths/lines. Are `patience` vs `early_stop_epochs` same concept? Verify history (when introduced, original logic, 20→10, `eval_freq=2`, "Keep patience equivalent" meaning). Provide comparison table + S1 timeline with best at epoch 20.

### Answer
**MF — `baseline_train_mf_ood.py:82-127,208,281,292,318,327-333`:**
```python
early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])  # :208
# update :94-109: strict `>` resets reach_count=0 else reach_count+=1
def is_stop(self): return self.reach_count>=self.patience  # :123-127
for epoch in range(start_epoch, train_config['epoch']):  # :281 0..4999 ceiling
    if epoch%train_config['eval_epoch']==0:  # :292 =1 → every epoch
        valid_auc=roc_auc_score(...); updated=early_stop.update({...})  # :303,318
        if early_stop.is_stop(): stop_reason="early_stop ..."; break  # :327-332
```
Monitored: `valid_auc`; improvement = strict `>`. With `eval_epoch=1`, 100 = 100 epochs = 100 evals (emergent 1:1, no conversion). `epoch=5000` is loop bound only. Stop 567 = 467+100; log shows `early_count:93` at 560 and `best epoch 467`.

**LLM — `minigpt4/runners/runner_base.py:382-385,404,416-417,432-446,472-475`:**
```python
early_stop_epochs=get("early_stop_epochs",20)  # :385
for cur_epoch in range(start_epoch,max_epoch):  # :404 0..79
  eval_freq=get("eval_freq",1)  # :416
  if valid_splits and cur_epoch%eval_freq==0:  # :417 evals at 0,2,4...
    if agg_metrics>best and split=="valid": best=...; save_best; not_change=0  # :432-437
    not_change+=1  # :446 unconditional, even on new best
  if not_change>early_stop_epochs: break  # :472-475
```
Monitored: `val_log["agg_metrics"]` = valid AUC (`rec_base_task.py:311-312`, else `-loss`). Strict `>`. With `eval_freq=2`, `early_stop_epochs=10` = 10 non-improving validations ≈20 real epochs (needs `not_change=11` due to always-increment). `max_epoch=80` is loop bound + default `stop_reason="max_epoch reached"` (`:387,484-489`).

**Same concept?** Same idea (consecutive non-improving validations, strict `>` on valid AUC) but different classes/names/operators (`>=` vs `>`+always-increment). 100 vs 10 incomparable (MF validates every epoch cheaply; LLM validates every 2nd epoch at ~1300 batches).

**History:** Originally hardcoded `if not_change>20:break` (eval every epoch). `69d369f` introduced YAML `early_stop_epochs`+`save_ckpt_freq` (20). `6fdbbe8` added `eval_freq` gate (validation dominant cost). `a78b2ac` "Keep patience equivalent under eval_freq=2" halved 20→10 to preserve ~20 real epochs (20×1 vs 10×2). No code conversion; manual equivalence.

| Training | Parameter | Unit | Counts | Eval freq | Stop |
|---|---|---|---|---|---|
| MF | `patience` | evals (=epochs here) | non-improving `valid_auc` (`reach_count`) | every epoch | `>=100` (467→567) |
| LLM S1/S2 | `early_stop_epochs` | validations | non-improving valid AUC (`not_change`) | every 2nd epoch | `>10` (11 incl. best ≈20 epochs) |

**S1 timeline (best @20, `eval_freq=2`, `early=10`):** evals 0,2,...,20(BEST→1),22(2),24(3),26(4),28(5),30(6),32(7),34(8),36(9),38(10),40(11→break). Stop 20 real epochs after best.

---

## Q4 — finetune/eval YAML origins and OOD meaning

### Question
> Why does fork have `collm_finetune_mf_ood.yaml` if upstream lacks it? Why separate `collm_eval_mf_ood.yaml`? Where does `ood`/`ood2` in `baseline_train_mf_ood.py`, `collm_*_mf_ood.yaml`, `train_ood2.pkl` etc. come from and what does OOD mean? Do not infer from filenames; trace code/history/preprocessing/upstream. Distinguish original vs fork vs inference. Establish upstream commit, per-file existence, introducing commit/problem/motivation/rename, Stage 2 mechanics (freeze→optimizer→forward→loss→backward, trainable params), whether upstream used same-config/manual-changes/other-config/overrides, algorithm vs organization difference, eval path (`evaluate=True`/ckpt/freeze/loop/metrics/optimizer/grads/merged ckpt/training vs inference/why reuse S2 freezes), paper→repo→fork→Phase mapping fidelity, full config diff with execution effects, chronological timeline with hash/date/msg/files/concept, marking undocumented reasons.

### Answer
**Upstream base:** local `66ddd4d` (Yang Zhang 2023-11-19 `init`), remote now `rahelehv/CoLLM-thesis`. Init already had `baseline_train_mf_ood.py` and `collm_pretrain_mf_ood.yaml`, plus `processing_ood.ipynb`; **no** `collm_finetune_mf_ood.yaml` or `collm_eval_mf_ood.yaml` (only `collm_pretrain_*`, `personlized_*`, `soft_prompt_*`). Upstream Stage 2/3 was manual edits to the same pretrain file per `README.md:167-181` (`freeze_proj False, freeze_lora True, pretrained_path, collm_movie.txt, ckpt`) then "run same command"; eval per `:186-192` (`ckpt` + `evaluate:True`).

**Fork additions (verified diffs):** `d783524` 2026-08-13 "Add Stage 2 (CIE) config..." created `collm_finetune_mf_ood.yaml` (copy with 5-line S2 switch + env-vars + `output_dir collm-stage2`); `e1ee937` 2026-08-14 "Add Phase 4 evaluate-only config..." created `collm_eval_mf_ood.yaml` (merged ckpt + `evaluate:True` + `output_dir collm-eval`). Separation is **configuration convenience + distinct outputs/ckpt wiring**, not paper/algorithm change. Algorithm identical.

**Stage 2 mechanics (`minigpt4rec_v2.py`):** `from_config:1123,1130-1131` → `__init__:102-106` MF frozen/eval, `:131-132` base always frozen, `:160-163` LoRA frozen when `freeze_lora True`, `:201-206` skipped when `freeze_proj False` leaving `llama_proj=Linear(256→2560)→ReLU→Linear(2560→4096)` (`:179-184`) trainable; `ckpt` loaded `strict=False` (`:1175-1178`) then MF reloaded if `freeze_rec` (`:1180-1181`); optimizer (`runner_base:107-109`) sees only `requires_grad` (i.e. `llama_proj`, ~44 MB/4 keys vs 17 MB LoRA S1); `forward_v2:726-747` BCE on Yes-logit; backward updates only `llama_proj`.

**Eval path:** `evaluate=True` → `runner_base:331-336 evaluate_only` → skips training loop (`:403`), runs `evaluate(test_splits)` → `eval_epoch:527 @torch.no_grad()` → `rec_base_task:224-323` gathers Yes-logits/labels/UserIDs (same as AUC/UAUC inputs, both `all_gather` and single-process branches) → AUC/UAUC/NDCG. No optimizer step, no grads, trains nothing. Expects merged LoRA+CIE (`merge_stage_ckpts.py:a6839c0`) because stage ckpts are disjoint. S2 freezes reused for identical construction semantics (irrelevant under `no_grad`). Not a paper stage.

**OOD (verified `dataset/ml-1m/processing_ood.ipynb`, `rec_datasets.py:220`, `rec_pair_builder.py:94`):** Temporal OOD: `time=(year-min)*12+month`, `train 14-24, valid 24-29, test 29-34` (cells 15,20), label `rating>=4` (cell 16), history via `deal_with_each_u` + `flag -1/0/1`. `filter_cold_start` defined (cell 23) but commented out (cell 24; `valid_f=valid` cell 22), so cold retained; `not_cold` (cell 42), `warm` (>3 train rows, `read_info:2-11`), `cold` (uid+iid unseen), `test_warm_cold_ood2.pkl` are subsets/flags inside OOD. **OOD ≠ cold.** `ood2` version: only `*_ood2.pkl` exist; no `ood`/`ood1` in code/history; why `2` is **not determined** (inference: second split version).

**Mapping:** Paper (a)MF→(b)S1 LoRA text-only→(c)S2 CIE→Table II eval; upstream = MF scripts + one YAML hand-edited; fork Phase1/2/3/4 naming (`IMPLEMENTATION_REPORT §1.2`) is fork organization, faithful semantically.

**Config diff:** Only `freeze_proj/lora`, `prompt_path` (tallrec vs collm), `pretrained_path` (`not_have` vs MF), `ckpt` (none vs S1 vs merged), `evaluate/output_dir` differ; rest identical. Effects per above; eval training keys marked "irrelevant for evaluate-only".

**Timeline:** `66ddd4d` init OOD+pretrain → author `2aab75c` comments → fork `6f793e3/502bffd` MF env/resume → `5805f4d-b68d8cd-ee4d3e8-69d369f` Kaggle/OOM/early-stop → `d783524` finetune copy → `e1ee937` eval copy → `8d10dc3` warm/cold gate, `67be072` NDCG, `a6839c0` merge. Undocumented file-split motives marked as inference.

---

## Q5 — Stage 2 / CIE execution order for one sample

### Question
> Trace actual Stage 2 execution order from source for one sample to loss/backward: entry command/config/file/function to `forward()`; dataset class/`__getitem__`/fields/MF source shapes; where `collm_movie.txt` loaded/placeholders filled/final prompt example; exact tokenizer call/returns/shapes and whether CIE involved yet; `embed_tokens` line/shapes/`<unk>` vector/dim 4096/grad hook effect; MF→CIE class/function/shapes verifying 256→2560→4096 and `merged_embs`; `prompt_embeds[replaced_idx...]=merged_embs` (`replaced_idx` creation/meaning, `merged_embs` shape/meaning, embedding-vs-ID replacement, before/after model); post-replace `inputs_embeds` flow/`input_ids`/`attention_mask`/`position_ids`; transformer class/layers/LoRA frozen-trainable split; `recprompt_wrap_v2`/Yes-No logits extraction; loss function/inputs/shapes/return; `loss.backward()` grad paths and optimizer updates; validation vs Phase-4 eval splits/purpose/`no_grad`/metrics collection. Build code-verified diagram (do not blindly reuse example). Final YES/NO: does it tokenize→embed normally while separately MF→CIE→4096 vectors inserted at `<unk>` before transformer? Use exact names/lines/snippets/shapes; say "Not verified" if unverifiable; beginner-friendly but technical.

### Answer
**Entry:** `train_collm_mf_din.py --cfg-path=collm_finetune_mf_ood.yaml` (`arch mini_gpt4rec_v2`, freezes True/False/True, collm prompt) → `main():100-101,121-125` task/datasets/model/runner → `runner.train→train_epoch→base_task.train_step:model(samples)["loss"]` → `v2.py:566-569 forward→forward_v2` (v2 mode).

**Sample:** `MoiveOOData:214` (`*_ood2.pkl:220`, history cap 10:269). `__getitem__:273-303` → `{UserID, InteractedItemIDs_pad[10], InteractedItemTitles str, TargetItemID, TargetItemTitle quoted, InteractedNum, label}`. Batch B=4. MF `MatrixFactorization:66-84` tables `[users,256]/[items,256]` from Phase-1 ckpt (`v2.py:95-96`), frozen (`:102-106`).

**Prompt (`recprompt_wrap_v2:493-521`):** template loaded `:221-226`; `<UserID>→<unk>`, `<TargetItemID>→<unk>`, `<ItemIDList>→k×<unk>` (k=`InteractedNum`), titles filled. Example: `<s>#Question: ... "Toy Story", "Jumanji". ... feature <unk>. ... "Grumpier Old Men" with feature <unk>? ...` plus `<unk>,<unk>` history slots. Order via `get_ids_order:17-25`.

**Tokenize (`:529-537`):** `LlamaTokenizer` left-pad, `max_length=1024`, returns `input_ids [4,S]/attention_mask [4,S]`. CIE not involved — **text→IDs only**.

**Embed (`:544,551`):** `replaced_idx=nonzero(input_ids==unk_id)`; `prompt_embeds=embed_tokens(input_ids).clone() [4,S,4096]` (`modeling_llama:427 Embedding(vocab,4096)`). `<unk>` rows initially normal unk vectors. Hook `:140-142` makes output grad-leaf for checkpointing; `.clone()` avoids in-place-leaf error while preserving CIE grad.

**CIE (`encode_recdata_v2:311-390`, called first in `prompt_based_encode_v2:644-648`):** `llama_proj:179-184` verified `Linear(256→2560)→ReLU→Linear(2560→4096)` (hidden 4096). Inputs `[4,1,256]/[4,10,256]` → per-ID `[*,1,4096]` → ordered concat (`:347-349`), non-pad mask (`:350-356`) → `merged_embs [N_valid,4096] (N_valid=user+nonpad-history+target) (:373)`. 4096 is embedding dim, not vocab.

**Replace (`:554`):** `prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]]=merged_embs`. `replaced_idx [N_valid,2]` = (batch,seq) unk slots; counts match `merged_embs`. Replaces **vectors**, not IDs. Before: `[tok,tok,unk-row,...]`; after: `[tok,tok,CIE(user),...]`. Confirmed.

**Final (`:561,694,722-724`):** `prompt_embeds` returned as `sample_embeds`; `inputs_embeds=cat([sample_embeds, answer_embeds])`, `attention_mask` likewise; only `inputs_embeds/attention_mask/labels` passed to transformer (`:735-740`). `position_ids` inside Llama stack **not traced**.

**Transformer:** vendored `modeling_llama.LlamaForCausalLM` (32 decoders) via `llama_model_lora` (LoRA r8/α16 q/v wrappers active but frozen S2). Trainable: only `llama_proj`; frozen: base, LoRA (`:160-163`), MF. Optimizer (`runner_base:107-109`) steps only `requires_grad`.

**Yes/No (`forward_v2:702-747`, eval `760-883` same):** `ans Yes/No` (`set_mode:273-274`), `logits=outputs.logits[:,-t_posi][:,Yes_id] [4]` (`:746,875` via `recprompt_wrap_v2` path + `generate_for_samples:1094-1098`).

**Loss (`:747,877`):** `BCEWithLogits(logits [4], label.float [4])`, returns `{"loss":loss}` (+`logits` in eval).

**Backward (`base_task:227-240`):** loss→frozen transformer→cloned prompts→`<unk>` slots→`llama_proj`. Updates **only `llama_proj`**; LoRA/base/MF excluded.

**Eval:** same forward via `valid_step→generate_for_samples_v2` under `runner.eval_epoch:527 @torch.no_grad()`, no backward/optimizer, gather logits/labels/users → `rec_base_task:292-294,311-316` AUC/UAUC/NDCG. S2 validation (every 2nd epoch, `valid_small`) for early stopping vs Phase-4 once over `test+valid` with merged ckpt.

**Diagram (verified):**
```text
__getitem__ batch [4] → prompt string (<unk> slots) → tokenizer → input_ids [4,S]
  ↓ embed.clone [4,S,4096] (+hook)  +  MF[256] → llama_proj 256→2560→4096 → merged [N,4096]
  ↓ replace unk rows → sample_embeds + answer embeds → inputs_embeds
  ↓ Llama + frozen LoRA → Yes-logit [4] → BCE vs label → backward → only llama_proj
Eval: same to logits_, no_grad → AUC/UAUC/NDCG
```

**Final:** `YES` — tokenize→normal embeddings and separate frozen-MF→trainable-CIE→4096 vectors inserted at `<unk>` before transformer, per lines above.

---

## Q6 — NDCG audit (paper vs dormant vs active, Table II)

### Question
> Audit whether Phase-4 NDCG matches paper/Table II NDCG. Inspect paper (formula, @K, K, ranking scope, negatives, candidates, per-user, aggregation, relevance, filtering; search whole paper, say "not specified" if absent), dormant `rec_base_task_ndcg.py` (`u_dcg`/`compute_dcg`/helpers/inputs/grouping/sort/DCG/IDCG/K/ties/single-class/single-sample/aggregation + math), active `rec_base_task.py` after `67be072` (both distributed/single branches, collected preds/labels/users, grouping timing, same samples as AUC, filtering), dataset→eval sample meaning (one interaction? TargetItemID? binary label? negatives meaning? per-user multiples? all included? top-K/candidate/NDCG sampling?), comparison table (Paper vs dormant vs active), Table II traceability (scripts/configs/logs/README/commits/filenames/hard-code/alternative impl; search NDCG/dcg/Table II/0.84/NDCG@/topk/candidate/exposed), dormant-file authorship (A used for Table II vs B merely present; commit `74df7e5`/surroundings/imports/configs), exact math (DCG/IDCG, all vs @K, binary vs graded, macro vs micro), formula vs protocol distinction, final Option A/B/C with known/unknown, numeric comparability + 2–4 sentence thesis wording. Do not guess; cite paths/lines.

### Answer
**Paper (`docs/CoLLM_paper.pdf`, pdftotext verified):** Metrics para (V-A1): AUC/UAUC [49] + NDCG [50]; UAUC = per-user AUC over exposed items averaged; NDCG = "Normalized Discounted Cumulative Gain metrics" only, user-level. Ref [50] Järvelin/Kekäläinen. **Unspecified:** formula, `NDCG@K`/K (Table II is image, no `@K` in text), candidate set, negative sampling, top-K generation, per-user grouping/aggregation for NDCG, tie handling, filtering. Binary relevance specified globally (`(u,i,y) y∈{0,1}`, ML-1M threshold 3 i.e. ≥4, Amazon 4). Splits temporal (ML-1M last 20mo 10/5/5). `exposed` appears only for UAUC.

**Dormant `rec_base_task_ndcg.py:36-92`:** `u_dcg:36-46` returns -1 if `<2` labels or single class; else `argsort(-predict)`, `flag=1/log2(arange(n)+2)`, `dcg=(ranked_label*flag).sum()`, `idcg=flag[:pos_num].sum()`. `compute_dcg:48-92` groups by user (`np.unique`), skips `n==1`, keeps `>=0`, returns macro mean. Math: `DCG_u=Σ rel_{π_i}/log2(i+1)`, `IDCG_u=Σ_{i≤P_u}1/log2(i+1)`, `NDCG=mean kept NDCG_u`. No K (full-list), binary, deterministic `argsort` ties **not specified**.

**Active `rec_base_task.py:69-123` after `67be072`:** Verbatim port (diff proves). Both branches (`:287-289` all_gather, `:292-294` single-process now `distributed=False`) call `compute_dcg` on identical arrays as AUC/UAUC (`users/logits/labels :240-267`). No pre-filter; filtering inside `compute_dcg`. `evaluation:224-323` returns `agg_metrics:AUC` + `uauc/ndcg`.

**Samples:** `MoiveOOData` one row = one `(u,target-y)` + history; `TargetItemID` ranked item; label binary; negatives are real low ratings in window (no sampling code); per-user multiples included (except 33 singles); all split rows ranked; no candidate/top-K/NDCG sampling.

| Aspect | Paper | Dormant | Active |
|---|---|---|---|
| Formula | not specified | binary full-list DCG/IDCG above | identical |
| Per-user | "user-level" only | yes desc-score | yes |
| Candidate/exposed/neg/top-K | not specified (exposed only for UAUC) | all split rows / none / none | same |
| Relevance | binary global | 0/1 | same |
| Aggregation/filter/ties | not specified | macro mean / drop n<2/single-class / unspecified | same |

**Table II:** No scripts/configs/logs/README/alternative impl found (search `NDCG@|topk|candidate` empty; Table II image-only). Reported MF 0.7295/0.6875/0.8714 per report (values themselves not in extractable text).

**Authorship (`74df7e5` 2025-08-03 "ndcg metrics" adds only `rec_base_task_ndcg.py`; never imported — `tasks/__init__.py` + `rec_pretrain.py:10` import only `rec_base_task`): **B. Merely present, not proven used for Table II** (duplicates eval swapping `uAUC→compute_dcg`, likely experiment).

**Verdict: Option B — Partially verified.** Formula + per-user full-list setup match; K/candidates/aggregation/filtering/tie authorship unverifiable. Compare 0.8663 vs 0.8714 only directionally.

**Thesis wording:** "We compute binary full-list macro-averaged per-user NDCG over all test-split interactions ranked by the Yes-logit (1/log2(rank+1), IDCG from positive count; <2-sample/single-class users excluded), ported verbatim from dormant `rec_base_task_ndcg.py`. The paper specifies NDCG only by citation without K/candidate/aggregation details and the file was never wired in, so Table II equivalence cannot be proven; we report directional comparability."
