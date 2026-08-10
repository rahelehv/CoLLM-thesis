# CoLLM

Research code (arXiv 2310.19488): LoRA-tuned LLM (Vicuna-7B v0 by default) + a collaborative recommender (MF / LightGCN / SASRec) whose embeddings are projected into the LLM embedding space. Built on MiniGPT-4/LAVIS. **No tests, no lint, no CI.** All hyperparameters live in `train_configs/*.yaml` — edit the YAML, never the entry scripts.

## Environment (this repo is run on Kaggle, not local)

- Authoritative env: `environment.yml` (Python 3.9, `cudatoolkit`, pytorch 1.12.1, transformers 4.28.0, bitsandbytes 0.37.0). `requirements.txt` is only the pip-section mirror — it does **not** pin torch/python/cudatoolkit.
- Kaggle has no conda by default. Do **not** rely on `pip install -r requirements.txt` — the pinned torch 1.12 + py3.9 + bitsandbytes 0.37 combo won't match Kaggle's preinstalled stack. Instead install Miniconda in the notebook (`wget Miniconda3... && bash Miniconda3-latest-Linux-x86_64.sh -b`), then `conda env create -f environment.yml && conda activate minigpt4`.
- Hardware: 2x T4 (16GB each). Vicuna-7B fp16 is ~13GB+; keep `run.world_size: 2`, `run.distributed: True`, and launch with both GPUs (`CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2`).
- Quota reality: 30 GPU-h/week, ~9h max session. Checkpoint to Kaggle output (`/kaggle/working` resets between sessions) or a dataset you re-attach next session.
- LLM weights: `PrepareVicuna.md` requires merging LLaMA-7B + `vicuna-7b-delta-v0`; the original LLaMA-7B is HF-gated, so **request access first** — it's the longest lead-time step. Alternative (skips the merge): use the already-merged Vicuna-7B v0 weights from `mvsoom/pandagpt-vicuna-v0-7b`, subdir `pretrained_ckpt/vicuna_ckpt/7b_v0/` — it matches the expected file structure (`config.json`, `generation_config.json`, `pytorch_model-*-of-*.bin`, `tokenizer.model`). Point `model.llama_model` at that directory.

## Layout

- `train_collm_*.py` — CoLLM entrypoints (`_mf_din`, `_lgcn`, `_sasrec`). `--cfg-path` selects the config.
- `baseline_train_*.py` — standalone collaborative-model trainers; their `.pth` output becomes `model.rec_config.pretrained_path`.
- `minigpt4/` — LAVIS-style core package: `models/` (`minigpt4rec_v2.py` = main model, `rec_base_models.py` = rec backbones, `modeling_llama.py`), plus `datasets/`, `runners/`, `tasks/`, `common/`.
- `prompts/` — templates. `tallrec_*.txt` = text-only (no IDs, stage 1); `collm_*.txt` = with `<UserID>`/`<ItemIDList>`/`<TargetItemID>` (stage 2).
- `dataset/` = preprocessing ipynb; `collm-datasets/` = preprocessed zips (incl. `ml-1m.zip`).

## Training (two stages, config flags)

Launch: `torchrun --nproc-per-node 2 --master_port=PORT train_collm_mf_din.py --cfg-path=train_configs/<cfg>.yaml`

- Stage 1 (LoRA, text-only): `freeze_rec: True`, `freeze_proj: True`, `freeze_lora: False`, `prompt_path: prompts/tallrec_movie.txt`, `ckpt: None`, `evaluate: False`.
- Stage 2 (CIE, IDs): `freeze_rec: True`, `freeze_proj: False`, `freeze_lora: True`, `prompt_path: prompts/collm_movie.txt`, `ckpt: <stage1 ckpt>`, `rec_config.pretrained_path: <pretrained rec .pth>`.
- Evaluate only: set `ckpt` + `evaluate: True`.

## ML-1M + MF: exact path changes (first pipeline)

Script: `train_collm_mf_din.py`. Config: `train_configs/collm_pretrain_mf_ood.yaml`.

**Gotcha:** the script's default `--cfg-path` (`train_configs/minigpt4rec_pretrain_ood_cc.yaml`) does not exist in this repo — always pass `--cfg-path` explicitly.

Keys in `collm_pretrain_mf_ood.yaml` that must change (authors' scratch paths won't exist on Kaggle):

| Key | Current (author) value | Replace with |
|---|---|---|
| `model.llama_model` | `/data/zyang/LLM/PretrainedModels/vicuna/working-v0/` | the merged Vicuna v0 weights dir (e.g. Pand-GPT `pretrained_ckpt/vicuna_ckpt/7b_v0/`) |
| `model.rec_config.pretrained_path` | `/data2/zyang/minigpt4rec-log/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth` | your trained MF `.pth` from `baseline_train_mf_ood.py`; a nonexistent path raises, unless literally `"not_have"` |
| `datasets.movie_ood.path` | `/data/zyang/datasets/ml-1m/` | dir from `collm-datasets/ml-1m.zip` containing `train_ood2.pkl` / `valid_ood2.pkl` / `test_ood2.pkl` |
| `datasets.movie_ood.build_info.storage` | `/data/zyang/datasets/ml-1m/` | same dir |
| `run.output_dir` | `/data2/zyang/minigpt4rec-log` | Kaggle output/checkpoint dir |

For stage 2 / eval, additionally set `model.ckpt` (the commented key in that file). The same five keys appear in every other `train_configs/*.yaml`; only the ML-1M+MF file is listed here per instruction.

## Gotchas

- `run.mode` must stay `'v2'` (v2 = BCE on the "Yes"/"No" logit; v1 is legacy). `model.arch` must be `mini_gpt4rec_v2`.
- `user_num`/`item_num` in the YAML are placeholders — scripts recompute them from the pickles at startup.
- Dataset classes hard-append `_ood2.pkl` to the storage path; files must be named `train_ood2.pkl`, `valid_small_ood2.pkl`, `test_ood2.pkl` (+`test_warm_cold_ood2.pkl` for warm/cold).
- `minigpt4rec_v2_qwen.py` (used by `0924-*-qwen-*.yaml`, `llama_model: Qwen/Qwen2-1.5B`) registers the same model name as `minigpt4rec_v2.py` but is **not imported** by `minigpt4/models/__init__.py`, so the Qwen configs are not runnable as-is.
- `baseline_train_*.py` have data dirs and hyperparameter grids hard-coded in their `__main__`; parse their logs with `search_result.py`.