# Kaggle runbook: V-B2 warm/cold export

All commands are CPU/GPU notebook cells run from the repo root. The CoLLM
export is **single-process**: one Python process, one visible GPU. Do not use
`torchrun`, and do not set `run.distributed=True` (the exporter rejects it).

## Environment variables

| Variable | Must point to |
|---|---|
| `COLLM_ML1M_DIR` | Directory containing `train_ood2.pkl`, `valid_ood2.pkl`, `test_ood2.pkl` (the extracted `collm-datasets/ml-1m.zip` layout). |
| `COLLM_LLAMA_DIR` | Merged Vicuna-7B v0 weights directory (`config.json`, tokenizer, `pytorch_model-*.bin`). |
| `COLLM_MF_REC_PTH` | Trained MF checkpoint, e.g. `weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth`. |
| `COLLM_STAGE2_CKPT` | **Merged LoRA** stage-1 + stage-2 checkpoint used by `train_configs/collm_eval_mf_ood.yaml`. |
| `COLLM_MICA_STAGE2_CKPT` | **Frozen merged MiCA** checkpoint `merged_mica_s12.pth` only. No unfrozen warm-start checkpoint is supported. |
| `COLLM_OUTPUT_DIR` | Writable output root. Predictions/partials go under `${COLLM_OUTPUT_DIR}/vb2_out`. |
| `REPO` | Repo root, e.g. `/kaggle/working/CoLLM`. |
| `COLLM_REQUIRE_OFFICIAL_PARITY` | Test-only; set to `1` in the unit-test cell so a missing official `uAUC_me` import fails instead of skipping. |

Optional knobs: `COLLM_DEVICE` (default `cuda:0`), `COLLM_CHECKPOINT_EVERY`
(default `20` batches), `COLLM_OVERWRITE=1` (restart both CoLLM exports), and
`COLLM_NUM_WORKERS` (default `0`). `CUDA_VISIBLE_DEVICES=0` keeps the export on
one GPU; `COLLM_DEVICE` must index that visible device.

If `COLLM_STAGE2_CKPT` / `COLLM_MICA_STAGE2_CKPT` are not merged files, create
them first with `merge_stage_ckpts.py`; the exporter does not merge stage
checkpoints itself.

## Cell order

### 0. Environment and GPU sanity

Adjust `CONDA_SH` if Miniconda is installed somewhere else.

```bash
%%bash
set -euo pipefail
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate minigpt4
cd "${REPO:-/kaggle/working/CoLLM}"
python -c "import torch, transformers, peft, omegaconf; print('torch', torch.__version__); print('transformers', transformers.__version__); print('peft', peft.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected: `cuda True 1` (or more, but the exporter uses one process/one device).

### 1. Set paths

Use `%env` so later shell cells inherit the values. Replace every path with the
actual Kaggle input mount.

```python
%env REPO=/kaggle/working/CoLLM
%env CONDA_SH=/opt/conda/etc/profile.d/conda.sh
%env COLLM_ML1M_DIR=/kaggle/input/collm-datasets/ml-1m/ml-1m
%env COLLM_LLAMA_DIR=/kaggle/input/vicuna-v0/7b_v0
%env COLLM_MF_REC_PTH=/kaggle/input/collm-weights/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth
%env COLLM_STAGE2_CKPT=/kaggle/input/collm-weights/merged_lora_s12.pth
%env COLLM_MICA_STAGE2_CKPT=/kaggle/input/collm-weights/merged_mica_s12.pth
%env COLLM_OUTPUT_DIR=/kaggle/working/collm-eval
%env COLLM_DEVICE=cuda:0
%env CUDA_VISIBLE_DEVICES=0
```

### 2. Full unit test suite (must not skip official `uAUC_me` parity)

`COLLM_REQUIRE_OFFICIAL_PARITY=1` turns a missing official-import into a test
failure. The parity test must run, not be skipped, on Kaggle. The real-data test
also runs because `COLLM_ML1M_DIR` is set.

```bash
%%bash
set -euo pipefail
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate minigpt4
cd "${REPO:?}"
COLLM_REQUIRE_OFFICIAL_PARITY=1 python -m unittest discover -s warm_cold/tests -v
```

If you see `skipped 'official uAUC_me not importable...'`, fix the environment
before continuing. The official function lives in
`minigpt4/tasks/rec_base_task.py` and needs the repo's `omegaconf`/`transformers`
stack.

### 3. MF prediction + evaluation gate

This is the local/CPU anchor: it must print `PASS acceptance gate: MF`.
`results.csv` gets the per-subset user counts; `results.json` records the
threshold source and both primary and official UAUC values.

```bash
%%bash
set -euo pipefail
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate minigpt4
cd "${REPO:?}"
OUT="${COLLM_OUTPUT_DIR:?}/vb2_out"
mkdir -p "$OUT"
python -m warm_cold.predict \
  --model mf \
  --data-dir "$COLLM_ML1M_DIR" \
  --mf-ckpt "$COLLM_MF_REC_PTH" \
  --out "$OUT/pred_mf.csv"
python -m warm_cold.evaluate \
  --data-dir "$COLLM_ML1M_DIR" \
  --pred "MF=$OUT/pred_mf.csv" \
  --out-dir "$OUT" \
  --dump-masks
```

### 4. CoLLM-LoRA export

Uses `train_configs/collm_eval_mf_ood.yaml` and `COLLM_STAGE2_CKPT`. Partial
predictions are written to `pred_collm_lora.partial.csv` every 20 batches. If the
session dies, rerun this same cell; it resumes from the last completed batch.
Use `--overwrite` only when starting over.

```bash
%%bash
set -euo pipefail
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate minigpt4
cd "${REPO:?}"
OUT="${COLLM_OUTPUT_DIR:?}/vb2_out"
python -m warm_cold.predict \
  --model collm-lora \
  --cfg-path train_configs/collm_eval_mf_ood.yaml \
  --data-dir "$COLLM_ML1M_DIR" \
  --out "$OUT/pred_collm_lora.csv" \
  --checkpoint-every 20 \
  --num-workers 0 \
  --device "${COLLM_DEVICE:-cuda:0}"
```

### 5. CoLLM-MiCA export (frozen merged checkpoint only)

Uses `train_configs/collm_mica_eval_mf_ood.yaml` and
`COLLM_MICA_STAGE2_CKPT=merged_mica_s12.pth`. The exporter also rejects any
checkpoint whose basename is not `merged_mica_s12.pth`; it asserts
`arch == mini_gpt4rec_mica`, `lora_config.mica == true`, and that every `lora_B`
parameter is nonzero after the SVD rebuild. It prints the patched-module count.
If any assertion fails, stop and fix the checkpoint/config; do not silently
continue.

```bash
%%bash
set -euo pipefail
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate minigpt4
cd "${REPO:?}"
OUT="${COLLM_OUTPUT_DIR:?}/vb2_out"
python -m warm_cold.predict \
  --model collm-mica \
  --cfg-path train_configs/collm_mica_eval_mf_ood.yaml \
  --data-dir "$COLLM_ML1M_DIR" \
  --out "$OUT/pred_collm_mica.csv" \
  --checkpoint-every 20 \
  --num-workers 0 \
  --device "${COLLM_DEVICE:-cuda:0}"
```

### 6. Evaluate all three models

Each prediction CSV is validated for full coverage, row IDs, labels, and finite
logits before scoring. Acceptance gates print `PASS`/`FAIL` and the process exits
non-zero on a gate failure. CoLLM gates check overall AUC; MF additionally checks
warm/cold AUC.

```bash
%%bash
set -euo pipefail
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate minigpt4
cd "${REPO:?}"
OUT="${COLLM_OUTPUT_DIR:?}/vb2_out"
python -m warm_cold.evaluate \
  --data-dir "$COLLM_ML1M_DIR" \
  --pred "MF=$OUT/pred_mf.csv" \
  --pred "CoLLM-LoRA=$OUT/pred_collm_lora.csv" \
  --pred "CoLLM-MICA=$OUT/pred_collm_mica.csv" \
  --out-dir "$OUT" \
  --dump-masks
```

### 7. Inspect outputs

```bash
%%bash
set -euo pipefail
cd "${REPO:?}"
OUT="${COLLM_OUTPUT_DIR:?}/vb2_out"
ls -lh "$OUT"
cat "$OUT/results.csv"
python - <<'PY'
import json, os
out = os.path.join(os.environ["COLLM_OUTPUT_DIR"], "vb2_out")
with open(os.path.join(out, "results.json")) as f:
    data = json.load(f)
print("threshold", data["threshold"], "source", data["threshold_source"])
print("acceptance", data["acceptance"]["passed"])
for name in data["models"]:
    m = data["models"][name]
    print(name, "overall", m["overall_auc"])
PY
```

## Resume/restart rules

- Re-run an export cell after an interruption: it validates the partial prefix,
  requires it to end on a completed batch boundary, skips those batches, and
  continues.
- `--overwrite` deletes the final CSV, partial CSV, and metadata before
  exporting.
- The final CSV is written atomically only after all rows are present, then
  re-read with full-coverage validation.
- `run_all.sh` runs MF, LoRA, MiCA, and the final evaluation in the same order.
