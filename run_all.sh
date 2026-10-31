#!/usr/bin/env bash
set -euo pipefail

for name in COLLM_ML1M_DIR COLLM_LLAMA_DIR COLLM_MF_REC_PTH COLLM_STAGE2_CKPT COLLM_MICA_STAGE2_CKPT COLLM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: $name is not set" >&2
    exit 2
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
OUT_DIR="${COLLM_OUTPUT_DIR}/vb2_out"
DEVICE="${COLLM_DEVICE:-cuda:0}"
CHECKPOINT_EVERY="${COLLM_CHECKPOINT_EVERY:-20}"
NUM_WORKERS="${COLLM_NUM_WORKERS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUT_DIR"

COLLM_FLAGS=()
if [[ "${COLLM_OVERWRITE:-0}" == "1" ]]; then
  COLLM_FLAGS+=(--overwrite)
fi

"$PYTHON" -m warm_cold.predict \
  --model mf \
  --data-dir "$COLLM_ML1M_DIR" \
  --mf-ckpt "$COLLM_MF_REC_PTH" \
  --out "$OUT_DIR/pred_mf.csv"

"$PYTHON" -m warm_cold.predict \
  --model collm-lora \
  --cfg-path train_configs/collm_eval_mf_ood.yaml \
  --data-dir "$COLLM_ML1M_DIR" \
  --out "$OUT_DIR/pred_collm_lora.csv" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  ${COLLM_FLAGS[@]+"${COLLM_FLAGS[@]}"}

"$PYTHON" -m warm_cold.predict \
  --model collm-mica \
  --cfg-path train_configs/collm_mica_eval_mf_ood.yaml \
  --data-dir "$COLLM_ML1M_DIR" \
  --out "$OUT_DIR/pred_collm_mica.csv" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  ${COLLM_FLAGS[@]+"${COLLM_FLAGS[@]}"}

"$PYTHON" -m warm_cold.evaluate \
  --data-dir "$COLLM_ML1M_DIR" \
  --pred "MF=$OUT_DIR/pred_mf.csv" \
  --pred "CoLLM-LoRA=$OUT_DIR/pred_collm_lora.csv" \
  --pred "CoLLM-MICA=$OUT_DIR/pred_collm_mica.csv" \
  --out-dir "$OUT_DIR" \
  --dump-masks
