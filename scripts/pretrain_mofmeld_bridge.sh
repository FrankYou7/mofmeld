#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Stage-I pretraining for MOFMeld
# This script preserves the full pretraining workflow used
# in the manuscript and is intended for full reproduction,
# not for lightweight demo execution.
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# -------------------------
# Paths
# -------------------------
LLAMA_PATH="${PROJECT_ROOT}/checkpoints/MOFLLaMA"
EMBED_ROOT="${PROJECT_ROOT}/data/embeddings/qmof_embeddings"
DATA_DIR="${PROJECT_ROOT}/data/mofmeld_pretrain"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/mofmeld_pretrain"
RESUME_CKPT=""

# -------------------------
# Paper training settings
# -------------------------
DEVICE="cuda"
BATCH_SIZE=8
ACCUM_STEPS=32
TOTAL_STEPS=260000
SAVE_EVERY=50000
LR=2e-4
WEIGHT_DECAY=0.05
WARMUP_STEPS=1000
NUM_WORKERS=4
PREDICTION_MAX_LENGTH=64
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Launching Stage-I MOFMeld pretraining"
echo "Project root          : ${PROJECT_ROOT}"
echo "LLAMA_PATH            : ${LLAMA_PATH}"
echo "EMBED_ROOT            : ${EMBED_ROOT}"
echo "DATA_DIR              : ${DATA_DIR}"
echo "OUTPUT_DIR            : ${OUTPUT_DIR}"
echo "RESUME_CKPT           : ${RESUME_CKPT}"
echo "Device                : ${DEVICE}"
echo "Batch size            : ${BATCH_SIZE}"
echo "Accum steps           : ${ACCUM_STEPS}"
echo "Total steps           : ${TOTAL_STEPS}"
echo "Save every            : ${SAVE_EVERY}"
echo "Learning rate         : ${LR}"
echo "Weight decay          : ${WEIGHT_DECAY}"
echo "Warmup steps          : ${WARMUP_STEPS}"
echo "Prediction max length : ${PREDICTION_MAX_LENGTH}"
echo "Num workers           : ${NUM_WORKERS}"
echo "Seed                  : ${SEED}"
echo "=================================================="

if [[ ! -d "${LLAMA_PATH}" ]]; then
  echo "[ERROR] MOFLLaMA directory not found: ${LLAMA_PATH}"
  echo "Please place the MOFLLaMA checkpoint under checkpoints/."
  exit 1
fi

if [[ ! -d "${EMBED_ROOT}" ]]; then
  echo "[ERROR] Embedding root not found: ${EMBED_ROOT}"
  echo "Please prepare the QMOF embedding files under the expected path."
  exit 1
fi

for file in prediction.jsonl correlation.jsonl association.jsonl; do
  if [[ ! -f "${DATA_DIR}/${file}" ]]; then
    echo "[ERROR] Missing pretraining task file: ${DATA_DIR}/${file}"
    echo "Please prepare Stage-I pretraining data under data/mofmeld_pretrain/."
    exit 1
  fi
done

python src/mofmeld/training/pretrain_bridge.py \
  --llama_path "${LLAMA_PATH}" \
  --embed_root "${EMBED_ROOT}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_ckpt "${RESUME_CKPT}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --total_steps "${TOTAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --prediction_max_length "${PREDICTION_MAX_LENGTH}" \
  --seed "${SEED}"

echo "Stage-I MOFMeld pretraining completed."