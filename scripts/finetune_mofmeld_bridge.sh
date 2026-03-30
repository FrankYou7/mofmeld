#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Stage-II fine-tuning for MOFMeld
# This script preserves the full training workflow used in
# the manuscript and is intended for full reproduction,
# not for lightweight demo execution.
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# -------------------------
# Paths
# -------------------------
LLAMA_PATH="${PROJECT_ROOT}/checkpoints/MOFLLaMA"
BRIDGE_CKPT="${PROJECT_ROOT}/checkpoints/pretrain_result.pt"
EMBED_ROOT="${PROJECT_ROOT}/data/embeddings/qmof_hmof_embeddings"
DATA_PATH="${PROJECT_ROOT}/data/mofmeld_finetune/finetune_qa.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/mofmeld_finetune"

# -------------------------
# Paper training settings
# -------------------------
NPROC_PER_NODE=4
BATCH_SIZE=2
ACCUM_STEPS=16
TOTAL_STEPS=250500
SAVE_EVERY=50000
LR=2e-4
WEIGHT_DECAY=0.05
WARMUP_STEPS=500
MAX_LENGTH=256
NUM_WORKERS=4
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Launching Stage-II MOFMeld fine-tuning"
echo "Project root   : ${PROJECT_ROOT}"
echo "LLAMA_PATH     : ${LLAMA_PATH}"
echo "BRIDGE_CKPT    : ${BRIDGE_CKPT}"
echo "EMBED_ROOT     : ${EMBED_ROOT}"
echo "DATA_PATH      : ${DATA_PATH}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "GPUs           : ${NPROC_PER_NODE}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Accum steps    : ${ACCUM_STEPS}"
echo "Total steps    : ${TOTAL_STEPS}"
echo "Save every     : ${SAVE_EVERY}"
echo "Learning rate  : ${LR}"
echo "Weight decay   : ${WEIGHT_DECAY}"
echo "Warmup steps   : ${WARMUP_STEPS}"
echo "Max length     : ${MAX_LENGTH}"
echo "Num workers    : ${NUM_WORKERS}"
echo "Seed           : ${SEED}"
echo "=================================================="

if [[ ! -d "${LLAMA_PATH}" ]]; then
  echo "[ERROR] MOFLLaMA directory not found: ${LLAMA_PATH}"
  echo "Please place the MOFLLaMA checkpoint under checkpoints/."
  exit 1
fi

if [[ ! -f "${BRIDGE_CKPT}" ]]; then
  echo "[ERROR] Pretrained bridge checkpoint not found: ${BRIDGE_CKPT}"
  echo "Please place the stage-I pretrained bridge checkpoint under checkpoints/."
  exit 1
fi

if [[ ! -d "${EMBED_ROOT}" ]]; then
  echo "[ERROR] Embedding root not found: ${EMBED_ROOT}"
  echo "Please prepare the stage-II embedding files under the expected path."
  exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[ERROR] Fine-tuning JSONL not found: ${DATA_PATH}"
  echo "Please prepare the stage-II fine-tuning data under data/mofmeld_finetune/."
  exit 1
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  src/mofmeld/training/finetune_ddp.py \
  --llama_path "${LLAMA_PATH}" \
  --bridge_ckpt "${BRIDGE_CKPT}" \
  --embed_root "${EMBED_ROOT}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --total_steps "${TOTAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}"

echo "Stage-II MOFMeld fine-tuning completed."