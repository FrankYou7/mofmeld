#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# MOFLLaMA supervised fine-tuning
# This script preserves the full training workflow used in
# the manuscript and is intended for full reproduction,
# not for lightweight demo execution.
#
# Before running:
# 1. Prepare the base LLaMA instruct model under:
#    checkpoints/base_llama_3_1_8b_instruct
# 2. Prepare the MOFLLaMA training dataset under:
#    data/mofllama/mofllama_train_dataset.jsonl
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# -------------------------
# Paths
# -------------------------
MODEL_PATH="${PROJECT_ROOT}/checkpoints/base_llama_3_1_8b_instruct"
DATA_PATH="${PROJECT_ROOT}/data/mofllama/mofllama_train_dataset.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/mofllama_sft"

# -------------------------
# Paper training settings
# -------------------------
MAX_LENGTH=2048
TRAIN_SPLIT_RATIO=0.99
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=2e-5
LOGGING_STEPS=10
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=2
WARMUP_RATIO=0.03
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Launching MOFLLaMA supervised fine-tuning"
echo "Project root                  : ${PROJECT_ROOT}"
echo "MODEL_PATH                    : ${MODEL_PATH}"
echo "DATA_PATH                     : ${DATA_PATH}"
echo "OUTPUT_DIR                    : ${OUTPUT_DIR}"
echo "MAX_LENGTH                    : ${MAX_LENGTH}"
echo "TRAIN_SPLIT_RATIO             : ${TRAIN_SPLIT_RATIO}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE   : ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS   : ${GRADIENT_ACCUMULATION_STEPS}"
echo "NUM_TRAIN_EPOCHS              : ${NUM_TRAIN_EPOCHS}"
echo "LEARNING_RATE                 : ${LEARNING_RATE}"
echo "LOGGING_STEPS                 : ${LOGGING_STEPS}"
echo "SAVE_STEPS                    : ${SAVE_STEPS}"
echo "SAVE_TOTAL_LIMIT              : ${SAVE_TOTAL_LIMIT}"
echo "WARMUP_RATIO                  : ${WARMUP_RATIO}"
echo "SEED                          : ${SEED}"
echo "=================================================="

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Base model directory not found: ${MODEL_PATH}"
  echo "Please prepare the base LLaMA instruct model under checkpoints/."
  exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[ERROR] Training dataset not found: ${DATA_PATH}"
  echo "Please place the MOFLLaMA training dataset under data/mofllama/."
  exit 1
fi

python src/mofllama/training/train_mofllama.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --train_split_ratio "${TRAIN_SPLIT_RATIO}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --seed "${SEED}"

echo "MOFLLaMA fine-tuning completed."