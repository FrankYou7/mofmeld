#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Interactive MOFMeld demo
# This script launches a single-sample interactive property
# prediction demo using precomputed demo embeddings.
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# -------------------------
# Demo paths
# -------------------------
LLAMA_PATH="${PROJECT_ROOT}/checkpoints/MOFLLaMA"
BRIDGE_CKPT="${PROJECT_ROOT}/checkpoints/finetune_result.pt"
CIF_DIR="${PROJECT_ROOT}/data_demo/mofmeld/sample_cifs"
EMBEDDING_DIR="${PROJECT_ROOT}/data_demo/mofmeld/sample_embeddings"

# -------------------------
# Inference settings
# -------------------------
DEVICE="cuda"
MAX_LENGTH=256
MAX_NEW_TOKENS=64

echo "=================================================="
echo "Launching MOFMeld interactive demo"
echo "Project root    : ${PROJECT_ROOT}"
echo "LLAMA_PATH      : ${LLAMA_PATH}"
echo "BRIDGE_CKPT     : ${BRIDGE_CKPT}"
echo "CIF_DIR         : ${CIF_DIR}"
echo "EMBEDDING_DIR   : ${EMBEDDING_DIR}"
echo "Device          : ${DEVICE}"
echo "Max length      : ${MAX_LENGTH}"
echo "Max new tokens  : ${MAX_NEW_TOKENS}"
echo "=================================================="

python -m src/mofmeld/inference/run_property_prediction_demo.py \
  --llama_path "${LLAMA_PATH}" \
  --bridge_ckpt "${BRIDGE_CKPT}" \
  --cif_dir "${CIF_DIR}" \
  --embedding_dir "${EMBEDDING_DIR}" \
  --device "${DEVICE}" \
  --max_length "${MAX_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}"
