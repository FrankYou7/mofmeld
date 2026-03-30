#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Interactive MOFLLaMA + KG demo
# This script launches a small retrieval-augmented demo
# using a demo FAISS store and citation metadata.
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# -------------------------
# Demo paths
# -------------------------
MODEL_PATH="${PROJECT_ROOT}/checkpoints/MOFLLaMA"
EMBEDDING_MODEL_PATH="${PROJECT_ROOT}/checkpoints/instructor_xl"
VECTOR_STORE_PATH="${PROJECT_ROOT}/data_demo/mofllama/retrieval_store/embedding"
CITATION_CSV="${PROJECT_ROOT}/data_demo/mofllama/metadata/citations_metadata.csv"

# -------------------------
# Inference settings
# -------------------------
DEVICE="cuda"
EMBED_DEVICE="cpu"
TOP_K=5
MAX_NEW_TOKENS=512

echo "=================================================="
echo "Launching MOFLLaMA + KG interactive demo"
echo "Project root         : ${PROJECT_ROOT}"
echo "MODEL_PATH           : ${MODEL_PATH}"
echo "EMBEDDING_MODEL_PATH : ${EMBEDDING_MODEL_PATH}"
echo "VECTOR_STORE_PATH    : ${VECTOR_STORE_PATH}"
echo "CITATION_CSV         : ${CITATION_CSV}"
echo "Device               : ${DEVICE}"
echo "Embed device         : ${EMBED_DEVICE}"
echo "Top-k                : ${TOP_K}"
echo "Max new tokens       : ${MAX_NEW_TOKENS}"
echo "=================================================="

python src/mofllama/inference/run_kg_grounded_inference_demo.py \
  --model_path "${MODEL_PATH}" \
  --embedding_model_path "${EMBEDDING_MODEL_PATH}" \
  --vector_store_path "${VECTOR_STORE_PATH}" \
  --citation_csv "${CITATION_CSV}" \
  --device "${DEVICE}" \
  --embed_device "${EMBED_DEVICE}" \
  --top_k "${TOP_K}" \
  --max_new_tokens "${MAX_NEW_TOKENS}"