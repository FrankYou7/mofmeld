#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Extract 64-dimensional CHGNet structure embeddings from CIF files
# This script is intended for full preprocessing used in the
# manuscript workflow, not for lightweight demo execution.
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# -------------------------
# Paths
# -------------------------
CIF_DIR="${CIF_DIR:-${PROJECT_ROOT}/data/cifs/train_cif}"
JSON_DIR="${JSON_DIR:-${PROJECT_ROOT}/data/jsons}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/data/embeddings/train_all_embeddings}"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/outputs/logs/failed_train_all_embedding.txt}"

mkdir -p "${SAVE_DIR}"
mkdir -p "$(dirname "${LOG_PATH}")"

echo "=================================================="
echo "Extracting CHGNet structure embeddings"
echo "Project root : ${PROJECT_ROOT}"
echo "CIF_DIR      : ${CIF_DIR}"
echo "JSON_DIR     : ${JSON_DIR}"
echo "SAVE_DIR     : ${SAVE_DIR}"
echo "LOG_PATH     : ${LOG_PATH}"
echo "=================================================="

if [[ ! -d "${CIF_DIR}" ]]; then
  echo "[ERROR] CIF directory not found: ${CIF_DIR}"
  exit 1
fi

if [[ ! -d "${JSON_DIR}" ]]; then
  echo "[ERROR] JSON directory not found: ${JSON_DIR}"
  exit 1
fi

python src/mofmeld/embeddings/extract_chgnet_embeddings.py \
  --cif_dir "${CIF_DIR}" \
  --json_dir "${JSON_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --log_path "${LOG_PATH}"

echo "CHGNet embedding extraction completed."