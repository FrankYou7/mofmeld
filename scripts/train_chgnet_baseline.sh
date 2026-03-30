#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Train a CHGNet baseline for a selected MOF property
#
# Before running this script, prepare the required baseline
# training data under:
#
#   data/chgnet_hmof/
#
# Expected structure:
#   data/chgnet_hmof/
#   ├── labels.json
#   ├── train/cifs/
#   ├── val/cifs/
#   └── test/cifs/
#
# Example usage:
#   bash scripts/train_chgnet_baseline.sh pld
#   bash scripts/train_chgnet_baseline.sh lcd
#   bash scripts/train_chgnet_baseline.sh surface_area
#   bash scripts/train_chgnet_baseline.sh void_fraction
#   bash scripts/train_chgnet_baseline.sh co2_0p01bar
#   bash scripts/train_chgnet_baseline.sh co2_2p5bar
#
# Optional:
#   DEVICE=cpu bash scripts/train_chgnet_baseline.sh pld
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET="${1:-pld}"

BASE_DIR="${PROJECT_ROOT}/data/chgnet_hmof"
LABELS_PATH="${BASE_DIR}/labels.json"
SAVE_DIR="${PROJECT_ROOT}/outputs/chgnet/${TARGET}"

BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=1e-3
NUM_WORKERS=0
DEVICE="${DEVICE:-cuda}"

mkdir -p "${SAVE_DIR}"

echo "=================================================="
echo "Launching CHGNet baseline training"
echo "Project root   : ${PROJECT_ROOT}"
echo "Target         : ${TARGET}"
echo "BASE_DIR       : ${BASE_DIR}"
echo "LABELS_PATH    : ${LABELS_PATH}"
echo "SAVE_DIR       : ${SAVE_DIR}"
echo "BATCH_SIZE     : ${BATCH_SIZE}"
echo "EPOCHS         : ${EPOCHS}"
echo "LEARNING_RATE  : ${LEARNING_RATE}"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "DEVICE         : ${DEVICE}"
echo "=================================================="

if [[ ! -f "${LABELS_PATH}" ]]; then
  echo "[ERROR] labels.json not found: ${LABELS_PATH}"
  echo "Please prepare the CHGNet baseline data under data/chgnet_hmof/."
  exit 1
fi

for split in train val test; do
  if [[ ! -d "${BASE_DIR}/${split}/cifs" ]]; then
    echo "[ERROR] Missing CIF directory: ${BASE_DIR}/${split}/cifs"
    echo "Please make sure the CHGNet baseline data is placed correctly."
    exit 1
  fi
done

python src/baselines/chgnet/train_chgnet_baseline.py \
  --base_dir "${BASE_DIR}" \
  --labels_path "${LABELS_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --target "${TARGET}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

echo "CHGNet baseline training completed."