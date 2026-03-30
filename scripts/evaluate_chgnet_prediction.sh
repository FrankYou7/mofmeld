#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Filter a CHGNet prediction CSV by a MOF subset list and
# compute MAD / RMSE / R2
#
# Before running this script:
# 1. Prepare a prediction CSV (e.g. from your own CHGNet run
#    or from released prediction files)
# 2. Prepare a MOF subset file:
#    - .txt with one MOF name per line, or
#    - .csv with a column named 'mof_name'
#
# Example usage:
#   bash scripts/evaluate_chgnet_prediction.sh \
#     prediction/prediction_co2_max.csv \
#     data/mofmeld_metadata/test_hmof_2769.txt \
#     outputs/chgnet_eval/co2_2p5bar_filtered.csv
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PRED_CSV="${1:-}"
MOF_LIST="${2:-}"
OUT_CSV="${3:-}"

FILTER_BAD_PRED=false
PRED_ABS_MAX=5.0

if [[ -z "${PRED_CSV}" || -z "${MOF_LIST}" || -z "${OUT_CSV}" ]]; then
  echo "Usage:"
  echo "  bash scripts/evaluate_chgnet_prediction.sh <pred_csv> <mof_list> <out_csv>"
  echo ""
  echo "mof_list supports:"
  echo "  - .txt file with one MOF name per line"
  echo "  - .csv file with a column named 'mof_name'"
  echo ""
  echo "Example:"
  echo "  bash scripts/evaluate_chgnet_prediction.sh \\"
  echo "    prediction/prediction_co2_max.csv \\"
  echo "    data/mofmeld_metadata/test_hmof_2769.txt \\"
  echo "    outputs/chgnet_eval/co2_2p5bar_filtered.csv"
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"

echo "=================================================="
echo "Running CHGNet prediction evaluation"
echo "Project root      : ${PROJECT_ROOT}"
echo "PRED_CSV          : ${PRED_CSV}"
echo "MOF_LIST          : ${MOF_LIST}"
echo "OUT_CSV           : ${OUT_CSV}"
echo "FILTER_BAD_PRED   : ${FILTER_BAD_PRED}"
echo "PRED_ABS_MAX      : ${PRED_ABS_MAX}"
echo "=================================================="

if [[ ! -f "${PRED_CSV}" ]]; then
  echo "[ERROR] Prediction CSV not found: ${PRED_CSV}"
  exit 1
fi

if [[ ! -f "${MOF_LIST}" ]]; then
  echo "[ERROR] MOF list file not found: ${MOF_LIST}"
  echo "Please provide either:"
  echo "  - a .txt file with one MOF name per line, or"
  echo "  - a .csv file with a column named 'mof_name'."
  exit 1
fi

CMD=(
  python src/baselines/chgnet/evaluate_prediction.py
  --pred_csv "${PRED_CSV}"
  --mof_list "${MOF_LIST}"
  --out_csv "${OUT_CSV}"
)

if [[ "${FILTER_BAD_PRED}" == "true" ]]; then
  CMD+=(--filter_bad_pred --pred_abs_max "${PRED_ABS_MAX}")
fi

"${CMD[@]}"

echo "CHGNet prediction evaluation completed."