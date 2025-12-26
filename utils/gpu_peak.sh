#!/usr/bin/env bash
set -e

# Measure peak CUDA memory for a small number of initial steps in epoch 0.

STEPS="${STEPS:-1}"
MEM_DIR="${MEM_DIR:-./mem_profiles}"  # MOD: output directory for per-dataset CSVs
mkdir -p "${MEM_DIR}"

run_one() {
  local name="$1"
  local data_dir="$2"
  local tf="$3"
  local maxlen="$4"
  local batch_size="$5"
  local mem_csv="$6"

  echo "============================================================"
  echo "[mem] dataset=${name}"
  echo "[mem] mem_csv=${mem_csv}"
  echo "[mem] tf=${tf} maxlen=${maxlen} batch=${batch_size} steps=${STEPS}"

  python main.py \
    --epochs 1 \
    --batch-size "${batch_size}" \
    --data-dir "${data_dir}" \
    --maxlen "${maxlen}" \
    --tf "${tf}" \
    --noloss \
    --compile \
    --expname gpu_ram \
    --mem-profile \
    --mem-profile-steps "${STEPS}" \
    --mem-profile-out "${mem_csv}"
}


# AmazonTitles-670K
mem_csv_at670k="${MEM_DIR}/mem_AmazonTitles-670K.csv"  # MOD: unique CSV per dataset
run_one \
  "AmazonTitles-670K" \
  "Datasets/AmazonTitles-670K" \
  "sentence-transformers/all-roberta-large-v1" \
  "32" \
  "256" \
  "${mem_csv_at670k}"

# Amazon-670K
# mem_csv_a670k="${MEM_DIR}/mem_Amazon-670K.csv"
# run_one \
#   "Amazon-670K" \
#   "Datasets/Amazon-670K" \
#   "sentence-transformers/all-roberta-large-v1" \
#   "512" \
#   "256" \
#   "${mem_csv_a670k}"

# LF-AmazonTitles-131K
mem_csv_lfat131k="${MEM_DIR}/mem_LF-AmazonTitles-131K.csv"
run_one \
  "LF-AmazonTitles-131K" \
  "Datasets/LF-AmazonTitles-131K-Aug" \
  "sentence-transformers/msmarco-distilbert-base-v4" \
  "32" \
  "512" \
  "${mem_csv_lfat131k}"

# LF-Amazon-131K
mem_csv_lfa131k="${MEM_DIR}/mem_LF-Amazon-131K.csv"
run_one \
  "LF-Amazon-131K" \
  "Datasets/LF-Amazon-131K-Aug" \
  "sentence-transformers/msmarco-distilbert-base-v4" \
  "128" \
  "512" \
  "${mem_csv_lfa131k}"

echo "============================================================"
echo "Done. CSVs written under: ${MEM_DIR}"
