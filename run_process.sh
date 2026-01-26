#!/usr/bin/env bash
set -euo pipefail

RUN_SH="utils/process.sh"
MEM_DIR="./mem_profiles"
SAMPLE_RATES="0.1 0.3 0.5 0.7 0.9"


# LF-AmazonTitles-131K
bash "${RUN_SH}" \
  "LF-AmazonTitles-131K" \
  "32" \
  "bert-base-uncased" \
  "${SAMPLE_RATES}" \
  "sentence-transformers/msmarco-distilbert-base-v4" \
  "512" \
  "${MEM_DIR}"

# LF-Amazon-131K
bash "${RUN_SH}" \
  "LF-Amazon-131K" \
  "128" \
  "bert-base-uncased" \
  "${SAMPLE_RATES}" \
  "sentence-transformers/msmarco-distilbert-base-v4" \
  "512" \
  "${MEM_DIR}"

# AmazonTitles-670K
bash "${RUN_SH}" \
  "AmazonTitles-670K" \
  "32" \
  "roberta-base" \
  "${SAMPLE_RATES}" \
  "sentence-transformers/all-roberta-large-v1" \
  "256" \
  "${MEM_DIR}"
