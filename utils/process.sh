#!/usr/bin/env bash
set -euo pipefail

data_name="${1:?Missing <data_name> argument}"
max_length="${2:?Missing <max_length> argument}"
tokenizer_type="${3:?Missing <tokenizer_type> argument}"

sample_rates="${4:?Missing <sample_rates> argument (space-separated list, e.g. '0.1 0.05')}"
TF_MODEL="${5:?Missing <tf_model> argument}"
BATCH_SIZE="${6:?Missing <batch_size> argument}"
MEM_DIR="${7:?Missing <mem_dir> argument}"
mkdir -p "${MEM_DIR}"

raw_dir="Datasets/${data_name}/raw"
trn_json="${raw_dir}/trn.json"
tst_json="${raw_dir}/tst.json"
lbl_json="${raw_dir}/lbl.json"


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
  echo "[mem] tf=${tf} maxlen=${maxlen} batch=${batch_size}"

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
    --mem-profile-steps 1 \
    --mem-profile-out "${mem_csv}" \
    --default-impl
}

for rate in ${sample_rates}; do
  out_dir="Datasets/${data_name}-${rate}"
  name="${data_name}-${rate}"
  mem_csv="${MEM_DIR}/mem_${data_name}_${rate}.csv"

  # tokenize_label_flag=()
  # if [[ -f "${lbl_json}" ]]; then
  #   tokenize_label_flag+=(--tokenize-label-texts)
  #   cp "Datasets/${data_name}/trn_filter_labels.txt" "${out_dir}/trn_filter_labels.txt"
  #   cp "Datasets/${data_name}/tst_filter_labels.txt" "${out_dir}/tst_filter_labels.txt"
  #   cp "Datasets/${data_name}/Y.txt" "${out_dir}/Y.txt"
  # fi

  echo "============================================================"
  echo "[prep] data_name=${data_name} rate=${rate}"
  echo "[prep] out_dir=${out_dir}"
  echo "[prep] max_length=${max_length} tokenizer_type=${tokenizer_type}"

  # 1) Sample + clean training split
  python utils/process.py \
    --input "${trn_json}" \
    --out-dir "${out_dir}/" \
    --prefix trn \
    --sample-rate "${rate}" \
    --drop-empty-instances

  # 2) Sample + clean test split
  python utils/process.py \
    --input "${tst_json}" \
    --out-dir "${out_dir}/" \
    --prefix tst \
    --sample-rate "${rate}" \
    --drop-empty-instances

  # 3) Tokenize with provided max_length + tokenizer_type
  python utils/CreateTokenizedFiles.py \
    --data-dir "${out_dir}" \
    --max-length "${max_length}" \
    --tokenizer-type "${tokenizer_type}" \
  #   "${tokenize_label_flag[@]}"

  # final_dir="${out_dir}"
  # # 3.5) Optional: create augmented data if lbl.json exists
  # if [[ -f "${lbl_json}" ]]; then
  #   python utils/CreateAugData.py \
  #     --data-dir "${out_dir}" \
  #     --tokenization-folder "${tokenizer_type}-${max_length}" \
  #     --max-len "${max_length}"
  #   final_dir="${out_dir}-Aug"
  # fi

  # 4) Measure peak CUDA memory on the tokenized dataset
  run_one \
    "${name}" \
    "${out_dir}" \
    "${TF_MODEL}" \
    "${max_length}" \
    "${BATCH_SIZE}" \
    "${mem_csv}"
done
