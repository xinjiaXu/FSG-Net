#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_ids="${GPU_IDS:-0}"
num_workers="${NUM_WORKERS:-8}"
batch_size="${BATCH_SIZE:-32}"
epochs="${EPOCHS:-10}"
learning_rate="${LR:-0.001}"
input_size="${INPUT_SIZE:-256}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset_root="$data_root/LEVIR_CD"; image_a="A"; image_b="B"; label_dir="label" ;;
  levircdplus) dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a="A"; image_b="B"; label_dir="label" ;;
  cdd) dataset_root="$data_root/CDD"; image_a="A"; image_b="B"; label_dir="OUT" ;;
  whucd) dataset_root="$data_root/WHUCD"; image_a="A"; image_b="B"; label_dir="label" ;;
  sysucd) dataset_root="$data_root/SYSU_CD"; image_a="time1"; image_b="time2"; label_dir="label" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

for split in train val test; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\nRun prepare_server_data.sh first or set DATA_ROOT.\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

cd "$repo_root"
python train.py \
  --dataset-dir "$dataset_root" \
  --dataset-name "$1" \
  --output-dir "$repo_root/FSG-Net_work_dirs" \
  --image-dir-a "$image_a" \
  --image-dir-b "$image_b" \
  --label-dir "$label_dir" \
  --cuda "$gpu_ids" \
  --num-workers "$num_workers" \
  --batch-size "$batch_size" \
  --epochs "$epochs" \
  --learning-rate "$learning_rate" \
  --input-size "$input_size"
