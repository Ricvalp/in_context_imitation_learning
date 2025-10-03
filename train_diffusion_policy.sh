#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root to keep relative imports working.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DEFAULT_DATASET="/mnt/external_storage/robotics/rlbench/temporal/close_drawer/42ac11cdf83fe09eb455ebc06db2c6176f6e7666.h5"
DATASET_PATH="${1:-$DEFAULT_DATASET}"
if [[ $# -gt 0 ]]; then
  shift
fi

python -m reworked_diffusion_policy.train \
  --dataset_path "${DATASET_PATH}" \
  --task close_drawer \
  --task open_drawer \
  --task close_fridge \
  --device cuda \
  --batch_size 64 \
  --epochs 300 \
  --config.horizon=16 \
  "$@"
