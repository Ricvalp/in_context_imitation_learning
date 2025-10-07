#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root to keep relative imports working.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DEFAULT_DATASET="/mnt/external_storage/robotics/rlbench/temporal/close_drawer/8d7bf9a9c0f3ba8074dfc82b0d01c251fd566099.h5"
DATASET_PATH="${1:-$DEFAULT_DATASET}"
if [[ $# -gt 0 ]]; then
  shift
fi

python -m reworked_diffusion_policy.train \
  --dataset_path "${DATASET_PATH}" \
  --task close_drawer \
  --device cuda \
  --batch_size 64 \
  --epochs 300 \
  --config.horizon=16 \
  "$@"

sudo docker run --rm -it --gpus all \
  -v /home/riccardo/Documents/Robotics/in_context_imitation_learning:/workspace \
  -v /mnt/external_storage/robotics/rlbench:/mnt/external_storage/robotics/rlbench \
  -w /workspace \
  rlbench bash