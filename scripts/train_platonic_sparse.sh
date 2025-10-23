#!/usr/bin/env bash

set -euo pipefail

PYTHONPATH="$(pwd)" \
python -m reworked_diffusion_policy.train_platonic_transformer \
  --dataset_path /mnt/external_storage/robotics/rlbench/temporal/ \
  --task close_drawer --task press_switch --task close_fridge \
  --device cuda \
  --batch_size 16 \
  --epochs 100 \
  --config.horizon=16 \
  --dataset_mode=sparse \
  --sparse_max_points=10000 \
  --debug_dataset \
  --debug_max_samples=24

