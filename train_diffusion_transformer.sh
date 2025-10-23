#!/usr/bin/env bash

set -euo pipefail

PYTHONPATH="$(pwd)" \
python -m scripts.train_diffusion_transformer \
  --dataset_path /mnt/external_storage/robotics/rlbench/temporal/ \
  --task close_drawer --task press_switch --task close_fridge \
  --device cuda \
  --batch_size 32 \
  --epochs 200 \
  --config.horizon=16 \
  --checkpoint_interval=10
  # --config.debug.limit_dataset=True \
  # --config.debug.max_samples_per_task=32 \

