#!/usr/bin/env bash
set -euo pipefail

python -m reworked_diffusion_policy.eval_rlbench \
    --checkpoint ./checkpoints/diffusion_policy_k7lvy6uv_latest.pt \
    --tasks setup_checkers \
    --no-wandb \
    --no-headless \
    --variations 0 \
    --episodes 10 \
    --max-steps 300 \
    --use-ema "$@"
