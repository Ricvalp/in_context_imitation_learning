#!/usr/bin/env bash
set -euo pipefail

python -m reworked_diffusion_policy.eval_rlbench \
    --checkpoint ./checkpoints/diffusion_policy_latest.pt \
    --tasks play_jenga \
    --variations 0 \
    --episodes 1 \
    --max-steps 100 \
    --use-ema "$@"
