#!/usr/bin/env bash
set -euo pipefail

python -m reworked_diffusion_policy.eval_rlbench \
    --checkpoint ./checkpoints/diffusion_policy_k3veppvb_latest.pt \
    --tasks close_drawer \
    --variations 0 \
    --episodes 10 \
    --max-steps 300 \
    --use-ema "$@"
