#!/usr/bin/env bash
set -euo pipefail

python -m reworked_diffusion_policy.eval_rlbench \
    --checkpoint ./checkpoints/diffusion_policy_fdr2pbsd_latest.pt \
    --tasks close_drawer \
    --variations 0 \
    --episodes 20 \
    --max-steps 100 \
    --use-ema "$@"
