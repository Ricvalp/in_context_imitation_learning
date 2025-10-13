#!/usr/bin/env bash
set -euo pipefail

python -m reworked_diffusion_policy.eval_rlbench \
    --checkpoint ./checkpoints/platonic_diffusion_policy_cooh4txe_epoch0049_step018750_metric0.005244.pt \
    --config reworked_diffusion_policy/platonic_config.py \
    --tasks close_drawer \
    --variations 0 \
    --episodes 10 \
    --max-steps 300 \
    --device cuda \
    --renderer opengl3 \
    --no-headless \
    --use-ema \
    "$@"
