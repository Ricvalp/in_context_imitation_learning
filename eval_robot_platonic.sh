#!/usr/bin/env bash
set -euo pipefail

export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-/home/riccardo/Documents/Robotics/misc/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}

xvfb-run --auto-servernum --server-args="-screen 0 1280x720x24" \
python -m reworked_diffusion_policy.eval_rlbench \
    --checkpoint ./checkpoints/platonic_diffusion_policy_cooh4txe_epoch0049_step018750_metric0.005244.pt \
    --config reworked_diffusion_policy/platonic_config.py \
    --tasks close_drawer \
    --variations 0 \
    --episodes 10 \
    --max-steps 300 \
    --device cuda \
    --renderer opengl \
    --use-ema \
    "$@"
