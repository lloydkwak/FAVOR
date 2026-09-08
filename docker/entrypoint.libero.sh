#!/usr/bin/env bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate libero_dp
# LIBERO is bind-mounted rather than pip-installed, so put it (and our own
# docker/ helpers) on the path here instead of baking it into the image.
export PYTHONPATH="/workspace/docker:/workspace/LIBERO:${PYTHONPATH}"
# git refuses to operate on the bind-mounted repos otherwise (root in
# container vs host uid), which breaks wandb's repo probing at startup.
git config --global --add safe.directory /workspace 2>/dev/null || true
git config --global --add safe.directory /workspace/diffusion_policy 2>/dev/null || true
git config --global --add safe.directory /workspace/LIBERO 2>/dev/null || true
exec "$@"
