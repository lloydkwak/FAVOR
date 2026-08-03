#!/usr/bin/env bash
# `verify` runs the Phase-0 DoD check; anything else runs as-is inside robodiff.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate robodiff
if [ "$1" = "verify" ]; then
    exec python /opt/verify_env.py
fi
exec "$@"
