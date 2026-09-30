#!/usr/bin/env bash
# usage: ./run_x_queue.sh <method> <spec...>   spec = "locked" or "range:<lvl>"
set -uo pipefail
cd "$(dirname "$0")"
M=$1; shift
DC="docker compose -f docker/docker-compose.libero.yml run --rm -v $PWD/analysis_out:/workspace/analysis_out libero bash -c"
for spec in "$@"; do
  for t in alphabet_soup milk bowl_ramekin bowl_stove; do
    a=$(echo $spec | tr ':' ' ')
    echo "=== [$(date '+%m-%d %H:%M')] $M $t $a ==="
    $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
         cd /workspace/diffusion_policy && python -u /workspace/docker/run_libero_fault_sweep_x.py $M $t $a" 2>&1 \
      | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^RESULT|^SKIP|^DONE|Error|Traceback" | grep --line-buffered -v BrokenPipe \
      || echo "!!! FAILED: $M $t $a"
  done
done
echo "X_QUEUE_DONE $M"
