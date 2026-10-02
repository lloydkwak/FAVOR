#!/usr/bin/env bash
cd "$(dirname "$0")"
# (no wait: GPU slot free)
DC="docker compose -f docker/docker-compose.libero.yml run --rm -e MS_JOINTS=1,5 libero bash -c"
echo "=== [$(date '+%m-%d %H:%M')] sanity: lock offset 0.0 (alphabet_soup j1, j5; exact = 0.75, 0.80) ==="
$DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
     cd /workspace/diffusion_policy && python -u /workspace/docker/run_libero_fault_sweep_ms.py lop000 alphabet_soup locked" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^RESULT|Error"
./run_ms_queue.sh none lop005:locked lop002:locked
