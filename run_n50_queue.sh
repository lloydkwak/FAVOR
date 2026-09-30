#!/usr/bin/env bash
cd ~/favor_project
DC="docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c"
for t in milk bowl_stove alphabet_soup bowl_ramekin; do
  echo "=== [$(date '+%m-%d %H:%M')] n50 $t ==="
  $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
       cd /workspace/diffusion_policy && python -u /workspace/docker/run_confirm_n50_prio.py $t" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^RESULT|^SKIP|^DONE|Error" || echo "!!! FAILED $t"
done
echo "N50_QUEUE_DONE"
