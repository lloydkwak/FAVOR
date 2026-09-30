#!/usr/bin/env bash
# usage: ./run_ms_queue.sh <wait_tmux_session|none> <tag:spec> ...   spec = locked | range:<lvl>
set -uo pipefail
cd ~/favor_project
W=$1; shift
if [ "$W" != "none" ]; then
  echo "waiting for tmux session '$W' to finish ..."
  while tmux has-session -t "$W" 2>/dev/null; do sleep 300; done
fi
DC="docker compose -f docker/docker-compose.libero.yml run --rm -v $HOME/favor_project/analysis_out:/workspace/analysis_out libero bash -c"
for item in "$@"; do
  tag=${item%%:*}; a=$(echo ${item#*:} | tr ':' ' ')
  for t in alphabet_soup milk bowl_ramekin bowl_stove; do
    echo "=== [$(date '+%m-%d %H:%M')] ms $tag $t $a ==="
    $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
         cd /workspace/diffusion_policy && python -u /workspace/docker/run_libero_fault_sweep_ms.py $tag $t $a" 2>&1 \
      | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^RESULT|^SKIP|^DONE|Error|Traceback" | grep --line-buffered -v BrokenPipe \
      || echo "!!! FAILED: $tag $t $a"
  done
done
echo "MS_QUEUE_DONE"
