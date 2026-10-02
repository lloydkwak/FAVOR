#!/usr/bin/env bash
# Qualitative figures + supplementary video from existing sweep seeds (no new experiments).
#   1 select   scenarios from results/libero_fault_sweep_locked_*  -> results/qual/scenarios.json
#   2 run      re-run the selected episodes with rendering on, check against the sweep JSONs
#   3 snaps    initial scene of every task (setup figure)
#   4 compose  paper/figs/fig_setup, fig_qual_*, paper/video/*.mp4, results/qual/compose_report.md
# usage: ./run_qual_media.sh [steps...]   (default: select run snaps compose)
#        METHODS=b1,pos,pose,prio,rg ./run_qual_media.sh run   to add RG-DDPM to the video
set -uo pipefail
cd "$(dirname "$0")"
STEPS=${*:-select run snaps compose}
METHODS=${METHODS:-b1,pos,pose,prio}
DC="docker compose -f docker/docker-compose.libero.yml run --rm libero bash -c"
for s in $STEPS; do
  echo "=== [$(date '+%m-%d %H:%M')] $s ==="
  case $s in
    select)  $DC "cd /workspace && python scripts_paper/select_qual_scenarios.py --results results --out results/qual/scenarios.json" ;;
    run)     $DC "pip install pytorch_kinematics --break-system-packages -q 2>&1 | tail -1
                  cd /workspace/diffusion_policy && python -u /workspace/docker/render_qualitative.py run --methods $METHODS" 2>&1 \
               | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^RERUN|^SKIP|^DONE|render_recorder|Error|Traceback" ;;
    snaps)   $DC "cd /workspace/diffusion_policy && python -u /workspace/docker/render_qualitative.py snapshots" 2>&1 \
               | stdbuf -oL tr '\r' '\n' | grep --line-buffered -aE "^SNAP|^DONE|Error|Traceback" ;;
    compose) $DC "pip install -q matplotlib pillow av --break-system-packages 2>&1 | tail -1
                  cd /workspace && python scripts_paper/compose_qualitative.py --qual results/qual --out paper" ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
# files written in the container are root-owned
$DC "chown -R $(id -u):$(id -g) /workspace/paper /workspace/results/qual" 2>/dev/null
echo "QUAL_DONE"
